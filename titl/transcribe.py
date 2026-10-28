"""Chunked, resumable transcription."""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

from . import media
from .clean import clean_segments
from .gpu import preload_cuda_libs
from .lang import profile_for, to_latin
from .segment import Word
from .timing import clamp_segment


def work_key(video: Path, language: str, model: str, track: int, chunk_s: float) -> str:
    st = video.stat()
    raw = f"{video.resolve()}|{st.st_size}|{int(st.st_mtime)}|{language}|{model}|{track}|{chunk_s}|v1"
    return hashlib.sha1(raw.encode()).hexdigest()[:12]


def load_model(model: str, device: str, compute_type: str | None):
    preload_cuda_libs()
    import ctranslate2
    from faster_whisper import WhisperModel

    if device == "auto":
        device = "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
    ct = compute_type or ("int8_float16" if device == "cuda" else "int8")
    print(f"model {model} on {device} ({ct})", file=sys.stderr)
    return WhisperModel(model, device=device, compute_type=ct)


def _segments(model, audio, language: str, offset: float) -> list[dict]:
    segs, _ = model.transcribe(
        audio, language=language, word_timestamps=True, vad_filter=True, beam_size=5,
        condition_on_previous_text=False, hallucination_silence_threshold=2.0)
    out = []
    for s in segs:
        out.append({
            "text": s.text, "no_speech_prob": s.no_speech_prob, "avg_logprob": s.avg_logprob,
            "compression_ratio": s.compression_ratio,
            "words": [{"w": w.word.strip(), "start": round(w.start + offset, 3), "end": round(w.end + offset, 3),
                       "p": round(w.probability, 3)} for w in (s.words or []) if w.word.strip()],
        })
    return out


def transcribe_film(video: Path, workdir: Path, language: str, model_name: str = "large-v3", track: int | None = None,
                    device: str = "auto", compute_type: str | None = None, chunk_s: float = 600.0) -> list[Word]:
    streams, duration = media.probe_audio(video)
    stream = media.pick_track(streams, language, track)
    print(f"audio stream {stream.index} ({stream.language or '??'}, {stream.channels}ch {stream.layout})", file=sys.stderr)
    wd = workdir / work_key(video, language, model_name, stream.index, chunk_s)
    (wd / "chunks").mkdir(parents=True, exist_ok=True)

    pcm_path = wd / "audio.s16"
    if not pcm_path.exists():
        print("extracting audio...", file=sys.stderr)
        media.extract_pcm(video, stream, pcm_path)
    pcm = media.open_pcm(pcm_path)
    bounds = media.chunk_bounds(pcm, chunk_s)
    n = len(bounds) - 1
    todo = [i for i in range(n) if not (wd / "chunks" / f"{i}.json").exists()]
    if todo:
        model = load_model(model_name, device, compute_type)
        t0 = time.time()
        for done, i in enumerate(todo):
            a, b = bounds[i], bounds[i + 1]
            segs = _segments(model, media.to_float(pcm[a:b]), language, a / media.SR)
            tmp = wd / "chunks" / f"{i}.tmp"
            tmp.write_text(json.dumps(segs), encoding="utf-8")
            tmp.replace(wd / "chunks" / f"{i}.json")
            el = time.time() - t0
            eta = el / (done + 1) * (len(todo) - done - 1)
            print(f"chunk {i + 1}/{n} done, {el / 60:.1f} min elapsed, ~{eta / 60:.1f} min left", file=sys.stderr)
    else:
        print(f"all {n} chunks already transcribed, resuming from {wd}", file=sys.stderr)

    segs: list[dict] = []
    for i in range(n):
        segs += json.loads((wd / "chunks" / f"{i}.json").read_text(encoding="utf-8"))
    latin = profile_for(language).latin
    if latin:  # script first, so the cleaning step sees the same text the viewer will
        for s in segs:
            s["text"] = to_latin(s["text"])
            for w in s["words"]:
                w["w"] = to_latin(w["w"])
    kept = clean_segments(segs)
    print(f"kept {len(kept)} of {len(segs)} segments ({len(segs) - len(kept)} dropped as non-speech/hallucination)",
          file=sys.stderr)
    words = []
    for s in kept:
        seg_words = [Word(w["w"], w["start"], max(w["end"], w["start"] + 0.05)) for w in s["words"]]
        words += clamp_segment(seg_words)
    words.sort(key=lambda w: w.start)
    return words
