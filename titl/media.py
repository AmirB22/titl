"""ffprobe and ffmpeg helpers."""
from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SR = 16000

LANG_TAGS = {
    "en": {"en", "eng"},
    "sr": {"sr", "srp", "scc", "scr", "hbs"},
    "bs": {"bs", "bos", "hbs"},
}


class MediaError(RuntimeError):
    pass


@dataclass
class AudioStream:
    index: int
    channels: int
    layout: str
    language: str
    codec: str
    title: str


def _run(cmd: list[str]) -> subprocess.CompletedProcess:
    p = subprocess.run(cmd, capture_output=True)
    if p.returncode != 0:
        raise MediaError(p.stderr.decode("utf-8", "replace").strip()[-500:] or "ffmpeg failed")
    return p


def probe_audio(path: Path | str) -> tuple[list[AudioStream], float]:
    p = _run(["ffprobe", "-v", "error", "-print_format", "json", "-show_streams", "-show_format", str(path)])
    d = json.loads(p.stdout)
    out = []
    for s in d.get("streams", []):
        if s.get("codec_type") != "audio":
            continue
        tags = {k.lower(): v for k, v in (s.get("tags") or {}).items()}
        out.append(AudioStream(s["index"], int(s.get("channels", 2)), s.get("channel_layout", ""),
                               tags.get("language", "").lower(), s.get("codec_name", ""), tags.get("title", "")))
    dur = float(d.get("format", {}).get("duration") or 0)
    return out, dur


def pick_track(streams: list[AudioStream], language: str, forced: int | None = None) -> AudioStream:
    """--audio-track N is the Nth audio stream (0 based). Otherwise prefer the stream tagged with the language."""
    if not streams:
        raise MediaError("no audio streams in this file")
    if forced is not None:
        if not 0 <= forced < len(streams):
            raise MediaError(f"--audio-track {forced} out of range, file has {len(streams)} audio streams")
        return streams[forced]
    tags = LANG_TAGS.get(language.lower()[:2], {language.lower()})
    for s in streams:
        if s.language in tags:
            return s
    return streams[0]


def audio_filter(s: AudioStream) -> str:
    """Dialogue lives in the centre channel of 5.1/7.1 mixes; a plain downmix buries it under music and effects."""
    if s.channels >= 6:
        return "pan=mono|c0=0.8*FC+0.2*FL+0.2*FR"
    return "pan=mono|c0=0.5*c0+0.5*c1" if s.channels == 2 else "anull"


def extract_pcm(path: Path | str, s: AudioStream, out: Path) -> None:
    tmp = out.with_suffix(".part")
    _run(["ffmpeg", "-v", "error", "-y", "-i", str(path), "-map", f"0:{s.index}", "-vn", "-af", audio_filter(s),
          "-ar", str(SR), "-ac", "1", "-f", "s16le", str(tmp)])
    tmp.replace(out)


def open_pcm(path: Path) -> np.memmap:
    return np.memmap(path, dtype=np.int16, mode="r")


def to_float(x: np.ndarray) -> np.ndarray:
    return np.asarray(x, dtype=np.float32) / 32768.0


def chunk_bounds(pcm: np.ndarray, chunk_s: float = 600.0, window_s: float = 15.0) -> list[int]:
    """Sample indices for chunk edges. Each edge sits at the quietest 50 ms inside +-window_s of the nominal cut."""
    step, w = int(chunk_s * SR), int(window_s * SR)
    n = len(pcm)
    cuts = [0]
    while cuts[-1] + step + w < n:
        a = cuts[-1] + step
        seg = to_float(pcm[a - w: a + w])
        win = SR // 20
        e = np.convolve(seg ** 2, np.ones(win) / win, "same")
        e[:win] = e[-win:] = np.inf
        cuts.append(a - w + int(np.argmin(e)))
    cuts.append(n)
    return cuts
