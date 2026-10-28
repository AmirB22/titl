"""Check ffmpeg and run a real GPU inference."""
from __future__ import annotations

import shutil
import subprocess
import tempfile
import time
from pathlib import Path


def _line(ok: bool, name: str, detail: str) -> bool:
    print(f"[{'ok' if ok else 'FAIL'}] {name}: {detail}")
    return ok


def run(model: str = "large-v3") -> int:
    good = True
    for tool in ("ffmpeg", "ffprobe"):
        good &= _line(bool(shutil.which(tool)), tool, shutil.which(tool) or "not found in PATH")

    from .gpu import preload_cuda_libs
    preload_cuda_libs()
    try:
        import ctranslate2
        n = ctranslate2.get_cuda_device_count()
    except Exception as e:
        return int(not _line(False, "ctranslate2", str(e)))
    good &= _line(n > 0, "cuda devices", str(n))
    if n == 0:
        print("no CUDA device: titl will fall back to the CPU (much slower)")
        return int(not good)

    with tempfile.TemporaryDirectory() as d:
        wav = Path(d) / "t.wav"
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "sine=f=220:d=2",
                        "-ar", "16000", "-ac", "1", str(wav)], check=True)
        try:
            from faster_whisper import WhisperModel
            t0 = time.time()
            m = WhisperModel(model, device="cuda", compute_type="int8_float16")
            t1 = time.time()
            import numpy as np
            raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(wav), "-f", "f32le", "-ac", "1",
                                  "-ar", "16000", "-"], capture_output=True, check=True).stdout
            segs, _ = m.transcribe(np.frombuffer(raw, np.float32), language="en")
            list(segs)
            _line(True, f"gpu inference ({model})", f"load {t1 - t0:.1f}s, run {time.time() - t1:.1f}s")
        except Exception as e:
            good = _line(False, "gpu inference", f"{type(e).__name__}: {e}") and good
            print("hint: pip install nvidia-cublas-cu12 'nvidia-cudnn-cu12==9.*'")
    return int(not good)
