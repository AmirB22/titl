from __future__ import annotations

from .segment import Cue


def ts(t: float) -> str:
    ms = max(0, round(t * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def to_srt(cues: list[Cue]) -> str:
    return "".join(f"{i}\n{ts(c.start)} --> {ts(c.end)}\n{c.text}\n\n" for i, c in enumerate(cues, 1))
