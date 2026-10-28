"""Drops text Whisper invents over silence and music."""
from __future__ import annotations

import re

BLOCKLIST = [
    "thanks for watching", "thank you for watching", "please subscribe", "subtitles by",
    "subtitles by the amaraorg community", "transcribed by", "translated by", "like and subscribe",
    "see you in the next video", "titlovao", "titlovi", "prevod", "hvala na gledanju", "hvala sto ste gledali",
    "hvala vam na paznji", "pretplatite se", "podrska titlovi", "preveo", "amaraorg",
]
# real dialogue most of the time: only dropped when Whisper itself is unsure there was speech
SOFT_BLOCKLIST = ["thank you", "thanks", "bye", "hvala", "hvala vam", "hvala lijepo"]
_BLOCK = {norm_ for norm_ in (re.sub(r"\s+", " ", re.sub(r"[^\w ]|_", " ", b)).strip() for b in BLOCKLIST)}
_SOFT = set(SOFT_BLOCKLIST)

_FOLD = str.maketrans({"č": "c", "ć": "c", "š": "s", "ž": "z", "đ": "d", "Č": "c", "Ć": "c", "Š": "s", "Ž": "z", "Đ": "d"})


def norm(text: str) -> str:
    t = text.lower().translate(_FOLD)
    return re.sub(r"\s+", " ", re.sub(r"[^\w ]|_", " ", t)).strip()


def is_hallucination(seg: dict) -> bool:
    t = norm(seg["text"])
    if not t:
        return True
    if t in _BLOCK:
        return True
    if t in _SOFT and (seg.get("no_speech_prob", 0) > 0.3 or seg.get("avg_logprob", 0) < -0.8):
        return True
    if seg.get("compression_ratio", 0) > 2.4:
        return True
    if seg.get("no_speech_prob", 0) > 0.6 and seg.get("avg_logprob", 0) < -1.0:
        return True
    return False


def drop_repeats(segs: list[dict], limit: int = 2) -> list[dict]:
    """The same segment text more than `limit` times in a row is a decoding loop; keep the first `limit`."""
    out: list[dict] = []
    run = 0
    for s in segs:
        if out and norm(s["text"]) == norm(out[-1]["text"]):
            run += 1
            if run >= limit:
                continue
        else:
            run = 0
        out.append(s)
    return out


def clean_segments(segs: list[dict]) -> list[dict]:
    return drop_repeats([s for s in segs if not is_hallucination(s)])
