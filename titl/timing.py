"""Clamp stretched first and last word timings relative to the segment's median word duration,
after stable-ts clamp_max."""
from __future__ import annotations

import statistics

from .segment import Word


def clamp_segment(words: list[Word], factor: float = 2.5, abs_max: float = 2.0) -> list[Word]:
    if not words:
        return words
    durs = [w.end - w.start for w in words]
    limit = abs_max
    if len(words) > 1:
        limit = min(abs_max, factor * statistics.median(durs))
    else:
        limit = min(abs_max, 0.35 + 0.08 * len(words[0].w))
    limit = max(limit, 0.12)
    last = words[-1]
    if last.end - last.start > limit:
        last.end = last.start + limit
    first = words[0]
    if first is not last and first.end - first.start > limit:
        first.start = first.end - limit
    return words
