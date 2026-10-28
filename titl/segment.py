"""Words with timings to subtitle cues."""
from __future__ import annotations

import re
from dataclasses import dataclass

from .lang import Profile

END_PUNCT = re.compile(r"[.!?…]+[\"')\]]*$")
ABBREVIATIONS = frozenset("mr mrs ms dr prof st jr sr mt gen col sgt capt lt cmdr gov sen rep rev hon inc ltd co corp vs".split())
_DOTTED = re.compile(r"^(?:[A-Za-z]\.){2,}$")


def is_sentence_end(w: str) -> bool:
    if not END_PUNCT.search(w):
        return False
    bare = w.strip("\"')]")
    if _DOTTED.match(bare):
        return False
    return not (bare.endswith(".") and not bare.endswith("..") and bare[:-1].lower() in ABBREVIATIONS)
SOFT_PUNCT = re.compile(r"[,;:\-–—]+[\"')\]]*$")


@dataclass
class Word:
    w: str
    start: float
    end: float


@dataclass
class Cue:
    start: float
    end: float
    lines: list[str]

    @property
    def text(self) -> str:
        return "\n".join(self.lines)


def _bare(w: str) -> str:
    return re.sub(r"[^\w]", "", w.lower())


def _break_cost(words: list[Word], i: int, p: Profile) -> float:
    if i >= len(words) - 1:
        return 0.0
    w = words[i].w
    if is_sentence_end(w):
        return 0.0
    if SOFT_PUNCT.search(w):
        return 1.0
    cost = 4.0
    if _bare(w) in p.no_end or (w.endswith(".") and w[:-1].lower() in ABBREVIATIONS):
        cost += 6.0
    if _bare(words[i + 1].w) in p.conj:
        cost -= 2.0
    return cost


def split_units(words: list[Word], pause: float = 0.9) -> list[list[Word]]:
    units: list[list[Word]] = []
    cur: list[Word] = []
    for k, w in enumerate(words):
        cur.append(w)
        nxt = words[k + 1] if k + 1 < len(words) else None
        if nxt is None or is_sentence_end(w.w) or nxt.start - w.end > pause:
            units.append(cur)
            cur = []
    return units


def _chars(words: list[Word]) -> int:
    return sum(len(w.w) for w in words) + max(0, len(words) - 1)


def _cue_cost(words: list[Word], i: int, j: int, p: Profile, target: int) -> float | None:
    seg = words[i: j + 1]
    n = _chars(seg)
    if n > p.max_chars * p.max_lines:
        return None
    if n > p.max_chars and _best_split(seg, p) is None:  # fits by count but has no legal line break
        return None
    dur = seg[-1].end - seg[0].start
    if dur > p.max_dur:
        return None
    cost = _break_cost(words, j, p) + 1.5
    cost += abs(n - target) / 6
    if n < 20 and len(words) > j + 1:
        cost += (20 - n) / 3
    if dur < p.min_dur * 0.6:
        cost += 3.0
    cps = n / (dur + p.linger)  # no min_dur floor: a short cue cannot really borrow time from the next one
    if cps > p.max_cps:
        cost += (cps - p.max_cps) * 1.5
    return cost


def split_cues_dp(unit: list[Word], p: Profile) -> list[list[Word]]:
    n = len(unit)
    target = int(p.max_chars * 1.2)
    best = [float("inf")] * (n + 1)
    back = [0] * (n + 1)
    best[0] = 0.0
    for j in range(n):
        for i in range(j, -1, -1):
            if _chars(unit[i: j + 1]) > p.max_chars * p.max_lines + 1:
                break
            c = _cue_cost(unit, i, j, p, target)
            if c is None:
                continue
            if best[i] + c < best[j + 1]:
                best[j + 1], back[j + 1] = best[i] + c, i
    if best[n] == float("inf"):
        return [[w] for w in unit]
    out, j = [], n
    while j > 0:
        i = back[j]
        out.append(unit[i:j])
        j = i
    return out[::-1]


def _best_split(words: list[Word], p: Profile) -> list[str] | None:
    best, best_cost = None, float("inf")
    for k in range(len(words) - 1):
        a, b = " ".join(w.w for w in words[: k + 1]), " ".join(w.w for w in words[k + 1:])
        if len(a) > p.max_chars or len(b) > p.max_chars:
            continue
        cost = _break_cost(words, k, p) * 2 + abs(len(a) - len(b)) / 2.5 + (0.5 if len(a) > len(b) else 0)
        if cost < best_cost:
            best, best_cost = [a, b], cost
    return best


def break_lines(words: list[Word], p: Profile) -> list[str]:
    text = " ".join(w.w for w in words)
    if len(text) <= p.max_chars or p.max_lines < 2:
        return [text]
    return _best_split(words, p) or [text]


def build_cues(words: list[Word], p: Profile) -> list[Cue]:
    cues: list[Cue] = []
    for unit in split_units(words, p.pause):
        for grp in split_cues_dp(unit, p):
            cues.append(Cue(grp[0].start, grp[-1].end + p.linger, break_lines(grp, p)))
    return fix_timing(cues, p)


def fix_timing(cues: list[Cue], p: Profile) -> list[Cue]:
    """Minimum duration, no overlaps, minimum gap. Starts stay where the speech starts; only ends move."""
    for k, c in enumerate(cues):
        nxt = cues[k + 1].start if k + 1 < len(cues) else float("inf")
        c.end = max(c.end, c.start + p.min_dur)
        c.end = min(c.end, c.start + p.max_dur, nxt - p.min_gap)
        if c.end <= c.start:
            c.end = c.start + 0.2
            if k + 1 < len(cues) and cues[k + 1].start < c.end + p.min_gap:
                cues[k + 1].start = c.end + p.min_gap
    return cues
