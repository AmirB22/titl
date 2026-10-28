from __future__ import annotations

from dataclasses import dataclass

from .lang import Profile
from .segment import Cue
from .srt import ts


@dataclass
class Issue:
    index: int
    time: str
    kind: str
    detail: str


def check(cues: list[Cue], p: Profile) -> list[Issue]:
    out: list[Issue] = []
    for i, c in enumerate(cues, 1):
        dur = c.end - c.start
        chars = len(c.text.replace("\n", ""))
        t = ts(c.start)
        if dur > 0 and chars / dur > p.max_cps + 0.01:
            out.append(Issue(i, t, "cps", f"{chars / dur:.1f} chars/s (max {p.max_cps:g})"))
        for ln in c.lines:
            if len(ln) > p.max_chars:
                out.append(Issue(i, t, "line-length", f"{len(ln)} chars (max {p.max_chars})"))
        if len(c.lines) > p.max_lines:
            out.append(Issue(i, t, "lines", f"{len(c.lines)} lines"))
        if dur < p.min_dur - 0.01:
            out.append(Issue(i, t, "short", f"{dur:.2f}s (min {p.min_dur:.2f})"))
        if dur > p.max_dur + 0.01:
            out.append(Issue(i, t, "long", f"{dur:.2f}s (max {p.max_dur:g})"))
        if i < len(cues):
            gap = cues[i].start - c.end
            if gap < -0.001:
                out.append(Issue(i, t, "overlap", f"{-gap:.2f}s over next cue"))
            elif gap < p.min_gap - 0.001:
                out.append(Issue(i, t, "gap", f"{gap * 1000:.0f} ms to next cue"))
    return out


def report(cues: list[Cue], issues: list[Issue]) -> str:
    kinds: dict[str, int] = {}
    for x in issues:
        kinds[x.kind] = kinds.get(x.kind, 0) + 1
    head = f"{len(cues)} cues, {len(issues)} issues" + (": " + ", ".join(f"{k} {v}" for k, v in sorted(kinds.items())) if issues else "")
    return head + "".join(f"\n  #{x.index} {x.time} {x.kind}: {x.detail}" for x in issues[:50])
