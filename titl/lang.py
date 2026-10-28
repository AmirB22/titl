"""Per-language subtitle rules."""
from __future__ import annotations

from dataclasses import dataclass, field, replace

_CYR = {
    "А": "A", "Б": "B", "В": "V", "Г": "G", "Д": "D", "Ђ": "Đ", "Е": "E", "Ж": "Ž", "З": "Z", "И": "I",
    "Ј": "J", "К": "K", "Л": "L", "Љ": "Lj", "М": "M", "Н": "N", "Њ": "Nj", "О": "O", "П": "P", "Р": "R",
    "С": "S", "Т": "T", "Ћ": "Ć", "У": "U", "Ф": "F", "Х": "H", "Ц": "C", "Ч": "Č", "Џ": "Dž", "Ш": "Š",
}
_CYR.update({k.lower(): v.lower() for k, v in list(_CYR.items())})


def to_latin(text: str) -> str:
    """Serbian Cyrillic to Latin. Latin text is returned unchanged. Lj/Nj/Dž become LJ/NJ/DŽ next to capitals."""
    out: list[str] = []
    for i, ch in enumerate(text):
        rep = _CYR.get(ch)
        if rep is None:
            out.append(ch)
            continue
        if ch in "ЉЊЏ":
            nxt = text[i + 1] if i + 1 < len(text) else ""
            prv = text[i - 1] if i > 0 else ""
            if (nxt.isalpha() and nxt.isupper()) or (prv.isalpha() and prv.isupper()):
                rep = rep.upper()
        out.append(rep)
    return "".join(out)


@dataclass(frozen=True)
class Profile:
    code: str
    max_chars: int = 42
    max_lines: int = 2
    max_cps: float = 17.0
    min_dur: float = 5 / 6
    max_dur: float = 7.0
    min_gap: float = 2 / 24
    linger: float = 0.12
    pause: float = 0.9
    latin: bool = False
    no_end: frozenset = field(default_factory=frozenset)
    conj: frozenset = field(default_factory=frozenset)


_EN = Profile(
    "en", max_cps=20.0,
    no_end=frozenset("a an the to of in on at for with from by and or but nor so if that than as is are was were be "
                     "my your his her its our their this these those not no".split()),
    conj=frozenset("and but or so because although while when if that which who whose where".split()),
)
_SRP = Profile(
    "sr", latin=True,
    no_end=frozenset("i a ali da u na je se su sam si smo ste sa s za od do o po iz ni ili što sto koji koja koje kao bi "
                     "će ce bih to ovo ono moj tvoj njegov njen naš vaš njihov".split()),
    conj=frozenset("i a ali da jer kad kada ako dok pa te ili nego već vec koji koja koje što sto gdje gde".split()),
)

PROFILES = {"en": _EN, "sr": _SRP, "bs": replace(_SRP, code="bs")}


def profile_for(language: str) -> Profile:
    code = language.lower()[:2]
    if code not in PROFILES:
        raise ValueError(f"no subtitle profile for '{language}' yet, supported: {', '.join(PROFILES)}")
    return PROFILES[code]
