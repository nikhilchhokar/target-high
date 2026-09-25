"""Multi-view normalisation of business names and addresses.

Every record gets several views (clean / core / skeleton / postal / numbers ...)
so blocking and features can each use the representation that suits them.
Rules are chosen by country code; unknown countries get only the common rules.
"""
import re
import unicodedata

import pandas as pd

from . import dictionaries as D

_ALL_LEGAL = set().union(*D.LEGAL_TOKENS.values())
_NON_ALNUM = re.compile(r"[^a-z0-9]+")
_FR_ELISION = re.compile(r"\b([ldjmnst])['’]")
_APOS = re.compile(r"['’`]")
_DIGITS = re.compile(r"\d+")
_VOWELS = re.compile(r"[aeiouy]")
_SKEL_MAP = [("x", "ks"), ("ph", "f"), ("sh", "s"), ("kh", "k"), ("gh", "g"), ("th", "t"),
             ("bh", "b"), ("dh", "d"), ("ck", "k"), ("q", "k"), ("w", "v"), ("z", "s")]


def strip_accents(s: str) -> str:
    s = unicodedata.normalize("NFKD", s)
    return "".join(ch for ch in s if not unicodedata.combining(ch))


def country_code(label: str) -> str:
    """Map a raw country label to 'us'/'in'/'fr', else 'other' (open set)."""
    key = " ".join(_NON_ALNUM.sub(" ", strip_accents(str(label).lower())).split())
    return D.COUNTRY_ALIASES.get(key, "other")


def block_country(label: str) -> str:
    """Blocking key for country: canonical code if known, else the cleaned label."""
    cc = country_code(label)
    if cc != "other":
        return cc
    return " ".join(_NON_ALNUM.sub(" ", strip_accents(str(label).lower())).split()) or "unknown"


def _join_single_letters(tokens):
    """'m g road' -> 'mg road', 's b i' -> 'sbi' (runs of >=2 single letters)."""
    out, run = [], []
    for t in tokens:
        if len(t) == 1 and t.isalpha():
            run.append(t)
            continue
        if run:
            out.extend(["".join(run)] if len(run) > 1 else run)
            run = []
        out.append(t)
    if run:
        out.extend(["".join(run)] if len(run) > 1 else run)
    return out


def basic_tokens(text: str, cc: str) -> list:
    """Lowercase, strip accents, handle apostrophes/&, split on non-alphanumerics."""
    s = strip_accents(text.lower())
    if cc == "fr":
        s = _FR_ELISION.sub(r"\1 ", s)
    s = _APOS.sub("", s).replace("&", " and ").replace("+", " and ")
    return _join_single_letters(_NON_ALNUM.sub(" ", s).split())


def _expand(tokens, table_by_cc, cc):
    table = dict(table_by_cc["common"])
    table.update(table_by_cc.get(cc, {}))
    return [table.get(t, t) for t in tokens]


def _strip_legal(tokens):
    toks = list(tokens)
    legal = []
    while toks and (toks[-1] in _ALL_LEGAL or toks[-1] == "and"):
        t = toks.pop()
        if t != "and":
            legal.append(t)
    while toks and toks[0] in _ALL_LEGAL:
        legal.append(toks.pop(0))
    return toks, legal


def skeleton(token: str) -> str:
    """Consonant skeleton for transliteration robustness (Laxmi~Lakshmi, Shree~Sri)."""
    t = token
    for a, b in _SKEL_MAP:
        t = t.replace(a, b)
    if not t:
        return t
    body = _VOWELS.sub("", t[1:])
    out = t[0]
    for ch in body:
        if ch != out[-1]:
            out += ch
    return out


def normalize_name(raw: str, cc: str) -> dict:
    text = strip_accents(str(raw).lower())
    parts = re.split(D.DBA_PATTERN, text, maxsplit=1)
    main, dba = parts[0], (parts[1] if len(parts) > 1 else "")

    def core_of(s):
        toks = _expand(basic_tokens(s, cc), D.NAME_ABBR, cc)
        stripped, legal = _strip_legal(toks)
        core = [t for t in stripped if t not in D.NAME_STOPWORDS] or stripped or toks
        return toks, core, legal

    toks, core, legal = core_of(main)
    _, dba_core, _ = core_of(dba) if dba.strip() else ([], [], [])
    return {
        "name_clean": " ".join(toks),
        "name_core": " ".join(core),
        "legal": " ".join(sorted(set(legal))),
        "dba_core": " ".join(dba_core),
        "name_skel": " ".join(skeleton(t) for t in core),
        "name_nums": " ".join(sorted({n.lstrip("0") or "0" for t in core for n in _DIGITS.findall(t)})),
        "name_acr": "".join(t[0] for t in core if t) if len(core) >= 2 else "",
    }


_POSTAL = {
    "in": re.compile(r"(?<!\d)(\d{3})\s?(\d{3})(?!\d)"),
    "us": re.compile(r"(?<!\d)(\d{5})(?:\s?-\s?\d{4})?(?!\d)"),
    "fr": re.compile(r"(?<!\d)(\d{2})\s?(\d{3})(?!\d)"),
    "other": re.compile(r"(?<!\d)(\d{5,6})(?!\d)"),
}
_FR_POSTBOX = re.compile(r"\b(?:bp|cs)\s*\d+|\bcedex(?:\s*\d+)?")
_LANDMARK = re.compile(D.LANDMARK_PATTERN)
_UNIT = re.compile(D.UNIT_PATTERN)


def extract_postal(text: str, cc: str):
    """Return (postal, span). A lone match at the very start is a house number, not a postcode."""
    matches = list(_POSTAL.get(cc, _POSTAL["other"]).finditer(text))
    if not matches:
        return "", None
    m = matches[-1]
    if len(matches) == 1 and m.start() < 0.3 * len(text):
        return "", None
    return "".join(g for g in m.groups() if g), m.span()


def _phrase_replace(text: str, table: dict) -> str:
    for full, short in table.items():
        if " " in full or len(full) > 3:
            text = re.sub(rf"\b{full}\b", short, text)
    return text


def normalize_address(raw: str, cc: str) -> dict:
    text = strip_accents(str(raw).lower())
    if cc == "fr":
        text = _FR_POSTBOX.sub(" ", text)
    postal, span = extract_postal(text, cc)
    if span:
        text = text[:span[0]] + " " + text[span[1]:]
    landmarks = [m.group(0) for m in _LANDMARK.finditer(text)]
    text = _LANDMARK.sub(" ", text)
    units = {u.lstrip("0") or "0" for u in _UNIT.findall(text)}
    text = _UNIT.sub(" ", text)
    if cc == "us":
        text = _phrase_replace(text, D.US_STATES)
    elif cc == "in":
        text = _phrase_replace(text, D.IN_STATES)
    toks = _expand(basic_tokens(text, cc), D.ADDR_ABBR, cc)
    if cc == "in":
        toks = [D.IN_STATES.get(t, t) for t in toks]
    nums = {n.lstrip("0") or "0" for t in toks for n in _DIGITS.findall(t) if len(n) <= 6}
    core = [t for t in toks if not any(c.isdigit() for c in t) and t not in D.ADDR_STOPWORDS]
    lm_toks = []
    for lm in landmarks:
        lm_toks += basic_tokens(lm, cc)
    return {
        "addr_clean": " ".join(toks),
        "addr_core": " ".join(core),
        "postal": postal,
        "house_nums": " ".join(sorted(nums)),
        "units": " ".join(sorted(units)),
        "landmark": " ".join(lm_toks),
    }


def normalize_records(recs: pd.DataFrame) -> pd.DataFrame:
    """Add all normalised views to a stacked records frame (from io.load_split)."""
    out = recs.copy()
    out["cc"] = [country_code(c) for c in out.country]
    out["bc"] = [block_country(c) for c in out.country]
    names = [normalize_name(n, cc) for n, cc in zip(out.business_name, out.cc)]
    addrs = [normalize_address(a, cc) for a, cc in zip(out.business_address, out.cc)]
    out = pd.concat([out.reset_index(drop=True), pd.DataFrame(names), pd.DataFrame(addrs)], axis=1)
    # fall back so no view is empty when the raw field has content
    out["name_core"] = out.name_core.where(out.name_core != "", out.name_clean)
    out["name_addr"] = (out.name_core + " " + out.addr_core).str.strip()
    return out
