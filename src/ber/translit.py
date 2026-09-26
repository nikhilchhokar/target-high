"""Rule-based Indic-script -> Latin transliteration (stdlib only, no external data).

About 1 in 5 Indian records writes the business name in Devanagari, Gujarati, Tamil, ...
("राम मार्केटिंग प्राइवेट लिमिटेड" = "Ram Marketing Private Limited"). All Indic Unicode
blocks share the ISCII layout, so one offset table covers every script. Output is rough
phonetic Latin ("raam maarketing praaivet limited" -> vowel runs collapsed); the consonant
skeleton in normalize.py then makes it comparable with the English spelling.
"""

# offset within a 128-codepoint Indic block -> Latin
_CONS = {
    0x15: "k", 0x16: "kh", 0x17: "g", 0x18: "gh", 0x19: "n", 0x1A: "ch", 0x1B: "chh",
    0x1C: "j", 0x1D: "jh", 0x1E: "n", 0x1F: "t", 0x20: "th", 0x21: "d", 0x22: "dh",
    0x23: "n", 0x24: "t", 0x25: "th", 0x26: "d", 0x27: "dh", 0x28: "n", 0x29: "n",
    0x2A: "p", 0x2B: "ph", 0x2C: "b", 0x2D: "bh", 0x2E: "m", 0x2F: "y", 0x30: "r",
    0x31: "r", 0x32: "l", 0x33: "l", 0x34: "l", 0x35: "v", 0x36: "sh", 0x37: "sh",
    0x38: "s", 0x39: "h",
    0x58: "q", 0x59: "kh", 0x5A: "g", 0x5B: "z", 0x5C: "r", 0x5D: "rh", 0x5E: "f", 0x5F: "y",
}
_VOWEL = {  # independent vowels
    0x04: "a", 0x05: "a", 0x06: "aa", 0x07: "i", 0x08: "ii", 0x09: "u", 0x0A: "uu",
    0x0B: "ri", 0x0C: "li", 0x0D: "e", 0x0E: "e", 0x0F: "e", 0x10: "ai", 0x11: "o",
    0x12: "o", 0x13: "o", 0x14: "au", 0x60: "ri", 0x61: "li",
}
_MATRA = {  # dependent vowel signs (replace the inherent 'a')
    0x3E: "aa", 0x3F: "i", 0x40: "ii", 0x41: "u", 0x42: "uu", 0x43: "ri", 0x44: "ri",
    0x45: "e", 0x46: "e", 0x47: "e", 0x48: "ai", 0x49: "o", 0x4A: "o", 0x4B: "o",
    0x4C: "au", 0x62: "li", 0x63: "li",
}
_NASAL = {0x01: "n", 0x02: "n", 0x03: "h"}
_VIRAMA = 0x4D
_IGNORE = {0x3C, 0x3D, 0x00, 0x51, 0x52, 0x70, 0x71}  # nukta, avagraha, misc marks
_START, _END = 0x0900, 0x0DFF


def _is_indic(ch: str) -> bool:
    return _START <= ord(ch) <= _END


def has_indic(s: str) -> bool:
    if s.isascii():
        return False
    return any(_START <= ord(c) <= _END for c in s)


def indic_to_latin(s: str) -> str:
    """Transliterate Indic characters; everything else passes through unchanged."""
    if not has_indic(s):
        return s
    # zero-width (non-)joiners sit INSIDE words (e.g. Telugu 'infra'); they must not split words
    s = s.replace(chr(0x200C), "").replace(chr(0x200D), "")
    out = []
    pending = False  # a consonant waiting for its inherent vowel

    def flush(next_is_word_char: bool):
        nonlocal pending
        if pending and next_is_word_char:
            out.append("a")
        pending = False

    for ch in s:
        if not _is_indic(ch):
            flush(False)  # schwa deletion at the end of a word
            out.append(ch)
            continue
        off = ord(ch) & 0x7F
        if 0x66 <= off <= 0x6F:  # native digits
            flush(False)
            out.append(str(off - 0x66))
        elif off in _CONS:
            flush(True)
            out.append(_CONS[off])
            pending = True
        elif off in _MATRA:
            pending = False
            out.append(_MATRA[off])
        elif off == _VIRAMA:
            pending = False
        elif off in _NASAL:
            flush(True)
            out.append(_NASAL[off])
        elif off in _VOWEL:
            pending = False  # a vowel after a consonant replaces its inherent 'a'
            out.append(_VOWEL[off])
        elif off in _IGNORE:
            continue
        else:
            flush(False)
            out.append(" ")
    flush(False)
    txt = "".join(out)
    for a, b in (("aa", "a"), ("ii", "i"), ("uu", "u"), ("ee", "i"), ("oo", "u")):
        txt = txt.replace(a, b)
    return txt
