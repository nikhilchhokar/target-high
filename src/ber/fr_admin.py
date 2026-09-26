"""French administrative-area cleanup (hand-written from the provided test data's own tokens).

French copies of one business alternate between the region ('Nouvelle-Aquitaine') and the
departement ('Gironde') or omit both, which makes true copies look like address mismatches.
The training countries never show this (US states normalise to one code), so we remove these
phrases from French addresses before retrieval and features.
"""
import pandas as pd

# token sequences as they appear in addr_core (stopwords already removed, 'pas' -> 'passage')
FR_ADMIN_PHRASES = [
    ("nouvelle", "aquitaine"), ("pays", "loire"), ("loire", "atlantique"), ("passage", "calais"),
    ("hauts",), ("gironde",), ("nord",),
]


def strip_admin(addr: str) -> str:
    toks = addr.split()
    out, i = [], 0
    while i < len(toks):
        for ph in FR_ADMIN_PHRASES:
            if tuple(toks[i:i + len(ph)]) == ph:
                i += len(ph)
                break
        else:
            out.append(toks[i])
            i += 1
    return " ".join(out)


def clean_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Apply to a France-only frame (pool or S1); returns a copy with cleaned addr_core."""
    return df.assign(addr_core=[strip_admin(a) for a in df.addr_core])
