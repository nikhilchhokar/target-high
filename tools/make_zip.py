"""Build the final submission package in the organisers' layout.

    python tools/make_zip.py --team <team_name> --sub submissions/sub_<...>

<team_name>_submission.zip
├── output/{matching_results.tsv, candidate_pairs.tsv}      (from the chosen submission dir)
├── code/business_entity_resolution/{src/, configs/, tools/, README.md, requirements.txt, pyproject.toml}
└── Documentation_template.md
"""
import argparse
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CODE_ITEMS = ["src", "configs", "tools", "README.md", "requirements.txt", "pyproject.toml",
              "run_e010.ps1", "run_after.ps1"]
SKIP_PARTS = {"__pycache__", ".egg-info"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--team", required=True)
    ap.add_argument("--sub", required=True, help="submission folder holding both TSVs")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    sub = Path(a.sub)
    for f in ("matching_results.tsv", "candidate_pairs.tsv"):
        assert (sub / f).is_file(), f"missing {sub / f}"
    out = Path(a.out or ROOT.parent / f"{a.team}_submission.zip")
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for f in ("matching_results.tsv", "candidate_pairs.tsv"):
            z.write(sub / f, f"output/{f}")
        for item in CODE_ITEMS:
            p = ROOT / item
            if not p.exists():
                continue
            files = [p] if p.is_file() else [q for q in p.rglob("*") if q.is_file()]
            for q in files:
                if any(part in SKIP_PARTS or part.endswith(".egg-info") for part in q.parts):
                    continue
                z.write(q, "code/business_entity_resolution/" + q.relative_to(ROOT).as_posix())
        z.write(ROOT / "Documentation_template.md", "Documentation_template.md")
    print(f"wrote {out} ({out.stat().st_size / 1e6:.0f} MB)")
    with zipfile.ZipFile(out) as z:
        names = z.namelist()
    print(f"{len(names)} files; top-level:", sorted({n.split('/')[0] for n in names}))


if __name__ == "__main__":
    main()
