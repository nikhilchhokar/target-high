# Business Entity Resolution — Amazon ML Challenge 2026

For every Source 1 business, find the Source 2 / Source 3 records that are the same real-world
business (possibly none). Pipeline: parallel normalisation → per-country hashed TF-IDF retrieval
over combination tokens → learned pruner (25 candidates) → pair + context features → LightGBM →
stacked stage 2 → one-to-one resolution + thresholds tuned on the exact macro F0.5.

Uses only the provided data. No external databases, APIs, geocoding or internet data; the
normalisation dictionaries and the transliteration table are hand-written. Models: LightGBM
(MIT). Retrieval: scikit-learn HashingVectorizer + sparse_dot_topn (Apache-2.0).

## Setup (Python 3.12, Windows / Linux / macOS)

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt   # Linux/macOS: .venv/bin/python
.venv/Scripts/python -m pip install -e .
```

Hardware used: 8-core laptop CPU, 14 GB RAM. Close memory-heavy apps during full runs;
peak usage is ~8 GB.

## Data

Set the paths at the top of `configs/base.yaml`:

- `paths.data_dir` — folder containing `train/` and `test/` with the official TSV files
- `paths.work_dir`, `paths.cache_dir` — large intermediate files (~6 GB). Keep them outside
  any cloud-synced folder.
- `paths.validator` — the official `utils/validate_submission.py`

## Reproduce the final submission

```bash
python -m ber.train_pruner configs/exp/e008_india.yaml
python -m ber.make_submission configs/exp/e008_india.yaml
```

The first command normalises all records (cached by code hash, ~30 min on 7 cores) and trains
the candidate pruner on a separate S1 sample. The second runs the training experiment
(out-of-fold predictions, stage-2 stacking, threshold search), fits the final models, scores
every test S1 country by country, writes `output/matching_results.tsv` and
`output/candidate_pairs.tsv`, self-checks both, runs the official validator and records the run
in `submissions/registry.csv`. `run_e008.sh` chains all steps.

## Commands

| Command | What it does |
| --- | --- |
| `python -m ber.run_experiment <cfg>` | Training experiment → `results/<exp_id>/` (metrics, OOF scores, errors) + row in `results/leaderboard.csv` |
| `python -m ber.run_experiment <cfg> --blocking-only` | Blocking recall / ceiling / candidate size only |
| `python -m ber.train_pruner <cfg>` | Train the learned candidate pruner |
| `python -m ber.compare <new> <old>` | Paired per-entity comparison of two experiments |
| `python -m ber.make_submission <cfg>` | Test outputs, checks, validator, registry |
| `python -m ber.make_submission <cfg> --empty` | All-empty probe |
| `python tools/make_synthetic.py` | Tiny synthetic dataset in the same format, for code tests only |

## Layout

```
src/ber/
  io.py              safe TSV read/write
  translit.py        rule-based Indic-script -> Latin transliteration
  dictionaries.py    legal suffixes, abbreviations, states (US / India / France + generic)
  normalize.py       name views (core, legal, DBA, skeleton) and address views
  prep.py            parallel normalisation to parquet, cached by code hash
  retrieve.py        region detection, combination tokens, hashed TF-IDF, top-k, union, pruning
  train_pruner.py    learned pruner training
  featurize.py       pair / context / reverse-competition / sibling features (parallel)
  model.py           LightGBM, group k-fold by S1, out-of-fold predictions
  stage2.py          stacked re-scoring from each pair's neighbourhood
  decide.py          one-to-one resolution, thresholds, grid search
  metrics.py         exact macro F0.5, blocking metrics, calibration
  errors.py          error buckets and side-by-side dumps
  pipeline.py        configs (inherit), region-complete sampling, cached train build, test chunks
  run_experiment.py  training experiment + logging;  compare.py  paired comparison
  make_submission.py test outputs + checks + validator + registry
configs/             base.yaml + exp/e###.yaml (each changes one thing vs its parent)
```

## Key design decisions

- **Macro F0.5 per entity.** A false merge on a singleton and an empty prediction on a matched
  entity both score 0, so thresholds are tuned on the exact metric, not pair-level F0.5.
- **Combination tokens for retrieval.** Single common words (city, street type, state) are
  dropped for speed; word pairs and region-tagged words keep that information cheaply.
- **Candidate-set size is judged.** A learned pruner keeps 25 candidates per S1;
  `candidate_pairs.tsv` is exactly what the model scores, and every match is a candidate.
- **France is test-only.** No country one-hot; country is an open set of labels.
- **Region-complete training sample.** Whole states are sampled so each business's competitors
  are present, as at test time; test S1 are processed in region-aligned chunks.
