# Business Entity Resolution — Amazon ML Challenge 2026

For every Source 1 business, find the Source 2 / Source 3 records that are the same real-world
business (possibly none). Pipeline: normalise → multi-retriever blocking → prune →
pair + context features → LightGBM (out-of-fold) → decision layer tuned on exact macro F0.5.

Uses only the provided data. No external databases, APIs, geocoding or internet data; the
normalisation dictionaries in `src/ber/dictionaries.py` are hand-written. The final model is
LightGBM (MIT licence).

## Setup (Windows / Linux / macOS, Python 3.12)

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt   # Linux/macOS: .venv/bin/python
.venv/Scripts/python -m pip install -e .
```

## Data

Place the official dataset so that these paths exist (or change `paths.data_dir` in `configs/base.yaml`):

```
data/train/train_source1.tsv  train_source2.tsv  train_source3.tsv  train_ground_truth.tsv
data/test/test_source1.tsv    test_source2.tsv   test_source3.tsv
```

Set `paths.validator` in `configs/base.yaml` to the official `utils/validate_submission.py`.

## Reproduce the final submission end to end

```bash
python -m ber.make_submission configs/exp/<final_exp>.yaml
```

This runs the train experiment (thresholds + iteration count from out-of-fold predictions),
fits the final models on all training pairs, builds test candidates and features, predicts,
writes `output/matching_results.tsv` and `output/candidate_pairs.tsv`, self-checks both files,
runs the official validator, and records the run in `submissions/registry.csv`.

## Commands

| Command | What it does |
| --- | --- |
| `python -m ber.audit configs/base.yaml` | Data facts: singleton rate, match shape, one-to-one check, country mix, postcode coverage → `reports/audit.json` |
| `python -m ber.run_experiment configs/exp/e001_baseline.yaml` | Full train experiment → `results/<exp_id>/` + a row in `results/leaderboard.csv` |
| `python -m ber.run_experiment <cfg> --blocking-only` | Blocking recall / ceiling / candidate size only (seconds) |
| `python -m ber.compare e002 e001` | Paired per-entity delta ± SE, metric diffs, error-bucket diffs |
| `python -m ber.make_submission <cfg>` | Test predictions, both TSVs, validator, registry |
| `python -m ber.make_submission <cfg> --empty` | All-empty probe: its public score = public singleton rate |
| `python tools/make_synthetic.py` | Synthetic data in the exact format (`data_synth/`) for testing code; scores on it mean nothing |

## Experiment workflow

1. Copy a config in `configs/exp/`, set `inherit: ../base.yaml`, a new `exp_id`, its `parent`,
   a one-line `note`, and change **one** thing.
2. Run it; read the summary line and the error-bucket table (ranked by metric lost).
3. `python -m ber.compare <new> <parent>` — keep the change only if z is clearly above 2.
4. Upload only for a real gain (≥ 0.5 pt paired) or a question CV cannot answer.

Stages are cached under `cache/` by a hash of their config, the input files and the stage
source code, so a threshold or model change re-uses normalised data, candidates and features.

## Layout

```
src/ber/
  io.py              safe TSV read/write (tab, no NA conversion, no quoting)
  dictionaries.py    legal suffixes, abbreviations, states (US / India / France + generic)
  normalize.py       name views (core, legal, DBA, skeleton) and address views
                     (core, postcode, house numbers, unit, landmark)
  blocking.py        TF-IDF index, top-k and key retrievers, union with provenance, pruning
  features.py        pairwise name/address/number features + context (rank, gap, reverse rank)
  model.py           LightGBM, GroupKFold by S1 id, out-of-fold predictions
  decide.py          one-to-one resolution, t_first / t_other / t_extra, grid search
  metrics.py         exact macro F0.5 replica, breakdowns, blocking metrics, calibration
  errors.py          entity error buckets and side-by-side FP/FN pair dumps
  pipeline.py        config loading (inherit), cached split building
  run_experiment.py  train experiment + logging;   compare.py  paired comparison
  make_submission.py test outputs, checks, validator, registry;   audit.py  data audit
configs/             base.yaml + exp/*.yaml (one change each)
tools/               make_synthetic.py
```

## Key design decisions

- **Macro F0.5 per entity.** A wrong match on a singleton and an empty prediction on a matched
  entity both score 0, so the first match uses a lower threshold (`t_first`) than extra matches
  (`t_extra`, break-even ≈ 0.73). Thresholds are tuned on out-of-fold predictions with the exact
  metric, taking the centre of a flat region.
- **Candidate-set size is judged.** Retrievers are unioned, then pruned to `max_k` with a score
  floor; `candidate_pairs.tsv` is exactly what the model scores, and every match is a candidate.
- **France is unseen in training.** No country one-hot or country-specific features; country is
  an open set of labels; leave-one-country-out scores are logged as a proxy.
- **S1 is deduplicated.** If the audit confirms each S2/S3 id appears in at most one ground-truth
  list, each S2/S3 record keeps only its best S1 (`decision.one_to_one`).
