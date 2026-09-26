#!/usr/bin/env bash
# e008 chain: re-normalise -> pruner -> training experiment -> test submission (stops on first failure)
set -e
PY=.venv/Scripts/python
export PYTHONIOENCODING=utf-8
echo "[chain] prep start $(date +%H:%M)"
$PY -u -c "
from ber import pipeline, prep
cfg = pipeline.load_config('configs/exp/e008_india.yaml')
prep.prep_split(cfg, 'train', workers=7)
prep.prep_split(cfg, 'test', workers=7)
"
echo "[chain] pruner start $(date +%H:%M)"
$PY -u -m ber.train_pruner configs/exp/e008_india.yaml
echo "[chain] experiment start $(date +%H:%M)"
$PY -u -m ber.run_experiment configs/exp/e008_india.yaml
echo "[chain] submission start $(date +%H:%M)"
$PY -u -m ber.make_submission configs/exp/e008_india.yaml --note "e008: India script fixes + stacking + 25 cands"
echo "[chain] ALL DONE $(date +%H:%M)"
