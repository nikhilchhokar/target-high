#!/usr/bin/env bash
# Full chain for the final submission (stops on first failure). ~10-12 h on 8 cores / 16 GB.
#   1. pruner (trained on its own S1 sample; also normalises all records, cached)
#   2. e011: training experiment + test scoring (its test scores give the test ambiguity profile)
#   3. test-density weights from e011 (train vs test ambiguity buckets)
#   4. e012: e011 retrained / re-thresholded with those weights -> output/
set -e
PY=.venv/Scripts/python
export PYTHONIOENCODING=utf-8
W=$($PY -c "from ber import pipeline; print(pipeline.load_config('configs/exp/e012_density.yaml')['model']['weight_file'])")
echo "[chain] pruner $(date +%H:%M)"
$PY -u -m ber.train_pruner configs/exp/e008_india.yaml
echo "[chain] e011 $(date +%H:%M)"
$PY -u -m ber.run_experiment configs/exp/e011_hop.yaml
$PY -u -m ber.make_submission configs/exp/e011_hop.yaml --note "e011"
echo "[chain] density weights $(date +%H:%M)"
$PY -u tools/density_weights.py --config configs/exp/e011_hop.yaml --out "$W"
echo "[chain] e012 $(date +%H:%M)"
$PY -u -m ber.run_experiment configs/exp/e012_density.yaml
$PY -u -m ber.make_submission configs/exp/e012_density.yaml --note "e012"
echo "[chain] ALL DONE $(date +%H:%M)"
