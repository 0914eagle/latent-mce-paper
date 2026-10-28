#!/usr/bin/env bash
# Original MCE (mce_upstream, unchanged) on Cancer-Myth, the balanced folds (data/fpq/cancer_myth_balanced_splits.json):
# MCE (MCE_LATENT=none) vs latent MCE (MCE_LATENT=lens: get_context(question, latent) and the Oracle-lens reading in the
# training records). Paper settings: 5 iterations (epochs), training batch 50; all 100 training rows (50 FPQ / 50 NFP;
# TRAIN_LIMIT), validation 96 (48 / 48), test = the fold's test (241-242), as the balanced GEPA runs.
# Test: Base (no context; shared by both arms) and the best iteration by validation accuracy of each arm.
# Servers (start before; see docs/ONBOARDING.md section 4):
#   lens (step 1 only): CUDA_VISIBLE_DEVICES=a,b READ_LAYERS=36,44,52,60 PYTHONPATH=/data2/heejae/uv/fla_pkgs \
#       .venv-fpq/bin/python experiments/latent_mce/src/experiment/serve27b_lens.py 8796 <cache dir>
#   answers: vLLM 0.18.1 Qwen3.6-27B, TP 2, --served-model-name qwen36-27b, thinking off
#       (--default-chat-template-kwargs '{"enable_thinking": false}'), max 2048 new tokens as every latent_mce run
#       (--override-generation-config '{"max_new_tokens": 2048}'); one port per fold in GEN_PORTS.
# Usage: GEN_PORTS=8802,8803,8804 [TRAIN_LIMIT=100] [FOLDS="0 1 2"] [ARMS="none lens"] experiments/latent_mce/run_mce_cancer.sh [lens]
#   "lens" runs step 1 (read every question once on LENS_URL) and exits.
set -uo pipefail
R=$(cd "$(dirname "$0")/../.." && pwd); X=$R/experiments/latent_mce; U=$X/mce_upstream; PY=/data2/heejae/uv/mce/bin/python
D=$X/results/data/mce_cancer; LOG=$X/results/logs/mce_cancer; mkdir -p $LOG
export MCE_GENERATOR_BACKEND=chat GENERATOR_API_KEY=local MCE_GENERATOR_TIMEOUT=600
if [ "${1:-}" = lens ]; then cd $R && MCE_LATENT=lens $PY -m experiments.latent_mce.src.experiment.mce_cancer_env; exit; fi
IFS=, read -ra P <<< "$GEN_PORTS"; TL=${TRAIN_LIMIT:-100}
cd $U
best() { d=$(grep -a -o "Best iteration: iter[0-9]*" $1 | tail -1 | grep -oE "iter[0-9]+"); ls -d $2/${d}_sub* | sort -V | tail -1; }
ev() { # fold port arm iter_dir tag
  MCE_LATENT=$3 GENERATOR_API_BASE=http://127.0.0.1:$2/v1 $PY $X/src/experiment/run_mce_cancer.py eval --iter_dir "$4" --env cancer_myth \
    --data $D/fold$1_test.jsonl --limit 1000 --model qwen36-27b --save-results-to $D/test_fold$1_$5.json > $LOG/test_fold$1_$5.log 2>&1; }
run() { # fold port arm
  w=$D/fold$1_$3
  MCE_LATENT=$3 GENERATOR_API_BASE=http://127.0.0.1:$2/v1 $PY $X/src/experiment/run_mce_cancer.py main --workspace $w --env cancer_myth \
    --train-data $D/fold$1_train.jsonl --val-data $D/fold$1_val.jsonl --iterations 5 --train-limit $TL --val-limit 1000 \
    --train-batch-size 50 --model qwen36-27b --log-dir $LOG/fold$1_$3 > $LOG/fold$1_$3.log 2>&1
  ev $1 $2 $3 "$(best $LOG/fold$1_$3.log $w)" $3; }
for f in ${FOLDS:-0 1 2}; do p=${P[$f % ${#P[@]}]}
  ev $f $p none "" base &
  for a in ${ARMS:-none lens}; do run $f $p $a & done
done
wait
echo "mce_cancer done $(date -u +%H:%M)"
