#!/usr/bin/env bash
# The paper's Ours (latent MCE, src/experiment/lens_feedback.py) on one dataset, with the same code, rows, model, judge
# and budget as that dataset's MCE row: original MCE (mce_upstream) with the environment <DATASET>_lens, whose training
# records carry the fixed Oracle-lens reading of each training rollout's input WITH the context it used
# (LENS_STATE=context, default; plain = the no-context reading reused). Servers during the whole run:
#   answers: the dataset's vLLM server (GEN_PORT);
#   lens:    serve27b_lens.py, READ_LAYERS=36,44,52,60 (QWEN36_PATH / OLENS_ADAPTER / OLENS_SRC / LENS_MAX_MEMORY on another
#            server), LENS_URLS=http://127.0.0.1:<port>/run[,...].
# Step 1 (once): STEP=precompute -> <TRAIN stem>_lens.jsonl (the meta agent's training rows with the no-context reading).
# Usage: DATASET=<cancer_myth|hotpotqa|hover|medcalc_cot> TRAIN=... VAL=... TEST=... GEN_PORT=... LENS_URLS=... TAG=...
#        [ITER=5] [TL=...] [VL=...] [BS=50] [STEP=precompute|loop] experiments/latent_mce/run_ours.sh
# Dataset defaults (the MCE rows): cancer_myth  TRAIN=results/data/mce_cancer/fold<k>_train.jsonl VAL=..._val TEST=..._test TL=100 VL=96
#   hotpotqa / hover: mce_repro_qwen3 gepa_splits, TL=150 VL=100;  medcalc_cot: mce_medcalc_cot results/data, TL=220 VL=110.
set -uo pipefail
R=$(cd "$(dirname "$0")/../.." && pwd); U=$R/experiments/latent_mce/mce_upstream; X=$R/experiments/latent_mce/src/experiment
PY=${PY:-/data2/heejae/uv/mce/bin/python}
: "${DATASET:?}" "${TRAIN:?}" "${LENS_URLS:?}"; export LENS_URLS LENS_STATE=${LENS_STATE:-context}
cd $U
if [ "${STEP:-loop}" = precompute ]; then $PY $X/lens_feedback.py precompute $DATASET $TRAIN; exit; fi
: "${VAL:?}" "${TEST:?}" "${GEN_PORT:?}" "${TAG:?}"
T=$(dirname $TRAIN)/$(basename $TRAIN .jsonl)_lens.jsonl; [ -f $T ] || { echo "missing $T (STEP=precompute first)"; exit 1; }
W=$(dirname $TRAIN)/ws_ours_$TAG; LOG=$R/experiments/latent_mce/results/logs/ours_$TAG; mkdir -p $LOG
export MCE_GENERATOR_BACKEND=chat GENERATOR_API_KEY=local MCE_GENERATOR_TIMEOUT=3600 GENERATOR_API_BASE=http://127.0.0.1:$GEN_PORT/v1
$PY $X/run_mce_latent.py main --workspace $W --env ${DATASET}_lens --train-data $T --val-data $VAL --iterations ${ITER:-5} \
  --train-limit ${TL:?} --val-limit ${VL:?} --train-batch-size ${BS:-50} --model qwen36-27b --log-dir $LOG > $LOG/main.log 2>&1
d=$(grep -a -o "Best iteration: iter[0-9]*" $LOG/main.log | tail -1 | grep -oE "iter[0-9]+"); b=$(ls -d $W/${d}_sub* 2>/dev/null | sort -V | tail -1)
[ -n "$b" ] && $PY $X/run_mce_latent.py eval --iter_dir "$b" --env $DATASET --data $TEST --limit 100000 --model qwen36-27b \
  --save-results-to $(dirname $TRAIN)/test_ours_$TAG.json > $LOG/test.log 2>&1
echo "ours $DATASET $TAG done $(date -u +%H:%M)"
