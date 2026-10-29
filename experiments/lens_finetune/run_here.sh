#!/usr/bin/env bash
# Oracle-lens fine-tuning on Cancer-Myth (rejection-sampling SFT judged by the frozen AR), every step on this server
# (4x RTX 4090). See README.md. All GPUs must be free (stop vLLM first); each step loads Qwen3.6-27B over the 4 GPUs.
# Usage (any directory): [DATASET=cancer_myth|hotpotqa|hover|synqa2|medcalc] [FOLDS="0 1 2"] [KEEP=2] [LR=2e-5] [N_POOL=8] [N_EVAL=4] [LIMIT=0] experiments/lens_finetune/run_here.sh
#   LIMIT=N: first N questions only (with RESULTS_SUBDIR=<name> so the stored results are not touched).
# Steps are resumable: sampling and scoring skip what is done; training restarts its fold.
set -euo pipefail
source ${ENV_SH:-/data1/heejae/scripts/env.sh}  # this server's paths and venvs (env_a6000.sh on the 8x A6000 server)
R=$(cd "$(dirname "$0")/../.." && pwd); cd $R
PY=${PY:-$D/uv/hf27b/bin/python}; MCE=${MCE:-$D/uv/mce/bin/python}; SK=${SK:-$D/uv/cancer_myth_internals/bin/python}
M=experiments.lens_finetune.src
FOLDS=${FOLDS:-0 1 2}; LIM=${LIMIT:-0}; NP=${N_POOL:-8}; NE=${N_EVAL:-4}
export PYTHONPATH=$R:$OLENS_SRC${EXTRA_PYTHONPATH:+:$EXTRA_PYTHONPATH} PYTORCH_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1
OUT=$R/experiments/lens_finetune/results/${RESULTS_SUBDIR:-}/data
GPUS=${CUDA_VISIBLE_DEVICES:-$(nvidia-smi --query-gpu=index --format=csv,noheader | paste -sd,)}  # the GPUs this run uses must be free
if nvidia-smi -i $GPUS --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1 > 1000' | grep -q .; then echo "GPUs $GPUS are in use (stop vLLM)"; exit 1; fi

[ -f ${OLENS_ARTIFACTS:-$D/olens_artifacts}/hf/ckpts/ar/ptag/ar.asst.ptag.pooled.s0/ex16013184/lora_hf/adapter_model.safetensors ] || $PY -m $M.prepare.fetch_ar
DATASET=${DATASET:-cancer_myth}  # hotpotqa / hover / synqa2 / medcalc: the lens FVE table's items (prepare/questions_other.py)
if [ "$DATASET" != cancer_myth ]; then export RESULTS_SUBDIR=${RESULTS_SUBDIR:-$DATASET}; OUT=$R/experiments/lens_finetune/results/$RESULTS_SUBDIR/data; fi
if [ "$DATASET" = cancer_myth ]; then [ -f $OUT/questions.jsonl ] || PAPER_V4_DIR=$PAPER_V4_DIR $MCE -m $M.prepare.questions
elif [ "${SPLIT:-}" = native ]; then [ -f $OUT/questions.jsonl ] || ${QPY:-$PY} -m $M.prepare.questions_other $DATASET --native
else [ -f $OUT/questions.jsonl ] || $PY -m $M.prepare.questions_other $DATASET; fi
[ -f $OUT/residuals.safetensors ] || $PY -m $M.experiment.residuals --limit $LIM
# training candidates (unclamped sampler) and the base lens at the served setting, every question
$PY -m $M.experiment.sample --tag pool --questions ${POOL_Q:-all} --n $NP --temperature 1.0 --top-p 1.0 --top-k 0 --limit $LIM
$PY -m $M.experiment.sample --tag eval_base --questions ${EVAL_Q:-all} --n $NE --temperature 0.8 --top-p 0.95 --limit $LIM
$PY -m $M.experiment.score --tags pool eval_base
[ -z "${STOP_BEFORE_TRAIN:-}" ] || { echo "stopped before training (STOP_BEFORE_TRAIN)"; exit 0; }
for f in $FOLDS; do
  $PY -m $M.experiment.train --fold $f --keep ${KEEP:-2} --lr ${LR:-2e-5} ${TRAIN_ARGS:-}
  $PY -m $M.experiment.sample --tag eval_fold$f --adapter $OUT/adapters/fold$f --questions test:$f --n $NE \
    --temperature 0.8 --top-p 0.95 --limit $LIM
  $PY -m $M.experiment.score --tags eval_fold$f
done
$PY -m $M.experiment.report --folds $FOLDS | tee $OUT/report.md
[ "$DATASET" = cancer_myth ] || exit 0  # diag.py uses the Cancer-Myth MCE test runs
$SK -m $M.experiment.diag --cache $R/experiments/latent_mce/results/data/mce_cancer/lens_cache.jsonl \
  $OUT/lens_cache/eval_base.jsonl $(for f in $FOLDS; do echo $OUT/lens_cache/eval_fold$f.jsonl; done) | tee $OUT/diag.md
