#!/usr/bin/env bash
# One SFT setting on one dataset and fold, checkpoints picked by validation FVE (not loss): train with --save-all, read
# and score the fold's validation questions with the base lens and every saved step (2 samples, served setting), read
# and score the test with the best step (4 samples), print sweep_summary. Needs a finished run_here.sh run of the
# dataset (residuals, pool and eval_base readings and scores).
# Usage: CUDA_VISIBLE_DEVICES=a,b ENV_SH=experiments/lens_finetune/env_a6000.sh DATASET=hotpotqa FOLD=0 NAME=lr1e-4_k1_e2 \
#        TRAIN_ARGS="--lr 1e-4 --keep 1 --epochs 2" experiments/lens_finetune/sweep.sh
set -euo pipefail
source ${ENV_SH:-/data1/heejae/scripts/env.sh}
R=$(cd "$(dirname "$0")/../.." && pwd); cd $R
PY=${PY:-$D/uv/hf27b/bin/python}; M=experiments.lens_finetune.src
export PYTHONPATH=$R:$OLENS_SRC${EXTRA_PYTHONPATH:+:$EXTRA_PYTHONPATH} PYTORCH_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1
export RESULTS_SUBDIR=${RESULTS_SUBDIR:-$DATASET}; OUT=$R/experiments/lens_finetune/results/$RESULTS_SUBDIR/data
F=$FOLD; S="--temperature 0.8 --top-p 0.95"
$PY -m $M.experiment.train --fold $F --name $NAME --save-all ${TRAIN_ARGS:-}
[ -f $OUT/scores/base_val$F.jsonl ] || { $PY -m $M.experiment.sample --tag base_val$F --questions val:$F --n 2 $S; $PY -m $M.experiment.score --tags base_val$F; }
for d in $OUT/adapters/$NAME/step*; do s=$(basename $d)
  [ -f $OUT/scores/${NAME}_val_$s.jsonl ] || { $PY -m $M.experiment.sample --tag ${NAME}_val_$s --adapter $d --questions val:$F --n 2 $S; $PY -m $M.experiment.score --tags ${NAME}_val_$s; }
done
b=$($PY -m $M.experiment.sweep_summary $NAME $F --pick)
if [ "$b" != 0 ]; then
  $PY -m $M.experiment.sample --tag ${NAME}_test --adapter $OUT/adapters/$NAME/step$b --questions test:$F --n 4 $S
  $PY -m $M.experiment.score --tags ${NAME}_test
fi
$PY -m $M.experiment.sweep_summary $NAME $F | tee $OUT/sweep_${NAME}.md
