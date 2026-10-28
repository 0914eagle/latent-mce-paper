#!/usr/bin/env bash
# E21 run (prereg_v4_test.md): 3 iterations x {OUT3, FTY} x 3 replicates from an empty memory; E20 procedure and
# train-aware meta agent in both arms; after each rollout both get failure counts by answer behaviour, FTY also the
# probe-based split of FPQ failures (possibly_silent / likely_unaware); one rewrite if a version attaches no context on
# dev100. dev100 after every version, fold-0 test for each run's dev-selected version.
# Generation: vLLM 8802 (GPUs 0,1); hidden states: 8798 (GPUs 2,3).
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=.venv-fpq/bin/python; X=experiments.latent_mce.src.experiment
export PATH=/data2/heejae/uv/mce/bin:$PATH E18_GEN_PORTS=8802
V=experiments/latent_mce/results/data/v4/versions
RUNS="OUT3_a FTY_a OUT3_b FTY_b OUT3_c FTY_c"
until curl -s -m 5 http://127.0.0.1:8802/v1/models > /dev/null; do sleep 15; done  # vLLM ready
until curl -s -m 5 -o /dev/null http://127.0.0.1:8798/; do sleep 15; done          # hidden-state server up
# Workspaces, version-0 snapshots, version-0 dev100 (Plain, cached)
$PY -m experiments.latent_mce.src.prepare.e21_workspaces a,b,c
for r in $RUNS; do $PY -m $X.e18_meta snapshot $r 0; done
$PY -m $X.e18_loop dev OUT3_a 0
for r in FTY_a OUT3_b FTY_b OUT3_c FTY_c; do cp $V/e18_OUT3_a/memory/dev_0.json $V/e18_$r/memory/dev_0.json; done
for k in 1 2 3; do
  # Rollout of version k-1 on the iteration's training sample, then failure counts (FTY one at a time: shared server)
  for r in $RUNS; do E18_CACHE_TAG=$r $PY -m $X.e18_loop rollout $r $k & done; wait
  for r in OUT3_a OUT3_b OUT3_c; do $PY -m $X.e21_loop types $r $k & done
  for r in FTY_a FTY_b FTY_c; do $PY -m $X.e21_loop types $r $k; done; wait
  # From iteration 2: meta agents
  if [ $k -ge 2 ]; then for r in $RUNS; do $PY -m $X.e21_loop meta $r $k & done; wait; fi
  # Base agents (with the dev100 coverage check), snapshots, dev100
  for r in $RUNS; do $PY -m $X.e21_loop agent $r $k & done; wait
  for r in $RUNS; do $PY -m $X.e18_meta snapshot $r $k; done
  for r in $RUNS; do E18_CACHE_TAG=$r $PY -m $X.e18_loop dev $r $k & done; wait
  echo "iteration $k done $(date -u +%H:%M:%S)"
done
for r in $RUNS; do E18_CACHE_TAG=$r $PY -m $X.e18_meta test $r & done; wait
echo "e21 done $(date -u +%H:%M:%S)"
