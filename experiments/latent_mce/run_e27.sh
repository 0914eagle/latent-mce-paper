#!/usr/bin/env bash
# E27 run (prereg_v4_test.md): 3 iterations x {OUT5, LAT} x 5 replicates on fold 0. OUT5 = E21 OUT3 procedure; LAT = the
# same + probe gate on every attached context (rollout, dev100, test) + lens failure types and their changes (base
# agent) + internal aggregates per version (meta agent). vLLM 8802 + 8803; lens 8796 (LAT a, c, e) and 8797 (LAT b, d).
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=.venv-fpq/bin/python; X=experiments.latent_mce.src.experiment
export PATH=/data2/heejae/uv/mce/bin:$PATH E18_GEN_PORTS=8802,8803
V=experiments/latent_mce/results/data/v4/versions
RUNS="OUT5_a LAT_a OUT5_b LAT_b OUT5_c LAT_c OUT5_d LAT_d OUT5_e LAT_e"
for p in 8802 8803; do until curl -s -m 5 http://127.0.0.1:$p/v1/models > /dev/null; do sleep 15; done; done
for p in 8796 8797; do until curl -s -m 5 -o /dev/null http://127.0.0.1:$p/; do sleep 15; done; done
$PY -m experiments.latent_mce.src.prepare.e27_workspaces a,b,c,d,e
for r in $RUNS; do $PY -m $X.e18_meta snapshot $r 0; cp $V/e18_OUT3_a/memory/dev_0.json $V/e18_$r/memory/dev_0.json; done
for k in 1 2 3; do
  for r in $RUNS; do E18_CACHE_TAG=$r $PY -m $X.e27_loop rollout $r $k & done; wait
  for r in OUT5_a OUT5_b OUT5_c OUT5_d OUT5_e; do $PY -m $X.e27_loop types $r $k & done
  (for r in LAT_a LAT_c LAT_e; do E18_LENS_PORT=8796 $PY -m $X.e27_loop types $r $k; done) &
  (for r in LAT_b LAT_d; do E18_LENS_PORT=8797 $PY -m $X.e27_loop types $r $k; done) &
  wait
  if [ $k -ge 2 ]; then for r in $RUNS; do $PY -m $X.e27_loop meta $r $k & done; wait; fi
  for r in $RUNS; do $PY -m $X.e27_loop agent $r $k & done; wait
  for r in $RUNS; do $PY -m $X.e18_meta snapshot $r $k; done
  for r in $RUNS; do E18_CACHE_TAG=$r $PY -m $X.e27_loop dev $r $k & done; wait
  echo "iteration $k done $(date -u +%H:%M:%S)"
done
for r in $RUNS; do E18_CACHE_TAG=$r $PY -m $X.e27_loop test $r & done; wait
echo "e27 done $(date -u +%H:%M:%S)"
