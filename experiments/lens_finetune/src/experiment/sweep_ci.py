"""Paired bootstrap 95% intervals (2000 resamples over questions) of tuned - base test FVE and specificity for one
sweep.sh setting. Usage: RESULTS_SUBDIR=<dataset>_split python -m experiments.lens_finetune.src.experiment.sweep_ci NAME FOLD"""
import random
import sys
from collections import defaultdict
from statistics import mean

from experiments.lens_finetune.src.experiment.lens_common import OUT, read_jsonl, split

name, fold = sys.argv[1], int(sys.argv[2])
test = set(split(fold)[2])


def per_q(tag):
    by = defaultdict(lambda: defaultdict(list))
    for r in read_jsonl(OUT / "scores" / f"{tag}.jsonl"):
        if r["qid"] in test:
            by[r["qid"]][r["layer"]].append(r)
    return {q: (mean(mean(x["fve"] for x in v) for v in L.values()), mean(mean(x["fve"] - x["fve_shuf"] for x in v) for v in L.values()))
            for q, L in by.items()}


b, t = per_q("eval_base"), per_q(f"{name}_test")
qs = sorted(set(b) & set(t))
for k, nm in ((0, "FVE"), (1, "specificity")):
    d = [t[q][k] - b[q][k] for q in qs]
    rng = random.Random(0)
    bs = sorted(mean(rng.choice(d) for _ in d) for _ in range(2000))
    print(f"{nm}: base {mean(b[q][k] for q in qs):.3f} tuned {mean(t[q][k] for q in qs):.3f} diff {mean(d):+.4f} "
          f"[{bs[50]:+.4f}, {bs[1949]:+.4f}] n {len(d)}, improved {sum(x > 0 for x in d)}")
