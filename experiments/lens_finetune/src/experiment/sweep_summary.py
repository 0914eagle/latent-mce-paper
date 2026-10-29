"""Summary of one SFT setting (sweep.sh): validation FVE per saved step, the step picked, and test FVE vs the base lens.

Mean over (question, layer) of the mean over a question's samples; specificity = FVE - shuffled FVE. Test: the base
lens's eval_base readings of the fold's test questions vs the picked step's readings (both at the served setting).
Usage: python -m experiments.lens_finetune.src.experiment.sweep_summary NAME FOLD [--pick]   (--pick prints the step only)
"""
import json
import sys
from collections import defaultdict
from statistics import mean

from experiments.lens_finetune.src.experiment.lens_common import OUT, read_jsonl, split

name, fold = sys.argv[1], int(sys.argv[2])
train, val, test = split(fold)


def stats(tag, qids):
    by = defaultdict(list)
    for r in read_jsonl(OUT / "scores" / f"{tag}.jsonl"):
        if r["qid"] in qids:
            by[(r["qid"], r["layer"])].append(r)
    if not by:
        return None
    f = mean(mean(x["fve"] for x in v) for v in by.values())
    s = mean(mean(x["fve"] - x["fve_shuf"] for x in v) for v in by.values())
    return {"fve": f, "spec": s, "n": len(by)}


steps = sorted(int(p.name[4:]) for p in (OUT / "adapters" / name).glob("step*"))
curve = {0: stats(f"base_val{fold}", set(val))} | {s: stats(f"{name}_val_step{s}", set(val)) for s in steps}
curve = {s: c for s, c in curve.items() if c}
best = max(curve, key=lambda s: curve[s]["fve"])
if "--pick" in sys.argv:
    print(best); sys.exit()
print(f"## {name} (fold {fold})\n\nvalidation ({len(val)} questions): " + ", ".join(f"step {s} FVE {c['fve']:.3f} spec {c['spec']:.3f}" for s, c in curve.items()))
print(f"picked step {best}")
b, t = stats("eval_base", set(test)), stats(f"{name}_test", set(test))
if t:
    print(f"test ({len(test)} questions): base FVE {b['fve']:.3f} spec {b['spec']:.3f} | tuned FVE {t['fve']:.3f} spec {t['spec']:.3f} | "
          f"diff FVE {t['fve'] - b['fve']:+.3f} spec {t['spec'] - b['spec']:+.3f}")
