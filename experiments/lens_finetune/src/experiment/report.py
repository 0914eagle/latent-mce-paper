"""Compare the fine-tuned lens with the base lens on each fold's test questions, and export the readings.

Inputs: eval readings sampled at the served setting (sample.py, temperature 0.8, top-p 0.95) — base lens on every
question (tag eval_base), fold-f lens on fold f's test (tag eval_fold<f>) — and their AR scores (score.py).
Per fold and layer: mean joint FVE (mean over a question's samples, then over questions), the shuffled-residual FVE
(fve_shuf) and specificity = fve - fve_shuf, and the EOS rate; tuned - base with a paired bootstrap 95% interval over
questions (2000 resamples). Writes results/data/report.json and prints markdown tables.
Export: results/data/lens_cache/<tag>.jsonl in the latent_mce lens-cache format ({"k": key, "v": {"36": text, ...}},
sample 0 of each layer) for the test questions, for diag.py and for a latent MCE run.
Usage (repo root): python -m experiments.lens_finetune.src.experiment.report [--folds 0 1 2]
"""
import argparse
import json
import random
from collections import defaultdict

from experiments.lens_finetune.src.experiment.lens_common import LAYERS, OUT, lens_key, questions, read_jsonl, split

ap = argparse.ArgumentParser()
ap.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2])
args = ap.parse_args()
Q = {q["qid"]: q for q in questions()}


def per_question(tag, qids):
    """qid -> layer -> mean of fve, fve_shuf, eos over the samples."""
    acc = defaultdict(lambda: defaultdict(list))
    for s in read_jsonl(OUT / "scores" / f"{tag}.jsonl"):
        if s["qid"] in qids:
            acc[s["qid"]][s["layer"]].append(s)
    return {q: {L: {k: sum(float(x[k]) for x in v) / len(v) for k in ("fve", "fve_shuf", "eos")} for L, v in d.items()}
            for q, d in acc.items()}


def boot(diffs, n=2000):
    rng = random.Random(0)
    means = sorted(sum(rng.choice(diffs) for _ in diffs) / len(diffs) for _ in range(n))
    return means[int(0.025 * n)], means[int(0.975 * n)]


def export(tag, qids):
    by = defaultdict(dict)
    for r in read_jsonl(OUT / "readings" / f"{tag}.jsonl"):
        if r["qid"] in qids and r["s"] == 0:
            by[r["qid"]][str(r["layer"])] = r["text"]
    (OUT / "lens_cache").mkdir(exist_ok=True)
    with open(OUT / "lens_cache" / f"{tag}.jsonl", "w") as f:
        for q, v in by.items():
            f.write(json.dumps({"k": lens_key(Q[q]["question"], Q[q].get("system")), "v": v}, ensure_ascii=False) + "\n")


report = {}
for fold in args.folds:
    test = set(split(fold)[2])
    base, tuned = per_question("eval_base", test), per_question(f"eval_fold{fold}", test)
    both = sorted(set(base) & set(tuned))
    rows = {}
    for L in list(LAYERS) + ["all"]:
        Ls = LAYERS if L == "all" else [L]
        m = lambda d, q, k: sum(d[q][l][k] for l in Ls) / len(Ls)
        r = {}
        for k in ("fve", "fve_shuf", "eos"):
            r[f"base_{k}"] = sum(m(base, q, k) for q in both) / len(both)
            r[f"tuned_{k}"] = sum(m(tuned, q, k) for q in both) / len(both)
        diff = [m(tuned, q, "fve") - m(base, q, "fve") for q in both]
        spec = [(m(tuned, q, "fve") - m(tuned, q, "fve_shuf")) - (m(base, q, "fve") - m(base, q, "fve_shuf")) for q in both]
        r["diff_fve"], r["diff_fve_ci"] = sum(diff) / len(diff), boot(diff)
        r["diff_specificity"], r["diff_specificity_ci"] = sum(spec) / len(spec), boot(spec)
        for lab in ("FPQ", "NFP"):
            sub = [q for q in both if Q[q]["label"] == lab]
            r[f"diff_fve_{lab}"] = sum(m(tuned, q, "fve") - m(base, q, "fve") for q in sub) / max(1, len(sub))
        rows[str(L)] = r
    report[fold] = {"n_questions": len(both), "selection": json.loads((OUT / "adapters" / f"fold{fold}" / "selection.json").read_text()),
                    "layers": rows}
    export("eval_base", test)
    export(f"eval_fold{fold}", test)
    print(f"\n### fold {fold} test ({len(both)} questions)\n")
    print("| layer | base FVE | tuned FVE | tuned − base [95% CI] | base / tuned shuffled FVE | specificity Δ [95% CI] | EOS base / tuned |")
    print("|---|---|---|---|---|---|---|")
    for L, r in rows.items():
        print(f"| {L} | {r['base_fve']:.3f} | {r['tuned_fve']:.3f} | {r['diff_fve']:+.3f} [{r['diff_fve_ci'][0]:+.3f}, {r['diff_fve_ci'][1]:+.3f}]"
              f" | {r['base_fve_shuf']:.3f} / {r['tuned_fve_shuf']:.3f} | {r['diff_specificity']:+.3f} "
              f"[{r['diff_specificity_ci'][0]:+.3f}, {r['diff_specificity_ci'][1]:+.3f}] | {r['base_eos']:.2f} / {r['tuned_eos']:.2f} |")
(OUT / "report.json").write_text(json.dumps(report, indent=1))
