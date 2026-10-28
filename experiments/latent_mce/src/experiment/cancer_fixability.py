"""Do Oracle-lens cells pick out instruction-fixable errors on Cancer-Myth (27B, stored data)?

Among FPQ items whose Plain answer fails, compare how often the fold's selected GEPA prompt fixes the answer when the
Plain-prefix Oracle readout shows a correction frame vs not, and the same for size-matched probe and TF-IDF splits.
Pre-registration: results/logs/prereg_cancer_fixability.md. Inputs are read from experiments/latent_c1c2_audit_20261024
(normalized.json, supervised_text_control.json); the FRAME regex is copied from its src/experiment/analyze.py.
Usage: .venv-fpq/bin/python -m experiments.latent_mce.src.experiment.cancer_fixability [SMOKE_N]
"""
import json
import re
import sys

import numpy as np

from common.paths import EXPERIMENTS_DIR, results_dir

AUDIT = EXPERIMENTS_DIR / "latent_c1c2_audit_20261024" / "results" / "data"
OUT = results_dir("latent_mce")
FRAME = re.compile(r"misconception|misunderstand|premise|incorrect|not actually|myth|mistaken|\bfalse\b|inaccurate|is not true|contrary to|误解|纠正|澄清|误区", re.I)
PASS, BOOT, SEED = 4, 2000, 0


def fix_gap(on, fixed):
    """Fix rate (percentage points) of the ON group minus the OFF group."""
    return 100 * (fixed[on].mean() - fixed[~on].mean())


def main(smoke_n=None):
    # Load items, lens frames and the out-of-fold TF-IDF scores
    items = json.loads((AUDIT / "normalized.json").read_text())["items"]
    task = json.loads((AUDIT / "supervised_text_control.json").read_text())["tasks"][0]
    tfidf = dict(zip(task["keys"], task["predictions"]))
    items = [r for r in items if r["id"] in tfidf][:smoke_n]
    frame = np.array([bool(FRAME.search("\n".join(r["plain_reads"][L] for L in ("52", "60")))) for r in items])
    label = np.array([r["label"] for r in items])
    plain_ok = np.array([r["plain_rating"] >= PASS for r in items])
    gepa_ok = np.array([r["gepa_rating"] >= PASS for r in items])
    need = np.array([r["need"] for r in items])
    text = np.array([tfidf[r["id"]] for r in items])

    # Population: FPQ items the Plain prompt fails; splits size-matched to the lens split
    pop = (label == 1) & ~plain_ok
    fixed, lens_on = gepa_ok[pop], frame[pop]
    k = int(lens_on.sum())
    top = lambda s: np.isin(np.arange(len(s)), np.argsort(-s)[:k])
    splits = {"lens frame": lens_on, "probe (size-matched)": top(need[pop]), "TF-IDF (size-matched)": top(text[pop]),
              "probe >= 0.5": need[pop] >= 0.5, "TF-IDF >= 0.5": text[pop] >= 0.5}

    # Point estimates and item-bootstrap CIs, including lens-minus-control differences
    rng = np.random.default_rng(SEED)
    boot = {name: [] for name in splits} | {"lens - probe": [], "lens - TF-IDF": []}
    for _ in range(BOOT):
        b = rng.integers(0, len(fixed), len(fixed))
        g = {name: fix_gap(on[b], fixed[b]) for name, on in splits.items()}
        for name in splits:
            boot[name].append(g[name])
        boot["lens - probe"].append(g["lens frame"] - g["probe (size-matched)"])
        boot["lens - TF-IDF"].append(g["lens frame"] - g["TF-IDF (size-matched)"])
    ci = lambda v: [round(float(x), 1) for x in np.nanpercentile(v, [2.5, 97.5])]
    rows = {name: {"n_on": int(on.sum()), "n_off": int((~on).sum()), "fix_on": round(100 * fixed[on].mean(), 1),
                   "fix_off": round(100 * fixed[~on].mean(), 1), "gap": round(fix_gap(on, fixed), 1), "ci95": ci(boot[name])}
            for name, on in splits.items()}
    lens_gap = rows["lens frame"]["gap"]
    go = lens_gap >= 15 and lens_gap > rows["probe (size-matched)"]["gap"] and lens_gap > rows["TF-IDF (size-matched)"]["gap"]

    # Secondary: NFP Plain passes broken by GEPA, by frame; full cell counts
    nfp_ok = (label == 0) & plain_ok
    nfp = {"on": [int((nfp_ok & frame).sum()), round(100 * (~gepa_ok[nfp_ok & frame]).mean(), 1)],
           "off": [int((nfp_ok & ~frame).sum()), round(100 * (~gepa_ok[nfp_ok & ~frame]).mean(), 1)]}
    cells = {f"{'FPQ' if l else 'NFP'} | Plain {'pass' if p else 'fail'} | frame {'on' if f else 'off'}":
             int(((label == l) & (plain_ok == p) & (frame == f)).sum()) for l in (1, 0) for p in (True, False) for f in (True, False)}

    # Write the summary
    out = {"n_items": len(items), "population_FPQ_plain_fail": int(pop.sum()), "splits": rows,
           "lens_minus_probe_ci95": ci(boot["lens - probe"]), "lens_minus_tfidf_ci95": ci(boot["lens - TF-IDF"]),
           "GO": bool(go), "nfp_plain_pass_broken_by_gepa_pct": nfp, "cells": cells}
    (OUT / "data").mkdir(parents=True, exist_ok=True)
    (OUT / "data" / ("cancer_fixability_smoke.json" if smoke_n else "cancer_fixability.json")).write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else None)
