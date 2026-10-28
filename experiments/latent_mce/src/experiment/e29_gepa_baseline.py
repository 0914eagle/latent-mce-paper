"""Baselines for E29 on the vLLM pipeline (prereg_v4_test.md, E29): Plain, the GEPA fold-0 system prompt
(gepa_fold0_system.txt replaces the whole system prompt), and GEPA under the probe gate (GEPA answer when the question's
Plain-state p_fpq >= 0.5, Plain answer otherwise), on the fold-0 test questions; same answer and judge caches as E29.
Usage: E18_GEN_PORTS=8802,8803 E18_CACHE_TAG=gepa_base .venv-fpq/bin/python -m experiments.latent_mce.src.experiment.e29_gepa_baseline
"""
import json

import numpy as np

from experiments.latent_mce.src.experiment import e18_loop as L
from experiments.latent_mce.src.experiment.v4_eval import V4


def main():
    # Test questions with their probe scores; Plain and GEPA answers and ratings
    base, items, by_role, refs, rows, bridge = L.setup()
    test = {x["id"]: x for x in json.loads((V4 / "items.json").read_text()) if x["role"] == "test"}
    gepa = (V4 / "gepa_fold0_system.txt").read_text()
    ids = sorted(test)
    rated = L.answer_and_rate([(i, s) for i in ids for s in (base, gepa)], rows, bridge)

    # Macro = (FPQ pass % + NFP pass %) / 2 for each condition
    ok = {c: {i: (rated[(i, s)][1] or 0) >= 4 for i in ids} for c, s in (("plain", base), ("gepa", gepa))}
    ok["gepa_gate"] = {i: ok["gepa"][i] if test[i]["p_fpq"] >= 0.5 else ok["plain"][i] for i in ids}
    out = {}
    for c, o in ok.items():
        fpq = 100 * np.mean([o[i] for i in ids if test[i]["label"] == 1]); nfp = 100 * np.mean([o[i] for i in ids if test[i]["label"] == 0])
        out[c] = {"macro": round((fpq + nfp) / 2, 1), "FPQ": round(fpq, 1), "NFP": round(nfp, 1)}
    out["n"] = {"test": len(ids), "FPQ": sum(test[i]["label"] == 1 for i in ids), "gate_on": sum(test[i]["p_fpq"] >= 0.5 for i in ids)}
    print(json.dumps(out, indent=1))
    (V4 / "e29_gepa_baseline.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
