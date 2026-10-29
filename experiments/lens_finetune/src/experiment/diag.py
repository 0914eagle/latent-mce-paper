"""Do the lens readings tell apart the test questions MCE rescues from those it does not? (behavioral diagnostic)

Groups from latent_mce's test runs on the balanced folds (results/data/mce_cancer/test_fold<f>_{base,none}.json):
Base = Plain, no context; MCE = the original MCE's best context. For each comparison, the AUROC of a 5-fold
cross-validated TF-IDF logistic regression on the reading (four layers joined) vs on the question text alone; the
reading carries information beyond the question only where it beats the question. Readings: any lens-cache file
(report.py exports; the latent_mce cache for the base lens as served).
Usage (repo root, a venv with scikit-learn): python -m experiments.lens_finetune.src.experiment.diag --cache <jsonl> [...]
"""
import argparse
import json
from collections import defaultdict

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from common.paths import EXPERIMENTS_DIR
from experiments.lens_finetune.src.experiment.lens_common import LAYERS, lens_key

ap = argparse.ArgumentParser()
ap.add_argument("--cache", nargs="+", required=True, help="lens-cache jsonl files; each is scored on the items it covers")
args = ap.parse_args()
M = EXPERIMENTS_DIR / "latent_mce" / "results" / "data" / "mce_cancer"  # the stored test runs, whatever RESULTS_SUBDIR says


def outcomes():
    rows = []
    for f in range(3):
        B = {r["sample"]["qid"]: r for r in json.load(open(M / f"test_fold{f}_base.json" / "evaluation.json"))["results"]}
        C = {r["sample"]["qid"]: r for r in json.load(open(M / f"test_fold{f}_none.json" / "evaluation.json"))["results"]}
        for q, r in B.items():
            b, m = r["evaluation"]["metrics"]["accuracy"] == 1, C[q]["evaluation"]["metrics"]["accuracy"] == 1
            rows.append({"question": r["sample"]["question"], "label": r["sample"]["label"],
                         "group": {(0, 1): "rescued", (0, 0): "both_fail", (1, 1): "both_pass", (1, 0): "lost"}[(int(b), int(m))]})
    return rows


def cv_auroc(pos, neg, field):
    X, y = [r[field] for r in pos + neg], np.array([1] * len(pos) + [0] * len(neg))
    if min(len(pos), len(neg)) < 5:
        return float("nan")
    p = cross_val_predict(LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced"),
                          TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True).fit_transform(X), y,
                          cv=StratifiedKFold(5, shuffle=True, random_state=0), method="predict_proba")[:, 1]
    return roc_auc_score(y, p)


base_rows = outcomes()
for path in args.cache:
    lens = {json.loads(l)["k"]: json.loads(l)["v"] for l in open(path)}
    rows = [dict(r, reading=" ".join(lens[lens_key(r["question"])].get(str(L), "") for L in LAYERS))
            for r in base_rows if lens_key(r["question"]) in lens]
    G = defaultdict(list)
    for r in rows:
        G[(r["label"], r["group"])].append(r)
    fpq, nfp = [r for r in rows if r["label"] == "FPQ"], [r for r in rows if r["label"] == "NFP"]
    print(f"\n{path}: {len(rows)} items  " + " ".join(f"{k[0]}-{k[1]} {len(v)}" for k, v in sorted(G.items())))
    print("  comparison                       n pos/neg   reading | question")
    for name, pos, neg in [
        ("FPQ vs NFP", fpq, nfp),
        ("FPQ rescued vs both_fail", G[("FPQ", "rescued")], G[("FPQ", "both_fail")]),
        ("FPQ Base-pass vs Base-fail", G[("FPQ", "both_pass")] + G[("FPQ", "lost")], G[("FPQ", "rescued")] + G[("FPQ", "both_fail")]),
        ("NFP lost vs kept-pass", G[("NFP", "lost")], G[("NFP", "both_pass")]),
    ]:
        print(f"  {name:30s} {len(pos):4d}/{len(neg):<4d}   {cv_auroc(pos, neg, 'reading'):.3f} | {cv_auroc(pos, neg, 'question'):.3f}")
