"""Per-item signals for the Latent-MCE (cells) run on Cancer-Myth fold 0: train 156, dev 40, test 242, and dev100
(dev 40 + 60 unused non-test FPQ items, seed 0; fold 0 has no unused normal questions).

mode  = Oracle-lens correction frame (FRAME regex on the stored Plain-prefix readouts, blocks 52/60).
p_fpq = need probe trained on the 156 training items only (per-layer StandardScaler + balanced LR, C=0.003, mean of the
        11 stored layers, same recipe as latent_c1c2_audit_20261024 src/experiment/layers.py); training items get
        5-fold cross-fit scores, dev/test items the full-train probe. No dev/test label enters the probe.
text_neighbors = top-8 training ids by word(1-2)+char(3-5) TF-IDF fitted on training questions, the recipe of
        latent_cancer_memory_20261025 src/prepare/build.py, used only to run the other session's text-MCE context.
Usage: .venv-fpq/bin/python -m experiments.latent_mce.src.prepare.v4_features
"""
import json

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import FeatureUnion, make_pipeline
from sklearn.preprocessing import StandardScaler

from common.paths import DATA_DIR, results_dir
from experiments.latent_mce.src.experiment.cancer_fixability import AUDIT, FRAME

OUT = results_dir("latent_mce") / "data" / "v4"


def probe_scores(X_train, y, X_other):
    """(cross-fit train scores, other scores): mean over layers of per-layer balanced LR probabilities."""
    fit = lambda X, t: [make_pipeline(StandardScaler(), LogisticRegression(C=.003, max_iter=3000, class_weight="balanced")).fit(X[:, L], t) for L in range(X.shape[1])]
    score = lambda models, X: np.mean([m.predict_proba(X[:, L])[:, 1] for L, m in enumerate(models)], axis=0)
    oof = np.zeros(len(y))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(X_train[:, 0], y):
        oof[te] = score(fit(X_train[tr], y[tr]), X_train[te])
    return oof, score(fit(X_train, y), X_other)


def main():
    # Items, roles, states and the Cancer-Myth questions
    norm = json.loads((AUDIT / "normalized.json").read_text())
    items = {r["id"]: r for r in norm["items"]}
    split = next(s for s in norm["splits"] if s["fold"] == "fold0")
    roles = {"train": split["train_ids"], "dev": split["val_ids"], "test": sorted(i for i, r in items.items() if r["fold"] == "fold0")}
    z = np.load(AUDIT / "states_Plain.npz")
    row = {i: k for k, i in enumerate(z["ids"])}
    # Unused non-test items (all FPQ in fold 0) with stored states: 60 of them extend dev to "dev100" (seed 0)
    used = set(roles["train"]) | set(roles["dev"]) | set(roles["test"])
    pool = sorted(i for i in items if i not in used and i in row)
    roles["dev100"] = roles["dev"] + sorted(np.random.default_rng(0).choice(pool, 60, replace=False).tolist())
    states = z["states"]

    # Probe trained on training items only
    tr = roles["train"]; other = sorted(set(roles["dev"] + roles["test"] + roles["dev100"]))
    y = np.array([items[i]["label"] for i in tr])
    oof, rest = probe_scores(states[[row[i] for i in tr]], y, states[[row[i] for i in other]])
    p = dict(zip(tr, oof)) | dict(zip(other, rest))

    # TF-IDF text neighbours over training questions
    tfidf = FeatureUnion([("word", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, max_features=50000)),
                          ("char", TfidfVectorizer(analyzer="char", ngram_range=(3, 5), sublinear_tf=True, max_features=50000))])
    T = tfidf.fit_transform([items[i]["question"] for i in tr])
    out = []
    for role, ids in roles.items():
        S = (tfidf.transform([items[i]["question"] for i in ids]) @ T.T).toarray()
        for j, i in enumerate(ids):
            order = [k for k in np.argsort(-S[j], kind="stable") if tr[k] != i][:8]
            out.append({"id": i, "role": role, "question": items[i]["question"], "label": items[i]["label"],
                        "mode": bool(FRAME.search("\n".join(items[i]["plain_reads"][L] for L in ("52", "60")))),
                        "p_fpq": round(float(p[i]), 4), "text_neighbors": [tr[k] for k in order]})

    # Write, with the probe AUC on dev+test as a sanity number (evaluation only)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "items.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
    from sklearn.metrics import roc_auc_score
    for role in roles:
        rr = [x for x in out if x["role"] == role]
        print(role, len(rr), "probe AUC", round(roc_auc_score([x["label"] for x in rr], [x["p_fpq"] for x in rr]), 3),
              "mode on", sum(x["mode"] for x in rr))


if __name__ == "__main__":
    main()
