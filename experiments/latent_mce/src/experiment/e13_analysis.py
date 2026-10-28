"""E13 analysis: does the with-context hidden state tell which context helps a training question? (prereg_v4_test.md, E13)

Pairs (training question, context) from the given training rollouts plus no context; outcome = pass (Luna rating >= 4)
of that answer. Three logistic-regression predictors of pass, compared under 5-fold cross-validation grouped by
question: T (label, no-context flag, the context's net pass effect on the other training-fold questions, TF-IDF
question-context similarity), T+P (+ out-of-fold with-context probe score and its difference from the question's Plain score; nested
inside each outer fold), T+P+O (+ Oracle-lens plan with the context, plan without, interactions with the label).
Only pairs with every input are used. Metrics: within-question pairwise accuracy, AUC for fixing a Plain failure and
for breaking a Plain pass; 95% intervals of the differences by bootstrap over questions. Writes results/data/v4/e13.json.
Usage: .venv-fpq/bin/python -m experiments.latent_mce.src.experiment.e13_analysis ROLLOUT[,ROLLOUT...]
"""
import hashlib
import json
import sys

import numpy as np
from safetensors.torch import load_file
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common.paths import results_dir
from experiments.latent_mce.src.experiment.cancer_fixability import AUDIT, FRAME
from experiments.latent_mce.src.experiment.v4_eval import CM, CONTEXT_FORMAT, P4

V4 = results_dir("latent_mce") / "data" / "v4"


def probe(X_tr, y_tr, X_te):
    """Mean over layers of per-layer balanced LR probabilities (the v4_features recipe)."""
    models = [make_pipeline(StandardScaler(), LogisticRegression(C=.003, max_iter=3000, class_weight="balanced")).fit(X_tr[:, L], y_tr)
              for L in range(X_tr.shape[1])]
    return np.mean([m.predict_proba(X_te[:, L])[:, 1] for L, m in enumerate(models)], axis=0)


def pairwise(q, y, s):
    """Share of same-question pairs with different outcomes that the scores order correctly (ties = 0.5)."""
    hit = n = 0
    for g in np.unique(q):
        i = np.where(q == g)[0]
        for a in i:
            for b in i:
                if y[a] > y[b]:
                    n += 1; hit += 1.0 if s[a] > s[b] else 0.5 if s[a] == s[b] else 0.0
    return hit / n if n else float("nan"), n


def main(rollouts):
    # Pairs with outcome, state and lens plan
    items = {x["id"]: x for x in json.loads((V4 / "items.json").read_text()) if x["role"] == "train"}
    norm = {r["id"]: r for r in json.loads((AUDIT / "normalized.json").read_text())["items"]}
    reg = json.load(open(CM / json.load(open(P4 / "run/qwen3_8b/identity.json"))["full_config"]["registry"]))
    base = reg["transport_common"] + "\n\n" + reg["prompts"]["plain"]
    sha = lambda *xs: hashlib.sha256("\x1f".join(xs).encode()).hexdigest()[:32]
    plans = {json.loads(l)["k"]: json.loads(l)["plan"] for f in sorted(V4.glob("plans*.jsonl")) for l in f.read_text().splitlines()}
    ratings = {}
    for f in sorted(V4.glob("ratings_train_plain_*.json")):
        for c, v in json.loads(f.read_text())["per_item"].items():
            ratings.setdefault(c, v)
    index = json.loads((V4 / "e13_states" / "index.json").read_text())
    states = {}
    for name in sorted({v["file"] for v in index.values()}):
        states.update({k: v.float().numpy() for k, v in load_file(str(V4 / "e13_states" / name)).items()})
    rows = {}
    for i, x in items.items():
        k = sha(base, x["question"])
        rows[k] = dict(id=i, context="", rating=ratings["plain"][i]["rating"],
                       plan=bool(FRAME.search("\n".join(norm[i]["plain_reads"][L] for L in ("52", "60")))))
    for r in rollouts.split(","):
        ctx = json.loads((V4 / f"ctx_train_{r}.json").read_text())
        for i, x in items.items():
            t = ctx[i]["context"]
            k = sha(base + CONTEXT_FORMAT.format(context=t), x["question"]) if t else None
            if t and k not in rows and k in plans:
                p = plans[k]
                rows[k] = dict(id=i, context=t, rating=ratings[r][i]["rating"], plan=bool(FRAME.search(p["52"] + "\n" + p["60"])))
    keys = [k for k in rows if k in states]
    print("pairs", len(rows), "| with state", len(keys), "| contexts", len({rows[k]["context"] for k in keys}))

    # Arrays
    q = np.array([rows[k]["id"] for k in keys]); ctxt = [rows[k]["context"] for k in keys]
    y = np.array([rows[k]["rating"] >= 4 for k in keys]).astype(int)
    lab = np.array([items[i]["label"] for i in q])
    plain = np.array([c == "" for c in ctxt]).astype(int)
    plan = np.array([rows[k]["plan"] for k in keys]).astype(int)
    plain_key = {i: sha(base, items[i]["question"]) for i in items}
    plan0 = np.array([rows[plain_key[i]]["plan"] for i in q]).astype(int)
    X = np.stack([states[k] for k in keys])
    tf = TfidfVectorizer(sublinear_tf=True).fit([x["question"] for x in items.values()] + sorted(set(ctxt) - {""}))
    sim = np.array([0.0 if c == "" else float((tf.transform([items[i]["question"]]) @ tf.transform([c]).T).toarray()[0, 0]) for i, c in zip(q, ctxt)])
    pass_plain = {i: rows[plain_key[i]]["rating"] >= 4 for i in items}
    gain = np.array([y[m] - pass_plain[q[m]] for m in range(len(keys))])

    def effect_from(rows_seen):
        """Net pass effect of each pair's context on the other questions among rows_seen (training fold only)."""
        seen = set(rows_seen)
        out = np.zeros(len(keys))
        for j, (i, c) in enumerate(zip(q, ctxt)):
            if c:
                d = [gain[m] for m in seen if ctxt[m] == c and q[m] != i]
                out[j] = np.mean(d) if d else 0.0
        return out

    # Grouped cross-validation, probe nested inside each outer fold
    folds = list(GroupKFold(5).split(X, y, q))
    score = {name: np.zeros(len(keys)) for name in ("T", "T+P", "T+P+O")}
    for tr, te in folds:
        inner = np.zeros(len(tr))
        for itr, ite in GroupKFold(5).split(X[tr], y[tr], q[tr]):
            inner[ite] = probe(X[tr][itr], y[tr][itr], X[tr][ite])
        p_te = probe(X[tr], y[tr], X[te])
        p_all = np.zeros(len(keys)); p_all[tr] = inner; p_all[te] = p_te
        p_plain = {i: p_all[j] for j, i in enumerate(q) if plain[j]}
        dp = np.array([p_all[j] - p_plain.get(i, p_all[j]) for j, i in enumerate(q)])
        base_f = np.c_[lab, plain, effect_from(tr), sim]  # outcomes of the held-out fold never enter a feature
        feats = {"T": base_f, "T+P": np.c_[base_f, p_all, dp],
                 "T+P+O": np.c_[base_f, p_all, dp, plan, plan0, plan * lab, plan * (1 - lab)]}
        for name, F in feats.items():
            m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000)).fit(F[tr], y[tr])
            score[name][te] = m.predict_proba(F[te])[:, 1]

    # Metrics and bootstrap over questions
    pp = np.array([pass_plain[i] for i in q])
    fix = (plain == 0) & (pp == 0); brk = (plain == 0) & (pp == 1)
    res = {"pairs": len(keys), "questions": int(len(set(q))), "contexts": len(set(ctxt)), "pass_rate": float(y.mean()), "by": {}}
    unseen = sum(1 for tr, te in folds for j in te if ctxt[j] and ctxt[j] not in {ctxt[m] for m in tr})
    res["held_out_card_pairs_with_context_unseen_in_training_fold"] = unseen
    for name, s in score.items():
        acc, n = pairwise(q, y, s)
        top = [y[np.where(q == g)[0][np.argmax(s[q == g])]] for g in np.unique(q)]  # pass of the top-scored context
        res["by"][name] = {"pairwise": acc, "pairs_compared": n, "top_pick_pass": float(np.mean(top)),
                           "auc_fix": roc_auc_score(y[fix], s[fix]) if len(set(y[fix])) > 1 else None,
                           "auc_break": roc_auc_score(1 - y[brk], 1 - s[brk]) if len(set(y[brk])) > 1 else None}
    rng = np.random.default_rng(0); qs = np.unique(q); at = {g: np.where(q == g)[0] for g in qs}
    for a, b in (("T+P", "T"), ("T+P+O", "T+P"), ("T+P+O", "T")):
        diffs = []
        for _ in range(1000):
            draw = rng.choice(qs, len(qs))
            idx = np.concatenate([at[g] for g in draw])
            gid = np.concatenate([np.full(len(at[g]), k) for k, g in enumerate(draw)])  # each drawn copy is its own group
            diffs.append(pairwise(gid, y[idx], score[a][idx])[0] - pairwise(gid, y[idx], score[b][idx])[0])
        res[f"{a} - {b}"] = {"pairwise_diff": res["by"][a]["pairwise"] - res["by"][b]["pairwise"],
                             "ci95": [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))]}
    (V4 / "e13.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main(sys.argv[1])
