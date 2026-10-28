"""E15 analysis: choose one context per held-out question by the predicted pass of its with-context reaction.

Predictors (label-free logistic regressions, trained on the 640 E13 training pairs only):
  T      no-context flag, the context's net pass effect on the training questions (0 if it never occurred there),
         TF-IDF question-context similarity
  T+P    T + with-context pass-probe score (per-layer LR mean, trained on training pairs), its difference from the
         question's no-context score, p_fpq (train-only Plain probe from items.json)
  T+P+O  T+P + Oracle-lens plan with the context, plan without, both times p_fpq
Candidates per dev100/test question: no context plus every distinct context one of the listed memories attached; their
stored answers and Luna ratings are used, nothing is generated. Per question the candidate with the highest predicted
pass is chosen. Reported: macro of the chosen answers per predictor, the mean single memory, t0, a uniformly random
candidate (expected), the best candidate per question (upper bound, uses outcomes), and a bootstrap over questions for
T+P+O - T. Writes results/data/v4/e15_<role>.json.
Usage: .venv-fpq/bin/python -m experiments.latent_mce.src.experiment.e15_select ROLE MEMORY[,MEMORY...]
"""
import glob
import hashlib
import json
import sys

import numpy as np
from safetensors.torch import load_file
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common.paths import results_dir
from experiments.latent_mce.src.experiment.cancer_fixability import AUDIT, FRAME
from experiments.latent_mce.src.experiment.e13_analysis import probe
from experiments.latent_mce.src.experiment.v4_eval import CM, CONTEXT_FORMAT, P4

V4 = results_dir("latent_mce") / "data" / "v4"
TRAIN_ROLLOUTS = "t0,e9a_T,e9a_TP,e9a_TPO,e9b_T,e9b_TP,e9b_TPO"


def pairs_of(role, sources, items, base, sha, plans, norm):
    """{key: row} for no context plus each (question, context) from the sources, with rating and lens plan."""
    per = {}
    for f in sorted(glob.glob(str(V4 / f"ratings_{role}_*.json"))):
        for c, v in json.loads(open(f).read())["per_item"].items():
            per.setdefault(c, v)
    rows = {}
    for i, x in items.items():
        rows[sha(base, x["question"])] = dict(id=i, context="", rating=per["plain"][i]["rating"], memory="plain",
                                              plan=bool(FRAME.search("\n".join(norm[i]["plain_reads"][L] for L in ("52", "60")))))
    for m in sources:
        ctx = json.loads((V4 / f"ctx_{role}_{m}.json").read_text())
        for i, x in items.items():
            t = ctx[i]["context"]
            k = sha(base + CONTEXT_FORMAT.format(context=t), x["question"]) if t else None
            if t and k not in rows and k in plans:
                p = plans[k]
                rows[k] = dict(id=i, context=t, rating=per[m][i]["rating"], memory=m, plan=bool(FRAME.search(p["52"] + "\n" + p["60"])))
    return rows, per


def features(rows, keys, items, states, effect, tf, plain_key, pass_probe):
    """T, T+P, T+P+O feature matrices for the given pair keys (label-free)."""
    q = [rows[k]["id"] for k in keys]; c = [rows[k]["context"] for k in keys]
    plain = np.array([t == "" for t in c]).astype(float)
    eff = np.array([effect.get(t, 0.0) if t else 0.0 for t in c])
    sim = np.array([0.0 if t == "" else float((tf.transform([items[i]["question"]]) @ tf.transform([t]).T).toarray()[0, 0]) for i, t in zip(q, c)])
    pf = np.array([items[i]["p_fpq"] for i in q])
    X = np.stack([states[k] for k in keys]); s = pass_probe(X)
    s0 = pass_probe(np.stack([states[plain_key[i]] for i in q]))
    plan = np.array([rows[k]["plan"] for k in keys]).astype(float)
    plan0 = np.array([rows[plain_key[i]]["plan"] for i in q]).astype(float)
    T = np.c_[plain, eff, sim]
    return {"T": T, "T+P": np.c_[T, s, s - s0, pf], "T+P+O": np.c_[T, s, s - s0, pf, plan, plan0, plan * pf, plan0 * pf]}


def main(role, memories):
    # Shared inputs and states
    items_all = {x["id"]: x for x in json.loads((V4 / "items.json").read_text())}
    norm = {r["id"]: r for r in json.loads((AUDIT / "normalized.json").read_text())["items"]}
    reg = json.load(open(CM / json.load(open(P4 / "run/qwen3_8b/identity.json"))["full_config"]["registry"]))
    base = reg["transport_common"] + "\n\n" + reg["prompts"]["plain"]
    sha = lambda *xs: hashlib.sha256("\x1f".join(xs).encode()).hexdigest()[:32]
    plans = {json.loads(l)["k"]: json.loads(l)["plan"] for f in sorted(V4.glob("plans*.jsonl")) for l in f.read_text().splitlines()}
    index = json.loads((V4 / "e13_states" / "index.json").read_text())
    states = {}
    for name in sorted({v["file"] for v in index.values()}):
        states.update({k: v.float().numpy() for k, v in load_file(str(V4 / "e13_states" / name)).items()})

    # Training pairs: outcomes, the pass probe, context effects, TF-IDF
    tr_items = {i: x for i, x in items_all.items() if x["role"] == "train"}
    tr_rows, _ = pairs_of("train", TRAIN_ROLLOUTS.split(","), tr_items, base, sha, plans, norm)
    tr_keys = [k for k in tr_rows if k in states]
    y_tr = np.array([tr_rows[k]["rating"] >= 4 for k in tr_keys]).astype(int)
    X_tr = np.stack([states[k] for k in tr_keys])
    pass_probe = lambda X: probe(X_tr, y_tr, X)
    plain_tr = {i: sha(base, x["question"]) for i, x in tr_items.items()}
    gains = {}
    for k in tr_keys:
        r = tr_rows[k]
        if r["context"]:
            gains.setdefault(r["context"], []).append((r["rating"] >= 4) - (tr_rows[plain_tr[r["id"]]]["rating"] >= 4))
    effect = {t: float(np.mean(v)) for t, v in gains.items()}
    ev_items = {i: x for i, x in items_all.items() if x["role"] == role}
    ev_rows, per = pairs_of(role, memories.split(","), ev_items, base, sha, plans, norm)
    ev_keys = [k for k in ev_rows if k in states]
    tf = TfidfVectorizer(sublinear_tf=True).fit([x["question"] for x in items_all.values() if x["role"] in ("train", role)]
                                                + sorted({r["context"] for r in list(tr_rows.values()) + list(ev_rows.values())} - {""}))
    plain_ev = {i: sha(base, x["question"]) for i, x in ev_items.items()}
    F_tr = features(tr_rows, tr_keys, tr_items, states, effect, tf, plain_tr, pass_probe)
    F_ev = features(ev_rows, ev_keys, ev_items, states, effect, tf, plain_ev, pass_probe)
    print(role, "candidates", len(ev_keys), "| questions", len({ev_rows[k]["id"] for k in ev_keys}), "| contexts unseen in training",
          len({ev_rows[k]["context"] for k in ev_keys if ev_rows[k]["context"] and ev_rows[k]["context"] not in effect}), flush=True)

    # Choice per question and macro of the chosen answers
    q = np.array([ev_rows[k]["id"] for k in ev_keys]); r = np.array([ev_rows[k]["rating"] for k in ev_keys])
    lab = {i: x["label"] for i, x in ev_items.items()}
    def macro(pass_of):
        p = {L: np.mean([pass_of[i] for i in pass_of if lab[i] == L]) * 100 for L in (0, 1)}
        return round((p[0] + p[1]) / 2, 2), round(p[1], 1), round(p[0], 1)
    res = {"role": role, "candidates": len(ev_keys), "by": {}}
    chosen = {}
    for name in F_tr:
        m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000)).fit(F_tr[name], y_tr)
        s = m.predict_proba(F_ev[name])[:, 1]
        pick = {g: int(np.argmax(np.where(q == g, s, -1))) for g in np.unique(q)}
        chosen[name] = {g: int(r[j] >= 4) for g, j in pick.items()}
        res["by"][name] = macro(chosen[name])
    mems = memories.split(",")
    res["single_memories_mean"] = round(float(np.mean([macro({i: per[m][i]["rating"] >= 4 for i in ev_items})[0] for m in mems])), 2)
    res["t0"] = macro({i: per["t0"][i]["rating"] >= 4 for i in ev_items})
    res["plain"] = macro({i: per["plain"][i]["rating"] >= 4 for i in ev_items})
    res["random_candidate"] = macro({g: float(np.mean(r[q == g] >= 4)) for g in np.unique(q)})
    res["best_candidate_upper_bound"] = macro({g: bool((r[q == g] >= 4).any()) for g in np.unique(q)})
    rng = np.random.default_rng(0); qs = np.unique(q); diffs = []
    for _ in range(2000):
        draw = rng.choice(qs, len(qs))
        d = {L: [chosen["T+P+O"][g] - chosen["T"][g] for g in draw if lab[g] == L] for L in (0, 1)}
        diffs.append(50 * (np.mean(d[0]) + np.mean(d[1])))  # macro points
    res["T+P+O - T"] = {"macro_diff": round(res["by"]["T+P+O"][0] - res["by"]["T"][0], 2),
                        "ci95": [round(float(np.percentile(diffs, 2.5)), 2), round(float(np.percentile(diffs, 97.5)), 2)]}
    (V4 / f"e15_{role}.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main(*sys.argv[1:3])
