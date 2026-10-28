"""E25: test-time choice of where to attach a learned context (prereg_v4_test.md, E25).

Contexts: the dev-selected versions of E18-E24 whose test contexts are one text attached to all 242 fold-0 test
questions. A gate decides per question whether that context is attached; the gated answer is the stored with-context
answer or the stored Plain answer (vLLM answers and Luna Well ratings from the e18_loop caches; no new generation).
Gates: all, none, probe (items.json p_fpq >= 0.5), text (TF-IDF logistic regression on the training questions),
qwen (the answering model's yes/no premise check on vLLM, cached in e25_qwen_gate.jsonl), label (upper bound).
Writes results/data/v4/e25.json and prints the summary.
Usage: .venv-fpq/bin/python -m experiments.latent_mce.src.experiment.e25_gate PORT
"""
import glob
import hashlib
import json
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from experiments.latent_mce.src.experiment import e18_loop as L
from experiments.latent_mce.src.experiment.v4_eval import CONTEXT_FORMAT, V4

QWEN_Q = "Does this question rest on a false or mistaken assumption? Answer yes or no.\n\nQuestion: {q}"


def qwen_gate(port, te):
    """{id: True if Qwen answers yes}, cached."""
    path = V4 / "e25_qwen_gate.jsonl"
    have = {json.loads(l)["id"]: json.loads(l)["yes"] for l in path.read_text().splitlines()} if path.exists() else {}
    todo = [i for i in te if i not in have]

    def one(i):
        body = {"model": "qwen36-27b", "messages": [{"role": "user", "content": QWEN_Q.format(q=te[i]["question"])}],
                "temperature": 0, "max_tokens": 8, "chat_template_kwargs": {"enable_thinking": False}}
        req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=600) as resp:
            return i, json.load(resp)["choices"][0]["message"]["content"].strip().lower().startswith("yes")
    with ThreadPoolExecutor(32) as pool, path.open("a") as f:
        for i, y in pool.map(one, todo):
            have[i] = y; f.write(json.dumps({"id": i, "yes": y}) + "\n")
    return have


def main(port):
    # Items, stored answers and ratings
    base, items, by_role, refs, rows, bridge = L.setup()
    allit = {x["id"]: x for x in json.loads((V4 / "items.json").read_text())}
    te = {i: x for i, x in allit.items() if x["role"] == "test"}; tr = {i: x for i, x in allit.items() if x["role"] == "train"}
    ans = {}
    for f in sorted(V4.glob("vllm_answers*.jsonl")):
        if f.name != "vllm_answers_plain_dev100.jsonl":
            ans.update({json.loads(l)["k"]: json.loads(l)["answer"] for l in f.read_text().splitlines()})
    jc = {json.loads(l)["k"]: json.loads(l)["rating"] for f in sorted(V4.glob("judge*.jsonl")) for l in f.read_text().splitlines()}
    def passed(i, system):
        a = ans[L.sha(system, rows[i]["question"])]
        return (jc[hashlib.sha256((i + "\x1f" + a).encode()).hexdigest()[:32]] or 0) >= 4
    plain = {i: passed(i, base) for i in te}

    # Contexts: one text attached to every test question
    ctxs = {}
    for p in sorted(glob.glob(str(V4 / "versions" / "e18_*" / "memory" / "test.json"))):
        t = json.load(open(p)); f = V4 / f"ctx_test_e18_{t['run']}_v{t['selected_version']}.json"
        texts = {c["context"] for c in json.load(open(f)).values()} if f.exists() else set()
        if len(texts) == 1 and "" not in texts:
            ctxs[f"{t['run']}_v{t['selected_version']}"] = texts.pop()
    withc = {name: {i: passed(i, base + CONTEXT_FORMAT.format(context=t)) for i in te} for name, t in ctxs.items()}
    print(len(ctxs), "contexts:", sorted(ctxs), flush=True)

    # Gates
    tf = TfidfVectorizer().fit([x["question"] for x in tr.values()])
    lr = LogisticRegression(class_weight="balanced", max_iter=2000).fit(tf.transform([x["question"] for x in tr.values()]), [x["label"] for x in tr.values()])
    text_p = dict(zip(te, lr.predict_proba(tf.transform([te[i]["question"] for i in te]))[:, 1]))
    qy = qwen_gate(port, te)
    gates = {"all": {i: True for i in te}, "none": {i: False for i in te}, "probe": {i: te[i]["p_fpq"] >= 0.5 for i in te},
             "text": {i: text_p[i] >= 0.5 for i in te}, "qwen": qy, "label": {i: te[i]["label"] == 1 for i in te}}
    lab = np.array([te[i]["label"] for i in te])
    auc = {"probe": roc_auc_score(lab, [te[i]["p_fpq"] for i in te]), "text": roc_auc_score(lab, [text_p[i] for i in te]),
           "qwen": roc_auc_score(lab, [qy[i] for i in te])}
    attach = {g: {"FPQ": round(100 * np.mean([gates[g][i] for i in te if te[i]["label"] == 1]), 1),
                  "NFP": round(100 * np.mean([gates[g][i] for i in te if te[i]["label"] == 0]), 1)} for g in gates}

    # Gated macro per context, means, paired bootstrap of probe - text and probe - all
    ids = list(te)
    def macro(g, name, idx):
        ok = np.array([withc[name][ids[j]] if gates[g][ids[j]] else plain[ids[j]] for j in idx]); y = lab[idx]
        return 50 * (ok[y == 1].mean() + ok[y == 0].mean()), 100 * ok[y == 1].mean(), 100 * ok[y == 0].mean()
    full = np.arange(len(ids))
    table = {g: {name: [round(v, 1) for v in macro(g, name, full)] for name in ctxs} for g in gates}
    mean = {g: [round(float(np.mean([table[g][n][k] for n in ctxs])), 1) for k in range(3)] for g in gates}
    rng = np.random.RandomState(0); boot = {"probe-text": [], "probe-all": [], "probe-qwen": []}
    for _ in range(2000):
        idx = rng.choice(full, len(full))
        m = {g: np.mean([macro(g, n, idx)[0] for n in ctxs]) for g in ("probe", "text", "all", "qwen")}
        boot["probe-text"].append(m["probe"] - m["text"]); boot["probe-all"].append(m["probe"] - m["all"]); boot["probe-qwen"].append(m["probe"] - m["qwen"])
    diff = {k: {"mean": round(float(np.mean(v)), 2), "ci95": [round(float(np.percentile(v, 2.5)), 2), round(float(np.percentile(v, 97.5)), 2)]} for k, v in boot.items()}
    summary = {"contexts": len(ctxs), "gate_auc_test": {k: round(v, 3) for k, v in auc.items()}, "attach_rate_%": attach,
               "mean_macro_FPQ_NFP": mean, "paired_difference_macro": diff}
    (V4 / "e25.json").write_text(json.dumps({"summary": summary, "per_context": table, "contexts": ctxs}, indent=1, ensure_ascii=False))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main(sys.argv[1])
