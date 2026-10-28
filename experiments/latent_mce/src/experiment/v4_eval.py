"""Generate and grade one role under several conditions (Plain, our cell versions, text MCE) for the Latent-MCE run.

Each condition is a contexts file from v4_contexts.py ({id: {"cell", "context"}}; "plain" needs none). The fixed prompt
is transport_common + plain with "\n\nInstructional context:\n{context}" (as latent_mce_optimization_20261025). Unique
(system, question) pairs are split over the given 27B /run servers (greedy, thinking off, max 2048). Luna (medium) grades
with the Well rubric through the cancer-myth WellBridge, blind, in shuffled order, cached by (id, answer) with its text.
Judge code adapted from latent_harness scratch_scripts_2026-10-24/recall_toy.py (an earlier, untracked experiment).
Usage: .venv-fpq/bin/python -m experiments.latent_mce.src.experiment.v4_eval ROLE PORTS COND[=CONTEXTS] ...
       e.g. v4_eval test 8792,8795 plain textmce=ctx_test_textmce.json v1=ctx_test_v1.json   (paths relative to v4/)
       COND@SYSTEM_FILE replaces the whole system prompt (e.g. gepa@gepa_fold0_system.txt).
       CACHE_TAG=<tag> lets two runs work at once: each appends to answers_<tag>.jsonl / judge_<tag>.jsonl and reads all.
       READ_CARDED=1 also reads the fixed Oracle lens (blocks 52/60) on every prompt that carries a context, in the same
       server call, and stores the readouts in plans[_<tag>].jsonl (used as hidden-state feedback for training rollouts).
"""
import hashlib
import json
import os
import random
import sys
import threading
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from common.paths import results_dir

CM = Path(os.environ.get("CANCER_MYTH_DIR", "/home/eagle0914/cancer-myth"))  # private repo 0914eagle/cancer-myth-internals (judge, prompts)
sys.path[:0] = [str(CM), str(CM / "results/paper_v4_qwen3")]
import qwen3_patches as Q  # noqa: E402
import src.fpqa_cli_backend as B  # noqa: E402
from src.fpqa_prompting import parse_rating  # noqa: E402
from src.paper_baselines import WellBridge  # noqa: E402

P4 = Path(os.environ.get("PAPER_V4_DIR", "/data2/heejae/cancer-myth/paper_v4_qwen3"))  # v4 protocol: question bundle, run identity
V4 = results_dir("latent_mce") / "data" / "v4"
LUNA = {"backend": "openai", "model": "gpt-6-luna", "effort": "medium", "timeout_seconds": 600}
CONTEXT_FORMAT = "\n\nInstructional context:\n{context}"


def generate(port, pairs, read=frozenset()):
    """27B results [{"answer", "plan"}] for [(system, question)] from one server; pairs in `read` also get lens plans."""
    body = {"items": [{"system": s, "question": q, "read": (s, q) in read} for s, q in pairs]}
    req = urllib.request.Request(f"http://127.0.0.1:{port}/run", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=24 * 3600) as resp:
        return json.load(resp)["results"]


def main(role, ports, conds):
    # Inputs: items of the role, Cancer-Myth rows, fixed prompt, contexts per condition
    conf = json.load(open(P4 / "run/qwen3_8b/identity.json"))["full_config"]
    reg = json.load(open(CM / conf["registry"]))
    rows = {r["id"]: r for r in json.load(open(P4 / "protocol/bundle.json"))["rows"] if r["dataset"] == "cancer_myth"}
    items = {x["id"]: x for x in json.loads((V4 / "items.json").read_text()) if x["role"] == role}
    base = reg["transport_common"] + "\n\n" + reg["prompts"]["plain"]
    ctx, whole = {}, {}
    for c in conds:
        if "@" in c:
            name, _, path = c.partition("@")
            whole[name] = (V4 / path).read_text()
            ctx[name] = {i: {"cell": None, "context": ""} for i in items}
            continue
        name, _, path = c.partition("=")
        ctx[name] = json.loads((V4 / path).read_text()) if path else {i: {"cell": None, "context": ""} for i in items}
    system = {(n, i): whole.get(n, base) + (CONTEXT_FORMAT.format(context=ctx[n][i]["context"]) if ctx[n][i]["context"] else "") for n in ctx for i in items}
    tag = os.environ.get("CACHE_TAG", "")

    # Generate unique prompts, split round-robin over the servers
    cache_path = V4 / (f"answers_{tag}.jsonl" if tag else "answers.jsonl")
    have = {json.loads(l)["k"]: json.loads(l)["answer"] for f in sorted(V4.glob("answers*.jsonl")) for l in f.read_text().splitlines()}
    key = lambda s, q: hashlib.sha256((s + "\x1f" + q).encode()).hexdigest()[:32]
    plan_path = V4 / (f"plans_{tag}.jsonl" if tag else "plans.jsonl")
    plans = {json.loads(l)["k"]: json.loads(l)["plan"] for f in sorted(V4.glob("plans*.jsonl")) for l in f.read_text().splitlines()}
    want = {(s, rows[i]["question"]) for (n, i), s in system.items() if os.environ.get("READ_CARDED") and ctx[n][i]["context"]}
    read = frozenset(p for p in want if key(*p) not in plans)
    todo = sorted({(s, rows[i]["question"]) for (n, i), s in system.items() if key(s, rows[i]["question"]) not in have} | read)
    parts = [todo[k::len(ports)] for k in range(len(ports))]
    with ThreadPoolExecutor(len(ports)) as pool:
        outs = list(pool.map(lambda pp: generate(pp[0], pp[1], read) if pp[1] else [], zip(ports, parts)))
    with cache_path.open("a") as f, plan_path.open("a") as g:
        for part, out in zip(parts, outs):
            for (s, q), r in zip(part, out):
                if key(s, q) not in have:
                    have[key(s, q)] = r["answer"]
                    f.write(json.dumps({"k": key(s, q), "answer": r["answer"]}, ensure_ascii=False) + "\n")
                if r.get("plan"):
                    plans[key(s, q)] = r["plan"]
                    g.write(json.dumps({"k": key(s, q), "plan": r["plan"]}, ensure_ascii=False) + "\n")
    answers = {(n, i): have[key(s, rows[i]["question"])] for (n, i), s in system.items()}

    # Blind, shuffled, cached Luna Well ratings (judge text kept for training feedback)
    bridge, lock = WellBridge(conf), threading.Lock()
    jpath = V4 / (f"judge_{tag}.jsonl" if tag else "judge.jsonl")
    jc = {json.loads(l)["k"]: json.loads(l) for f in sorted(V4.glob("judge*.jsonl")) for l in f.read_text().splitlines()}

    def judge(pair):
        i, a = pair
        k = hashlib.sha256((i + "\x1f" + a).encode()).hexdigest()[:32]
        if k in jc:
            return jc[k]["rating"]
        sys_msg, user = B.envelope(bridge(op="judge", row=rows[i], answer=a), exact=True)
        rating, text = None, ""
        for _ in range(3):
            text = Q._openai(LUNA, sys_msg, user)[0]
            try:
                rating = parse_rating(text)
            except ValueError:
                continue
            if rating is not None:
                break
        with lock:
            jc[k] = {"k": k, "id": i, "rating": rating, "text": text}
            with jpath.open("a") as f:
                f.write(json.dumps(jc[k], ensure_ascii=False) + "\n")
        return rating
    unique = sorted({(i, a) for (n, i), a in answers.items()})
    random.Random(11).shuffle(unique)
    with ThreadPoolExecutor(16) as pool:
        rated = dict(zip(unique, pool.map(judge, unique)))

    # Per-item ratings and the summary (macro pass, per predicted cell)
    per = {n: {i: {"cell": ctx[n][i]["cell"], "rating": rated[(i, answers[(n, i)])], "has_context": bool(ctx[n][i]["context"])} for i in items} for n in ctx}
    def macro(n, ids):
        f = [per[n][i]["rating"] >= 4 for i in ids if items[i]["label"] == 1]
        g = [per[n][i]["rating"] >= 4 for i in ids if items[i]["label"] == 0]
        F = 100 * sum(f) / len(f) if f else float("nan"); G = 100 * sum(g) / len(g) if g else float("nan")
        return {"macro": round((F + G) / 2, 1), "FPQ": round(F, 1), "NFP": round(G, 1), "n_fpq": len(f), "n_nfp": len(g)}
    summary = {"role": role, "n": len(items), "missing_ratings": sum(r is None for r in rated.values()),
               "conditions": {n: macro(n, list(items)) for n in ctx}}
    (V4 / f"ratings_{role}_{'_'.join(ctx)}.json").write_text(json.dumps({"per_item": per, "summary": summary}, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main(sys.argv[1], [int(p) for p in sys.argv[2].split(",")], sys.argv[3:])
