"""E17 evaluation: answers from the vLLM servers, rated by the same blind, cached Luna Well judge as v4_eval.

Conditions are "plain" or NAME=ctx_<role>_<workspace>.json (contexts chosen by the workspace's get_context). Answers are
greedy, thinking off, max 2048 tokens, with v4_eval's prompt and context block; they are cached in
results/data/v4/vllm_answers.jsonl by sha(system, question) (outside the HF answers*.jsonl cache: different engine).
Ratings: results/data/v4/ratings_vllm_<role>_<conditions>.json (outside the ratings_<role>_*.json pattern).
Usage: .venv-fpq/bin/python -m experiments.latent_mce.src.experiment.e17_eval ROLE PORT[,PORT] COND ...
"""
import hashlib
import json
import random
import sys
import threading
import urllib.request
from concurrent.futures import ThreadPoolExecutor

from experiments.latent_mce.src.experiment.v4_eval import B, CM, CONTEXT_FORMAT, LUNA, P4, Q, V4, WellBridge, parse_rating


def ask(port, system, question):
    body = {"model": "qwen36-27b", "messages": [{"role": "system", "content": system}, {"role": "user", "content": question}],
            "temperature": 0, "max_tokens": 2048, "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=3600) as resp:
        return json.load(resp)["choices"][0]["message"]["content"]


def main(role, ports, conds):
    # Items, prompts, contexts per condition
    conf = json.load(open(P4 / "run/qwen3_8b/identity.json"))["full_config"]
    reg = json.load(open(CM / conf["registry"]))
    rows = {r["id"]: r for r in json.load(open(P4 / "protocol/bundle.json"))["rows"] if r["dataset"] == "cancer_myth"}
    items = {x["id"]: x for x in json.loads((V4 / "items.json").read_text()) if x["role"] == role}
    base = reg["transport_common"] + "\n\n" + reg["prompts"]["plain"]
    ctx = {}
    for c in conds:
        name, _, path = c.partition("=")
        ctx[name] = json.loads((V4 / path).read_text()) if path else {i: {"context": ""} for i in items}
    system = {(n, i): base + (CONTEXT_FORMAT.format(context=ctx[n][i]["context"]) if ctx[n][i]["context"] else "") for n in ctx for i in items}
    key = lambda s, q: hashlib.sha256((s + "\x1f" + q).encode()).hexdigest()[:32]

    # vLLM answers for the prompts not cached yet, spread over the servers
    path = V4 / "vllm_answers.jsonl"
    have = {json.loads(l)["k"]: json.loads(l)["answer"] for l in path.read_text().splitlines()} if path.exists() else {}
    todo = sorted({(s, rows[i]["question"]) for (n, i), s in system.items() if key(s, rows[i]["question"]) not in have})
    lock = threading.Lock()
    with path.open("a") as f, ThreadPoolExecutor(64 * len(ports)) as pool:
        def one(jp):
            j, (s, q) = jp
            a = ask(ports[j % len(ports)], s, q)
            with lock:
                have[key(s, q)] = a
                f.write(json.dumps({"k": key(s, q), "answer": a}, ensure_ascii=False) + "\n"); f.flush()
        list(pool.map(one, enumerate(todo)))
    answers = {(n, i): have[key(s, rows[i]["question"])] for (n, i), s in system.items()}
    print(role, "generated", len(todo), flush=True)

    # Blind, shuffled, cached Luna Well ratings (same request and cache key as v4_eval)
    bridge = WellBridge(conf)
    jpath = V4 / "judge_vllm.jsonl"
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
            with jpath.open("a") as g:
                g.write(json.dumps(jc[k], ensure_ascii=False) + "\n")
        return rating
    unique = sorted({(i, a) for (n, i), a in answers.items()})
    random.Random(11).shuffle(unique)
    with ThreadPoolExecutor(16) as pool:
        rated = dict(zip(unique, pool.map(judge, unique)))

    # Per-item ratings and summary
    per = {n: {i: {"rating": rated[(i, answers[(n, i)])], "has_context": bool(ctx[n][i]["context"])} for i in items} for n in ctx}
    def macro(n):
        f = [per[n][i]["rating"] >= 4 for i in items if items[i]["label"] == 1]; g = [per[n][i]["rating"] >= 4 for i in items if items[i]["label"] == 0]
        return {"macro": round((100 * sum(f) / len(f) + 100 * sum(g) / len(g)) / 2, 1), "FPQ": round(100 * sum(f) / len(f), 1),
                "NFP": round(100 * sum(g) / len(g), 1), "carded": sum(per[n][i]["has_context"] for i in items)}
    summary = {"role": role, "engine": "vllm 0.18.1", "conditions": {n: macro(n) for n in ctx}}
    (V4 / f"ratings_vllm_{role}_{'_'.join(ctx)}.json").write_text(json.dumps({"per_item": per, "summary": summary}, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main(sys.argv[1], [int(p) for p in sys.argv[2].split(",")], sys.argv[3:])
