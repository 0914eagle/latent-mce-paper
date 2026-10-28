"""E18 loop steps for one run (versions/e18_<ARM>_<rep>) and one iteration k (prereg_v4_test.md, E18).

rollout RUN K: the current memory (version k-1) answers the iteration's 80 training items (40 FPQ / 40 NFP, seed 1000+k,
the same for every run): contexts from the workspace's get_context, answers from vLLM, Luna Well ratings, the fixed
answer-behaviour summary; INT also gets, on the same input re-run on the HF servers, the four-block Oracle-lens reading
(lens4_e18.jsonl), the with-context state (pass probe) and the fixed reading interpretation. One record per item is
appended to memory/records.jsonl, then data/memory.md is rendered (per-context summary, timelines of failed or changed
questions, retrieved cases; items tagged [observed] / [retrieved]).
agent RUN K: the base agent (gpt-6-luna medium via codex) revises context/ and get_context with memory.md inlined.
dev RUN K: the revised memory answers dev100 (vLLM + Luna); writes memory/dev_<k>.json.
Usage: .venv-fpq/bin/python -m experiments.latent_mce.src.experiment.e18_loop {rollout|agent|dev} RUN K
"""
import hashlib
import json
import os
import random
import subprocess
import sys
import threading
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from safetensors.torch import load_file

from common.paths import DATA_DIR, ROOT
from experiments.latent_mce.src.experiment.cancer_fixability import AUDIT, FRAME
from experiments.latent_mce.src.experiment.e13_analysis import probe
from experiments.latent_mce.src.experiment.e15_select import pairs_of
from experiments.latent_mce.src.experiment.e17_eval import ask
from experiments.latent_mce.src.experiment.e18_tools import answer_behaviour, interpret_reading, reference
from experiments.latent_mce.src.experiment.v4_eval import B, CM, CONTEXT_FORMAT, LUNA, P4, Q, V4, WellBridge, parse_rating

GEN_PORTS = [int(x) for x in os.environ.get("E18_GEN_PORTS", "8802,8803").split(",")]
LENS_PORT = int(os.environ.get("E18_LENS_PORT", "8796")); HIDDEN_PORT = 8798
LAYERS = [20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60]
CONTEXTS = V4.parents[2] / "src" / "experiment" / "v4_contexts.py"
PY = str(ROOT / ".venv-fpq" / "bin" / "python")
sha = lambda *xs: hashlib.sha256("\x1f".join(xs).encode()).hexdigest()[:32]
LOCK = threading.Lock()


def setup():
    """Prompts, items, references, Cancer-Myth rows, judge bridge."""
    conf = json.load(open(P4 / "run/qwen3_8b/identity.json"))["full_config"]
    reg = json.load(open(CM / conf["registry"]))
    base = reg["transport_common"] + "\n\n" + reg["prompts"]["plain"]
    items = {x["id"]: x for x in json.loads((V4 / "items.json").read_text()) if x["role"] in ("train", "dev100")}
    by_role = {r: {x["id"]: x for x in json.loads((V4 / "items.json").read_text()) if x["role"] == r} for r in ("train", "dev100")}
    refs = {json.loads(l)["id"]: json.loads(l) for l in (DATA_DIR / "fpq" / "cancer_myth_questions.jsonl").read_text().splitlines()}
    rows = {r["id"]: r for r in json.load(open(P4 / "protocol/bundle.json"))["rows"] if r["dataset"] == "cancer_myth"}
    return base, items, by_role, refs, rows, WellBridge(conf)


def contexts(run, role):
    """{id: context} from the workspace's get_context (subprocess inside the workspace)."""
    out = V4 / f"ctx_{role}_e18_{run}_now.json"
    subprocess.run([PY, str(CONTEXTS), str(V4 / "items.json"), role, "nocell", str(out)], cwd=V4 / "versions" / f"e18_{run}", check=True,
                   stdout=subprocess.DEVNULL)
    return {i: c["context"] for i, c in json.loads(out.read_text()).items()}


def answer_and_rate(pairs, rows, bridge):
    """[(id, system)] -> {(id, system): (answer, rating)} with the vLLM answer cache and the Luna judge cache."""
    tag = os.environ.get("E18_CACHE_TAG", "")  # one cache file per concurrent process; every file is read
    apath, jpath = V4 / f"vllm_answers{('_' + tag) if tag else ''}.jsonl", V4 / f"judge_vllm{('_' + tag) if tag else ''}.jsonl"
    have = {json.loads(l)["k"]: json.loads(l)["answer"] for f in sorted(V4.glob("vllm_answers*.jsonl")) if f.name != "vllm_answers_plain_dev100.jsonl"
            for l in f.read_text().splitlines()}
    jc = {json.loads(l)["k"]: json.loads(l) for f in sorted(V4.glob("judge*.jsonl")) for l in f.read_text().splitlines()}
    todo = sorted({(s, rows[i]["question"]) for i, s in pairs if sha(s, rows[i]["question"]) not in have})
    with apath.open("a") as f, ThreadPoolExecutor(128) as pool:
        def gen(jp):
            j, (s, q) = jp
            a = ask(GEN_PORTS[j % len(GEN_PORTS)], s, q)
            with LOCK:
                have[sha(s, q)] = a; f.write(json.dumps({"k": sha(s, q), "answer": a}, ensure_ascii=False) + "\n"); f.flush()
        list(pool.map(gen, enumerate(todo)))

    def judge(p):
        i, s = p; a = have[sha(s, rows[i]["question"])]
        k = hashlib.sha256((i + "\x1f" + a).encode()).hexdigest()[:32]
        if k in jc:
            return a, jc[k]["rating"]
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
        with LOCK:
            jc[k] = {"k": k, "id": i, "rating": rating, "text": text}
            with jpath.open("a") as g:
                g.write(json.dumps(jc[k], ensure_ascii=False) + "\n")
        return a, rating
    order = list(pairs); random.Random(11).shuffle(order)
    with ThreadPoolExecutor(16) as pool:
        return dict(zip(order, pool.map(judge, order)))


def read_lens4(pairs):
    """[(system, question)] -> {key: {"36","44","52","60"}} from the lens-only server, cached in lens4_e18.jsonl."""
    path = V4 / f"lens4_e18_{LENS_PORT}.jsonl"
    have = {json.loads(l)["k"]: json.loads(l)["plan"] for f in sorted(V4.glob("lens4_e18*.jsonl")) for l in f.read_text().splitlines()}
    todo = sorted({sha(s, q): (s, q) for s, q in pairs if sha(s, q) not in have}.items())
    for c in range(0, len(todo), 48):
        chunk = todo[c:c + 48]
        body = {"items": [{"system": s, "question": q, "read": True, "gen": False} for _, (s, q) in chunk]}
        req = urllib.request.Request(f"http://127.0.0.1:{LENS_PORT}/run", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=24 * 3600) as resp:
            res = json.load(resp)["results"]
        with path.open("a") as f:
            for (k, _), r in zip(chunk, res):
                assert r["plan"] and set(r["plan"]) == {"36", "44", "52", "60"}, k
                have[k] = r["plan"]; f.write(json.dumps({"k": k, "plan": r["plan"]}) + "\n")
    return have


def states_for(pairs):
    """[(system, question)] -> {key: [11, 5120]} with-context states at the last prompt token (serve27b_hidden /hidden)."""
    out = V4 / "e18_states"; out.mkdir(exist_ok=True)
    groups = {}
    for s, q in pairs:
        groups.setdefault(s, set()).add(q)
    got = {}
    for s, qs in groups.items():
        qs = sorted(qs); path = out / f"{hashlib.sha256((s + ''.join(qs)).encode()).hexdigest()[:16]}.safetensors"
        if not path.exists():
            body = {"system": s, "users": qs, "keys": [sha(s, q) for q in qs], "layers": LAYERS, "out": str(path)}
            req = urllib.request.Request(f"http://127.0.0.1:{HIDDEN_PORT}/hidden", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=24 * 3600) as resp:
                assert "out" in json.load(resp)
        got.update({k: v.float().numpy() for k, v in load_file(str(path)).items()})
    return got


def pass_probe(base, items_train):
    """Pass probe fit once on the earlier training pairs (no context, t0, E9 memories; their stored states)."""
    norm = {r["id"]: r for r in json.loads((AUDIT / "normalized.json").read_text())["items"]}
    plans = {json.loads(l)["k"]: json.loads(l)["plan"] for f in sorted(V4.glob("plans*.jsonl")) for l in f.read_text().splitlines()}
    rows, _ = pairs_of("train", ["t0", "e9a_T", "e9a_TP", "e9a_TPO", "e9b_T", "e9b_TP", "e9b_TPO"], items_train, base, sha, plans, norm)
    st = {}
    for path in sorted((V4 / "e13_states").glob("*.safetensors")):
        if not path.name.startswith(("dev100_", "test_")):
            st.update({k: v.float().numpy() for k, v in load_file(str(path)).items()})
    keys = [k for k in rows if k in st]
    X = np.stack([st[k] for k in keys]); y = np.array([rows[k]["rating"] >= 4 for k in keys]).astype(int)
    return lambda Z: probe(X, y, Z)


def rollout(run, k):
    # Inputs, the iteration's sample (same for every run), contexts of the current memory
    arm = run.split("_")[0]
    base, items, by_role, refs, rows, bridge = setup()
    train = by_role["train"]
    rng = random.Random(1000 + int(k))
    sample = sorted(rng.sample([i for i in train if train[i]["label"] == 1], 40) + rng.sample([i for i in train if train[i]["label"] == 0], 40))
    ctx = contexts(run, "train")
    system = {i: base + (CONTEXT_FORMAT.format(context=ctx[i]) if ctx[i] else "") for i in sample}
    rated = answer_and_rate([(i, system[i]) for i in sample], rows, bridge)
    ref = {i: reference(train[i]["label"], refs[i]["correction"] if train[i]["label"] else "") for i in sample}
    store = V4 / "versions" / f"e18_{run}" / "memory" / "records.jsonl"
    past = [json.loads(l) for l in store.read_text().splitlines()] if store.exists() else []
    last = {}
    for r in past:
        last[r["q"]] = r

    # Output summary (both arms) and internal observations (INT) on the same input
    with ThreadPoolExecutor(16) as pool:
        beh = dict(zip(sample, pool.map(lambda i: answer_behaviour(train[i]["question"], ref[i], rated[(i, system[i])][0]), sample)))
    internal = {}
    if arm == "INT":
        lens = read_lens4([(system[i], train[i]["question"]) for i in sample] + [(base, train[i]["question"]) for i in sample])
        st = states_for([(system[i], train[i]["question"]) for i in sample])
        pp = pass_probe(base, train)
        scores = dict(zip(sample, pp(np.stack([st[sha(system[i], train[i]["question"])] for i in sample]))))
        def interp(i):
            cur = lens[sha(system[i], train[i]["question"])]; plain = lens[sha(base, train[i]["question"])]
            prev = last[i]["internal"]["reading_raw"] if i in last and last[i].get("internal") else None
            return interpret_reading(train[i]["question"], ref[i], plain, cur if ctx[i] else plain, prev)
        with ThreadPoolExecutor(16) as pool:
            reads = dict(zip(sample, pool.map(interp, sample)))
        for i in sample:
            cur = lens[sha(system[i], train[i]["question"])]
            b, rd = beh[i].get("behavior"), reads[i]
            rel = ("unclear" if "unclear" in (b, rd.get("correction_plan")) else
                   "agree" if (b in ("corrects_target", "rejects_user_premise")) == (rd.get("correction_plan") == "yes") else
                   f"answer {b}, reading correction_plan {rd.get('correction_plan')}")
            internal[i] = {"p_fpq": round(train[i]["p_fpq"], 3), "probe_pass": round(float(scores[i]), 3), "reading": rd,
                           "reading_raw": cur, "frame_plan": bool(FRAME.search("\n".join(cur.values()))), "answer_vs_reading": rel}

    # Records (one per item, tied to this version's actual input and rating)
    version = int(k) - 1
    import re as _re
    dec_path = V4 / "versions" / f"e18_{run}" / "context" / "decisions.md"
    stated = {}
    if dec_path.exists():
        for line in dec_path.read_text().splitlines():
            m = _re.search(r"\b((?:fpq|nfp)_\d+)\b", line)
            if m:
                stated.setdefault(m.group(1), line.strip()[:400])
    with store.open("a") as f:
        for i in sample:
            a, rating = rated[(i, system[i])]
            rec = {"q": i, "label": "FPQ" if train[i]["label"] else "NFP", "version": version, "iteration": int(k),
                   "context_id": hashlib.sha256(ctx[i].encode()).hexdigest()[:8] if ctx[i] else "none",
                   "context_excerpt": ctx[i][:300], "rating": rating, "pass": rating is not None and rating >= 4,
                   "prev_rating": last[i]["rating"] if i in last else None, "behavior": beh[i],
                   "answer_preview": a[:800], "engine": "vllm-0.18.1 answer; internal = HF re-run of the same input",
                   "stated_decision_for_this_version": stated.get(i), "context_changed_since_last": (i in last and last[i]["context_excerpt"] != ctx[i][:300])}
            if arm == "INT":
                rec["internal"] = internal[i]
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    render(run, int(k), train, refs)


def render(run, k, train, refs):
    """data/memory.md from all records of the run."""
    arm = run.split("_")[0]
    recs = [json.loads(l) for l in (V4 / "versions" / f"e18_{run}" / "memory" / "records.jsonl").read_text().splitlines()]
    now = [r for r in recs if r["iteration"] == k]
    hist = {}
    for r in recs:
        hist.setdefault(r["q"], []).append(r)
    L = [f"# Memory after iteration {k} (training questions; {len(now)} answered with version {k - 1})", ""]

    # Per-context summary of this iteration
    L += ["## This iteration, per context [observed]"]
    by_c = {}
    for r in now:
        by_c.setdefault(r["context_id"], []).append(r)
    for c, rs in sorted(by_c.items(), key=lambda kv: -len(kv[1])):
        f = [r for r in rs if r["label"] == "FPQ"]; n = [r for r in rs if r["label"] == "NFP"]
        line = (f"- context {c} ({rs[0]['context_excerpt'][:80]!r}): {len(f)} FPQ pass {sum(r['pass'] for r in f)}, "
                f"{len(n)} NFP pass {sum(r['pass'] for r in n)}")
        if arm == "INT":
            plan = lambda g: sum(r["internal"]["reading"].get("correction_plan") == "yes" for r in g)
            line += f" | reading correction_plan yes: FPQ {plan(f)}/{len(f)}, NFP {plan(n)}/{len(n)}"
        L.append(line)
    L.append("")

    # Timelines of failed or changed questions
    L += ["## Timelines of training questions that failed or changed this iteration"]
    def output_changed(r):
        prev = hist[r["q"]][-2] if len(hist[r["q"]]) > 1 else None
        return prev is not None and (prev["pass"] != r["pass"] or prev["behavior"].get("behavior") != r["behavior"].get("behavior"))
    def reading_changed(r):
        return arm == "INT" and r["internal"]["reading"].get("change_vs_previous") in ("stronger", "weaker", "different_target")
    show = [r for r in now if not r["pass"] or output_changed(r) or reading_changed(r)]
    for r in sorted(show, key=lambda r: (r["label"], r["q"])):
        x = train[r["q"]]
        L.append(f"### {r['q']} {r['label']} | {x['question'][:220]}")
        if r["label"] == "FPQ":
            L.append(f"reference correction: {refs[r['q']]['correction'][:300]}")
        for h in hist[r["q"]]:
            tag = "[observed]" if h["iteration"] == k else "[observed, earlier version]"
            b = h["behavior"]
            s = (f"- v{h['version']} {tag} context {h['context_id']}: rating {h['rating']}, behavior {b.get('behavior')} "
                 f"\"{(b.get('evidence') or '')[:160]}\"")
            if arm == "INT":
                it = h["internal"]; rd = it["reading"]
                s += (f" | p_fpq {it['p_fpq']}, probe_pass {it['probe_pass']}, reading: correction_plan {rd.get('correction_plan')}, "
                      f"target {rd.get('target')}, change vs no context {rd.get('change_vs_no_context')}, vs previous "
                      f"{rd.get('change_vs_previous')}; \"{' | '.join(rd.get('evidence') or [])[:200]}\"; {it['answer_vs_reading']}")
            L.append(s)
            if h.get("stated_decision_for_this_version"):
                L.append(f"  the agent's stated decision that produced v{h['version']} [stated]: {h['stated_decision_for_this_version']}")
        L.append(f"  answer (v{r['version']}, first 800 characters): {r['answer_preview']}")
        L.append("")

    # Retrieved cases: other questions with the same context, earlier or now
    L += ["## Retrieved cases (other questions with the same context) [retrieved]"]
    for r in sorted(show, key=lambda r: r["q"]):
        if r["context_id"] == "none":
            continue
        same = [h for h in recs if h["context_id"] == r["context_id"] and h["q"] != r["q"]]
        if arm == "INT":
            key = (r["behavior"].get("behavior"), r["internal"]["reading"].get("correction_plan"))
            same.sort(key=lambda h: (h["behavior"].get("behavior"), h["internal"]["reading"].get("correction_plan")) != key)
        else:
            same.sort(key=lambda h: h["behavior"].get("behavior") != r["behavior"].get("behavior"))
        for h in same[:3]:
            s = f"- for {r['q']}: {h['q']} {h['label']} v{h['version']} rating {h['rating']}, behavior {h['behavior'].get('behavior')}"
            if arm == "INT":
                s += f" | reading correction_plan {h['internal']['reading'].get('correction_plan')}, target {h['internal']['reading'].get('target')}"
            L.append(s)
    L += ["", "## Other questions this iteration (passed, output unchanged) [observed]"]
    for r in sorted((r for r in now if r not in show), key=lambda r: r["q"]):
        line = f"- {r['q']} {r['label']} v{r['version']} context {r['context_id']}: rating {r['rating']}, behavior {r['behavior'].get('behavior')}"
        if arm == "INT":
            rd = r["internal"]["reading"]
            line += f" | correction_plan {rd.get('correction_plan')}, target {rd.get('target')}, change vs previous {rd.get('change_vs_previous')}"
        L.append(line)
    (V4 / "versions" / f"e18_{run}" / "data" / "memory.md").write_text("\n".join(L) + "\n")
    print(run, "iteration", k, "records", len(now), "| listed", len(show), "| memory.md chars", sum(len(l) for l in L))


def agent(run, k):
    ws = V4 / "versions" / f"e18_{run}"
    prompt = (V4 / "e18_agent_prompt.md").read_text() + "\n\n## data/memory.md (full copy)\n" + (ws / "data" / "memory.md").read_text()
    (V4 / f"e18_prompt_{run}_{k}.md").write_text(prompt)
    with open(V4 / f"agent_e18_{run}_{k}_events.jsonl", "w") as out, open(V4 / f"agent_e18_{run}_{k}_stderr.log", "w") as err:
        subprocess.run(["codex", "exec", "--skip-git-repo-check", "--json", "-C", str(ws), "--sandbox", "workspace-write",
                        "-c", 'model="gpt-6-luna"', "-c", 'model_reasoning_effort="medium"', "-"], input=prompt.encode(),
                       stdout=out, stderr=err, check=False)
    src = (ws / "interfaces" / "get_context.py").read_text()
    print(run, "agent done | get_context calls utils.llm:", "utils.llm" in src or "from utils import llm" in src)


def dev(run, k):
    base, items, by_role, refs, rows, bridge = setup()
    dv = by_role["dev100"]; ctx = contexts(run, "dev100")
    pairs = [(i, base + (CONTEXT_FORMAT.format(context=ctx[i]) if ctx[i] else "")) for i in dv]
    rated = answer_and_rate(pairs, rows, bridge)
    ok = lambda p: (rated[p][1] or 0) >= 4
    f = [ok(p) for p in pairs if dv[p[0]]["label"] == 1]; g = [ok(p) for p in pairs if dv[p[0]]["label"] == 0]
    res = {"run": run, "iteration": int(k), "macro": round(50 * (np.mean(f) + np.mean(g)), 1), "FPQ": round(100 * np.mean(f), 1),
           "NFP": round(100 * np.mean(g), 1), "with_context": sum(bool(c) for c in ctx.values())}
    (V4 / "versions" / f"e18_{run}" / "memory" / f"dev_{k}.json").write_text(json.dumps(res))
    print(json.dumps(res))


if __name__ == "__main__":
    {"rollout": rollout, "agent": agent, "dev": dev}[sys.argv[1]](sys.argv[2], sys.argv[3])
