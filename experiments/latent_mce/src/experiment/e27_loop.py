"""E27 steps for one run (versions/e18_<OUT5|LAT>_<rep>) and iteration K (prereg_v4_test.md, E27).

LAT runs: every context from get_context is kept only for questions with p_fpq >= 0.5 (items.json, train-only probe),
in rollout, dev100 and test (e18_loop.contexts is wrapped in this process). types: OUT5 = e21_loop.types; LAT =
failure counts with the gated counts, the lens split of the failed FPQs (four-block lens on E18_LENS_PORT, fixed Luna
interpretation; types kept in memory/lens_types.jsonl) and the type change from the previous iteration. meta: OUT5 =
e21_loop.meta; LAT = the E20 meta step with internal aggregates per version in meta/history.json (written before the agent runs). agent =
e21_loop.agent (dev100 coverage rewrite). dev = e18_loop.dev. test = e18_meta.test with the gate.
Usage: E18_GEN_PORTS=... E18_LENS_PORT=... .venv-fpq/bin/python -m experiments.latent_mce.src.experiment.e27_loop {rollout|types|meta|agent|dev|test} RUN [K]
"""
import json
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from experiments.latent_mce.src.experiment import e18_loop as L
from experiments.latent_mce.src.experiment import e20_loop as E20
from experiments.latent_mce.src.experiment import e21_loop as E21
from experiments.latent_mce.src.experiment.e18_meta import SKILL, section
from experiments.latent_mce.src.experiment.e18_tools import interpret_reading, reference
from experiments.latent_mce.src.experiment.v4_eval import CONTEXT_FORMAT, V4

P_FPQ = {x["id"]: x["p_fpq"] for x in json.loads((V4 / "items.json").read_text())}
_contexts = L.contexts


def gated_contexts(run, role):
    """e18_loop.contexts with the probe gate for LAT runs."""
    ctx = _contexts(run, role)
    return {i: (c if P_FPQ[i] >= 0.5 else "") for i, c in ctx.items()} if run.startswith("LAT") else ctx


L.contexts = gated_contexts


def types(run, k):
    if not run.startswith("LAT"):
        return E21.types(run, k)

    # Lens type of every failed FPQ of this iteration, kept per question and iteration
    ws = V4 / "versions" / f"e18_{run}"
    recs = [r for r in map(json.loads, (ws / "memory" / "records.jsonl").read_text().splitlines()) if r["iteration"] == int(k)]
    base, items, by_role, refs, rows, bridge = L.setup()
    train = by_role["train"]
    fails = [r["q"] for r in recs if r["label"] == "FPQ" and not r["pass"]]
    ctx = L.contexts(run, "train")
    system = {i: base + (CONTEXT_FORMAT.format(context=ctx[i]) if ctx[i] else "") for i in fails}
    lens = L.read_lens4([(system[i], train[i]["question"]) for i in fails] + [(base, train[i]["question"]) for i in fails])
    def one(i):
        rd = interpret_reading(train[i]["question"], reference(1, refs[i]["correction"]), lens[L.sha(base, train[i]["question"])], lens[L.sha(system[i], train[i]["question"])], None)
        return {"yes": "intent", "no": "unaware"}.get(rd.get("correction_plan"), "unclear")
    with ThreadPoolExecutor(16) as pool:
        typ = dict(zip(fails, pool.map(one, fails)))
    store = ws / "memory" / "lens_types.jsonl"
    with store.open("a") as f:
        for r in recs:
            f.write(json.dumps({"q": r["q"], "iteration": int(k), "label": r["label"], "pass": r["pass"], "gated": r["context_id"] != "none",
                                "type": typ.get(r["q"], "pass" if r["pass"] else "-")}) + "\n")
    prev = {r["q"]: r for r in map(json.loads, store.read_text().splitlines()) if r["iteration"] == int(k) - 1}

    # Counts per group, gated counts, lens split and type changes
    by_c = {}
    for r in recs:
        by_c.setdefault(r["context_id"], []).append(r)
    groups = {}
    for c, rs in by_c.items():
        key = "no context (gate off or empty)" if c == "none" else (f"context {c} ({rs[0]['context_excerpt'][:80]!r})" if len(rs) >= 3 else "other contexts (each on fewer than 3 questions), pooled")
        groups.setdefault(key, []).extend(rs)
    gated = [r for r in recs if r["context_id"] != "none"]
    out = [f"# Failure counts, iteration {k} (training questions answered with version {int(k) - 1}) [observed]", "",
           f"Gate: context attached to {sum(r['label'] == 'FPQ' for r in gated)} of 40 FPQ and {sum(r['label'] == 'NFP' for r in gated)} of 40 NFP questions.", ""]
    for g, rs in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        f = [r for r in rs if r["label"] == "FPQ"]; n = [r for r in rs if r["label"] == "NFP"]
        out.append(f"## {g}: FPQ {sum(r['pass'] for r in f)}/{len(f)} pass, NFP {sum(r['pass'] for r in n)}/{len(n)} pass")
        for lab, grp in (("FPQ", f), ("NFP", n)):
            by_b = {}
            for r in grp:
                if not r["pass"]:
                    by_b.setdefault(r["behavior"].get("behavior"), []).append(r["q"])
            for b, ids in sorted(by_b.items(), key=lambda kv: -len(kv[1])):
                out.append(f"- {lab} failures, behavior {b}: {len(ids)} ({', '.join(sorted(ids))})")
            if lab == "FPQ":
                for t in ("intent", "unaware", "unclear"):
                    ids = sorted(r["q"] for r in grp if not r["pass"] and typ.get(r["q"]) == t)
                    if ids:
                        out.append(f"- FPQ failures, internal {'correction intent' if t == 'intent' else t}: {len(ids)} ({', '.join(ids)})")
        out.append("")
    moves = sorted(f"{q}: {prev[q]['type']} -> {typ.get(q, 'pass' if r['pass'] else '-')}" for r in recs for q in [r["q"]] if r["label"] == "FPQ" and q in prev)
    out += ["## FPQ questions also answered in the previous iteration: internal type then -> now", *[f"- {m}" for m in moves], ""]
    (ws / "data" / "failure_types.md").write_text("\n".join(out) + "\n")
    print(run, "failure types", k, "| lens", {t: sum(v == t for v in typ.values()) for t in ("intent", "unaware", "unclear")}, "| repeated FPQ", len(moves))


def meta(run, k):
    if not run.startswith("LAT"):
        return E21.meta(run, k)

    # Isolated folder: history (training, dev100 and internal observations per earlier version) and earlier skills
    ws = V4 / "versions" / f"e18_{run}"
    mf = V4 / "versions" / f"e18_{run}_meta{k}"
    if mf.exists():
        shutil.rmtree(mf)
    (mf / "meta" / "skills").mkdir(parents=True)
    recs = [json.loads(l) for l in (ws / "memory" / "records.jsonl").read_text().splitlines()]
    lt = [json.loads(l) for l in (ws / "memory" / "lens_types.jsonl").read_text().splitlines()]
    hist = []
    for j in range(int(k)):
        g = [r for r in recs if r["version"] == j]
        fp = [r["pass"] for r in g if r["label"] == "FPQ"]; nf = [r["pass"] for r in g if r["label"] == "NFP"]
        d = json.loads((ws / "memory" / f"dev_{j}.json").read_text())
        t = [r for r in lt if r["iteration"] == j + 1]; fails = [r for r in t if r["label"] == "FPQ" and not r["pass"]]
        prev = {r["q"]: r["type"] for r in lt if r["iteration"] == j}
        trans = {}
        for r in t:
            if r["label"] == "FPQ" and r["q"] in prev:
                key = f"{prev[r['q']]} -> {r['type']}"; trans[key] = trans.get(key, 0) + 1
        hist.append({"version": j,
                     "train": {"macro": round(50 * (sum(fp) / len(fp) + sum(nf) / len(nf)), 1), "FPQ": round(100 * sum(fp) / len(fp), 1), "NFP": round(100 * sum(nf) / len(nf), 1), "questions": len(g)},
                     "dev": {"macro": d["macro"], "FPQ": d["FPQ"], "NFP": d["NFP"], "questions_with_context": d["with_context"]},
                     "internal": {"gated_FPQ": sum(r["gated"] for r in t if r["label"] == "FPQ"), "gated_NFP": sum(r["gated"] for r in t if r["label"] == "NFP"),
                                  "FPQ_failures_intent": sum(r["type"] == "intent" for r in fails), "FPQ_failures_unaware": sum(r["type"] == "unaware" for r in fails),
                                  "type_changes_from_previous_version": trans}})
        (mf / "meta" / "skills" / f"v{j}").mkdir()
        shutil.copy2(ws / "memory" / "snapshots" / f"v{j}" / "SKILL.md", mf / "meta" / "skills" / f"v{j}" / "SKILL.md")
    (mf / "meta" / "history.json").write_text(json.dumps(hist, indent=1))
    old = (ws / SKILL).read_text()
    shutil.copy2(ws / SKILL, mf / "SKILL.md")

    # Meta agent (E20 prompt naming the internal observations and this arm's locked sections), then restore them
    locked = ("## Memory format", "## Output fields", "## Failure counts", "## Gate", "## Failure types")
    prompt = E20.PROMPT.replace('"## Sentence table"', '"## Failure counts", "## Gate", "## Failure types"').replace(
        "and its development score on held-out questions; macro, FPQ, NFP)",
        "and its development score on held-out questions; macro, FPQ, NFP; and internal observations of its training run:\n"
        "gated questions and, among FPQ failures, the lens types correction intent / unaware and their changes from the\n"
        "previous version)")
    assert prompt != E20.PROMPT
    with open(V4 / f"agent_e18_{run}_meta{k}_events.jsonl", "w") as out, open(V4 / f"agent_e18_{run}_meta{k}_stderr.log", "w") as err:
        subprocess.run(["codex", "exec", "--skip-git-repo-check", "--json", "-C", str(mf), "--sandbox", "workspace-write",
                        "-c", 'model="gpt-6-luna"', "-c", 'model_reasoning_effort="medium"', "-"], input=prompt.encode(),
                       stdout=out, stderr=err, check=False)
    new = (mf / "SKILL.md").read_text()
    restored = []
    for head in locked:
        want, got = section(old, head), section(new, head)
        if want and got != want:
            new = new.replace(got, want) if got else new.rstrip() + "\n\n" + want
            restored.append(head)
    (ws / SKILL).write_text(new)
    print(run, "meta", k, "| skill lines", len(new.splitlines()), "| locked sections restored:", restored or "none")


def test(run):
    # e18_meta.test with the gate for LAT
    ws = V4 / "versions" / f"e18_{run}"
    devs = {int(p.stem.split("_")[1]): json.loads(p.read_text()) for p in (ws / "memory").glob("dev_*.json")}
    best = max(devs, key=lambda j: (devs[j]["macro"], j))
    tmp = V4 / "versions" / f"e18_{run}_testv{best}"
    if tmp.exists():
        shutil.rmtree(tmp)
    shutil.copytree(ws, tmp, ignore=shutil.ignore_patterns("memory", "__pycache__"))
    for d in ("context", "interfaces"):
        shutil.rmtree(tmp / d); shutil.copytree(ws / "memory" / "snapshots" / f"v{best}" / d, tmp / d)
    base, items, by_role, refs, rows, bridge = L.setup()
    te = {x["id"]: x for x in json.loads((V4 / "items.json").read_text()) if x["role"] == "test"}
    out = V4 / f"ctx_test_e18_{run}_v{best}.json"
    subprocess.run([L.PY, str(L.CONTEXTS), str(V4 / "items.json"), "test", "nocell", str(out)], cwd=tmp, check=True, stdout=subprocess.DEVNULL)
    ctx = {i: c["context"] for i, c in json.loads(out.read_text()).items()}
    if run.startswith("LAT"):
        ctx = {i: (c if P_FPQ[i] >= 0.5 else "") for i, c in ctx.items()}
    pairs = [(i, base + (CONTEXT_FORMAT.format(context=ctx[i]) if ctx[i] else "")) for i in te]
    rated = L.answer_and_rate(pairs, rows, bridge)
    ok = lambda p: (rated[p][1] or 0) >= 4
    f = [ok(p) for p in pairs if te[p[0]]["label"] == 1]; g = [ok(p) for p in pairs if te[p[0]]["label"] == 0]
    res = {"run": run, "selected_version": best, "dev": devs[best], "test_macro": round(50 * (np.mean(f) + np.mean(g)), 1),
           "test_FPQ": round(100 * np.mean(f), 1), "test_NFP": round(100 * np.mean(g), 1)}
    (ws / "memory" / "test.json").write_text(json.dumps(res))
    print(json.dumps(res))


if __name__ == "__main__":
    cmd, run = sys.argv[1], sys.argv[2]
    {"rollout": lambda: L.rollout(run, sys.argv[3]), "types": lambda: types(run, sys.argv[3]), "meta": lambda: meta(run, sys.argv[3]),
     "agent": lambda: E21.agent(run, sys.argv[3]), "dev": lambda: L.dev(run, sys.argv[3]), "test": lambda: test(run)}[cmd]()
