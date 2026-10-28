"""E18 meta step for one run before iteration K (K >= 2), and the per-version snapshots and test evaluation.

meta RUN K: in an isolated folder versions/e18_<RUN>_meta<K> that holds only meta/history.json (dev100 macro / FPQ /
NFP of every earlier version) and meta/skills/v<j>/SKILL.md (the SKILLs used so far), the meta agent (gpt-6-luna via
codex, sandbox limited to that folder) writes a new SKILL.md; the sections "Memory format", "Output fields" and, in
INT, "Internal fields" are restored verbatim if changed, then the SKILL is copied into the run. Prompt adapted from
metaevo-ai/meta-context-engineering c4b7a7c mce/prompts/meta_agent.py (role, history, overfitting check, crossover).
snapshot RUN K: copies context/, interfaces/ and SKILL.md of version K to memory/snapshots/v<K>.
test RUN: restores the snapshot with the best dev100 macro (ties: later version) and evaluates it on fold-0 test.
Usage: .venv-fpq/bin/python -m experiments.latent_mce.src.experiment.e18_meta {meta|snapshot|test} RUN [K]
"""
import json
import re
import shutil
import subprocess
import sys

import numpy as np

from experiments.latent_mce.src.experiment import e18_loop as L
from experiments.latent_mce.src.experiment.v4_eval import CONTEXT_FORMAT, V4

SKILL = ".claude/skills/learning-context/SKILL.md"
LOCKED = ("## Memory format", "## Output fields", "## Internal fields")
PROMPT = """# Meta-level agent: evolve the context-engineering skill (Cancer-Myth)

An answering model (Qwen3.6-27B) answers patients' cancer questions; some rest on a false premise (FPQ), the rest are
normal (NFP). A grader rewards correcting a real false premise and penalises inventing corrections; the score is
(FPQ pass% + NFP pass%) / 2. A base agent writes the instructional context (context/ and get_context) by following a
skill (SKILL.md). You evolve that skill.

Files (read-only, in ./meta/): history.json (development score of every earlier version: macro, FPQ, NFP) and
skills/v<j>/SKILL.md (the skills used so far). Write ONLY ./SKILL.md.

1. Read the history: what changed between skills and what it did to the development FPQ and NFP rates.
2. Identify the main limits and write procedure steps that address them; combine what worked.
3. Keep the sections "## Memory format", "## Output fields" and "## Internal fields" (if present) exactly as they are.
Finish with a short note: what you changed in the skill and why.
"""


def section(text, head):
    """The section starting at `head` up to the next '## ' heading (or the end)."""
    i = text.find(head)
    if i < 0:
        return None
    j = text.find("\n## ", i + len(head))
    return text[i:] if j < 0 else text[i:j]


def meta(run, k):
    # Isolated folder with only the history and the earlier skills
    ws = V4 / "versions" / f"e18_{run}"
    mf = V4 / "versions" / f"e18_{run}_meta{k}"
    if mf.exists():
        shutil.rmtree(mf)
    (mf / "meta" / "skills").mkdir(parents=True)
    hist = []
    for j in range(int(k)):
        d = ws / "memory" / f"dev_{j}.json"
        if d.exists():
            hist.append(json.loads(d.read_text()))
        snap = ws / "memory" / "snapshots" / f"v{j}" / "SKILL.md"
        if snap.exists():
            (mf / "meta" / "skills" / f"v{j}").mkdir()
            shutil.copy2(snap, mf / "meta" / "skills" / f"v{j}" / "SKILL.md")
    (mf / "meta" / "history.json").write_text(json.dumps(hist, indent=1))
    old = (ws / SKILL).read_text()
    shutil.copy2(ws / SKILL, mf / "SKILL.md")

    # Meta agent, then restore the locked sections and install the skill
    with open(V4 / f"agent_e18_{run}_meta{k}_events.jsonl", "w") as out, open(V4 / f"agent_e18_{run}_meta{k}_stderr.log", "w") as err:
        subprocess.run(["codex", "exec", "--skip-git-repo-check", "--json", "-C", str(mf), "--sandbox", "workspace-write",
                        "-c", 'model="gpt-6-luna"', "-c", 'model_reasoning_effort="medium"', "-"], input=PROMPT.encode(),
                       stdout=out, stderr=err, check=False)
    new = (mf / "SKILL.md").read_text()
    restored = []
    for head in LOCKED:
        want, got = section(old, head), section(new, head)
        if want and got != want:
            new = new.replace(got, want) if got else new.rstrip() + "\n\n" + want
            restored.append(head)
    (ws / SKILL).write_text(new)
    print(run, "meta", k, "| skill lines", len(new.splitlines()), "| locked sections restored:", restored or "none")


def snapshot(run, k):
    ws = V4 / "versions" / f"e18_{run}"; snap = ws / "memory" / "snapshots" / f"v{k}"
    if snap.exists():
        shutil.rmtree(snap)
    shutil.copytree(ws / "context", snap / "context"); shutil.copytree(ws / "interfaces", snap / "interfaces")
    shutil.copy2(ws / SKILL, snap / "SKILL.md")
    print(run, "snapshot", k)


def test(run):
    # Version with the best dev100 macro, restored into a scratch copy of the workspace
    ws = V4 / "versions" / f"e18_{run}"
    devs = {int(re.search(r"dev_(\d+)", p.name).group(1)): json.loads(p.read_text()) for p in (ws / "memory").glob("dev_*.json")}
    best = max(devs, key=lambda j: (devs[j]["macro"], j))
    tmp = V4 / "versions" / f"e18_{run}_testv{best}"
    if tmp.exists():
        shutil.rmtree(tmp)
    shutil.copytree(ws, tmp, ignore=shutil.ignore_patterns("memory", "__pycache__"))
    for d in ("context", "interfaces"):
        shutil.rmtree(tmp / d); shutil.copytree(ws / "memory" / "snapshots" / f"v{best}" / d, tmp / d)
    base, items, by_role, refs, rows, bridge = L.setup()
    te = {x["id"]: x for x in json.loads((V4 / "items.json").read_text()) if x["role"] == "test"}
    import subprocess as sp
    out = V4 / f"ctx_test_e18_{run}_v{best}.json"
    sp.run([L.PY, str(L.CONTEXTS), str(V4 / "items.json"), "test", "nocell", str(out)], cwd=tmp, check=True, stdout=sp.DEVNULL)
    ctx = {i: c["context"] for i, c in json.loads(out.read_text()).items()}
    pairs = [(i, base + (CONTEXT_FORMAT.format(context=ctx[i]) if ctx[i] else "")) for i in te]
    rated = L.answer_and_rate(pairs, rows, bridge)
    ok = lambda p: (rated[p][1] or 0) >= 4
    f = [ok(p) for p in pairs if te[p[0]]["label"] == 1]; g = [ok(p) for p in pairs if te[p[0]]["label"] == 0]
    res = {"run": run, "selected_version": best, "dev": devs[best], "test_macro": round(50 * (np.mean(f) + np.mean(g)), 1),
           "test_FPQ": round(100 * np.mean(f), 1), "test_NFP": round(100 * np.mean(g), 1)}
    (ws / "memory" / "test.json").write_text(json.dumps(res))
    print(json.dumps(res))


if __name__ == "__main__":
    {"meta": lambda: meta(sys.argv[2], sys.argv[3]), "snapshot": lambda: snapshot(sys.argv[2], sys.argv[3]),
     "test": lambda: test(sys.argv[2])}[sys.argv[1]]()
