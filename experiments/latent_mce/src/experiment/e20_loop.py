"""E20 agent and meta steps for one run (versions/e18_<OUT2|TAB>_<rep>) and iteration K (prereg_v4_test.md, E20).

agent RUN K: as E18 (gpt-6-luna medium via codex, e18_agent_prompt.md, data/memory.md inlined); TAB also gets
data/sentence_table.md inlined when it exists.
meta RUN K: as E18's meta step, but meta/history.json holds, per earlier version, the training score (the 80 items that
version answered, from memory/records.jsonl) next to the dev100 score, and the prompt asks for the MCE overfitting
check. Locked sections ("Memory format", "Output fields", "Sentence table") are restored verbatim if changed. Prompt
adapted from metaevo-ai/meta-context-engineering c4b7a7c mce/prompts/meta_agent.py.
Rollout, dev, snapshot and test are the E18 steps (e18_loop, e18_meta).
Usage: .venv-fpq/bin/python -m experiments.latent_mce.src.experiment.e20_loop {agent|meta} RUN K
"""
import json
import shutil
import subprocess
import sys

from experiments.latent_mce.src.experiment.e18_meta import SKILL, section
from experiments.latent_mce.src.experiment.v4_eval import V4

LOCKED = ("## Memory format", "## Output fields", "## Sentence table")
PROMPT = """# Meta-level agent: evolve the context-engineering skill (Cancer-Myth)

An answering model (Qwen3.6-27B) answers patients' cancer questions; some rest on a false premise (FPQ), the rest are
normal (NFP). A grader rewards correcting a real false premise and penalises inventing corrections; the score is
(FPQ pass% + NFP pass%) / 2. A base agent writes the instructional context (context/ and get_context) by following a
skill (SKILL.md). You evolve that skill.

Files (read-only, in ./meta/): history.json (for every earlier version: its training score on the 80 training questions
it answered, and its development score on held-out questions; macro, FPQ, NFP) and skills/v<j>/SKILL.md (the skills
used so far). Write ONLY ./SKILL.md.

1. Read the history: what changed between skills and what it did to the training and development FPQ and NFP rates.
2. Overfitting check: a version whose training score rose while its development score fell, or that attached no
   context on the development questions, learned the training questions rather than a rule; find what in the skill
   allowed that.
3. Identify the main limits and write procedure steps that address them; combine what worked.
4. Keep the sections "## Memory format", "## Output fields" and "## Sentence table" (if present) exactly as they are.
Finish with a short note: what you changed in the skill and why.
"""


def agent(run, k):
    # Prompt: the E18 base-agent prompt, memory.md and (TAB) the sentence table, all inlined
    ws = V4 / "versions" / f"e18_{run}"
    prompt = (V4 / "e18_agent_prompt.md").read_text() + "\n\n## data/memory.md (full copy)\n" + (ws / "data" / "memory.md").read_text()
    table = ws / "data" / "sentence_table.md"
    if run.startswith("TAB") and table.exists():
        prompt += "\n\n## data/sentence_table.md (full copy)\n" + table.read_text()
    (V4 / f"e18_prompt_{run}_{k}.md").write_text(prompt)

    # Base agent in the workspace
    with open(V4 / f"agent_e18_{run}_{k}_events.jsonl", "w") as out, open(V4 / f"agent_e18_{run}_{k}_stderr.log", "w") as err:
        subprocess.run(["codex", "exec", "--skip-git-repo-check", "--json", "-C", str(ws), "--sandbox", "workspace-write",
                        "-c", 'model="gpt-6-luna"', "-c", 'model_reasoning_effort="medium"', "-"], input=prompt.encode(),
                       stdout=out, stderr=err, check=False)
    src = (ws / "interfaces" / "get_context.py").read_text()
    print(run, "agent done | get_context calls utils.llm:", "utils.llm" in src or "from utils import llm" in src)


def meta(run, k):
    # Isolated folder: history (training + dev100 score per earlier version) and the earlier skills
    ws = V4 / "versions" / f"e18_{run}"
    mf = V4 / "versions" / f"e18_{run}_meta{k}"
    if mf.exists():
        shutil.rmtree(mf)
    (mf / "meta" / "skills").mkdir(parents=True)
    recs = [json.loads(l) for l in (ws / "memory" / "records.jsonl").read_text().splitlines()]
    hist = []
    for j in range(int(k)):
        g = [r for r in recs if r["version"] == j]
        f = [r["pass"] for r in g if r["label"] == "FPQ"]; n = [r["pass"] for r in g if r["label"] == "NFP"]
        d = json.loads((ws / "memory" / f"dev_{j}.json").read_text())
        hist.append({"version": j, "train": {"macro": round(50 * (sum(f) / len(f) + sum(n) / len(n)), 1), "FPQ": round(100 * sum(f) / len(f), 1),
                                             "NFP": round(100 * sum(n) / len(n), 1), "questions": len(g)},
                     "dev": {"macro": d["macro"], "FPQ": d["FPQ"], "NFP": d["NFP"], "questions_with_context": d["with_context"]}})
        snap = ws / "memory" / "snapshots" / f"v{j}" / "SKILL.md"
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


if __name__ == "__main__":
    {"agent": agent, "meta": meta}[sys.argv[1]](sys.argv[2], sys.argv[3])
