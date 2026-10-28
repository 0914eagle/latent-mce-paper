"""E21 steps for one run (versions/e18_<OUT3|FTY>_<rep>) and iteration K (prereg_v4_test.md, E21).

types RUN K: after the E18 rollout of iteration K, writes data/failure_types.md from this iteration's records: per
context group (a context on >= 3 of the 80 questions; other contexts pooled; no context) the FPQ / NFP counts and
passes and the failures by answer behaviour with ids. FTY adds the split of the FPQ failures by the with-context pass
probe (serve27b_hidden /hidden states, e18_loop.pass_probe), split at the group's median: above = possibly_silent,
at or below = likely_unaware (groups with >= 3 FPQ failures).
agent RUN K: the E18 base-agent prompt with data/memory.md and data/failure_types.md inlined (gpt-6-luna medium via
codex); if get_context then attaches context to 0 of the 100 dev100 questions (texts only), the agent is asked once
to rewrite it.
meta RUN K: the E20 meta step with "## Failure counts" and "## Failure types" as locked sections.
Usage: .venv-fpq/bin/python -m experiments.latent_mce.src.experiment.e21_loop {types|agent|meta} RUN K
"""
import json
import random
import subprocess
import sys

import numpy as np

from experiments.latent_mce.src.experiment import e18_loop as L
from experiments.latent_mce.src.experiment import e20_loop as E20
from experiments.latent_mce.src.experiment.v4_eval import CONTEXT_FORMAT, V4

RECHECK = """

## Check after your edit
get_context returned context for 0 of 100 held-out development questions, so its rules fit only the training
questions. Rewrite get_context so that every rule applies to a class of questions that unseen questions can fall into,
and update context/decisions.md to match.
"""


def types(run, k):
    # This iteration's records and, for FTY, the with-context pass-probe score of every sampled question
    ws = V4 / "versions" / f"e18_{run}"
    recs = [r for r in map(json.loads, (ws / "memory" / "records.jsonl").read_text().splitlines()) if r["iteration"] == int(k)]
    score = {}
    if run.startswith("FTY"):
        base, items, by_role, refs, rows, bridge = L.setup()
        train = by_role["train"]
        rng = random.Random(1000 + int(k))
        sample = sorted(rng.sample([i for i in train if train[i]["label"] == 1], 40) + rng.sample([i for i in train if train[i]["label"] == 0], 40))
        assert sample == sorted(r["q"] for r in recs)
        ctx = L.contexts(run, "train")
        pairs = [(base + (CONTEXT_FORMAT.format(context=ctx[i]) if ctx[i] else ""), train[i]["question"]) for i in sample]
        st = L.states_for(pairs)
        s = L.pass_probe(base, train)(np.stack([st[L.sha(p, q)] for p, q in pairs]))
        score = dict(zip(sample, s))

    # Groups: shared contexts, pooled others, no context
    by_c = {}
    for r in recs:
        by_c.setdefault(r["context_id"], []).append(r)
    groups = {}
    for c, rs in by_c.items():
        key = "no context" if c == "none" else (f"context {c} ({rs[0]['context_excerpt'][:80]!r})" if len(rs) >= 3 else "other contexts (each on fewer than 3 questions), pooled")
        groups.setdefault(key, []).extend(rs)

    # Counts per group
    out = [f"# Failure counts, iteration {k} (training questions answered with version {int(k) - 1}) [observed]", ""]
    if score:
        out += ["FPQ failures are also split by the internal probe at each group's median (see SKILL, Failure types):",
                "possibly_silent = above, likely_unaware = at or below. Earlier reliability: lens plan to correct the actual",
                "premise in 22% vs 4%.", ""]
    for g, rs in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        f = [r for r in rs if r["label"] == "FPQ"]; n = [r for r in rs if r["label"] == "NFP"]
        out.append(f"## {g}: FPQ {sum(r['pass'] for r in f)}/{len(f)} pass, NFP {sum(r['pass'] for r in n)}/{len(n)} pass")
        for lab, grp in (("FPQ", f), ("NFP", n)):
            fails = [r for r in grp if not r["pass"]]
            by_b = {}
            for r in fails:
                by_b.setdefault(r["behavior"].get("behavior"), []).append(r["q"])
            for b, ids in sorted(by_b.items(), key=lambda kv: -len(kv[1])):
                out.append(f"- {lab} failures, behavior {b}: {len(ids)} ({', '.join(sorted(ids))})")
            if lab == "FPQ" and score and len(fails) >= 3:
                m = np.median([score[r["q"]] for r in fails])
                sil = sorted(r["q"] for r in fails if score[r["q"]] > m); una = sorted(r["q"] for r in fails if score[r["q"]] <= m)
                out.append(f"- FPQ failures, internal possibly_silent: {len(sil)} ({', '.join(sil)})")
                out.append(f"- FPQ failures, internal likely_unaware: {len(una)} ({', '.join(una)})")
        out.append("")
    (ws / "data" / "failure_types.md").write_text("\n".join(out) + "\n")
    print(run, "failure types", k, "| groups", len(groups), "| internal split", bool(score))


def agent(run, k):
    # Prompt: E18 base-agent prompt, memory.md and failure_types.md inlined
    ws = V4 / "versions" / f"e18_{run}"
    prompt = ((V4 / "e18_agent_prompt.md").read_text() + "\n\n## data/memory.md (full copy)\n" + (ws / "data" / "memory.md").read_text()
              + "\n\n## data/failure_types.md (full copy)\n" + (ws / "data" / "failure_types.md").read_text())

    # Base agent; one rewrite if no dev100 question gets context
    for attempt in (1, 2):
        (V4 / f"e18_prompt_{run}_{k}_{attempt}.md").write_text(prompt)
        with open(V4 / f"agent_e18_{run}_{k}_{attempt}_events.jsonl", "w") as out, open(V4 / f"agent_e18_{run}_{k}_{attempt}_stderr.log", "w") as err:
            subprocess.run(["codex", "exec", "--skip-git-repo-check", "--json", "-C", str(ws), "--sandbox", "workspace-write",
                            "-c", 'model="gpt-6-luna"', "-c", 'model_reasoning_effort="medium"', "-"], input=prompt.encode(),
                           stdout=out, stderr=err, check=False)
        covered = sum(bool(c) for c in L.contexts(run, "dev100").values())
        print(run, "agent", k, "attempt", attempt, "| dev100 questions with context", covered)
        if covered or attempt == 2:
            break
        prompt += RECHECK


def meta(run, k):
    # E20 meta step with this experiment's locked sections
    E20.LOCKED = ("## Memory format", "## Output fields", "## Failure counts", "## Failure types")
    E20.PROMPT = E20.PROMPT.replace('"## Sentence table"', '"## Failure counts", "## Failure types"')
    E20.meta(run, k)


if __name__ == "__main__":
    {"types": types, "agent": agent, "meta": meta}[sys.argv[1]](sys.argv[2], sys.argv[3])
