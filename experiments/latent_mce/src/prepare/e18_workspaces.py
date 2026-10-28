"""E18 workspaces: empty-memory MCE runs for the output arm (OUT) and the internal arm (INT) (prereg_v4_test.md, E18).

Each versions/e18_<arm>_<rep> starts empty: context/ (free format), interfaces/get_context.py returning "" (no context),
utils/llm.py (gpt-6-luna, for the agent's own analysis scripts only), utils/qwen.py (the answering model on the vLLM
servers; the only model get_context may call), data/ (filled by the loop) and SKILL.md. The two SKILLs share every
sentence of the procedure; INT adds one paragraph that defines the internal fields.
Usage: .venv-fpq/bin/python -m experiments.latent_mce.src.prepare.e18_workspaces REP[,REP...]
"""
import shutil
import sys

from common.paths import results_dir

V4 = results_dir("latent_mce") / "data" / "v4"
GET_CONTEXT = '''"""Return the instructional context appended to the answering model's system prompt for one question ("" = none)."""


def get_context(question: str, latent: dict) -> str:
    return ""
'''
QWEN = '''"""The answering model itself (Qwen3.6-27B on the vLLM servers; greedy, thinking off). get_context may call only this."""
import hashlib
import json
import os
import urllib.request

PORTS = [int(p) for p in os.environ.get("E18_GEN_PORTS", "8802").split(",")]


def call_qwen(prompt: str, system: str = "You are a careful assistant.", max_tokens: int = 64) -> str:
    port = PORTS[int(hashlib.sha256(prompt.encode()).hexdigest(), 16) % len(PORTS)]
    body = {"model": "qwen36-27b", "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
            "temperature": 0, "max_tokens": max_tokens, "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=3600) as resp:
        return json.load(resp)["choices"][0]["message"]["content"]
'''
SKILL = """# Skill: learn context from accumulated experience (Cancer-Myth)

## Memory format
- context/ holds any files you choose (notes, items, rules); the format is free.
- interfaces/get_context.py defines get_context(question, latent) -> str: the text appended to the answering model's
  system prompt as "Instructional context" ("" = nothing). It may read files in context/ and may call only
  utils.qwen.call_qwen (the answering model itself); it must not call utils.llm. Return at most 1,500 characters.

## Skill Overview
Learn context that makes the answering model correct a real false premise and answer normal questions directly.
The score is (FPQ pass rate + NFP pass rate) / 2 under the grader (pass >= 4 of 5).

## Procedure
1. Read data/memory.md: the per-context summary of this iteration, the timeline of every training question that failed
   or changed, and retrieved past cases. Items are tagged [observed] (this version), [retrieved] (a record of another
   question or an earlier version) or [unknown].
2. For each failing or changed question, read its timeline and the retrieved cases: what was tried before, what changed,
   what happened to the answer and the rating.
3. Group the problems. For each group decide one change: add or change knowledge, change instructions, change when
   context is attached (narrower or wider), or attach nothing.
4. Write context/ and interfaces/get_context.py.
5. Write context/decisions.md with one line for every question listed under Timelines: the question id, what your
   change does for it (or "no change"), and the memory fields you relied on for it.

## Output fields
behavior = what the answer did about the premise: corrects_target (says the reference premise is false),
rejects_user_premise (says something in the question is wrong although it is not the reference premise, e.g. disputes a
clinician's plan), adds_caveat (adds qualifications without saying anything is wrong), complies (accepts the premise
and answers), hedges_or_refuses, unclear; with a quote from the answer.
"""
INTERNAL = """
## Internal fields (read from the answering model before it answers, on the same input)
p_fpq = a probe's probability, from the internal state without any context, that the question has a false premise.
probe_pass = a probe's probability, from the internal state with the context, that the answer passes.
reading = Oracle-lens readout of that state (blocks 36, 44, 52, 60); interpreted fields: correction_plan (yes / no /
unclear), target (target_premise / user_statement / other_content / none / unclear), change vs no context and vs the previous version,
with quotes. answer_vs_reading compares the answer's behavior with the reading. Readings are noisy.
"""


def main(reps):
    # One empty workspace per arm and replicate
    for rep in reps.split(","):
        for arm in ("OUT", "INT"):
            ws = V4 / "versions" / f"e18_{arm}_{rep}"
            assert not ws.exists()
            for d in ("context", "interfaces", "utils", "data", "memory", ".claude/skills/learning-context"):
                (ws / d).mkdir(parents=True)
            (ws / "interfaces" / "get_context.py").write_text(GET_CONTEXT)
            (ws / "interfaces" / "__init__.py").write_text("from .get_context import get_context\n")
            shutil.copy2(V4 / "versions" / "t1" / "utils" / "llm.py", ws / "utils" / "llm.py")
            (ws / "utils" / "qwen.py").write_text(QWEN)
            (ws / "utils" / "__init__.py").write_text("")
            (ws / ".claude/skills/learning-context/SKILL.md").write_text(SKILL + (INTERNAL if arm == "INT" else ""))
            print(ws.name, "ready")


if __name__ == "__main__":
    main(sys.argv[1])
