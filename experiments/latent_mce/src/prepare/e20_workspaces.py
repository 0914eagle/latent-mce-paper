"""E20 workspaces: empty-memory MCE runs for the output arm (OUT2) and the sentence-table arm (TAB) (prereg_v4_test.md, E20).

Same layout as E18 (context/, interfaces/get_context.py returning "", utils/llm.py, utils/qwen.py, data/, SKILL.md).
The procedure differs from E18 in steps 3-5, identically in both arms: groups are classes of questions, rules keyed to
single training questions are not allowed, and decisions are written per group and per earlier sentence. TAB adds one
section that defines data/sentence_table.md.
Usage: .venv-fpq/bin/python -m experiments.latent_mce.src.prepare.e20_workspaces REP[,REP...]
"""
import shutil
import sys

from experiments.latent_mce.src.prepare.e18_workspaces import GET_CONTEXT, QWEN, V4

SKILL = """# Skill: learn context from accumulated experience (Cancer-Myth)

## Memory format
- context/ holds any files you choose (notes, items, rules); the format is free.
- interfaces/get_context.py defines get_context(question, latent) -> str: the text appended to the answering model's
  system prompt as "Instructional context" ("" = nothing). It may read files in context/ and may call only
  utils.qwen.call_qwen (the answering model itself); it must not call utils.llm. Return at most 1,500 characters.

## Skill Overview
Learn context that makes the answering model correct a real false premise and answer normal questions directly.
The score is (FPQ pass rate + NFP pass rate) / 2 under the grader (pass >= 4 of 5). The training questions are a
sample; the score that counts is measured on other questions.

## Procedure
1. Read data/memory.md: the per-context summary of this iteration, the timeline of every training question that failed
   or changed, and retrieved past cases. Items are tagged [observed] (this version), [retrieved] (a record of another
   question or an earlier version) or [unknown].
2. For each failing or changed question, read its timeline and the retrieved cases: what was tried before, what changed,
   what happened to the answer and the rating.
3. Group the problems into classes of questions (a kind of premise or a kind of request) that unseen questions can also
   fall into. For each group decide one change: add or change knowledge, change instructions, change when context is
   attached (narrower or wider), or attach nothing.
4. Write context/ and interfaces/get_context.py. Do not key rules to the wording, names or facts of single training
   questions (for example one regular expression or one stored correction per question): every rule must apply to a
   class of questions you can name.
5. Write context/decisions.md: (a) one line per group: the class, the training questions it covers (ids), the change,
   and the memory fields you relied on; (b) if the earlier context is kept or edited, one line per sentence of the
   earlier context: keep / change / remove, and the evidence you relied on.

## Output fields
behavior = what the answer did about the premise: corrects_target (says the reference premise is false),
rejects_user_premise (says something in the question is wrong although it is not the reference premise, e.g. disputes a
clinician's plan), adds_caveat (adds qualifications without saying anything is wrong), complies (accepts the premise
and answers), hedges_or_refuses, unclear; with a quote from the answer.
"""
TABLE = """
## Sentence table (data/sentence_table.md; read from the answering model before it answers, on the same inputs)
For each context of the current version attached to at least 3 of this iteration's training questions: each sentence
is removed in turn, the answering model reads the shortened context with the same questions, and two probes read its
internal state (no answer is generated). B = probability that the state reads the question as false-premise; C =
probability that the answer would pass. Per sentence: the mean change (removal - full context) on NFP and on FPQ
questions, and the questions with the largest change in C. These are internal reactions, not outcomes, and they are
noisy; real outcomes for the full context are in data/memory.md.
"""


def main(reps):
    # One empty workspace per arm and replicate
    for rep in reps.split(","):
        for arm in ("OUT2", "TAB"):
            ws = V4 / "versions" / f"e18_{arm}_{rep}"
            assert not ws.exists()
            for d in ("context", "interfaces", "utils", "data", "memory", ".claude/skills/learning-context"):
                (ws / d).mkdir(parents=True)
            (ws / "interfaces" / "get_context.py").write_text(GET_CONTEXT)
            (ws / "interfaces" / "__init__.py").write_text("from .get_context import get_context\n")
            shutil.copy2(V4 / "versions" / "t1" / "utils" / "llm.py", ws / "utils" / "llm.py")
            (ws / "utils" / "qwen.py").write_text(QWEN)
            (ws / "utils" / "__init__.py").write_text("")
            (ws / ".claude/skills/learning-context/SKILL.md").write_text(SKILL + (TABLE if arm == "TAB" else ""))
            print(ws.name, "ready")


if __name__ == "__main__":
    main(sys.argv[1])
