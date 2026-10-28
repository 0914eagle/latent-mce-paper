"""E24 workspaces: empty-memory MCE runs for the output arm (OUT4) and the lens-routed arm (LNS) (prereg_v4_test.md, E24).

Same layout and procedure as E20 (e20_workspaces.SKILL) plus one sentence, in both arms, that the context may combine
instructions and knowledge entries. Both arms define data/failure_types.md with one experience line from E22: pooled
for OUT4, split by the lens failure type for LNS; LNS also defines the split itself.
Usage: .venv-fpq/bin/python -m experiments.latent_mce.src.prepare.e24_workspaces REP[,REP...]
"""
import shutil
import sys

from experiments.latent_mce.src.prepare.e18_workspaces import GET_CONTEXT, QWEN, V4
from experiments.latent_mce.src.prepare.e20_workspaces import SKILL

CONTENT = """The context may combine instructions and knowledge entries (general facts that correct a kind of misconception);
get_context may select the entries that fit a question.
"""
COUNTS = """
## Failure counts (data/failure_types.md; read it together with data/memory.md)
For each context group of this iteration (a context shared by at least 3 of the 80 training questions; all other
contexts pooled; no context): the number of FPQ and NFP questions and passes, and the failures by answer behavior
(see Output fields) with their question ids.
"""
POOLED = """Earlier experience on these training questions (81 FPQ failures, one change at a time): adding an instruction
to state the correction fixed 31%, adding the missing fact fixed 93%, adding a neutral sentence fixed 11%.
"""
SPLIT = """Earlier experience on these training questions (81 FPQ failures, one change at a time), by failure type (see
Failure types): correction intent (28) - an instruction to state the correction fixed 50%, the missing fact 82%, a
neutral sentence 14%; unaware (53) - the instruction 21%, the missing fact 98%, a neutral sentence 9%.
"""
TYPES = """
## Failure types (internal; in data/failure_types.md)
Each failed FPQ answer's input is also read before answering by an Oracle lens on the answering model's internal state
(no answer is generated), and the reading is summarized by a fixed prompt. correction intent = the reading shows a
plan to correct something (the model may hold a correction it did not voice, or aim it at the wrong claim); unaware =
the reading shows no plan to correct. Per context group, data/failure_types.md lists the FPQ failures of each type.
"""


def main(reps):
    # One empty workspace per arm and replicate
    for rep in reps.split(","):
        for arm in ("OUT4", "LNS"):
            ws = V4 / "versions" / f"e18_{arm}_{rep}"
            assert not ws.exists()
            for d in ("context", "interfaces", "utils", "data", "memory", ".claude/skills/learning-context"):
                (ws / d).mkdir(parents=True)
            (ws / "interfaces" / "get_context.py").write_text(GET_CONTEXT)
            (ws / "interfaces" / "__init__.py").write_text("from .get_context import get_context\n")
            shutil.copy2(V4 / "versions" / "t1" / "utils" / "llm.py", ws / "utils" / "llm.py")
            (ws / "utils" / "qwen.py").write_text(QWEN)
            (ws / "utils" / "__init__.py").write_text("")
            skill = SKILL.replace("## Output fields", CONTENT + "\n## Output fields")
            skill += COUNTS + (SPLIT + TYPES if arm == "LNS" else POOLED)
            (ws / ".claude/skills/learning-context/SKILL.md").write_text(skill)
            print(ws.name, "ready")


if __name__ == "__main__":
    main(sys.argv[1])
