"""E21 workspaces: empty-memory MCE runs for the output arm (OUT3) and the failure-type arm (FTY) (prereg_v4_test.md, E21).

Same layout and procedure as E20 (e20_workspaces.SKILL). Both arms add a section that defines data/failure_types.md
(failure counts by answer behaviour per context group); FTY adds one section that defines the internal split of the
FPQ failures (with-context pass probe) and its measured reliability.
Usage: .venv-fpq/bin/python -m experiments.latent_mce.src.prepare.e21_workspaces REP[,REP...]
"""
import shutil
import sys

from experiments.latent_mce.src.prepare.e18_workspaces import GET_CONTEXT, QWEN, V4
from experiments.latent_mce.src.prepare.e20_workspaces import SKILL

COUNTS = """
## Failure counts (data/failure_types.md; read it together with data/memory.md)
For each context group of this iteration (a context shared by at least 3 of the 80 training questions; all other
contexts pooled; no context): the number of FPQ and NFP questions and passes, and the failures by answer behavior
(see Output fields) with their question ids. Use it to see which kind of failure is common under which context.
"""
INTERNAL = """
## Failure types (internal; in data/failure_types.md)
In each context group, the FPQ failures are also split by a probe on the answering model's internal state with the
context, read before it answers (no answer is generated), at the group's median score: possibly_silent (above) = the
model may hold the correction but not say it, so an instruction to state the correction may help; likely_unaware (at
or below) = no internal plan to correct, so knowledge or attention to the premise may be missing. Reliability,
measured earlier on these training questions with the same rule: an Oracle-lens reading showed a plan to correct the
actual premise in 22% of possibly_silent failures (15/67) and in 4% of likely_unaware failures (3/70). The split is a
tendency, not a certainty.
"""


def main(reps):
    # One empty workspace per arm and replicate
    for rep in reps.split(","):
        for arm in ("OUT3", "FTY"):
            ws = V4 / "versions" / f"e18_{arm}_{rep}"
            assert not ws.exists()
            for d in ("context", "interfaces", "utils", "data", "memory", ".claude/skills/learning-context"):
                (ws / d).mkdir(parents=True)
            (ws / "interfaces" / "get_context.py").write_text(GET_CONTEXT)
            (ws / "interfaces" / "__init__.py").write_text("from .get_context import get_context\n")
            shutil.copy2(V4 / "versions" / "t1" / "utils" / "llm.py", ws / "utils" / "llm.py")
            (ws / "utils" / "qwen.py").write_text(QWEN)
            (ws / "utils" / "__init__.py").write_text("")
            (ws / ".claude/skills/learning-context/SKILL.md").write_text(SKILL + COUNTS + (INTERNAL if arm == "FTY" else ""))
            print(ws.name, "ready")


if __name__ == "__main__":
    main(sys.argv[1])
