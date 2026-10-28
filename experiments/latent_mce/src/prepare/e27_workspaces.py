"""E27 workspaces: OUT5 (the E21 OUT3 procedure) and LAT (the same plus the probe gate and the lens observations)
(prereg_v4_test.md, E27).
Usage: .venv-fpq/bin/python -m experiments.latent_mce.src.prepare.e27_workspaces REP[,REP...]
"""
import shutil
import sys

from experiments.latent_mce.src.prepare.e18_workspaces import GET_CONTEXT, QWEN, V4
from experiments.latent_mce.src.prepare.e20_workspaces import SKILL
from experiments.latent_mce.src.prepare.e21_workspaces import COUNTS
from experiments.latent_mce.src.prepare.e24_workspaces import SPLIT

GATE = """
## Gate (internal)
In this run the context returned by get_context is attached only to questions whose internal false-premise score is
at least 0.5 (a probe on the answering model's internal state before answering; on held-out questions it selects about
55% of FPQ and 20% of NFP questions); other questions get no context, in training, development and test alike. Write
the context for the questions the gate selects; it does not need to protect questions the gate leaves out.
"""
TYPES = """
## Failure types (internal; in data/failure_types.md)
Each failed FPQ answer's input is also read before answering by an Oracle lens on the answering model's internal state
(no answer is generated), and the reading is summarized by a fixed prompt. correction intent = the reading shows a
plan to correct something (the model may hold a correction it did not voice, or aim it at the wrong claim); unaware =
no plan to correct. For questions also answered in the previous iteration the change of type is listed (for example
unaware -> intent: the new context raised a plan to correct that the answer still did not voice).
""" + SPLIT


def main(reps):
    # One empty workspace per arm and replicate
    for rep in reps.split(","):
        for arm in ("OUT5", "LAT"):
            ws = V4 / "versions" / f"e18_{arm}_{rep}"
            assert not ws.exists()
            for d in ("context", "interfaces", "utils", "data", "memory", ".claude/skills/learning-context"):
                (ws / d).mkdir(parents=True)
            (ws / "interfaces" / "get_context.py").write_text(GET_CONTEXT)
            (ws / "interfaces" / "__init__.py").write_text("from .get_context import get_context\n")
            shutil.copy2(V4 / "versions" / "t1" / "utils" / "llm.py", ws / "utils" / "llm.py")
            (ws / "utils" / "qwen.py").write_text(QWEN)
            (ws / "utils" / "__init__.py").write_text("")
            (ws / ".claude/skills/learning-context/SKILL.md").write_text(SKILL + COUNTS + (GATE + TYPES if arm == "LAT" else ""))
            print(ws.name, "ready")


if __name__ == "__main__":
    main(sys.argv[1])
