"""Project folders. Every script resolves its inputs and outputs from here, so moving a folder is a one-line change."""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"  # shared inputs of this project
JLENS_DIR = DATA_DIR / "jlens"  # pre-fitted Jacobian lenses (HF neuronpedia/jacobian-lens)
JLENS_EVALS_DIR = DATA_DIR / "jlens_evals"  # six lens-quality prompt sets (github.com/anthropics/jacobian-lens data/evaluations, 581d398)
CREPE_DIR = DATA_DIR / "crepe"  # CREPE official splits (HF tasksource/crepe, cd11ed2)
FPQ_DIR = DATA_DIR / "fpq"  # frozen false-premise split lists and Cancer-Myth questions (tracked; see common/fpq.py)
WELL_DIR = DATA_DIR / "well"  # Well Actually prompt templates, judge rubrics, few-shot examples (github.com/ShenranTomWang/Well, 6c9770f)
MEDRAC_DIR = DATA_DIR / "medrac"  # MedCalc-Bench test set, formula library, one-shot examples, formula FAISS index (github.com/benluwang/EMNLP-2025-MedRaC data/, aad3aa3)
MEDCALC_TRAIN_DIR = DATA_DIR / "medcalc_v1.2"  # MedCalc-Bench v1.2 train split (HF ncbi/MedCalc-Bench-v1.2, pinned in the experiment config)
JPP_DIR = DATA_DIR / "jpp"  # our Qwen3-8B J++ lens (HF latentgpea/qwen3-8b-jpp-lens, bc90b8e)
EXPERIMENTS_DIR = ROOT / "experiments"
HF_CACHE = ROOT / ".cache" / "hf"


def results_dir(experiment: str) -> Path:
    """experiments/<experiment>/results, which holds data/ (inputs and outputs of each run) and figures/.

    With RESULTS_SUBDIR=<name> set, every script reads and writes under results/<name>/ instead, so a re-run never
    touches the stored results."""
    # Build the results path, with the RESULTS_SUBDIR override inserted when it is set
    subdir = os.environ.get("RESULTS_SUBDIR", "")
    return EXPERIMENTS_DIR / experiment / "results" / subdir


def use_project_hf_cache() -> None:
    """Keep model weights in the project's HF cache, never in $HOME. Call before importing transformers."""
    # Point HF_HOME at the project cache and read the HF token from .env
    os.environ["HF_HOME"] = str(HF_CACHE)
    env = ROOT / ".env"
    if env.exists() and "HF_TOKEN" not in os.environ:
        for line in env.read_text().splitlines():
            if line.startswith("HF_TOKEN="):
                os.environ["HF_TOKEN"] = line.split("=", 1)[1].strip()
