"""Run the vendored original MCE (mce_upstream) with the Cancer-Myth environment registered as "cancer_myth".

Nothing in mce_upstream is changed: this registers mce_cancer_env.CancerMythEnvironment in MCE's EnvironmentRegistry,
then runs mce.main (or mce.eval) with the given arguments. MCE_LATENT=none|lens picks the arm (see mce_cancer_env.py).
Usage (cwd = mce_upstream, MCE environment): python run_mce_cancer.py <main|eval> <mce arguments...>
"""
import os
import runpy
import sys
from pathlib import Path

sys.path.insert(0, os.getcwd())  # mce_upstream
sys.path.insert(1, str(Path(__file__).resolve().parents[4]))  # LatentGEPA root

from env.registry import EnvironmentRegistry  # noqa: E402
from experiments.latent_mce.src.experiment.mce_cancer_env import CancerMythEnvironment  # noqa: E402

EnvironmentRegistry.register("cancer_myth", CancerMythEnvironment)

if __name__ == "__main__":
    module = {"main": "mce.main", "eval": "mce.eval"}[sys.argv[1]]
    sys.argv = [module] + sys.argv[2:]
    runpy.run_module(module, run_name="__main__")
