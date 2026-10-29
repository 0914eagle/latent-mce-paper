"""Run the vendored original MCE (mce_upstream, unchanged) with the latent-MCE environments of lens_feedback.py
registered as "<dataset>_lens" (cancer_myth_lens, hotpotqa_lens, hover_lens, medcalc_cot_lens) next to the plain ones (cancer_myth, hotpotqa, hover,
medcalc_cot). Usage (cwd = mce_upstream, MCE venv): python run_mce_latent.py <main|eval> <mce arguments...>"""
import os
import runpy
import sys
from pathlib import Path

sys.path.insert(0, os.getcwd())  # mce_upstream
sys.path.insert(1, str(Path(__file__).resolve().parents[4]))  # LatentGEPA root

from env.registry import EnvironmentRegistry  # noqa: E402
from experiments.latent_mce.src.experiment.lens_feedback import lens_inputs, with_lens  # noqa: E402

for name, (Env, inp) in lens_inputs().items():
    EnvironmentRegistry.register(name, Env)
    EnvironmentRegistry.register(f"{name}_lens", with_lens(Env, inp))

if __name__ == "__main__":
    module = {"main": "mce.main", "eval": "mce.eval"}[sys.argv[1]]
    sys.argv = [module] + sys.argv[2:]
    runpy.run_module(module, run_name="__main__")
