"""Download the frozen AR the lens was RL-tuned against, and its whiteners (gated HF dataset agu18dec/local-workspace).

The lens (s3d.rl.ddp600, run.json) was GRPO-tuned on the joint bullet-FVE of the prompt-tag AR
ckpts/ar/ptag/ar.asst.ptag.pooled.s0/ex16013184 (layers 20-60, whitener prefix whitening_iolens_pooled, ridge 0.1);
the reward of this experiment uses that AR unchanged. Downloads the AR rung (LoRA, heads.pt, meta.json, 0.6 GB) and the
pooled whiteners of the read layers (36/44/52/60) under $OLENS_ARTIFACTS/hf, then writes lora_hf/: the AR LoRA with the
torch.compile wrapper prefix ('._orig_mod') stripped from every key, since the AR was trained with compiled blocks and
this experiment loads it uncompiled (PEFT would otherwise load nothing). Token: $HF_TOKEN or the default HF login.
Usage (repo root): python -m experiments.lens_finetune.src.prepare.fetch_ar
"""
import json
import os
import shutil
from pathlib import Path

from huggingface_hub import snapshot_download
from safetensors.torch import load_file, save_file

from experiments.lens_finetune.src.experiment.lens_common import AR_DIR, ARTIFACTS, LAYERS, WHITENER_DIR

REPO = "agu18dec/local-workspace"
token = os.environ.get("HF_TOKEN") or next((p.read_text().strip() for p in [Path.home() / ".cache/huggingface/token"]
                                           if p.exists()), None)
rung = "ckpts/ar/ptag/ar.asst.ptag.pooled.s0/ex16013184"
snapshot_download(REPO, repo_type="dataset", local_dir=ARTIFACTS / "hf", token=token,
                  allow_patterns=[f"{rung}/*", f"{rung}/lora/*"]
                  + [f"data/whiteners/pooled/whitening_iolens_pooled_L{L}.safetensors" for L in LAYERS]
                  + ["data/whiteners/pooled/whitening_iolens_pooled_provenance.json"])
cfg = json.loads((AR_DIR / "meta.json").read_text())["config"]
assert cfg["head_mode"] == "prompt_tag" and cfg["whitener_prefix"] == "whitening_iolens_pooled", cfg
for L in LAYERS:
    assert (WHITENER_DIR / f"whitening_iolens_pooled_L{L}.safetensors").exists(), L

out = AR_DIR / "lora_hf"
out.mkdir(exist_ok=True)
sd = load_file(str(AR_DIR / "lora/adapter_model.safetensors"))
n_pref = sum("._orig_mod" in k for k in sd)
save_file({k.replace("._orig_mod", ""): v for k, v in sd.items()}, str(out / "adapter_model.safetensors"))
ac = json.loads((AR_DIR / "lora/adapter_config.json").read_text())
ac["lora_dropout"] = 0.0
(out / "adapter_config.json").write_text(json.dumps(ac, indent=2))
shutil.copy(AR_DIR / "lora/README.md", out / "README.md")
print(f"AR {AR_DIR} ({len(sd)} LoRA tensors, {n_pref} with the compile prefix, stripped into lora_hf/); "
      f"whiteners {WHITENER_DIR}; val joint FVE of the AR: "
      + ", ".join(f"L{L} {json.loads((AR_DIR / 'meta.json').read_text())[f'val_jfve_L{L}']:.3f}" for L in LAYERS))
