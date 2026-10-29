"""Residuals the lens reads: block-L output (L = 36/44/52/60) at the last token of the Plain prompt, every question.

One forward of Qwen3.6-27B (no adapter) per question over the chat template (system = Plain, user = question,
generation prompt, thinking off) — the state right before the first answer token, as serve27b_lens.read_plans takes it.
Writes results/data/residuals.safetensors (h: [n, 4, 5120] fp32) and residuals_index.json (qid order).
Usage (repo root, hf27b venv, GPUs free): python -m experiments.lens_finetune.src.experiment.residuals [--limit N]
"""
import argparse
import json

import torch
from safetensors.torch import save_file

from experiments.lens_finetune.src.experiment.lens_common import LAYERS, OUT, load_base, load_tokenizer, plain, questions, system_of

ap = argparse.ArgumentParser()
ap.add_argument("--limit", type=int, default=0, help="N questions only, the first N/3 of each fold's test (a check run)")
args = ap.parse_args()

tok, base = load_tokenizer(), load_base()
blocks, dev = base.model.layers, base.get_input_embeddings().weight.device
qs = questions()
if args.limit:
    qs = [q for f in range(3) for q in [x for x in qs if x["test_fold"] == f][: args.limit // 3]]
acts = {}
hooks = [blocks[L].register_forward_hook(lambda m, i, o, L=L: acts.__setitem__(L, (o[0] if isinstance(o, tuple) else o)[0, -1].float().cpu()))
         for L in LAYERS]
H = torch.zeros(len(qs), len(LAYERS), base.config.get_text_config().hidden_size)
with torch.no_grad():
    for n, q in enumerate(qs):
        text = tok.apply_chat_template([{"role": "system", "content": system_of(q)}, {"role": "user", "content": q["question"]}],
                                       tokenize=False, add_generation_prompt=True, enable_thinking=False)
        base(input_ids=tok(text, return_tensors="pt", add_special_tokens=False).input_ids.to(dev), use_cache=False)
        H[n] = torch.stack([acts[L] for L in LAYERS])
        if n % 100 == 0:
            print(f"{n}/{len(qs)}", flush=True)
for h in hooks:
    h.remove()
save_file({"h": H}, str(OUT / "residuals.safetensors"))
(OUT / "residuals_index.json").write_text(json.dumps([q["qid"] for q in qs]))
print(f"residuals {tuple(H.shape)}, mean norm per layer", [round(float(H[:, i].norm(dim=-1).mean()), 1) for i in range(len(LAYERS))])
