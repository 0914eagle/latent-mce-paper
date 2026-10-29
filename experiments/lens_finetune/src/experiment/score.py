"""Score lens readings with the frozen AR: joint bullet FVE against the residual the reading came from.

The reward the lens was RL-tuned on (oracle_lens.pipeline.rl_reward.score_texts_joint, here for the prompt-tag AR):
the reading is split into <= 4 '- ' bullets (split_bullets); each bullet, bare-tokenized (<= 64 tokens), goes through
the AR with the layer's tag ('[Layer L]' prepended), which predicts the residual that precedes such a text at layer L;
predictions and the true residual are whitened with the layer's frozen whitener (ridge 0.1), predictions unit-
normalized, and the residual refit onto the bullet directions by non-negative least squares (nnls_refit):
  fve = 1 - |x_w - sum_i c_i a_i|^2 / |x_w|^2   (x_w = whitened residual, a_i = unit whitened bullet images, c_i >= 0).
fve_shuf: the same bullets refit onto the residual of another question (same layer, a fixed derangement within the
scored set) — how much of the score a reading earns without being about its own question.
A reading that hit the 128-token budget without EOS is scored but flagged (eos false). Rows: {qid, layer, s, fve,
fve_shuf, n_bullets, eos}. Resumable per tag.
Usage (repo root, GPUs free): python -m experiments.lens_finetune.src.experiment.score --tags pool [eval_base ...]
"""
import argparse
import json
import random
import time

import torch

from experiments.lens_finetune.src.experiment.lens_common import (
    AR_DIR, LAYERS, MAX_MEMORY, MODEL, OUT, WHITENER_DIR, append_jsonl, load_residuals, load_tokenizer, read_jsonl)
from oracle_lens.core.nnomp import nnls_refit
from oracle_lens.core.whitening import load_whitener
from oracle_lens.pipeline.rl_reward import reward_text_ids, split_bullets

K_MAX, RIDGE, MICRO = 4, 0.1, 32


class PromptTagAR(torch.nn.Module):
    """The prompt-tag AR (multilayer_reconstructor.PromptTagReconstructor, rebuilt here: that module pulls in the
    training stack): '[Layer L]' tag tokens prepended to the bare text, the final hidden state at the last real token,
    one shared head (LayerNorm without affine, then Linear) -> the predicted layer-L residual."""

    def __init__(self, backbone, state, d_model, layer_norm):
        super().__init__()
        from oracle_lens.core.reconstructor import ReconstructorHead
        self.backbone, self.layers, self.tag_ids = backbone, list(state["layers"]), state["tag_ids"].long()
        self.head = ReconstructorHead(d_model, layer_norm=layer_norm)
        self.head.load_state_dict(state["head"])


def load_ar():
    """The AR spread over the GPUs: Qwen3.6-27B with its final norm removed (truncate_backbone at the deepest AR layer,
    as ar_loader.load_ptag_reconstructor), AR LoRA with the compile prefix stripped (prepare/fetch_ar.py), head and
    tags from heads.pt (its own layer list is authoritative)."""
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    from oracle_lens.core.reconstructor import truncate_backbone

    cfg = json.loads((AR_DIR / "meta.json").read_text())["config"]
    state = torch.load(AR_DIR / "heads.pt", map_location="cpu", weights_only=True)
    assert all(L in state["layers"] for L in LAYERS), state["layers"]
    base = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="auto", max_memory=MAX_MEMORY)
    base.requires_grad_(False)
    inner = truncate_backbone(base, layer=max(state["layers"]))
    peft = PeftModel.from_pretrained(inner, str(AR_DIR / "lora_hf"))
    live = [p for n, p in peft.named_parameters() if "lora_B" in n]
    assert sum(float(p.abs().sum()) > 0 for p in live) >= 0.9 * len(live), "AR LoRA did not load (lora_B all zero)"
    peft.eval()
    recon = PromptTagAR(peft, state, int(base.config.get_text_config().hidden_size), bool(cfg.get("head_layer_norm", True)))
    out_dev = next(inner.layers[-1].parameters()).device
    recon.head.to(out_dev)
    recon.eval()
    return recon, out_dev


@torch.no_grad()
def bullet_images(recon, out_dev, id_rows, layer):
    """AR predictions [n, d] for bare-token rows at one layer (one tagged forward per micro-batch, right-padded)."""
    tag = recon.tag_ids[recon.layers.index(layer)].tolist()
    dev = recon.backbone.get_input_embeddings().weight.device
    preds = torch.zeros(len(id_rows), recon.head.linear.in_features)
    order = sorted(range(len(id_rows)), key=lambda j: len(id_rows[j]))
    for lo in range(0, len(order), MICRO):
        sel = order[lo: lo + MICRO]
        seqs = [tag + id_rows[j] for j in sel]
        W = max(len(x) for x in seqs)
        ids = torch.tensor([x + [0] * (W - len(x)) for x in seqs], device=dev)
        mask = torch.tensor([[1] * len(x) + [0] * (W - len(x)) for x in seqs], device=dev)
        hs = recon.backbone(input_ids=ids, attention_mask=mask, use_cache=False).last_hidden_state
        last = (mask.sum(1) - 1).to(hs.device)
        h = hs[torch.arange(len(sel), device=hs.device), last].float().to(out_dev)
        preds[sel] = recon.head(h).float().cpu()
    return preds


def joint_fve(vecs, valid, gold_w):
    _, fve = nnls_refit(vecs, gold_w, valid)
    return torch.where(valid.any(1), fve, torch.zeros_like(fve))


ap = argparse.ArgumentParser()
ap.add_argument("--tags", nargs="+", required=True)
args = ap.parse_args()

H = load_residuals()
W = {L: load_whitener(WHITENER_DIR / f"whitening_iolens_pooled_L{L}.safetensors", ridge_c=RIDGE) for L in LAYERS}
tok = load_tokenizer()
recon = out_dev = None
for tag in args.tags:
    path = OUT / "scores" / f"{tag}.jsonl"
    seen = {(r["qid"], r["layer"], r["s"]) for r in read_jsonl(path)}
    rows = [r for r in read_jsonl(OUT / "readings" / f"{tag}.jsonl") if (r["qid"], r["layer"], r["s"]) not in seen]
    print(f"{tag}: {len(rows)} readings to score", flush=True)
    if not rows:
        continue
    if recon is None:
        recon, out_dev = load_ar()
    qids = sorted({r["qid"] for r in read_jsonl(OUT / "readings" / f"{tag}.jsonl")})
    rng = random.Random(0)  # a fixed shuffle, then each question takes the next one's residual (a derangement)
    shuffled = qids[:]
    rng.shuffle(shuffled)
    other = dict(zip(shuffled, shuffled[1:] + shuffled[:1])) if len(qids) > 1 else {qids[0]: qids[0]}
    t0 = time.time()
    for L in LAYERS:
        part = [r for r in rows if r["layer"] == L]
        for lo in range(0, len(part), 512):
            chunk = part[lo: lo + 512]
            bullets = [split_bullets(r["text"], K_MAX) for r in chunk]
            flat = [(i, reward_text_ids(tok, b)) for i, bl in enumerate(bullets) for b in bl]
            flat = [(i, ids) for i, ids in flat if ids]
            img = bullet_images(recon, out_dev, [ids for _, ids in flat], L) if flat else torch.zeros(0, 1)
            d = H[chunk[0]["qid"]].shape[-1]
            vecs, valid = torch.zeros(len(chunk), K_MAX, d), torch.zeros(len(chunk), K_MAX, dtype=torch.bool)
            fill = [0] * len(chunk)
            for (i, _), v in zip(flat, img):
                v = W[L].whiten(v[None])[0]
                vecs[i, fill[i]] = v / (v.norm() + 1e-9)
                valid[i, fill[i]] = True
                fill[i] += 1
            gold = W[L].whiten(torch.stack([H[r["qid"]][LAYERS.index(L)] for r in chunk]))
            gold_o = W[L].whiten(torch.stack([H[other[r["qid"]]][LAYERS.index(L)] for r in chunk]))
            f, fs = joint_fve(vecs, valid, gold), joint_fve(vecs, valid, gold_o)
            append_jsonl(path, [{"qid": r["qid"], "layer": L, "s": r["s"], "fve": float(f[i]), "fve_shuf": float(fs[i]),
                                 "n_bullets": fill[i], "eos": r["eos"]} for i, r in enumerate(chunk)])
            print(f"  L{L} {lo + len(chunk)}/{len(part)}  mean fve {float(f.mean()):.3f} (shuffled {float(fs.mean()):.3f})"
                  f"  {time.time() - t0:.0f}s", flush=True)
