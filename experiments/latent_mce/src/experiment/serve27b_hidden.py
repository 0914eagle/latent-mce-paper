"""Qwen3.6-27B hidden states over HTTP on 127.0.0.1 (POST /hidden), for the probe and state-selector stages
(e13_states, e18_loop.pass_probe, e33_precheck, e37_state_selector, e39_selectors).

  POST /hidden {"system", "users", "keys", "layers", "out"} -> {"out": path}  block outputs of the given layers at the
               last prompt token (chat template, thinking off), saved as safetensors {key: [layers, hidden]} (bf16)

Rebuilt on 2026-10-27: the original (experiments/latent_harness/src/experiment/serve27b.py, untracked) was lost with the
deleted checkout. hidden() is the original function as recorded in the session log of 2026-10-25; model loading and the
chat template follow serve27b_lens.py (bf16 over two GPUs) and medqa_hint_gepa/src/experiment/serve27b.py (branch
gepa-hint-experiments). The original batch size came from a config file that was also lost; left padding with another
batch size can change the last bits of a state, so states extracted now may differ slightly from the stored ones
(e13_states/ e33_states/ e37_states/ in the HF results dataset).

Usage:  CUDA_VISIBLE_DEVICES=<a>,<b> /data2/heejae/uv/jpp_lens/bin/python -m experiments.latent_mce.src.experiment.serve27b_hidden PORT
"""
import json
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH = "/data4/heedong/repos/llm/params/Qwen3.6-27B"
BATCH = 8
TOK = AutoTokenizer.from_pretrained(MODEL_PATH)
TOK.padding_side = "left"
MODEL = AutoModelForCausalLM.from_pretrained(MODEL_PATH, dtype=torch.bfloat16, device_map="auto",
                                             max_memory={0: "40GiB", 1: "44GiB"}).eval()


def chat(system: str, user: str) -> str:
    """Chat template (system, user), thinking off, ready for the first answer token."""
    return TOK.apply_chat_template([{"role": "system", "content": system}, {"role": "user", "content": user}],
                                   tokenize=False, add_generation_prompt=True, enable_thinking=False)


@torch.no_grad()
def hidden(system: str, users: list, keys: list, layers: list, out: str) -> str:
    """Block outputs of the given layers at the last prompt token, one [layers, hidden] tensor per key."""
    from safetensors.torch import save_file
    blocks = MODEL.model.layers if hasattr(MODEL.model, "layers") else MODEL.model.language_model.layers
    saved = {}
    for b in range(0, len(users), BATCH):
        acts = {}
        hooks = [blocks[L].register_forward_hook(lambda m, i, o, L=L: acts.__setitem__(L, (o[0] if isinstance(o, tuple) else o)[:, -1].float().cpu()))
                 for L in layers]
        try:
            enc = TOK([chat(system, u) for u in users[b : b + BATCH]], return_tensors="pt", padding=True,
                      add_special_tokens=False).to(MODEL.device)
            MODEL(**enc, use_cache=False, logits_to_keep=1)
        finally:
            for h in hooks:
                h.remove()
        for k, key in enumerate(keys[b : b + BATCH]):
            saved[key] = torch.stack([acts[L][k] for L in layers]).to(torch.bfloat16).contiguous()
    save_file(saved, out)
    return out


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        # One request: parse, run, reply (errors are returned to the driver)
        req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        try:
            if self.path != "/hidden":
                raise ValueError(self.path)
            body, code = {"out": hidden(req["system"], req["users"], req["keys"], req["layers"], req["out"])}, 200
        except Exception as e:  # noqa: BLE001
            torch.cuda.empty_cache()
            body, code = {"error": repr(e)}, 500
        data = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args):
        pass


if __name__ == "__main__":
    print("serving on", sys.argv[1], flush=True)
    HTTPServer(("127.0.0.1", int(sys.argv[1])), Handler).serve_forever()
