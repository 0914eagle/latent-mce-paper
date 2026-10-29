"""Sample lens readings: N per (question, layer), from the base lens or a fine-tuned one.

Each reading = the lens's generation (<= 128 tokens) from the reader prompt with the question's residual injected
(lens_common.reader_embeds). Two settings are used:
  pool  (training candidates): temperature 1.0, top-p 1.0, top-k 0 — the unclamped sampler the lens was RL-tuned with;
  eval  (as served in latent_mce): temperature 0.8, top-p 0.95 and the model's generation config otherwise (top-k 20).
Rows: {qid, layer, s, ids (generated token ids up to the first EOS), text, eos (stopped by EOS, not by the budget)}.
The token ids are kept so training never re-tokenizes a reading. Resumable: (qid, layer) pairs with all N samples are
skipped. Batches hold whole pairs; each batch has its own seed (seed + batch number in the remaining list).
Usage (repo root, GPUs free): python -m experiments.lens_finetune.src.experiment.sample --tag pool --questions all
    --n 8 --temperature 1.0 --top-p 1.0 --top-k 0 [--adapter <dir>] [--rows 96] [--limit N]
"""
import argparse
import time

import torch

from experiments.lens_finetune.src.experiment.lens_common import (
    BASE_LENS, LAYERS, MAX_NEW, OUT, append_jsonl, eos_ids, load_base, load_lens, load_residuals, load_tokenizer,
    read_jsonl, reader_embeds, reader_prompts, select_qids)

ap = argparse.ArgumentParser()
ap.add_argument("--tag", required=True, help="output name: results/data/readings/<tag>.jsonl")
ap.add_argument("--adapter", default=BASE_LENS)
ap.add_argument("--questions", default="all", help="all | train:<f> | val:<f> | test:<f>")
ap.add_argument("--n", type=int, default=8)
ap.add_argument("--temperature", type=float, default=1.0)
ap.add_argument("--top-p", type=float, default=1.0)
ap.add_argument("--top-k", type=int, default=None, help="default: the model's generation config (20); 0 = off")
ap.add_argument("--rows", type=int, default=96, help="sequences per generate call")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--limit", type=int, default=0, help="first N questions only (a check run)")
args = ap.parse_args()

path = OUT / "readings" / f"{args.tag}.jsonl"
done = {}
for r in read_jsonl(path):
    done[(r["qid"], r["layer"])] = done.get((r["qid"], r["layer"]), 0) + 1
H = load_residuals()
qids = [q for q in select_qids(args.questions) if q in H][: args.limit or None]  # residuals.py --limit leaves some out
todo = [(q, L) for q in qids for L in LAYERS if done.get((q, L), 0) < args.n]
print(f"{args.tag}: {len(todo)} of {len(qids) * len(LAYERS)} (question, layer) pairs to sample, {args.n} each", flush=True)
if not todo:
    raise SystemExit

tok, base = load_tokenizer(), load_base()
pm = load_lens(base, args.adapter)
wv, eos = reader_prompts(tok), eos_ids(base)
gen = dict(do_sample=True, temperature=args.temperature, top_p=args.top_p, max_new_tokens=MAX_NEW,
           pad_token_id=base.generation_config.pad_token_id)
if args.top_k is not None:
    gen["top_k"] = args.top_k
per = max(1, args.rows // args.n)
t0 = time.time()
for b in range(0, len(todo), per):
    pairs = todo[b: b + per]
    rows = [(L, H[q][LAYERS.index(L)]) for q, L in pairs for _ in range(args.n)]
    e, m, _ = reader_embeds(pm, wv, rows)
    torch.manual_seed(args.seed + b // per)
    with torch.no_grad():
        g = pm.generate(inputs_embeds=e, attention_mask=m, **gen)
    out = []
    for i, seq in enumerate(g.tolist()):
        q, L = pairs[i // args.n]
        cut = next((j for j, t in enumerate(seq) if t in eos), None)
        ids = seq[:cut] if cut is not None else seq
        out.append({"qid": q, "layer": L, "s": i % args.n, "ids": ids, "text": tok.decode(ids, skip_special_tokens=True),
                    "eos": cut is not None})
    append_jsonl(path, out)
    k = min(b + per, len(todo))
    print(f"{k}/{len(todo)} pairs, {(time.time() - t0) / k * (len(todo) - k) / 60:.0f} min left", flush=True)
