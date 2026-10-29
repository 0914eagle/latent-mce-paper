"""Fine-tune the lens on one Cancer-Myth fold: rejection-sampling SFT with the frozen AR as the judge (RAFT-1 recipe of
olens_draft docs/pipeline.md, stage 8, with this experiment's residuals).

Data: the pool readings (sample.py --tag pool, N per question and layer, unclamped sampler) scored by score.py. For
each (question, layer) of the fold's training questions (every question outside the fold's test and validation),
the top --keep readings by joint FVE among those that ended with EOS become SFT targets (the generated token ids +
EOS). The fold's validation questions give the validation set the same way. Test questions are never read here.
Training: the lens LoRA (warm start from the base lens, fp32 master weights), the base frozen; the reader prompt with
the residual injected (lens_common.reader_embeds), cross-entropy on the target tokens only; AdamW, linear warmup then
constant, gradient clipping 1.0, micro-batch 2 (3 and 4 run out of memory on 24 GB: the gated-delta-rule layers fall
back to torch), gradient accumulation to --batch. Validation loss every --eval-every optimizer steps; the adapter with
the lowest validation loss is saved (results/data/adapters/fold<f>/), with train_log.jsonl and selection.json.
Usage (repo root, GPUs free): python -m experiments.lens_finetune.src.experiment.train --fold 0 [--keep 2] [--lr 2e-5]
"""
import argparse
import json
import random
import time
from collections import defaultdict

import torch

from experiments.lens_finetune.src.experiment.lens_common import (
    BASE_LENS, LAYERS, OUT, append_jsonl, load_base, load_lens, load_residuals, load_tokenizer, read_jsonl,
    reader_embeds, reader_prompts, split)

ap = argparse.ArgumentParser()
ap.add_argument("--fold", type=int, required=True)
ap.add_argument("--pool", default="pool", help="readings/scores tag of the candidates")
ap.add_argument("--keep", type=int, default=2, help="targets per (question, layer): the top-k by joint FVE")
ap.add_argument("--lr", type=float, default=2e-5)
ap.add_argument("--epochs", type=int, default=1)
ap.add_argument("--batch", type=int, default=16, help="examples per optimizer step")
ap.add_argument("--micro", type=int, default=2)
ap.add_argument("--warmup", type=int, default=10, help="optimizer steps")
ap.add_argument("--eval-every", type=int, default=25, help="optimizer steps")
ap.add_argument("--max-steps", type=int, default=0, help="stop after N optimizer steps (a check run)")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--name", default="", help="adapter folder under results/data/adapters (default fold<f>)")
ap.add_argument("--save-all", action="store_true", help="also save every evaluated step to <adapter>/step<N> (pick by validation FVE, sweep.sh)")
args = ap.parse_args()
assert args.batch % args.micro == 0

out = OUT / "adapters" / (args.name or f"fold{args.fold}")
out.mkdir(parents=True, exist_ok=True)
log = out / "train_log.jsonl"
log.unlink(missing_ok=True)
train_q, val_q, _ = split(args.fold)

# selection: top --keep EOS-terminated readings per (question, layer)
reads = {(r["qid"], r["layer"], r["s"]): r for r in read_jsonl(OUT / "readings" / f"{args.pool}.jsonl")}
cand = defaultdict(list)
for s in read_jsonl(OUT / "scores" / f"{args.pool}.jsonl"):
    cand[(s["qid"], s["layer"])].append(s)


def pick(qids):
    chosen, every = [], []
    for q in qids:
        for L in LAYERS:
            c = cand[(q, L)]
            every += [x["fve"] for x in c]
            ok = sorted((x for x in c if x["eos"] and x["n_bullets"] > 0), key=lambda x: -x["fve"])[: args.keep]
            chosen += [(q, L, reads[(q, L, x["s"])]["ids"], x["fve"]) for x in ok]
    return chosen, every


train, every_t = pick(train_q)
val, _ = pick(val_q)
sel = {"fold": args.fold, "train_questions": len(train_q), "val_questions": len(val_q), "keep": args.keep,
       "train_examples": len(train), "val_examples": len(val),
       "pool_mean_fve": sum(every_t) / max(1, len(every_t)), "selected_mean_fve": sum(x[3] for x in train) / max(1, len(train)),
       "args": vars(args)}
(out / "selection.json").write_text(json.dumps(sel, indent=1))
print(json.dumps({k: v for k, v in sel.items() if k != "args"}), flush=True)

tok, base = load_tokenizer(), load_base()
pm = load_lens(base, BASE_LENS, trainable=True)
params = [p for p in pm.parameters() if p.requires_grad]
for p in params:
    p.data = p.data.float()
base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
base.config.use_cache = False
H, wv = load_residuals(), reader_prompts(tok)
eos = tok.convert_tokens_to_ids("<|im_end|>")  # the turn end the lens emits; also the padding (masked out)
opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
accum = args.batch // args.micro
steps = args.epochs * len(train) // args.batch
if args.max_steps:
    steps = min(steps, args.max_steps)
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / args.warmup))


def loss_of(items):
    e, m, y = reader_embeds(pm, wv, [(L, H[q][LAYERS.index(L)]) for q, L, _, _ in items],
                            targets=[ids + [eos] for _, _, ids, _ in items], pad_id=eos)
    return pm(inputs_embeds=e, attention_mask=m, labels=y).loss


@torch.no_grad()
def val_loss():
    if not val:  # no validation questions (a --limit check run): every evaluation saves
        return float("inf")
    pm.eval()
    tot = sum(float(loss_of(val[i: i + args.micro])) * len(val[i: i + args.micro]) for i in range(0, len(val), args.micro))
    pm.train()
    return tot / max(1, len(val))


rng = random.Random(args.seed)
order = []
for _ in range(args.epochs):
    order += rng.sample(train, len(train))
pm.train()
best, t0 = val_loss(), time.time()
append_jsonl(log, [{"step": 0, "val_loss": best}])
print(f"step 0 val {best:.4f}; {steps} steps of {args.batch}", flush=True)
pm.save_pretrained(str(out))
for step in range(1, steps + 1):
    tr = 0.0
    for a in range(accum):
        items = order[((step - 1) * accum + a) * args.micro:][: args.micro]
        loss = loss_of(items) / accum
        loss.backward()
        tr += loss.item()
    gn = float(torch.nn.utils.clip_grad_norm_(params, 1.0))
    opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
    row = {"step": step, "train_loss": tr, "grad_norm": gn, "lr": sched.get_last_lr()[0], "sec": round(time.time() - t0)}
    if step % args.eval_every == 0 or step == steps:
        row["val_loss"] = v = val_loss()
        if v < best or not val:
            best, row["saved"] = v, True
            pm.save_pretrained(str(out))
        if args.save_all:
            pm.save_pretrained(str(out / f"step{step}"))
        print(f"step {step}/{steps} train {tr:.4f} val {v:.4f} (best {best:.4f}) {time.time() - t0:.0f}s", flush=True)
    append_jsonl(log, [row])
print(f"done: best val loss {best:.4f}, adapter {out}", flush=True)
