"""Lens-only variant of /data2/heejae/oracle_lens/gepa/serve27b.py (copied 2026-10-25, latent_mce).
One change: an item with "gen": false skips answer generation (answer = cached or None) and is only read by the
fixed Oracle reader, so a card can be checked on many questions for the cost of a prompt forward plus the reader.
Reader settings (layers, renderer, scale, sampling, per-batch seed) are unchanged. E29: an item may carry "nocache": true
and "seed": n to be read afresh with that seed (not cached; items with the same seed are read in one batch). E29 speed-up
(2026-10-26 11:50 KST): the four layers' reader prompts have the same length (56 tokens, slot 24), so they are sampled in
one generate call of 4 x batch rows instead of one call per layer; settings are unchanged, only the random stream differs
from earlier readings (cached readings are kept). E31: an item may carry "answer" and "upto" (with "gen": false) to be read
after the first `upto` tokens of that answer; cached under its own key. Original docstring follows.

Qwen3.6-27B answer server for GEPA with optional oracle-lens plans (2026-10-20).

POST /run {"items": [{"system": str, "question": str, "read": bool}, ...]}
  -> {"results": [{"answer": str, "plan": {"52": bullets, "60": bullets} | null}, ...]}
Answer: chat template (system, user=question), thinking off, greedy, max 2048 new tokens, LoRA disabled — the same
recipe as ../full/full_run.py. Plan: block outputs at layers 52 and 60 at the last prompt token (= right before the
first answer token; causal, so a forward over the prompt alone gives the same state), read by the oracle LoRA
(agu18dec olens_s3d_rl600, 'concepts_raw', 16000*h/|h|, T 0.8, top-p 0.95, 128 tokens, seed fixed per request).
Requests from concurrent clients are batched (up to BATCH, 0.3 s window). Answers and plans are cached on disk
(greedy answers are deterministic). Usage: CUDA_VISIBLE_DEVICES=a,b serve27b.py PORT CACHE_DIR"""
import hashlib, json, os, queue, sys, threading, time
from concurrent.futures import Future
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.environ.get('OLENS_SRC', '/data2/heejae/external/olens_draft/src'))  # other servers: env vars
from oracle_lens.pipeline.verbalizer import renderer_for  # noqa: E402

PORT, CACHE = int(sys.argv[1]), Path(sys.argv[2])
MODEL = os.environ.get('QWEN36_PATH', '/data4/heedong/repos/llm/params/Qwen3.6-27B')
ADAPTER = os.environ.get('OLENS_ADAPTER', '/data2/heejae/oracle_lens/olens_and_ar/olens_s3d_rl600')  # the fixed reader (a tuned adapter only as an ablation)
READ_LAYERS = [int(x) for x in os.environ.get('READ_LAYERS', '52,60').split(',')]   # layers read by the oracle
BATCH, WINDOW, MAX_NEW, ALPHA = 24, 0.3, 2048, 16000.0
CACHE.mkdir(parents=True, exist_ok=True)
sha = lambda *xs: hashlib.sha256('\x1f'.join(xs).encode()).hexdigest()[:32]


class Cache:
    def __init__(self, path):
        self.path, self.d, self.lock = path, {}, threading.Lock()
        if path.exists():
            for line in path.open():
                r = json.loads(line); self.d[r['k']] = r['v']

    def get(self, k):
        return self.d.get(k)

    def put(self, k, v):
        with self.lock:
            self.d[k] = v
            with self.path.open('a') as f:
                f.write(json.dumps({'k': k, 'v': v}, ensure_ascii=False) + '\n')


answers = Cache(CACHE / 'answers.jsonl')
plans = Cache(CACHE / ('plans.jsonl' if READ_LAYERS == [52, 60] else 'plans_L' + '_'.join(map(str, READ_LAYERS)) + '.jsonl'))
tok = AutoTokenizer.from_pretrained(MODEL); tok.padding_side = 'left'
base = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map='auto',
                                            max_memory=({int(k): v for k, v in (x.split(':') for x in os.environ['LENS_MAX_MEMORY'].split(','))}
                                                        if os.environ.get('LENS_MAX_MEMORY') else {0: '40GiB', 1: '44GiB'}))
base.eval(); base.requires_grad_(False)
from peft import PeftModel  # noqa: E402
pm = PeftModel.from_pretrained(base, ADAPTER); pm.eval()
layers = base.model.layers
embed = pm.get_input_embeddings(); dev0 = embed.weight.device
eos = base.generation_config.eos_token_id; eos = set(eos if isinstance(eos, list) else [eos])
pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
wv = {L: renderer_for('concepts_raw')(tok, layer=L) for L in READ_LAYERS}
Q: 'queue.Queue[tuple[dict, Future]]' = queue.Queue()
print('ready', flush=True)


def prompt_of(it):
    # E31: an item with "answer" and "upto" is read after the first `upto` tokens of that answer (inside the generated region)
    text = tok.apply_chat_template([{'role': 'system', 'content': it['system']}, {'role': 'user', 'content': it['question']}],
                                   tokenize=False, add_generation_prompt=True, enable_thinking=False)
    if it.get('answer') is not None:
        text += tok.decode(tok(it['answer'], add_special_tokens=False).input_ids[:int(it['upto'])])
    return text


def robust(fn, texts, *args):
    """Run fn on a list; on any error (e.g. CUDA OOM) log it, free memory, and retry the two halves."""
    try:
        return fn(texts, *args)
    except Exception:
        import traceback; traceback.print_exc(); torch.cuda.empty_cache()
        if len(texts) == 1:
            raise
        h = len(texts) // 2
        print(f'retrying {len(texts)} items as {h} + {len(texts) - h}', flush=True)
        return robust(fn, texts[:h], *args) + robust(fn, texts[h:], *args)


@torch.no_grad()
def generate(texts):
    out = []
    for b in range(0, len(texts), BATCH):
        enc = tok(texts[b:b + BATCH], return_tensors='pt', padding=True, add_special_tokens=False).to(dev0)
        with pm.disable_adapter():
            g = pm.generate(**enc, do_sample=False, max_new_tokens=MAX_NEW, pad_token_id=pad)
        W = enc['input_ids'].shape[1]
        for row in g:
            new = row[W:].tolist()
            cut = next((j for j, t in enumerate(new) if t in eos), None)
            out.append(tok.decode(new[:cut] if cut is not None else new, skip_special_tokens=True))
    return out


@torch.no_grad()
def read_plans(texts, seed):
    acts = {}
    hooks = [layers[L].register_forward_hook(lambda m, i, o, L=L: acts.__setitem__(L, (o[0] if isinstance(o, tuple) else o))) for L in READ_LAYERS]
    vec = {L: [] for L in READ_LAYERS}
    try:
        for t in texts:
            ids = tok(t, return_tensors='pt', add_special_tokens=False).input_ids.to(dev0)
            with pm.disable_adapter():
                pm(input_ids=ids, use_cache=False)
            for L in READ_LAYERS:
                vec[L].append(acts[L][0, -1].float())
    finally:
        for h in hooks:
            h.remove()
    res = [{} for _ in texts]
    torch.manual_seed(seed)
    pes = []
    for L in READ_LAYERS:  # one block of rows per layer; all reader prompts have the same length, so no padding
        p = wv[L]; v = torch.stack(vec[L]).to(dev0)
        inj = ALPHA * v / v.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        pe = embed(torch.tensor([p.input_ids], device=dev0)).expand(len(texts), -1, -1).clone()
        pe[:, p.slot, :] = inj.to(pe.dtype)
        pes.append(pe)
    pe = torch.cat(pes)
    g = pm.generate(inputs_embeds=pe, attention_mask=torch.ones(pe.shape[:2], dtype=torch.long, device=dev0),
                    max_new_tokens=128, do_sample=True, temperature=0.8, top_p=0.95, pad_token_id=pad)
    out = tok.batch_decode(g, skip_special_tokens=True)
    for b, L in enumerate(READ_LAYERS):
        for n, r in enumerate(res):
            r[str(L)] = out[b * len(texts) + n].strip()
    return res


def worker():
    while True:
        batch = [Q.get()]
        t0 = time.time()
        while len(batch) < BATCH and time.time() - t0 < WINDOW:
            try:
                batch.append(Q.get(timeout=0.05))
            except queue.Empty:
                pass
        try:
            items = [it for it, _ in batch]
            texts = [prompt_of(it) for it in items]
            keys = [sha(it['system'], it['question'], *((it['answer'], str(it['upto'])) if it.get('answer') is not None else ())) for it in items]
            need = sorted({k: t for k, t, a, it in zip(keys, texts, [answers.get(k) for k in keys], items) if a is None and it.get('gen', True)}.items())
            if need:
                for (k, _), a in zip(need, robust(generate, [t for _, t in need])):
                    answers.put(k, a)
            needp = sorted({k: t for k, t, it in zip(keys, texts, items) if it.get('read') and not it.get('nocache') and plans.get(k) is None}.items())
            if needp:
                for (k, _), p in zip(needp, robust(read_plans, [t for _, t in needp], int(needp[0][0][:8], 16) % 100000)):
                    plans.put(k, p)
            # E29 re-read check: an item with "nocache": true is read afresh with its own "seed" and not cached
            fresh = {}
            for sd in sorted({int(it.get('seed', 0)) for it in items if it.get('read') and it.get('nocache')}):
                ns = [n for n, it in enumerate(items) if it.get('read') and it.get('nocache') and int(it.get('seed', 0)) == sd]
                fresh.update(zip(ns, robust(read_plans, [texts[n] for n in ns], sd)))
            for n, ((it, fut), k) in enumerate(zip(batch, keys)):
                fut.set_result({'answer': answers.get(k), 'plan': fresh[n] if n in fresh else (plans.get(k) if it.get('read') else None)})
        except Exception as e:  # report to every waiting client
            import traceback; traceback.print_exc()
            for _, fut in batch:
                if not fut.done():
                    fut.set_exception(e)


class H(BaseHTTPRequestHandler):
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        futs = []
        for it in body['items']:
            f = Future(); Q.put((it, f)); futs.append(f)
        try:
            out = {'results': [f.result() for f in futs]}; code = 200
        except Exception as e:
            out = {'error': f'{type(e).__name__}: {e}'}; code = 500
        data = json.dumps(out, ensure_ascii=False).encode()
        self.send_response(code); self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(data))); self.end_headers(); self.wfile.write(data)

    def log_message(self, *a):
        pass


threading.Thread(target=worker, daemon=True).start()
ThreadingHTTPServer(('127.0.0.1', PORT), H).serve_forever()
