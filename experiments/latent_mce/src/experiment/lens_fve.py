"""Oracle-lens FVE on five datasets: how much of Qwen3.6-27B's pre-answer state the fixed lens reading explains.

Readings: the ones already taken (E42 HotpotQA / HoVer / Symptom2Disease, E43 MedCalc-Bench, the Cancer-Myth Plain
readings of the MCE env), fixed reader olens_s3d_rl600, blocks 36/44/52/60, at the last token of the chat input
(system + user, thinking off): looked up in the lens caches by sha256(system + "\\x1f" + user)[:32], as written.
  Cancer-Myth (724):     system = transport + Plain, user = the question
  HotpotQA / HoVer (300): system = "", user = the question / claim (GEPA artifact test split)
  Symptom2Disease (212): system = "", user = the MCE environment's diagnosis prompt (empty context)
  MedCalc-Bench (1048):  the paper's zero-shot CoT prompt (common/medrac/plain.cot_prompts)
  SynQA2 (600):          system = transport + Plain, user = the question (as Cancer-Myth); 300 normal / false-premise
                         pairs drawn with seed 0 (190 single-hop, 110 multi-hop, the source's 1115 : 643), read here
                         (`read`, lens servers LENS_URLS, same reader and blocks), cached in synqa2_readings.jsonl
Gold: the block output at that layer and token, recomputed with one forward over the same chat input (the lens server's
recipe: apply_chat_template([system, user], add_generation_prompt=True, enable_thinking=False), no adapter; causal, so
the prompt alone gives the state). FVE: the lens repo's joint bullet-FVE as in /data2/heejae/oracle_lens/full/fve_full.py
(bullets <= 4, AR ar_ptag_pooled, whitening_iolens_pooled ridge 0.1, unit-norm atoms, NNLS): 1 - |x - recon|^2 / |x|^2
of the whitened gold. Control "other": the bullets of another question of the same dataset at the same layer.
Usage (jpp_lens venv, PYTHONPATH=/data2/heejae/uv/fla_pkgs, two GPUs each):
  python lens_fve.py jobs            build jobs.jsonl (prompts + readings)
  python lens_fve.py read            SynQA2 readings on LENS_URLS (comma-separated /run URLs)
  python lens_fve.py gold [TAG]      forward pass for the items not yet in a gold file, gold[_TAG].safetensors
  python lens_fve.py fve L1,L2 TAG [DATASETS]   FVE at the given layers -> fve_TAG.jsonl
  python lens_fve.py table           summary.json and table.md
"""
import hashlib
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path[:0] = [str(ROOT), "/data2/heejae/external/olens_draft/src", "/data2/heejae/oracle_lens/pilot"]
OUT = ROOT / "experiments/latent_mce/results/data/lens_fve"
MCE = ROOT / "experiments/latent_mce/results/data/mce_cancer"
MODEL = "/data4/heedong/repos/llm/params/Qwen3.6-27B"
LAYERS = [36, 44, 52, 60]
key = lambda s, u: hashlib.sha256((s + "\x1f" + u).encode()).hexdigest()[:32]


def jobs():
    import pandas as pd  # noqa: F401  (common.medrac)
    from common.medrac.dataset import load_test_set
    from common.medrac.plain import cot_prompts
    cache = {}
    for f in (MCE / "lens_cache.jsonl", ROOT / "experiments/latent_mce/results/data/e43_medcalc/readings.jsonl", OUT / "synqa2_readings.jsonl"):
        if not f.exists():
            continue
        cache.update({json.loads(l)["k"]: json.loads(l)["v"] for l in f.read_text().splitlines()})
    reg = json.load(open("/home/eagle0914/cancer-myth/docs/reviews/paper_tables_2026-10-06/table2_prompt_registry_2026-10-07.json"))
    plain = reg["transport_common"] + "\n\n" + reg["prompts"]["plain"]
    rows = []
    bundle = json.load(open("/data2/heejae/cancer-myth/paper_v4_qwen3/protocol/bundle.json"))["rows"]
    rows += [("cancer_myth", r["id"], plain, r["question"]) for r in bundle if r["dataset"] == "cancer_myth"]
    sp = ROOT / "experiments/mce_repro_qwen3/results/data/gepa_splits"
    for t in ("hotpotqa", "hover"):
        rows += [(t, f"{t}_{i}", "", json.loads(l)["question"]) for i, l in enumerate((sp / f"{t}_test.jsonl").read_text().splitlines())]
    src = (ROOT / "experiments/latent_mce/mce_upstream/env/symptom_diagnosis/symptom_diagnosis_environment.py").read_text()
    ns = {}; exec(src[src.index("DISEASE_LIST"):].split("\n\n")[0], ns)
    s2d = json.load(open(ROOT / "experiments/mce_repro_qwen3/results/data/test_s2d_27b_base.json/evaluation.json"))["results"]
    rows += [("symptom2disease", f"s2d_{r['sample']['id']}", "", f"You are a medical diagnostician.\n\nPatient symptoms: {r['sample']['question']}\n\n"
              f"Possible diagnoses: {', '.join(ns['DISEASE_LIST'])}\n\nAnalyze and provide diagnosis in format: [DIAGNOSIS]disease_name[/DIAGNOSIS]") for r in s2d]
    df = load_test_set()
    for (s, u), row in zip(cot_prompts(df["Calculator ID"].astype(str).tolist(), df["Patient Note"].tolist(), df["Question"].tolist()), df["Row Number"].tolist()):
        rows.append(("medcalc", f"medcalc_{row}", s, u))
    syn = json.load(open("/home/eagle0914/cancer-myth/results/oracle_harness_qwen_final_20261024_v2/syn_qa2/paired_inputs.json"))
    pairs = {}
    for x in syn:
        pairs.setdefault(x["pair_id"], []).append(x)
    rng = random.Random(0)
    for hop, k in (("singlehop", 190), ("multihop", 110)):
        for pid in rng.sample(sorted(p for p in pairs if f":{hop}:" in p), k):
            rows += [("synqa2", x["id"], plain, x["question"]) for x in sorted(pairs[pid], key=lambda x: x["id"])]
    OUT.mkdir(parents=True, exist_ok=True)
    n, miss = {}, {}
    with open(OUT / "jobs.jsonl", "w") as f:
        for d, i, s, u in rows:
            v = cache.get(key(s, u))
            n[d] = n.get(d, 0) + 1
            if v is None:
                miss[d] = miss.get(d, 0) + 1
            f.write(json.dumps({"dataset": d, "id": i, "system": s, "user": u, "reading": v}, ensure_ascii=False) + "\n")
    print("items", n, "| no reading", miss)


def read():
    import os
    import urllib.request
    from concurrent.futures import ThreadPoolExecutor
    urls = os.environ["LENS_URLS"].split(",")
    J = [json.loads(l) for l in (OUT / "jobs.jsonl").read_text().splitlines()]
    todo = [j for j in J if j["reading"] is None]
    chunks = [todo[c:c + 24] for c in range(0, len(todo), 24)]
    def run(ci):
        body = {"items": [{"system": j["system"], "question": j["user"], "read": True, "gen": False} for j in chunks[ci]]}
        req = urllib.request.Request(urls[ci % len(urls)], data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=24 * 3600) as r:
            return ci, json.load(r)["results"]
    with ThreadPoolExecutor(len(urls)) as ex, open(OUT / "synqa2_readings.jsonl", "a") as f:
        for ci, res in ex.map(run, range(len(chunks))):
            for j, x in zip(chunks[ci], res):
                assert x["plan"] and set(x["plan"]) == {str(L) for L in LAYERS}
                f.write(json.dumps({"k": key(j["system"], j["user"]), "v": x["plan"]}, ensure_ascii=False) + "\n")
            f.flush(); print("read", (ci + 1) * 24, "/", len(todo), flush=True)


def gold(tag=""):
    import torch
    from safetensors.torch import save_file
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from safetensors import safe_open
    have = set()
    for g in OUT.glob("gold*.safetensors"):
        with safe_open(str(g), "pt") as h:
            have |= set(h.keys())
    J = [j for j in (json.loads(l) for l in (OUT / "jobs.jsonl").read_text().splitlines()) if f"{j['dataset']}|{j['id']}" not in have]
    tok = AutoTokenizer.from_pretrained(MODEL); tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="auto", max_memory={0: "40GiB", 1: "44GiB"})
    model.eval(); model.requires_grad_(False)
    dev0 = model.get_input_embeddings().weight.device
    acts = {}
    hs = [model.model.layers[L].register_forward_hook(lambda m, i, o, L=L: acts.__setitem__(L, (o[0] if isinstance(o, tuple) else o)[:, -1].detach())) for L in LAYERS]
    text = [tok.apply_chat_template([{"role": "system", "content": j["system"]}, {"role": "user", "content": j["user"]}],
                                    tokenize=False, add_generation_prompt=True, enable_thinking=False) for j in J]
    order = sorted(range(len(J)), key=lambda i: len(text[i]))
    out = {}
    for b in range(0, len(order), 8):
        idx = order[b:b + 8]
        enc = tok([text[i] for i in idx], return_tensors="pt", padding=True, add_special_tokens=False).to(dev0)
        with torch.inference_mode():
            model(**enc, use_cache=False)
        for r, i in enumerate(idx):
            out[f"{J[i]['dataset']}|{J[i]['id']}"] = torch.stack([acts[L][r].to(torch.bfloat16).cpu() for L in LAYERS]).contiguous()
        if b % 400 == 0:
            print("gold", b + len(idx), "/", len(J), flush=True)
    for h in hs:
        h.remove()
    save_file(out, str(OUT / (f"gold_{tag}.safetensors" if tag else "gold.safetensors")))
    print("gold done", len(out), flush=True)


def fve(layers, tag, datasets=None):
    import torch
    from safetensors.torch import load_file
    from transformers import AutoTokenizer
    from oracle_lens.core.nnomp import nnls_refit
    from oracle_lens.core.whitening import load_whitener
    from oracle_lens.pipeline.rl_reward import reward_text_ids, split_bullets
    from fve_repro import WH, ar_vectors, load_ar
    J = [j for j in (json.loads(l) for l in (OUT / "jobs.jsonl").read_text().splitlines()) if not datasets or j["dataset"] in datasets]
    G = {}
    for g in OUT.glob("gold*.safetensors"):
        G.update(load_file(str(g)))
    tok = AutoTokenizer.from_pretrained(MODEL)
    peft_inner, head, tags, ar_layers = load_ar()
    rng = random.Random(0)
    with open(OUT / f"fve_{tag}.jsonl", "w") as out:
        for L in layers:
            wh = load_whitener(WH / f"whitening_iolens_pooled_L{L}.safetensors", ridge_c=0.1)
            for d in sorted({j["dataset"] for j in J}):
                rs = [j for j in J if j["dataset"] == d]
                x = wh.whiten(torch.stack([G[f"{j['dataset']}|{j['id']}"][LAYERS.index(L)].float() for j in rs]))
                perm = list(range(len(rs)))
                while any(i == p for i, p in enumerate(perm)):
                    rng.shuffle(perm)
                res = {}
                for cond, texts in (("own", [j["reading"][str(L)] for j in rs]), ("other", [rs[p]["reading"][str(L)] for p in perm])):
                    bl = [split_bullets(t, 4) for t in texts]
                    flat = [(i, b) for i, bs in enumerate(bl) for b in bs]
                    ids = [reward_text_ids(tok, b) for _, b in flat]
                    keep = [k for k, s in enumerate(ids) if s]
                    vec = ar_vectors(peft_inner, head, tags, ar_layers.index(L), [ids[k] for k in keep])
                    V = torch.zeros(len(rs), 4, x.shape[1]); valid = torch.zeros(len(rs), 4, dtype=torch.bool); slot = [0] * len(rs)
                    for k, v in zip(keep, vec):
                        i = flat[k][0]; w = wh.whiten(v[None])[0]
                        V[i, slot[i]] = w / (w.norm() + 1e-9); valid[i, slot[i]] = True; slot[i] += 1
                    _, fv = nnls_refit(V, x, valid)
                    res[cond] = torch.where(valid.any(1), fv, torch.zeros_like(fv))
                for j, a, b in zip(rs, res["own"], res["other"]):
                    out.write(json.dumps({"dataset": d, "id": j["id"], "layer": L, "fve_own": float(a), "fve_other": float(b)}) + "\n")
                out.flush()
                print(f"L{L} {d}: own {float(res['own'].mean()):.3f} other {float(res['other'].mean()):.3f} n {len(rs)}", flush=True)


def table():
    import statistics as st
    R = [json.loads(l) for f in sorted(OUT.glob("fve_*.jsonl")) for l in f.read_text().splitlines()]
    names = {"cancer_myth": "Cancer-Myth", "hotpotqa": "HotpotQA", "hover": "HoVer", "symptom2disease": "Symptom2Disease", "medcalc": "MedCalc-Bench", "synqa2": "SynQA2"}
    summ = {}
    lines = ["| dataset | n | " + " | ".join(f"L{L} own / other" for L in LAYERS) + " | mean own / other |", "|---|---|" + "---|" * (len(LAYERS) + 1)]
    for d, nm in names.items():
        cells, own_all, oth_all = [], [], []
        for L in LAYERS:
            rs = [r for r in R if r["dataset"] == d and r["layer"] == L]
            if not rs:
                cells.append("–"); continue
            o, t = st.mean(r["fve_own"] for r in rs), st.mean(r["fve_other"] for r in rs)
            own_all.append(o); oth_all.append(t); cells.append(f"{o:.3f} / {t:.3f}")
            summ.setdefault(d, {})[f"L{L}"] = {"own": o, "other": t, "n": len(rs)}
        n = max((v["n"] for v in summ.get(d, {}).values()), default=0)
        lines.append(f"| {nm} | {n} | " + " | ".join(cells) + f" | {st.mean(own_all):.3f} / {st.mean(oth_all):.3f} |" if own_all else f"| {nm} | 0 |")
    (OUT / "summary.json").write_text(json.dumps(summ, indent=1))
    (OUT / "table.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "jobs":
        jobs()
    elif cmd == "read":
        read()
    elif cmd == "gold":
        gold(sys.argv[2] if len(sys.argv) > 2 else "")
    elif cmd == "fve":
        fve([int(x) for x in sys.argv[2].split(",")], sys.argv[3], sys.argv[4].split(",") if len(sys.argv) > 4 else None)
    else:
        table()
