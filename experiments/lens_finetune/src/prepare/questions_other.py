"""Write the question list of the lens fine-tuning experiment for a dataset other than Cancer-Myth.

Questions and chat inputs: the items of the Oracle-lens FVE table (latent_mce/results/data/lens_fve/jobs.jsonl, from
latent_mce/src/experiment/lens_fve.py), each with its own system prompt, so the residuals and the base readings are the
ones of that table and `lens_key` matches its readings:
  hotpotqa / hover (300 each): system "", user = the question / claim (GEPA artifact test split)
  synqa2 (600): system = transport + Plain, user = the question; 300 normal / false-premise pairs
  medcalc (1048): the paper's zero-shot CoT prompt (system + user)
Folds (three, as the Cancer-Myth folds; no labels are used in training): units shuffled with seed 0 (SynQA2: a pair,
both questions together; otherwise a question), unit i in fold i % 3's test; fold f's validation = the first half of
fold (f + 1) % 3's test units; fold f trains on the rest (about half of the dataset). label: SynQA2 "FPQ" (false
premise) / "NFP" (normal); the others "NA".
Usage (repo root, any venv): python -m experiments.lens_finetune.src.prepare.questions_other <dataset>
with RESULTS_SUBDIR=<dataset> so each dataset has its own results/<dataset>/data.

--native: the dataset's own split instead of folds, written as fold 0 (test_fold 0 = test, val_folds [0] = validation,
the rest train; test_fold -1):
  hotpotqa / hover: the GEPA artifact split (mce_repro_qwen3 gepa_splits): train 150, validation = the first 100 of its
    val 300, test 300 (the FVE table's items)
  synqa2: the registered pair-family partition (paired_evaluation_metadata.json role, 60/20/20, pairs together):
    train = 300 pairs of its train, validation = 50 pairs of its dev (seed 0), test = every evaluation pair
  medcalc (.venv-fpq): train 220 / validation 110 rows of the MedCalc-Bench v1.2 train split as the MedRaC MCE and GEPA
    runs build them (common/medrac/items.py, medrac_mce_qwen36_27b conf), test = the 1048 test rows; CoT prompt
Same chat inputs as above.
"""
import json
import random
import sys
from collections import defaultdict

from common.paths import EXPERIMENTS_DIR
from experiments.lens_finetune.src.experiment.lens_common import OUT, lens_key, read_jsonl

D = sys.argv[1]
NATIVE = "--native" in sys.argv
JOBS = EXPERIMENTS_DIR / "latent_mce" / "results" / "data" / "lens_fve" / "jobs.jsonl"
items = [j for j in read_jsonl(JOBS) if j["dataset"] == D]
assert items, f"no {D} items in {JOBS}"
if NATIVE:
    def write(rows):  # rows: (qid, system, user, label, part)
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / "plain.txt").write_text(rows[0][1])
        with open(OUT / "questions.jsonl", "w") as fo:
            for qid, system, user, label, part in rows:
                fo.write(json.dumps({"qid": qid, "question": user, "system": system, "label": label,
                                     "test_fold": 0 if part == "test" else -1, "val_folds": [0] if part == "val" else []},
                                    ensure_ascii=False) + "\n")
        print(D, "native split:", {p: sum(r[4] == p for r in rows) for p in ("train", "val", "test")})
    rng = random.Random(0)
    if D in ("hotpotqa", "hover"):
        sp = EXPERIMENTS_DIR / "mce_repro_qwen3" / "results" / "data" / "gepa_splits"
        rows = []
        for part, f, n in (("train", "train", None), ("val", "val", 100), ("test", "test", None)):
            for i, r in enumerate(read_jsonl(sp / f"{D}_{f}.jsonl")[:n]):
                rows.append((f"{D}_{f}_{i}" if f != "test" else f"{D}_{i}", "", r["question"], "NA", part))
        write(rows)
    elif D == "synqa2":
        S = "/home/eagle0914/cancer-myth/results/oracle_harness_qwen_final_20261024_v2/syn_qa2/"
        meta = {m["id"]: m for m in json.load(open(S + "paired_evaluation_metadata.json"))}
        text = {x["id"]: x["question"] for x in json.load(open(S + "paired_inputs.json"))}
        plain = items[0]["system"]
        pairs = defaultdict(list)
        for i, m in meta.items():
            pairs[(m["role"], m["pair_id"])].append(i)
        take = {"train": 300, "dev": 50, "evaluation": None}
        rows = []
        for role, part in (("train", "train"), ("dev", "val"), ("evaluation", "test")):
            ps = sorted(p for r, p in pairs if r == role)
            if take[role]:
                ps = sorted(rng.sample(ps, take[role]))
            for p in ps:
                for i in sorted(pairs[(role, p)]):
                    rows.append((i, plain, text[i], "FPQ" if i.endswith(":false") else "NFP", part))
        write(rows)
    elif D == "medcalc":
        import yaml
        from types import SimpleNamespace
        from common.medrac.dataset import load_test_set
        from common.medrac.items import build_items
        from common.medrac.plain import cot_prompts
        cfg = yaml.safe_load(open(EXPERIMENTS_DIR / "medrac_mce_qwen36_27b" / "conf" / "config.yaml"))["trainval"]
        tmp = OUT / "medcalc_trainval_items.json"
        build_items(SimpleNamespace(**cfg), tmp)
        tv = json.loads(tmp.read_text())
        rows = []
        for part, recs in (("train", tv["train"]), ("val", tv["val"])):
            ps = cot_prompts([str(r["Calculator ID"]) for r in recs], [r["Patient Note"] for r in recs], [r["Question"] for r in recs])
            rows += [(f"medcalc_{part}_{r['Row Number']}", s_, u_, "NA", part) for r, (s_, u_) in zip(recs, ps)]
        df = load_test_set()
        ps = cot_prompts(df["Calculator ID"].astype(str).tolist(), df["Patient Note"].tolist(), df["Question"].tolist())
        rows += [(f"medcalc_{n}", s_, u_, "NA", "test") for n, (s_, u_) in zip(df["Row Number"].tolist(), ps)]
        write(rows)
    sys.exit()
unit = (lambda j: j["id"].rsplit(":", 1)[0]) if D == "synqa2" else (lambda j: j["id"])
groups = defaultdict(list)
for j in items:
    groups[unit(j)].append(j)
units = sorted(groups)
random.Random(0).shuffle(units)
test_fold = {u: i % 3 for i, u in enumerate(units)}
by_fold = {f: [u for u in units if test_fold[u] == f] for f in range(3)}
val_of = defaultdict(list)
for f in range(3):
    nxt = by_fold[(f + 1) % 3]
    for u in nxt[: len(nxt) // 2]:
        val_of[u].append(f)
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "plain.txt").write_text(items[0]["system"])  # unused: every question carries its own system prompt
with open(OUT / "questions.jsonl", "w") as fo:
    for u in units:
        for j in groups[u]:
            label = ("FPQ" if j["id"].endswith(":false") else "NFP") if D == "synqa2" else "NA"
            assert lens_key(j["user"], j["system"]) == __import__("hashlib").sha256((j["system"] + "\x1f" + j["user"]).encode()).hexdigest()[:32]
            fo.write(json.dumps({"qid": j["id"], "question": j["user"], "system": j["system"], "label": label,
                                 "test_fold": test_fold[u], "val_folds": val_of[u]}, ensure_ascii=False) + "\n")
n = {f: sum(len(groups[u]) for u in by_fold[f]) for f in range(3)}
print(f"{D}: {len(items)} questions in {len(units)} units, test per fold {n}, "
      f"val per fold {[sum(len(groups[u]) for u in units if f in val_of[u]) for f in range(3)]}")
