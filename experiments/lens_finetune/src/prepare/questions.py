"""Write the question list and the Plain prompt of the lens fine-tuning experiment (results/data/questions.jsonl, plain.txt).

Questions: the balanced Cancer-Myth folds of latent_mce (results/data/mce_cancer/fold<f>_{train,val,test}.jsonl, from
data/fpq/cancer_myth_balanced_splits.json); every question is in exactly one fold's test, and in the validation of
zero or more folds. Plain: the latent_mce environment's own system prompt (mce_cancer_env.PLAIN), so the reading keys
equal the latent_mce lens cache; checked against that cache here.
Run with the mce venv from the repo root (needs PAPER_V4_DIR): python -m experiments.lens_finetune.src.prepare.questions
"""
import json
import sys

from common.paths import EXPERIMENTS_DIR

sys.path.insert(0, str(EXPERIMENTS_DIR / "latent_mce" / "mce_upstream"))
from experiments.latent_mce.src.experiment.mce_cancer_env import PLAIN  # noqa: E402
from experiments.lens_finetune.src.experiment.lens_common import OUT, lens_key, read_jsonl  # noqa: E402

DATA = EXPERIMENTS_DIR / "latent_mce" / "results" / "data" / "mce_cancer"  # the stored folds, whatever RESULTS_SUBDIR says
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "plain.txt").write_text(PLAIN)
rows = {}
for f in range(3):
    for r in read_jsonl(DATA / f"fold{f}_test.jsonl"):
        assert r["id"] not in rows, r["id"]
        rows[r["id"]] = {"qid": r["id"], "question": r["question"], "label": r["label"], "test_fold": f, "val_folds": []}
for f in range(3):
    for r in read_jsonl(DATA / f"fold{f}_val.jsonl"):
        rows[r["id"]]["val_folds"].append(f)
cache = {r["k"] for r in read_jsonl(DATA / "lens_cache.jsonl")}
missing = sum(lens_key(r["question"]) not in cache for r in rows.values())
assert rows and missing == 0, f"{missing} questions have no latent_mce lens reading: Plain prompt differs"
with open(OUT / "questions.jsonl", "w") as fo:
    for r in rows.values():
        fo.write(json.dumps(r, ensure_ascii=False) + "\n")
print(f"{len(rows)} questions ({sum(r['label'] == 'FPQ' for r in rows.values())} FPQ), keys match the latent_mce cache")
