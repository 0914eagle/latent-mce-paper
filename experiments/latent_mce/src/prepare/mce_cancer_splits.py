"""Write the balanced Cancer-Myth folds (data/fpq/cancer_myth_balanced_splits.json: train 50 FPQ / 50 NFP, validation
48 / 48, the test of data/fpq/splits.json; the balanced GEPA runs' folds) as jsonl for the original-MCE environment
(src/experiment/mce_cancer_env.py): results/data/mce_cancer/fold<k>_{train,val,test}.jsonl, val = the fold's dev.
One line per item: id, question (the v4 bundle text, as GEPA and the judge use), label (FPQ / NFP), reference correction.
Rows in id order; MCE samples and shuffles the training rows itself.
Usage: .venv-fpq/bin/python -m experiments.latent_mce.src.prepare.mce_cancer_splits
"""
import json
from pathlib import Path

from common.paths import results_dir
from experiments.latent_mce.src.experiment.v4_eval import P4

ROOT = Path(__file__).resolve().parents[4]


def main():
    rows = {r["id"]: r for r in json.load(open(P4 / "protocol/bundle.json"))["rows"] if r["dataset"] == "cancer_myth"}
    refs = {json.loads(l)["id"]: json.loads(l) for l in (ROOT / "data/fpq/cancer_myth_questions.jsonl").read_text().splitlines()}
    out = results_dir("latent_mce") / "data" / "mce_cancer"
    out.mkdir(parents=True, exist_ok=True)
    for s in json.load(open(ROOT / "data/fpq/cancer_myth_balanced_splits.json")):
        if s["dataset"] != "cancer_myth":
            continue
        for role, name in (("train", "train"), ("dev", "val"), ("test", "test")):
            ids = sorted(s[role])
            with open(out / f"{s['name']}_{name}.jsonl", "w") as f:
                for i in ids:
                    fpq = i.startswith("fpq")
                    f.write(json.dumps({"id": i, "question": rows[i]["question"], "label": "FPQ" if fpq else "NFP",
                                        "correction": refs[i].get("correction", "") if fpq else ""}, ensure_ascii=False) + "\n")
            print(s["name"], name, len(ids), "FPQ", sum(i.startswith("fpq") for i in ids), flush=True)


if __name__ == "__main__":
    main()
