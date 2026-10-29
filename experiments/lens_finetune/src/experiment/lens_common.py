"""Shared pieces of the lens fine-tuning experiment: paths, the Oracle-lens reader on Qwen3.6-27B, the injection, data.

The lens (oracle-lens s3d.rl.ddp600, a LoRA on frozen Qwen3.6-27B) reads ONE residual vector: the vector replaces the
embedding of the marker slot of the concepts_raw reader prompt ("An activation vector from layer L ... <activation>㈜
</activation>. Produce distinct concepts ..."), scaled as 16000 * h / |h| (the contract the lens was trained with), and
the lens writes '- ' bullets. h = output of block L at the last token of the Plain Cancer-Myth prompt (system = Plain,
user = question, chat template, thinking off), i.e. the state right before the first answer token, L in 36/44/52/60 —
the same reading as the latent_mce lens cache (serve27b_lens.py), so a key here equals a key there.
All weights are spread over the four 24 GB GPUs (device_map); nothing else may hold them. torch is imported inside the
functions, so prepare/questions.py can import the paths from the mce venv (no torch there).
"""
import hashlib
import json
import os
import sys
from pathlib import Path

from common.paths import results_dir

sys.path.insert(0, os.environ.get("OLENS_SRC", "/data1/heejae/repos/olens_draft/src"))

MODEL = os.environ.get("QWEN36_PATH", "/data1/heejae/models/Qwen3.6-27B")
BASE_LENS = os.environ.get("OLENS_ADAPTER", "/data1/heejae/models/oracle-lens-s3d-ddp600")
ARTIFACTS = Path(os.environ.get("OLENS_ARTIFACTS", "/data1/heejae/olens_artifacts"))
AR_DIR = ARTIFACTS / "hf/ckpts/ar/ptag/ar.asst.ptag.pooled.s0/ex16013184"  # the AR the lens was RL-tuned against
WHITENER_DIR = ARTIFACTS / "hf/data/whiteners/pooled"  # its whiteners (config.whitener_prefix whitening_iolens_pooled)
LAYERS = (36, 44, 52, 60)
ALPHA = 16000.0
MAX_NEW = 128  # reader output budget, as the lens was trained and served
OUT = results_dir("lens_finetune") / "data"
MAX_MEMORY = {0: "17GiB", 1: "19GiB", 2: "19GiB", 3: "19GiB"}  # GPU 0 also holds the embeddings and activations
if os.environ.get("LENS_MAX_MEMORY"):  # another server, e.g. "0:40GiB,1:44GiB" for two 48 GB GPUs (env_a6000.sh)
    MAX_MEMORY = {int(k): v for k, v in (x.split(":") for x in os.environ["LENS_MAX_MEMORY"].split(","))}


def read_jsonl(path):
    p = Path(path)
    return [json.loads(l) for l in p.open()] if p.exists() else []


def append_jsonl(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def questions():
    """All Cancer-Myth questions of the balanced folds: qid, question, label, test_fold, val_folds (prepare/questions.py)."""
    return read_jsonl(OUT / "questions.jsonl")


def plain():
    return (OUT / "plain.txt").read_text()


def system_of(q):
    """The question's own system prompt (other datasets, prepare/questions_other.py), else the Plain prompt."""
    return q["system"] if "system" in q else plain()


def lens_key(question, system=None):
    """Key of the latent_mce lens cache (mce_cancer_env.lens_reading; lens_fve.py for the other datasets)."""
    return hashlib.sha256(((plain() if system is None else system) + "\x1f" + question).encode()).hexdigest()[:32]


def split(fold):
    """qids of fold `fold`: train = every question outside its test and validation, val = its validation, test."""
    qs = questions()
    test = [q["qid"] for q in qs if q["test_fold"] == fold]
    val = [q["qid"] for q in qs if fold in q["val_folds"]]
    train = [q["qid"] for q in qs if q["test_fold"] != fold and fold not in q["val_folds"]]
    return train, val, test


def select_qids(spec):
    """'all', 'test:<f>', 'train:<f>', 'val:<f>', 'trainval:<f>' -> qids."""
    if spec == "all":
        return [q["qid"] for q in questions()]
    if spec.startswith("trainval:"):
        tr, va, _ = split(int(spec.split(":")[1]))
        return tr + va
    part, fold = spec.split(":")
    return dict(zip(("train", "val", "test"), split(int(fold))))[part]


def load_tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(MODEL)


def load_base():
    import torch
    from transformers import AutoModelForCausalLM
    base = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="auto", max_memory=MAX_MEMORY)
    base.requires_grad_(False)
    base.eval()
    return base


def load_lens(base, adapter, trainable=False):
    """Base + lens LoRA. Asserts the adapter is live: every saved LoRA tensor maps onto a module (PEFT loads nothing,
    silently, when the key schema drifts)."""
    from peft import PeftModel
    from safetensors import safe_open
    pm = PeftModel.from_pretrained(base, str(adapter), is_trainable=trainable)
    with safe_open(f"{adapter}/adapter_model.safetensors", "pt") as f:
        saved = len(list(f.keys()))
    lora = [n for n, _ in pm.named_parameters() if "lora_" in n]
    assert len(lora) >= 0.9 * saved, f"{adapter}: {len(lora)} LoRA tensors loaded of {saved} saved"
    if not trainable:
        pm.eval()
    return pm


def reader_prompts(tok):
    """layer -> rendered concepts_raw reader prompt (input_ids, slot of the marker)."""
    from oracle_lens.pipeline.verbalizer import renderer_for
    return {L: renderer_for("concepts_raw")(tok, layer=L) for L in LAYERS}


def eos_ids(model):
    e = model.generation_config.eos_token_id
    return set(e if isinstance(e, list) else [e])


def reader_embeds(model, wv, rows, targets=None, pad_id=0):
    """Input embeddings for reader rows [(layer, h)], marker slot replaced by ALPHA * h / |h|; with `targets` (token
    lists) the targets are appended, the batch right-padded, and labels returned (-100 outside the targets).
    Returns embeds, attention mask, labels (None without targets)."""
    import torch
    embed = model.get_input_embeddings()
    dev = embed.weight.device
    seqs = [wv[L].input_ids + (targets[i] if targets else []) for i, (L, _) in enumerate(rows)]
    W = max(len(s) for s in seqs)
    ids = torch.tensor([s + [pad_id] * (W - len(s)) for s in seqs], device=dev)
    e = embed(ids).clone()
    for i, (L, h) in enumerate(rows):
        h = h.to(dev, torch.float32)
        e[i, wv[L].slot] = (ALPHA * h / h.norm().clamp_min(1e-9)).to(e.dtype)
    mask = torch.tensor([[1] * len(s) + [0] * (W - len(s)) for s in seqs], device=dev)
    if not targets:
        return e, mask, None
    labels = torch.tensor([[-100] * len(wv[L].input_ids) + targets[i] + [-100] * (W - len(seqs[i]))
                           for i, (L, _) in enumerate(rows)], device=dev)
    return e, mask, labels


def load_residuals():
    """qid -> [len(LAYERS), d] fp32 residuals (residuals.py)."""
    from safetensors.torch import load_file
    h = load_file(str(OUT / "residuals.safetensors"))["h"]
    qids = json.loads((OUT / "residuals_index.json").read_text())
    return {q: h[i] for i, q in enumerate(qids)}
