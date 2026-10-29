"""Latent MCE for any MCE environment: Oracle-lens readings as training feedback only (the paper's Ours).

2026-10-29 (user): the reading in each training record is of the state WITH the context that rollout actually used
(LENS_STATE=context, the default): in every training rollout of every iteration, the answering model's input with the
current get_context output is read by the fixed lens on a live lens server (LENS_URLS), so the agents see how their
context changed what the model represents before answering. LENS_STATE=plain keeps the earlier definition (one
reading per training question of the input without context, precomputed and reused in every iteration). Validation
and test rollouts are not read (the agents never see them per item). Text below written for the plain definition
still holds for the record format and the meta agent's rows (which keep the no-context reading of precompute).

The Cancer-Myth latent MCE (mce_cancer_env.py, MCE_LATENT=lens) made general: `with_lens(Env, lens_input)` returns a
subclass of an original-format MCE environment whose interfaces, answer call, score and test are unchanged, and whose
agents additionally read, for every training rollout, "internal_reading" = the fixed Oracle-lens reading
(olens_s3d_rl600 on serve27b_lens.py, blocks 36/44/52/60) of the answering model's state right before its first
answer token for the task input without any context:
  - in every training record the base agent reads (data/train.json, next to the outcome);
  - in the training rows the meta agent reads: run MCE on the `<train>_lens.jsonl` file `precompute` writes (the
    training rows + "internal_reading"), as the Cancer-Myth run does with fold<k>_train_lens.jsonl;
  - the task instruction describes it (training data only; no internal input at test time).
`lens_input(sample) -> (system, user)` is the chat input whose state is read (the dataset's own prompt, no context):
  hotpotqa / hover (mce_repro_qwen3 mce_envs): ("", the question / claim), the first module's question as E42 read it
  medcalc_cot (mce_medcalc_cot env):            the paper's zero-shot CoT system and user messages
Readings are made once per training question (`precompute`, needs the lens server), cached in the latent_mce lens
cache (keyed by sha256(system + "\\x1f" + user)[:32], shared with mce_cancer_env and lens_fve), and looked up there in
the loop, so no lens server runs during MCE.
Usage: experiments/latent_mce/run_ours.sh (STEP=precompute once, then the loop with the lens server running).
"""
import hashlib
import json
import os
import sys
import threading
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
CACHE = ROOT / "experiments/latent_mce/results/data/mce_cancer/lens_cache.jsonl"
BLOCKS = ("36", "44", "52", "60")
_LOCK, _MEM = threading.Lock(), {}

INSTRUCTION = """
Internal information (training data only): every training record has "internal_reading": text decoded by a fixed
Oracle lens from the answering model's internal state right before its first answer token, for the task input
without any context, at four depths (blocks 36, 44, 52, 60 of 64). It shows what the model is representing before it
answers; it can be noisy, repeat the input or be unrelated. The interfaces do not receive it: it is there to help you
understand the training outcomes. The meta agent's training rows (meta_agent/train.jsonl) carry the same field.
"""


def key(system, user):
    return hashlib.sha256((system + "\x1f" + user).encode()).hexdigest()[:32]


EXTRA = [ROOT / "experiments/latent_mce/results/data/e43_medcalc/readings.jsonl",  # E43: MedCalc CoT inputs
         ROOT / "experiments/latent_mce/results/data/lens_fve/synqa2_readings.jsonl"]  # the FVE table's SynQA2 reads


def _cache():
    with _LOCK:
        if not _MEM:
            for f in EXTRA + [CACHE]:  # same reader, blocks and key; new readings go to CACHE
                if f.exists():
                    _MEM.update({json.loads(l)["k"]: json.loads(l)["v"] for l in f.read_text().splitlines()})
        return _MEM


def reading(system, user):
    return _cache().get(key(system, user))


LENS_STATE = os.environ.get("LENS_STATE", "context")
assert LENS_STATE in ("context", "plain"), LENS_STATE
INSTRUCTION_CONTEXT = """
Internal information (training data only): every training record has "internal_reading": text decoded by a fixed
Oracle lens from the answering model's internal state right before its first answer token, for the input of that
rollout WITH the context your interface returned (so it changes when your context changes), at four depths (blocks
36, 44, 52, 60 of 64). It shows what the model is representing before it answers; it can be noisy, repeat the input
or be unrelated. The interfaces do not receive it: it is there to help you see how the context changed the model's
internal state on the training questions. The meta agent's training rows (meta_agent/train.jsonl) carry the reading of
the input without any context.
"""


def read_live(system, user):
    """The cached reading of (system, user), else one read on the lens servers (LENS_URLS), cached."""
    r = reading(system, user)
    if r is not None:
        return r
    urls = os.environ["LENS_URLS"].split(",")
    k = key(system, user)
    body = {"items": [{"system": system, "question": user, "read": True, "gen": False}]}
    req = urllib.request.Request(urls[int(k, 16) % len(urls)], data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    for attempt in range(6):
        try:
            with urllib.request.urlopen(req, timeout=3600) as resp:
                plan = json.load(resp)["results"][0]["plan"]
            break
        except OSError:
            if attempt == 5:
                raise
            import time
            time.sleep(10 * 2 ** attempt)
    assert plan and set(plan) == set(BLOCKS), plan
    v = {b: plan[b] for b in BLOCKS}
    with _LOCK, CACHE.open("a") as f:
        _MEM[k] = v
        f.write(json.dumps({"k": k, "v": v}, ensure_ascii=False) + "\n")
    return v


def used_context(trajectory):
    """The context the rollout's first get_context-type call returned ("" without an interface or on an error)."""
    for s in trajectory:
        if str(s.get("step", "")).startswith("get_") and str(s.get("step", "")).endswith("context"):
            out = s.get("output", "")
            return "" if out == "(no interface)" else (s.get("full_output", out) or "")
    return ""


def with_lens(Env, lens_input, name=None):
    """The latent-MCE variant of an original-format MCE environment class (feedback only).
    lens_input(sample, context) -> (system, user): the chat input whose pre-answer state is read."""
    class LensEnv(Env):
        def get_task_instruction(self):
            return super().get_task_instruction() + (INSTRUCTION_CONTEXT if LENS_STATE == "context" else INSTRUCTION)

        def load_samples(self, path, *a, **k):
            out = super().load_samples(path, *a, **k)
            for smp in out:  # only training rows are read (their records are the agents' feedback)
                smp.extras["_lens_train"] = "train" in Path(path).name
            return out

        async def aevaluate(self, sample, interfaces, llm_client=None, context_dir=None, log_dir=None):
            res = await super().aevaluate(sample, interfaces, llm_client, context_dir, log_dir)
            if sample.extras.get("_lens_train"):
                ctx = used_context(res.trajectory) if LENS_STATE == "context" else ""
                import asyncio
                r = await asyncio.to_thread(read_live, *lens_input(sample, ctx))
                res.trajectory.append({"step": "internal_reading", "context_chars": len(ctx), "reading": r})
            return res

        def format_result_for_training(self, item):
            rec = super().format_result_for_training(item)
            step = next((s for s in item.get("evaluation", {}).get("trajectory", []) if s.get("step") == "internal_reading"), {})
            rec["internal_reading"] = step.get("reading")
            return rec
    LensEnv.__name__ = name or Env.__name__ + "Lens"
    return LensEnv


def lens_inputs():
    """dataset -> (environment class, lens_input) for the datasets whose environment modules are present (each import
    is tried on its own, so a checkout with only some experiments, e.g. the paper repository, still works)."""
    sys.path[:0] = [os.getcwd(), str(ROOT)]
    os.environ.setdefault("MCE_LATENT", "none")  # Cancer-Myth: the plain environment, wrapped here like the others
    out = {}
    try:
        from experiments.latent_mce.src.experiment import mce_cancer_env as C

        def cancer(sample, ctx=""):  # Plain + "\n\nInstructional context:\n" + context (mce_cancer_env)
            return C.PLAIN + (C.CONTEXT_FORMAT.format(context=ctx) if ctx else ""), sample.question
        out["cancer_myth"] = (C.CancerMythEnvironment, cancer)
    except ImportError:
        pass
    try:
        from experiments.mce_repro_qwen3.src.experiment.mce_envs import HotpotQAEnvironment, HoverEnvironment

        def multihop(sample, ctx=""):  # the first module's question field with its context appended (mce_envs.lm)
            return "", sample.question + (f"\n\nContext:\n{ctx}" if ctx else "")
        out["hotpotqa"], out["hover"] = (HotpotQAEnvironment, multihop), (HoverEnvironment, multihop)
    except ImportError:
        pass
    try:
        from experiments.mce_medcalc_cot.src.experiment.env import MedCalcCoTEnvironment
        from common.medrac.plain import cot_prompts

        def medcalc(sample, ctx=""):  # the CoT call: system + "\n\nContext:\n" + context (mce_medcalc_cot env)
            r = sample.extras["row"]
            system, user = cot_prompts([str(r["Calculator ID"])], [r["Patient Note"]], [r["Question"]])[0]
            return (system + f"\n\nContext:\n{ctx}" if ctx else system), user
        out["medcalc_cot"] = (MedCalcCoTEnvironment, medcalc)
    except ImportError:
        pass
    return out


def precompute(dataset, train_path):
    """Read every training row once on the lens servers (LENS_URLS, chunks of 24), add the readings to the cache and
    write <train>_lens.jsonl (the rows + "internal_reading")."""
    Env, inp = lens_inputs()[dataset]
    env = Env.__new__(Env)
    samples = env.load_samples(train_path, limit=0)
    lines = Path(train_path).read_text().splitlines()
    assert len(samples) == len(lines), "load_samples must keep every row in file order"
    pairs = [inp(s, "") for s in samples]
    todo = sorted({key(*p): p for p in pairs if reading(*p) is None}.items())
    urls = os.environ["LENS_URLS"].split(",")
    print(f"{len(pairs)} rows, {len(todo)} to read", flush=True)
    for c in range(0, len(todo), 24):
        chunk = todo[c:c + 24]
        body = {"items": [{"system": s, "question": u, "read": True, "gen": False} for _, (s, u) in chunk]}
        req = urllib.request.Request(urls[(c // 24) % len(urls)], data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=24 * 3600) as r:
            res = json.load(r)["results"]
        with _LOCK, CACHE.open("a") as f:
            for (k, _), x in zip(chunk, res):
                assert x["plan"] and set(x["plan"]) == set(BLOCKS), k
                _MEM[k] = {b: x["plan"][b] for b in BLOCKS}
                f.write(json.dumps({"k": k, "v": _MEM[k]}, ensure_ascii=False) + "\n")
        print("read", c + len(chunk), flush=True)
    out = Path(train_path).with_name(Path(train_path).stem + "_lens.jsonl")
    out.write_text("".join(json.dumps({**json.loads(l), "internal_reading": reading(*p)}, ensure_ascii=False) + "\n"
                           for l, p in zip(lines, pairs)))
    print(out, flush=True)


if __name__ == "__main__":
    assert sys.argv[1] == "precompute", sys.argv
    precompute(sys.argv[2], sys.argv[3])
