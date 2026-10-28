"""Cancer-Myth as an environment of the original MCE code (mce_upstream, c4b7a7c), with and without the Oracle lens.

The loop, agents, prompts, batching, validation and selection are the original mce.main / mce.eval, unchanged. This file
only adds the task, in the format of the original one-step environment (env/symptom_diagnosis): one interface, one LLM
call, a score. MCE_LATENT (env) picks the arm; the two arms differ only in the latent information:
  none  (MCE):        get_context(question) -> str
  lens  (latent MCE): get_context(question, latent) -> str, latent = the fixed Oracle-lens reading of the answering
                      model's state right before its first answer token, for the question with the Plain prompt
                      (no context): {"36": text, "44": text, "52": text, "60": text}. The same reading is added to
                      every training record ("internal_reading"), so the agents see it with the outcome.
Answer: Qwen3.6-27B through MCE's generator client (OpenAI-compatible vLLM server, greedy, thinking off on the server),
system = transport_common + Plain prompt, plus "\\n\\nInstructional context:\\n{context}" when the context is not empty
(the placement of all latent_mce runs; context in the user message cost NFP ~20 points, E31), user = the question.
Score: the Luna Well-rubric judge (v4_eval's), pass = rating >= 4 -> accuracy 1 / 0. Feedback and training records:
rating, label (FPQ / NFP) and, for an FPQ, the reference correction; no judge explanation (as the reference-only GEPA
runs and the E29/E35 loops). Data: one fold of data/fpq/splits.json (train / dev as validation / test), written by
src/prepare/mce_cancer_splits.py.
"""
import asyncio
import hashlib
import json
import logging
import os
import random
import threading
import time
import urllib.request
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from env.base import EnvironmentResult, InterfaceSignature, Sample, TaskEnvironment

from common.paths import results_dir
from experiments.latent_mce.src.experiment.v4_eval import B, CM, CONTEXT_FORMAT, LUNA, P4, Q, WellBridge, parse_rating

logger = logging.getLogger(__name__)
LATENT = os.environ.get("MCE_LATENT", "none")
assert LATENT in ("none", "lens"), LATENT
LENS_URL = os.environ.get("LENS_URL", "http://127.0.0.1:8796/run")  # serve27b_lens.py, READ_LAYERS=36,44,52,60
BLOCKS = ("36", "44", "52", "60")

CONF = json.load(open(P4 / "run/qwen3_8b/identity.json"))["full_config"]
REG = json.load(open(CM / CONF["registry"]))
PLAIN = REG["transport_common"] + "\n\n" + REG["prompts"]["plain"]
ROWS = {r["id"]: r for r in json.load(open(P4 / "protocol/bundle.json"))["rows"] if r["dataset"] == "cancer_myth"}
BRIDGE = WellBridge(CONF)
DATA = results_dir("latent_mce") / "data" / "mce_cancer"
LOCK = threading.Lock()
CACHE = {"judge": {}, "lens": {}}


def _load(kind):
    path = DATA / f"{kind}_cache.jsonl"
    if not CACHE[kind] and path.exists():
        CACHE[kind].update({json.loads(l)["k"]: json.loads(l)["v"] for l in path.read_text().splitlines()})


def _cached(kind, key, make):
    """One jsonl cache per kind (judge ratings, lens readings) shared by all runs; make() runs outside the lock."""
    path = DATA / f"{kind}_cache.jsonl"
    with LOCK:
        _load(kind)
        if key in CACHE[kind]:
            return CACHE[kind][key]
    v = make()
    with LOCK:
        CACHE[kind][key] = v
        with path.open("a") as f:
            f.write(json.dumps({"k": key, "v": v}, ensure_ascii=False) + "\n")
    return v


def judge(qid, answer):
    """Luna Well rating (1-5) of one answer, blind; retried with backoff on API errors (e.g. 429). None if unparsed."""
    def make():
        sys_msg, user = B.envelope(BRIDGE(op="judge", row=ROWS[qid], answer=answer), exact=True)
        for attempt in range(8):
            try:
                rating = parse_rating(Q._openai(LUNA, sys_msg, user)[0])
                if rating is not None:
                    return rating
            except ValueError:
                continue
            except Exception:
                time.sleep(min(600, 30 * 2 ** attempt) * (1 + random.random()))
        return None
    return _cached("judge", hashlib.sha256((qid + "\x1f" + answer).encode()).hexdigest()[:32], make)


def lens_reading(question):
    """Fixed Oracle-lens reading (four blocks) of the Plain-prompt state before the first answer token; no generation."""
    def make():
        body = {"items": [{"system": PLAIN, "question": question, "read": True, "gen": False}]}
        req = urllib.request.Request(LENS_URL, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=3600) as r:
            plan = json.load(r)["results"][0]["plan"]
        assert plan and set(plan) == set(BLOCKS), plan
        return {b: plan[b] for b in BLOCKS}
    return _cached("lens", hashlib.sha256((PLAIN + "\x1f" + question).encode()).hexdigest()[:32], make)


class CancerMythEnvironment(TaskEnvironment):
    """One-step answer: get_context -> Qwen answer (Plain prompt + context) -> Well judge."""

    def get_interface_signatures(self) -> List[InterfaceSignature]:
        inputs = [("question", "str", "The patient's question")]
        if LATENT == "lens":
            inputs.append(("latent", "dict", "Oracle-lens reading of the answering model's state before it answers: "
                                             "{'36': str, '44': str, '52': str, '60': str} (four depths)"))
        return [InterfaceSignature(name="get_context", inputs=inputs, output=("str", "Context for answering the question"),
                                   description="Return context that helps the LLM answer the patient's question well.")]

    def get_task_instruction(self) -> str:
        lens = ("""
Internal information: get_context also receives `latent`, and every training record has "internal_reading": text
decoded by a fixed Oracle lens from the answering model's internal state right before its first answer token (the
question with the system prompt, no context), at four depths (blocks 36, 44, 52, 60 of 64). It shows what the model
is representing before it answers; it can be noisy, repeat the question or be unrelated.
""" if LATENT == "lens" else "")
        sig = "get_context(question, latent)" if LATENT == "lens" else "get_context(question)"
        return f"""Cancer-Myth Patient Questions (One-Step)

Patients ask questions about cancer. Some questions rest on a false premise (a cancer myth or misconception, "FPQ");
the others have no false premise ("NFP"). A good answer corrects a false premise when there is one and otherwise
answers the question directly and helpfully. The answering LLM gets a fixed system prompt; the context returned by
get_context is appended to that system prompt. Each answer is rated 1-5 by a judge with a rubric for its label
(pass = 4 or 5). Training records give the label and, for an FPQ, the reference correction.
{lens}
Curate context files in context/ folder and implement {sig} to provide guidelines and knowledge.

IMPORTANT: In your interface implementation, use ABSOLUTE paths to access context files
(e.g., use the full path like /path/to/workspace/iter1_sub0/context/guide.md).
"""

    async def aevaluate(self, sample: Sample, interfaces: Dict[str, Callable], llm_client: Any = None,
                        context_dir: Optional[Path] = None, log_dir: Optional[Path] = None) -> EnvironmentResult:
        assert llm_client is not None, "LLM client is required"
        trajectory, qid = [], sample.extras["qid"]
        latent = await asyncio.to_thread(lens_reading, sample.question) if LATENT == "lens" else None
        get_context = (interfaces or {}).get("get_context")
        context = ""
        if get_context:
            try:
                context = (get_context(sample.question, latent) if LATENT == "lens" else get_context(sample.question)) or ""
                trajectory.append({"step": "get_context", "output": context})
            except Exception as e:
                logger.warning(f"get_context error: {e}")
                trajectory.append({"step": "get_context", "error": str(e)})
        else:
            trajectory.append({"step": "get_context", "output": "(no interface)", "note": "Using empty context"})
        system = PLAIN + (CONTEXT_FORMAT.format(context=context) if context else "")
        try:
            answer = await llm_client.ainvoke([{"role": "system", "content": system}, {"role": "user", "content": sample.question}])
        except Exception as e:
            logger.error(f"LLM inference error: {e}")
            return EnvironmentResult(feedback=f"LLM inference error: {e}", ground_truth=sample.ground_truth, metrics={"accuracy": 0.0},
                                     trajectory=trajectory + [{"step": "llm_inference", "error": str(e)}])
        rating = await asyncio.to_thread(judge, qid, answer)
        ok = (rating or 0) >= 4
        trajectory.append({"step": "evaluate", "prediction": answer, "rating": rating, "is_correct": ok, "latent": latent})
        return EnvironmentResult(feedback=f"The answer was rated {rating}/5 by the judge (pass = 4 or 5).\n{sample.ground_truth}",
                                 ground_truth=sample.ground_truth, metrics={"accuracy": 1.0 if ok else 0.0}, trajectory=trajectory)

    def format_result_for_training(self, item: Dict[str, Any]) -> Dict[str, Any]:
        sample, ev = item.get("sample", {}), item.get("evaluation", {})
        last = next((s for s in ev.get("trajectory", []) if s.get("step") == "evaluate"), {})
        rec = {"id": sample.get("id"), "question": sample.get("question"), "label": sample.get("extras", {}).get("label"),
               "ground_truth": sample.get("ground_truth"), "llm_prediction": last.get("prediction"), "rating": last.get("rating"),
               "is_correct": ev.get("metrics", {}).get("accuracy", 0.0) == 1.0}
        if LATENT == "lens":
            rec["internal_reading"] = last.get("latent")
        return rec

    def load_samples(self, path: str, limit: int = 10, random_sample: bool = False, shuffle: bool = False) -> List[Sample]:
        samples = []
        with open(path, encoding="utf-8") as f:
            for i, row in enumerate(f):
                if not random_sample and limit and i >= limit:
                    break
                d = json.loads(row)
                gold = (f"This question rests on a false premise that the answer should correct. Reference correction: {d['correction']}"
                        if d["label"] == "FPQ" else "This question has no false premise.")
                samples.append(Sample(id=i, question=d["question"], ground_truth=gold, extras={"qid": d["id"], "label": d["label"]}))
        if random_sample and limit and len(samples) > limit:
            samples = random.sample(samples, limit)
        if shuffle:
            random.shuffle(samples)
        return samples


def precompute_lens(port_url=LENS_URL):
    """Read every question of the three folds once (chunks of 48, file order) into the lens cache, so each question has
    one fixed reading for every run and the lens server is needed only for this step."""
    qs = sorted({json.loads(l)["question"] for f in sorted(DATA.glob("fold*_*.jsonl")) for l in f.read_text().splitlines()})
    key = lambda q: hashlib.sha256((PLAIN + "\x1f" + q).encode()).hexdigest()[:32]
    with LOCK:
        _load("lens")
    todo = [q for q in qs if key(q) not in CACHE["lens"]]
    print(len(qs), "questions,", len(todo), "to read", flush=True)
    for c in range(0, len(todo), 48):
        chunk = todo[c:c + 48]
        body = {"items": [{"system": PLAIN, "question": q, "read": True, "gen": False} for q in chunk]}
        req = urllib.request.Request(port_url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=24 * 3600) as r:
            res = json.load(r)["results"]
        for q, x in zip(chunk, res):
            assert x["plan"] and set(x["plan"]) == set(BLOCKS), q[:60]
            _cached("lens", key(q), lambda: {b: x["plan"][b] for b in BLOCKS})
        print("read", c + len(chunk), flush=True)


if __name__ == "__main__":
    precompute_lens()
