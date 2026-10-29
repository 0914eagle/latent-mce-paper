# latent-mce-paper

Code for the paper's experiments: does the answering model's **internal state** help **Meta Context Engineering
(MCE)** learn a better context than outputs and scores alone? Task: Cancer-Myth (patient questions that rest on a false
premise, FPQ, and normal questions, NFP). Answering model: Qwen3.6-27B.

This repository is a curated subset of the project repository (private, `JakeFRCSE/LatentGEPA`, `experiments/latent_mce`,
commit `320ec0f`). It holds only what the paper's method and setup need. Results data, model weights and the private
judge code are not included (see "External pieces").

- **Start here (co-authors, Korean):** [`docs/PAPER_BRIEF.md`](docs/PAPER_BRIEF.md) — why, what, method, results, references, how to write each section
- **Method and setup for the paper:** [`docs/METHOD_SETUP.md`](docs/METHOD_SETUP.md)
- **All results so far, with the experiment that produced each number:** [`docs/RESULTS.md`](docs/RESULTS.md)
- **Pre-registrations and result notes of every stage (E1–E40):** [`experiments/latent_mce/prereg/prereg_v4_test.md`](experiments/latent_mce/prereg/prereg_v4_test.md)

## What is compared

```
                       ┌────────── Plain: fixed system prompt, no context
question ─► get_context(...) ─► "Instructional context" appended to the system prompt ─► Qwen3.6-27B ─► answer ─► Luna judge (1-5, pass ≥ 4)
              ▲                                                                                              │
              └──────── written and revised by the MCE agents from training rollouts ◄────────────────────────┘

  MCE           get_context(question)                 agents see question, label, reference, answer, rating
  latent MCE    get_context(question)                 + the Oracle-lens reading of the pre-answer state WITH the rollout's
                                                      context (re-read every iteration), as feedback only: in the base agent's
                                                      records; the meta agent's meta_agent/train.jsonl has the no-context reading
  baselines     Plain, GEPA (optimized system prompt)
  (separate, E25-E27) probe gate: attach a learned context only when the hidden-state probe p_fpq >= 0.5
```

## Layout

| Path | What |
|---|---|
| `experiments/latent_mce/mce_upstream/` | the original MCE code, vendored unchanged (metaevo-ai/meta-context-engineering, c4b7a7c; see `VENDORED.md`, `LICENSE`) |
| `experiments/latent_mce/src/experiment/mce_cancer_env.py` | Cancer-Myth as an MCE environment; `MCE_LATENT=none` (MCE) or `lens` (latent MCE: readings in the agents' feedback) |
| `experiments/latent_mce/src/experiment/run_mce_cancer.py` | registers the environment and runs the original `mce.main` / `mce.eval` |
| `experiments/latent_mce/run_mce_cancer.sh` | **main run script**: folds 0–2, both arms, test of the validation-best iteration |
| `experiments/latent_mce/src/prepare/mce_cancer_splits.py` | writes the balanced folds as jsonl for the MCE environment |
| `experiments/latent_mce/src/experiment/serve27b_lens.py` | Oracle-lens server (fixed reader, blocks 36/44/52/60) |
| `experiments/latent_mce/src/experiment/serve27b_hidden.py` | hidden-state server (layers 20–60 at the last prompt token) |
| `experiments/latent_mce/src/prepare/v4_features.py` | train-only hidden-state probe `p_fpq` per item |
| `experiments/latent_mce/src/experiment/e25_gate.py` | probe gate vs other gates over stored answers (E25) |
| `experiments/latent_mce/src/experiment/e27_loop.py`, `run_e27.sh`, `run_e21.sh` | our earlier MCE-style loop with the probe gate inside (E27) and its base (E21) |
| `experiments/latent_mce/src/experiment/e29_gepa_baseline.py` | Plain and GEPA on the test split |
| `experiments/latent_mce/src/experiment/v4_eval.py` | answer generation and the blind, cached Luna Well judge |
| `data/fpq/` | Cancer-Myth questions and the frozen folds (`splits.json`, `cancer_myth_balanced_splits.json`) |

## Setup

These scripts were run on one 8-GPU server (RTX A6000, 48 GB). The paths below are that server's; change them for
another machine.

| Piece | Where (on the original server) |
|---|---|
| Python for `python -m experiments...` scripts | `.venv-fpq` (`requirements-fpq.txt`) |
| Python for the MCE loop and `get_context` | `/data2/heejae/uv/mce` (`experiments/latent_mce/mce_upstream/pyproject.toml`) |
| vLLM 0.18.1 (answers) | `/data2/heejae/uv/vllm_0181` |
| Lens and hidden-state servers (transformers 5, peft) | `/data2/heejae/uv/jpp_lens`; optional fast kernels `PYTHONPATH=/data2/heejae/uv/fla_pkgs` |
| Qwen3.6-27B | `/data4/heedong/repos/llm/params/Qwen3.6-27B` |
| Oracle-lens adapter / reader code | `/data2/heejae/oracle_lens/olens_and_ar/olens_s3d_rl600` / `/data2/heejae/external/olens_draft/src` |
| Agents | `codex` CLI logged in; model gpt-6-luna, reasoning effort medium |
| Judge | gpt-6-luna through the OpenAI API (`OPENAI_API_KEY`) |

### External pieces (not in this repository)
- **Judge bridge and prompt registry** (`qwen3_patches`, `WellBridge`, Plain prompt): private repo
  `0914eagle/cancer-myth-internals`; `v4_eval.py` reads it from `CANCER_MYTH_DIR` and `PAPER_V4_DIR`.
- **Oracle lens** reader code and adapter (paths above).
- **Results data** (answers, ratings, hidden states, lens readings, agent workspaces): private HF dataset
  `latentgpea/latent-feedback-cells-data` (folder name `latent_feedback_cells`, results up to 2026-10-26).
  `e25_gate.py`, `e27_loop.py` and `v4_features.py` need it under `experiments/latent_mce/results/`.

## Run

Servers (two GPUs each; vLLM TP 2, thinking off, max 2048 new tokens):
```bash
CUDA_VISIBLE_DEVICES=0,1 vllm serve <Qwen3.6-27B> --tensor-parallel-size 2 --port 8802 --served-model-name qwen36-27b \
  --max-model-len 16384 --default-chat-template-kwargs '{"enable_thinking": false}'
CUDA_VISIBLE_DEVICES=4,5 READ_LAYERS=36,44,52,60 python experiments/latent_mce/src/experiment/serve27b_lens.py 8796 <cache dir>
```
Main experiment (original MCE vs latent MCE, folds 0–2):
```bash
.venv-fpq/bin/python -m experiments.latent_mce.src.prepare.mce_cancer_splits     # balanced folds -> results/data/mce_cancer
GEN_PORTS=8802,8803,8804 ARMS=none experiments/latent_mce/run_mce_cancer.sh      # MCE row, 3 folds, then test
# latent MCE (Ours): the lens server stays up during the loop (readings of each rollout with its context)
D=experiments/latent_mce/results/data/mce_cancer; for k in 0 1 2; do
  DATASET=cancer_myth TRAIN=$D/fold${k}_train.jsonl LENS_URLS=http://127.0.0.1:8796/run STEP=precompute experiments/latent_mce/run_ours.sh
  DATASET=cancer_myth TRAIN=$D/fold${k}_train.jsonl VAL=$D/fold${k}_val.jsonl TEST=$D/fold${k}_test.jsonl TL=100 VL=96 \
    GEN_PORT=8802 LENS_URLS=http://127.0.0.1:8796/run TAG=fold$k experiments/latent_mce/run_ours.sh &
done; wait
```
Probe gate and baselines (need the results data):
```bash
.venv-fpq/bin/python -m experiments.latent_mce.src.prepare.v4_features           # p_fpq per item (train-only probe)
.venv-fpq/bin/python -m experiments.latent_mce.src.experiment.e25_gate 8802      # gates over 14 learned contexts
E18_GEN_PORTS=8802,8803 .venv-fpq/bin/python -m experiments.latent_mce.src.experiment.e29_gepa_baseline
experiments/latent_mce/run_e27.sh                                                 # probe gate inside our MCE-style loop
```

## Status (2026-10-28)
- Done: probe gate (E25–E27), GEPA and Plain baselines, our MCE-style loops with lens memory (E29/E30/E35), analyses
  (E31–E40). Numbers in `docs/RESULTS.md`.
- In progress: the original-MCE comparison (`run_mce_cancer.sh`); no complete result yet.
