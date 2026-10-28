# Method and experimental setup (for the paper draft)

Each item names the file that implements it, so every setting can be checked in the code. Where a value differs
between experiments, the experiment is named (E-numbers refer to `experiments/latent_mce/prereg/prereg_v4_test.md`).

## 1. Task and data

| Item | Setting | Source |
|---|---|---|
| Dataset | Cancer-Myth: patient questions that rest on a false premise (FPQ) and normal questions (NFP); each FPQ has a reference correction | `data/fpq/cancer_myth_questions.jsonl` |
| Folds (main MCE comparison) | 3 balanced folds: train 100 (50 FPQ / 50 NFP), validation 96 (48 / 48), test = the fold's test (241–242) | `data/fpq/cancer_myth_balanced_splits.json`, `src/prepare/mce_cancer_splits.py` |
| Folds (probe-gate and earlier loop experiments, E25–E35) | fold 0 of the frozen v4 splits: train 156 (78 / 78), dev100 (80 / 20), test 242 (193 / 49); folds 1 and 2 in E26 | `data/fpq/splits.json`, `src/prepare/v4_features.py` |
| Label use | labels and references are used only for training rows; test questions are given to `get_context` and the answering model without labels | `src/experiment/mce_cancer_env.py` |

## 2. Answering model and scoring

| Item | Setting | Source |
|---|---|---|
| Answering model | Qwen3.6-27B, vLLM 0.18.1, tensor parallel 2, greedy (temperature 0), thinking off, max 2048 new tokens | `src/experiment/e17_eval.py` (`ask`), `run_mce_cancer.sh` header |
| Prompt | system = transport text + the Plain prompt; the learned context is appended to the system prompt as `"\n\nInstructional context:\n{context}"`; user = the question. (Moving the context into the user message lost about 20 NFP points for a broad context, E31.) | `src/experiment/v4_eval.py` (`CONTEXT_FORMAT`), `mce_cancer_env.py` |
| Judge | gpt-6-luna with the Well rubric for the item's label (correct a real false premise; do not invent corrections for a normal question); rating 1–5, pass = rating ≥ 4; blind, shuffled order, cached by (item, answer) | `src/experiment/v4_eval.py`, `e18_loop.answer_and_rate` |
| Metric | macro = (FPQ pass % + NFP pass %) / 2 on the test split; MCE's own validation score is accuracy (pass = 1) | `e18_loop.py`, `mce_cancer_env.py` |

## 3. Internal signals

| Signal | Definition | Source |
|---|---|---|
| Hidden state | block outputs of layers 20, 24, …, 60 (11 layers, width 5120) at the last prompt token (the position before the first answer token), Plain prompt, chat template with thinking off | `src/experiment/serve27b_hidden.py` |
| Probe `p_fpq` | per layer: standardization + logistic regression (balanced classes, C = 0.003); score = mean of the 11 layer probabilities; trained on training items only (training items get 5-fold cross-fit scores); FPQ vs NFP AUC 0.77 on training (cross-fit) | `src/prepare/v4_features.py`, `src/experiment/e13_analysis.py` (`probe`) |
| Probe gate | attach the learned context to a question only when `p_fpq ≥ 0.5`; otherwise the Plain prompt | `src/experiment/e25_gate.py`, `e27_loop.py` |
| Oracle lens | a fixed LoRA reader (adapter `olens_s3d_rl600`) verbalizes the hidden state at the same position: the normalized state (scale 16000·h/‖h‖) is injected into a fixed reader prompt and the reader samples a short text (temperature 0.8, top-p 0.95, 128 tokens) for each of blocks 36, 44, 52, 60 of 64. The reader prompt and settings are never changed. | `src/experiment/serve27b_lens.py` |

## 4. Methods compared

### 4.1 Original MCE vs latent MCE (main comparison; `run_mce_cancer.sh`)
- **Loop:** the original MCE code (`mce_upstream`, unchanged): a meta agent writes a skill; a base agent follows it and
  writes `context/` files and the `get_context` interface; training rollouts are scored and fed back; the iteration
  with the best validation accuracy is evaluated on test. Agents: gpt-6-luna through the `codex` CLI, reasoning effort
  medium.
- **Settings:** 5 iterations, training batch 50, all 100 training rows, all validation rows; test = Base (no context,
  shared) and the validation-best iteration of each arm; folds 0, 1, 2.
- **Arms (differ only in the latent information):**
  - *MCE*: `get_context(question) -> str`; training records: question, label, reference correction (FPQ), answer,
    rating, pass.
  - *latent MCE*: `get_context(question, latent) -> str`, with `latent` = the Oracle-lens reading of the question under
    the Plain prompt (`{"36": text, "44": text, "52": text, "60": text}`); the same reading is added to every training
    record as `internal_reading`. Each question is read once, so it has one reading in all runs.
- No judge explanation is given to the agents in either arm.

### 4.2 Probe gate on learned contexts (E25–E27)
- **E25 (test-time gate):** 14 contexts learned by earlier MCE-style runs, each attached to all fold-0 test questions;
  gates decide per question whether the context is attached: probe gate, a TF-IDF text classifier trained on the
  training questions, the answering model asked whether the question has a false premise, and the label gate (upper
  bound).
- **E27 (gate inside the loop):** our MCE-style loop (3 iterations; base agent edits from training rollouts; meta agent
  from iteration 2; dev100 selection), with the probe gate applied to every attached context in rollouts, dev and test
  (LAT) vs the same loop without it (OUT5); 5 runs each.

### 4.3 Baselines
- **Plain:** the system prompt with no context.
- **GEPA:** a GEPA-optimized system prompt for fold 0 (Well settings, budget 500 metric calls), evaluated with the same
  answering model and judge (`src/experiment/e29_gepa_baseline.py`).

## 5. Compute

One server with 8 RTX A6000 (48 GB): vLLM answer servers and the lens / hidden-state servers take two GPUs each.
An MCE-style run (E29) took about 2 hours with 5 runs per arm in parallel; agents and the judge are API calls.

## 6. Known limitations to state
- Run-to-run variance is large (E29 A: test macro 64.3–69.6 over 5 identical runs); dev100 has 20 NFP and its score did
  not predict test across 13 runs (r = −0.01).
- vLLM answers are not batch-invariant (about 8% of pass/fail flip on regeneration).
- The judge is an LLM (gpt-6-luna); scores from different judges are never mixed.
