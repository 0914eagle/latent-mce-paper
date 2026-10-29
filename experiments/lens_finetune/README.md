# lens_finetune — fine-tune the Oracle lens on Cancer-Myth residuals, with the frozen AR as the judge

**Question.** The Oracle lens (`oracle-lens s3d.rl.ddp600`, a LoRA on frozen Qwen3.6-27B) was trained on WildChat /
FineWeb activations. On Cancer-Myth its reading of the pre-answer state mostly restates the opening of the answer and
often names the wrong cancer (a bladder-cancer question read as colorectal surgery, block 36). Does a short
fine-tuning on Cancer-Myth residuals, judged by the same frozen activation reconstructor (AR) the lens was RL-tuned
against, make the readings explain more of the state (joint FVE), and more of *this question's* state (FVE against
the question's own residual minus FVE against another question's)? And do the better readings carry information
about the answer's behavior that the question text does not (diagnostic AUROCs, `diag.py`)?

**Status (2026-10-28).** Code written and checked end to end on 12 questions (`RESULTS_SUBDIR=smoke`, below);
the full run has not been launched.

## Method

The lens recipe of record is unchanged; only the data are Cancer-Myth. Background: the lens repo
`/data1/heejae/repos/olens_draft` ([docs/pipeline.md](https://github.com/agu18dec/olens_draft/blob/a325900/docs/pipeline.md),
[docs/rl.md](https://github.com/agu18dec/olens_draft/blob/a325900/docs/rl.md)).

- **Input (what the lens reads).** h = output of block L (36 / 44 / 52 / 60) at the last token of the Plain Cancer-Myth
  prompt (system = transport + Plain, user = the question, chat template, thinking off): the state right before the
  first answer token, the reading latent_mce uses. h is injected as `16000 · h / |h|` in place of the marker `㈜` of
  the `concepts_raw` reader prompt ("An activation vector from layer L … Produce distinct concepts …"); the lens
  writes up to 128 tokens of `- ` bullets.
- **Judge (no labels).** The frozen prompt-tag AR the lens was RL-tuned on
  (`ckpts/ar/ptag/ar.asst.ptag.pooled.s0/ex16013184` in the gated HF dataset `agu18dec/local-workspace`; held-out
  joint FVE 0.59 / 0.59 / 0.59 / 0.53 at blocks 36 / 44 / 52 / 60). Each bullet (bare tokens, ≤ 64) goes through the
  AR with the tag `[Layer L]`; bullet images and h are whitened (`whitening_iolens_pooled_L{L}`, ridge 0.1), images
  unit-normalized, and h refit onto them by non-negative least squares:
  **joint FVE = 1 − |x_w − Σ cᵢ aᵢ|² / |x_w|²** — the lens's own RL reward (`rl_reward.score_texts_joint`), the same
  definition as `latent_mce/src/experiment/lens_fve.py`. **Shuffled FVE**: the same bullets refit onto another
  question's residual (a fixed derangement); a generic reading scores high on both.
- **Training (rejection-sampling SFT, RAFT-1 of the lens repo).** For every question and layer, 8 readings from the base
  lens with the unclamped sampler the lens was RL-tuned with (temperature 1, top-p 1, top-k 0); the top 2 by joint FVE
  among those that end with EOS become targets (the generated token ids + EOS; never re-tokenized). One epoch of
  cross-entropy on the target tokens, lens LoRA warm-started from the base lens (fp32 master weights), base frozen;
  AdamW lr 2e-5, 10 warmup steps then constant, clip 1.0, batch 16 (micro-batch 2 × 8). The adapter with the lowest
  validation loss is kept.
- **Folds.** The balanced folds of latent_mce (`data/fpq/cancer_myth_balanced_splits.json`, written to
  `latent_mce/results/data/mce_cancer/fold<f>_*.jsonl`). One adapter per fold: trained on every question outside the
  fold's test and validation (~386), validated on its validation (96), evaluated on its test (241–242). No labels,
  ratings or answers are used anywhere in training.
- **Evaluation.** Base and fine-tuned lens, 4 readings per question and layer at the served setting (temperature 0.8,
  top-p 0.95, top-k 20 from the model's generation config, as `serve27b_lens.py`), on the fold's test:
  mean joint FVE, shuffled FVE, specificity (FVE − shuffled FVE), EOS rate, tuned − base with a paired bootstrap 95%
  interval over questions. Then `diag.py`: cross-validated TF-IDF AUROCs of the reading vs the question text for
  FPQ vs NFP, MCE-rescued vs both-fail FPQs, Base-pass vs Base-fail FPQs, MCE-lost vs kept NFPs
  (groups from latent_mce's Base and original-MCE test runs).

## Run (this server, 4× RTX 4090 24 GB)

Every step loads Qwen3.6-27B bf16 over the four GPUs (`device_map`, 17/19/19/19 GiB), so **vLLM must be stopped**.

```bash
experiments/lens_finetune/run_here.sh            # all steps, folds 0 1 2; resumable
```

| Step | Module | Output (`results/data/`) |
|---|---|---|
| download the AR + whiteners (1.4 GB, once; strips the AR LoRA's torch.compile key prefix into `lora_hf/`) | `src/prepare/fetch_ar.py` | `$OLENS_ARTIFACTS/hf/…` |
| question list + Plain prompt (mce venv; checks every key against the latent_mce lens cache) | `src/prepare/questions.py` | `questions.jsonl`, `plain.txt` |
| residuals, 724 questions × 4 blocks | `src/experiment/residuals.py` | `residuals.safetensors` |
| readings: `pool` (8 / question / layer, training candidates), `eval_base`, `eval_fold<f>` (4, served setting) | `src/experiment/sample.py` | `readings/<tag>.jsonl` |
| AR scores (joint FVE, shuffled FVE) | `src/experiment/score.py` | `scores/<tag>.jsonl` |
| SFT per fold | `src/experiment/train.py` | `adapters/fold<f>/` (+ `train_log.jsonl`, `selection.json`) |
| base vs tuned table; readings exported in the lens-cache format | `src/experiment/report.py` | `report.json`, `report.md`, `lens_cache/<tag>.jsonl` |
| behavioral diagnostic (scikit-learn venv) | `src/experiment/diag.py` | `diag.md` |

Options: `FOLDS`, `KEEP` (targets per question and layer), `LR`, `N_POOL`, `N_EVAL`, `TRAIN_ARGS` (extra `train.py`
flags). Check run on a few questions, kept apart from the stored results:

```bash
RESULTS_SUBDIR=smoke LIMIT=12 FOLDS=0 N_POOL=4 N_EVAL=2 TRAIN_ARGS="--max-steps 2 --batch 4 --eval-every 1" experiments/lens_finetune/run_here.sh
```

Venvs: `hf27b` (torch 2.10, transformers 5.16, peft 0.20; no flash-attn / fla, so the gated-delta-rule layers run the
torch fallback), `mce` for `questions.py`, `cancer_myth_internals` for `diag.py`. `PYTHONPATH` = repo root + the lens
repo's `src` (set by `run_here.sh`). The lens repo's `multilayer_reconstructor` imports its training stack
(`mytorch_lightning`, `pydra`), so `score.py` rebuilds the small prompt-tag AR wrapper itself.

## Measured on this server

- Memory (LoRA SFT, gradient checkpointing): micro-batch 1 peaks at 13–17 GiB per GPU, 2 at 15–20 GiB; 3 and 4 run out
  of memory (torch fallback of the gated delta rule). 1.6–1.7 s per micro-batch of 2.
- Check run (12 questions; `results/smoke/`, not tracked): sampling ≈ 0.4 s per reading, AR scoring ≈ 0.1 s per
  reading. Base lens, served setting: joint FVE 0.27 / 0.40 / 0.57 / 0.58 at blocks 36 / 44 / 52 / 60, shuffled
  0.16 / 0.20 / 0.22 / 0.24 — the readings are about their own question, more so at the later blocks.
- Full run estimate: pool 23k readings ≈ 2.6 h, eval_base 11.6k ≈ 1.3 h, scoring ≈ 1 h, per fold SFT
  (~3,100 examples) ≈ 1.4 h + eval ≈ 0.5 h; about 10 hours in all.

## Caveats

- The FVE gain is bounded by the AR: the lens repo found the policy already near the AR's own reconstruction ceiling,
  so a large FVE gain is not expected; the specificity and the diagnostic AUROCs are the outcomes that matter here.
- Joint FVE rewards bullets that reconstruct the vector, not bullets about premises; better readings need not say
  whether the model will accept a false premise. `diag.py` tests exactly that against the question-text baseline.
- latent_mce's environment reads one shared lens cache keyed by question; a latent-MCE run with fine-tuned readings
  needs each fold's own readings of all its questions (train, validation and test) under that fold's adapter.
