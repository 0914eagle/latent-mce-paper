# Results so far (as of 2026-10-29)

Cancer-Myth, Qwen3.6-27B answering model (vLLM 0.18.1, greedy, thinking off), gpt-6-luna Well judge (pass ≥ 4),
macro = (FPQ pass % + NFP pass %) / 2. Fold 0 unless noted. Every entry points to its pre-registration and result note
in [`experiments/latent_mce/prereg/prereg_v4_test.md`](../experiments/latent_mce/prereg/prereg_v4_test.md) (search the
stage name, e.g. "## E25"). Intervals are 95% (question bootstrap or paired t over runs, as stated in the note).

## 1. What helped: the hidden state decides whether a context is attached

| Evidence | Result |
|---|---|
| E25: 14 learned contexts on fold-0 test, gates | Plain 64.1 · attach to all 65.6 · Qwen asked "false premise?" 65.3 · TF-IDF gate 66.3 · **probe gate 68.8** · label gate 73.2 |
| E25: paired differences | probe − Qwen +3.5 [−0.2, +7.7] · probe − text +2.5 [−0.6, +5.9] · probe − all +3.3 [−1.8, +8.6]; probe beats text on 14 of 14 contexts (not independent) |
| E26: folds 1 and 2 | probe gate 64.7 (Plain 63.7) · 65.5 (Plain 65.0) |
| E27: probe gate inside the MCE loop (LAT) vs the same loop (OUT5), 5 runs each | 69.2 vs 66.5, +2.6 (Welch p ≈ 0.08); 2 of 5 LAT runs selected v1, so the gain belongs to the gate, not the lens observations |
| E39: selectors that pick one of 15 candidates per question, trained on a training grid | a2 best context + probe gate 69.1 · text 69.6 · Plain state 69.6 · question+context state 69.9 · + lens 70.4 · oracle 92.3 |

The probe gate keeps NFP near Plain (78.0 vs 79.6) while most of the FPQ gain stays. It decides *whether*; nothing
tried so far closes the gap to the per-question oracle (92.3) for *what* to attach.

## 2. What did not help: Oracle-lens readings given to the agents

| Evidence | Result |
|---|---|
| E29: MCE A (outputs) vs B (+ lens change memory), 5 runs each | 66.6 vs 68.3, B − A +1.6 [−6.0, +9.2]; bugs: meta index after restart, SKILL paths missing in the base workspace |
| E30: B with an explicit internal step, 3 runs | 64.9 |
| E35: repaired loop, A vs B, 3 runs each | 69.3 vs 66.0, B − A −3.4 [−10.8, +4.1]; 11 of 50 codex calls failed with 429 (no edit), B's get_context narrowed to training-wording triggers (B_b attached to 1 of 193 test FPQ) |
| E22: lens failure type × fix, 81 training FPQ failures | correction seen: say-it 46.7 / reference fact 66.7; no correction plan seen: 20.8 / 98.1. The fact is best in both groups, so the type does not change the best fix (98.1 uses the item's own reference) |
| E34: four lens readings per item vs the probe | adding the 4-read consensus lowers FPQ AUC 0.765 → 0.724; frame errors are stable (90 of 158 FPQ never show a correction) |
| E37: with-context hidden state as a selector | 66.1 vs probe gate 68.8, −2.7 [−7.0, +1.0]; attaches to 60% of NFP |
| E39: + lens features in the learned selector | e − d +0.49 [−4.08, +5.44] |
| E40: lens read at the most uncertain answer token | does not show the path sampling would take: differing − same sample +0.004 [−0.029, +0.034] (post hoc at the pre-answer state: +0.039 [+0.016, +0.061], needs a fresh test) |

Why (E29 B records, 2,150 reading/answer pairs from 1,039 distinct inputs): the reading's status label points the same
way as the answer in 55.6%, the opposite way in 12.2%, and is off-topic or unclear in 32%. Candidates an edit could use
(correction in the reading but not in the answer; NFP reading starts correcting while the answer is fine) are 5.4% of
records, and an audit found the output-only baseline as good as internal signals for predicting the next edit's
outcome (AUC 0.718 vs 0.739, interval including 0). In 25 of 30 E29 B edits the base agent wrote that the reading adds
nothing beyond the answer; the meta agent cited specific readings but turned them into caution rules.

## 3. What the data say about the task

| Evidence | Result |
|---|---|
| E31: 4 sampled answers (T 0.8) per training question | 27% of FPQ sometimes corrected, 51% never; 32% of NFP sometimes over-corrected |
| E31: Plain hidden state vs question text for "sometimes corrected" FPQ | 0.76 (train) / 0.79 (dev) vs 0.54 / 0.51; label-free on dev 0.62 [0.49, 0.74] |
| E38: are question × context splits repeatable? (40 mixed questions × 4 contexts × 4 observations) | stored "A passes, B fails" splits reproduced in 81.9%, reversed in 1.2%; variance: question 0.42, context 0.04, question × context 0.47, noise 0.21 |
| E29 dev selection | dev100 macro of the selected version does not predict test (13 runs, r = −0.01); dev has 20 NFP |

## 4. Open issues

- No clean MCE A/B run: E29/E30 had loop bugs, E35 lost agent calls to 429. A rerun needs 429 detection and retry
  (`turn.failed` in the codex events) and at most 3 concurrent runs.
- E35/E36 summary prompt saw only the first 3,000 characters of the answer (81% of answers are longer);
  `E35_ANSWER_CHARS=0` passes the full text.
- E36 (dev NFP 40) is invalid (25 of 33 agent calls failed) and was removed.
- The MCE loop here uses 3 iterations over 50 new training questions each; the MCE paper runs 5 epochs over the whole
  training set, and its GEPA baseline gets at least MCE's rollout budget. Our GEPA (500 calls) has less than our MCE
  (about 730 rollouts per run).

## 5. Table runs on the other datasets (2026-10-29)

Qwen3.6-27B answering model, original MCE loop (c4b7a7c), codex exec agents. Ours = latent MCE with the fine-tuned
Oracle lens (`experiments/lens_finetune`), readings in the training records only. One run per row.

**HotpotQA** (GEPA artifact test 300; `experiments/mce_repro_qwen3/src/analysis/hotpot_metrics.py` in LatentGEPA)

| Method | EM | F1 | Both | Recall | EM\|both |
|---|---|---|---|---|---|
| CoT | 65.0 | | | | |
| GEPA (stopped at 918 / 2000 calls) | 73.3 | | | | |
| Ours, iter4 (val 76.0) | 73.67 | 80.55 | 73.0 | 83.5 | 84.02 |

EM = SQuAD-normalized exact match; Both = both gold supporting titles among the two retrievals' passages; Recall = mean
fraction of gold titles retrieved; EM|both = EM over the Both questions (219). One Ours row lost its program call and
counts as wrong. MCE (other server) not added yet.

**MedCalc-Bench, CoT start** (test 1048; Luna step-wise judge; regex as MedRaC)

| Method | Regex | Final | Form. | Extr. | Calc. | Ans. | Regex calc | Regex rule |
|---|---|---|---|---|---|---|---|---|
| CoT | 75.9 | 63.1 | 82.0 | 82.3 | 97.1 | 74.0 | 82.2 | 64.8 |
| MCE, original code, context appended (`experiments/mce_medcalc_cot`), iter1 (val 72.7) | 75.4 | 65.1 | 84.4 | 81.1 | 95.0 | 74.6 | 83.4 | 61.4 |
| Ours | running (val per iteration 63.6, 70.0, 71.8, 73.6) | | | | | | | |

- MCE val per iteration: 72.7, 68.2, 71.8, 67.3, 71.8 (110 rows, SE about 4.3); no gain after iteration 1.
- Paired against CoT (regex): calc 667 rows +1.2 (17 vs 9 discordant, McNemar p 0.17); rule 381 rows −3.4 (23 vs 36,
  p 0.12). Judge Final by type: calc 78.1 vs 75.0, rule 42.3 vs 42.3.
- The MCE context is one general text of about 6,000 characters for every row; rule calculators get one line each
  (e.g. APACHE II), not their point tables. Wrong rule answers count unmentioned criteria as absent, as the benchmark
  does; they miss or mis-score criteria (e.g. Wells DVT −2 item).

**Do the agents use the readings?** (Ours HotpotQA and MedCalc workspaces) The readings are in the training records
(`internal_reading`, blocks 36/44/52/60) and the agents open them (36 / 30 commands), but no context file, no
`get_context` interface and no skill mentions them. The meta agents call them "weak diagnostic evidence" / "noisy";
both runs had the default caveat in the task text (`LENS_NOTE` unset). At test, `get_context(question)` has no
reading, so a reading can change only how rules are written, not which rule a question gets.

**SynQA2** (other server; `synqa2-mce` branch of LatentGEPA): MCE 72.8 vs Plain 79.5 macro; FPQ 82.3 → 91.3, TPQ
76.7 → 54.3; iteration 1 best in both runs.
