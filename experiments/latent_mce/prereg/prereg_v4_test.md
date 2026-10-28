# Pre-registration: Latent-MCE (cells) vs baselines on Cancer-Myth fold-0 test (242)
Written 2026-10-25 10:37 KST, before any test generation.
Version choice (dev 40 only): higher macro pass, then higher mean rating, then the earlier iteration (v1 before v2),
the rule of latent_mce_optimization_20261025. Main test comparison: Plain, text MCE (other session's selected iter2
context, re-evaluated on the same servers and judge), ours (selected version). Ablations on the same test items, run
after the main comparison: Luna-only (B-card choice in every cell), 27B self-selection instead of Luna, probe-only
routing (p >= 0.5 -> B policy, no lens mode), shuffled cells (cell labels permuted across test items, seed 0).
Primary: macro pass (FPQ%+NFP%)/2, Luna Well >= 4, blind. Report all conditions; no test-based choice.
Version chosen 2026-10-25 10:40 KST: v2 (dev macro v1 = v2 = 65.0; mean rating v2 3.625 > v1 3.55).

## Clean rerun (written 2026-10-25 12:25 KST, before r1)
Fixed in advance: signals (train-only probe, Oracle-lens mode), cells, locked SKILL section and card format (r0),
harness tools (read_cards, choose_card with the step-1b question, call_llm), feedback fields (macro_points_vs_plain,
uncertain, mode_turned_on_by_card from the lens read with the card attached), exploration in every training rollout,
meta + base agent (gpt-6-luna medium) in every iteration, 3 iterations, selection on dev100 (dev 40 + 60 unused
non-test FPQ, seed 0) by macro, then mean rating, then the earlier iteration; then one test run (242). Budget is
counted as newly generated rollouts (prompts that differ from Plain). The development run (v0-v3) is reported separately.

## Hidden-state ablation of the card memory (written 2026-10-25 13:05 KST, fold 0 now used as development)
Question: do cards built with hidden-state cell feedback beat cards built by the same loop without any hidden-state
information, when both are used the same way at test time (gpt-6-luna choose_card on every question)?
Text arm t0 -> t1 -> t2: same base agent (gpt-6-luna medium, no meta agent, like v0 -> v2), same card format and tools,
training rows without cell / lens / probe fields, feedback per card only (no cells, no lens-with-card), training
rollouts attach the chosen card on every question. Compared on fold-0 test 242 against ours = v2 cards + Luna choice
on every question (70.6, already measured). Fold 0 is development now; this decides the design for folds 1-2.

## Memory-content check: does hidden-state information change what the base agent writes? (13:25 KST)
t0 (text fields only, already written) vs h0 (same rows + pass_forecast [train cross-fit probe of Plain pass from the
pre-answer state], plan_corrects [FRAME on the stored Plain-prefix lens readout], plan_vs_answer [plan vs FRAME on the
Plain answer's first 600 characters], plan_line [<=150 characters of the readout, flagged as noisy]); same skill text
apart from one paragraph that names the fields, same base agent, one writing pass. Compared by item counts/types and
coverage, then once on dev100 and fold-0 test with gpt-6-luna choose_card on every question.

## Generic memory check (written 2026-10-25 13:50 KST)
After t0/h0 (dev100: t0 72.5 > h0 70.0; h0's broad guard item took the single slot on 29 FPQ), the memory becomes generic:
one item file memory/items.md (no knowledge/guard split, agent-defined tags allowed), test-time choose_items picks up to
2 items (or none) with gpt-6-luna on every question. g_text = text fields only; g_hid = text fields + the four
hidden_state fields of h0 (same values). Same skill text apart from the fields paragraph, same base agent, one writing
pass each; evaluated once on dev100 and fold-0 test 242. Fold 0 is development.

## Before/after feedback comparison (written 2026-10-25 14:05 KST)
Same initial memory (t0 cards), same training rollout (t0 on 156 items: card choice on every question; 104 carded) for
four revision workspaces; one base-agent revision pass each (gpt-6-luna medium), then the revised memories are
evaluated with the same selector on dev100 and fold-0 test 242.
  T      card before/after answers, ratings, outcome, grader reason (text MCE feedback)
  T+P    T + Plain-state probes (p_fpq, pass_forecast)
  T+P+O  T+P + Oracle-lens correction plan without vs with the card (FRAME on blocks 52/60, read before the answer),
         one readout line each, and per-card counts of FPQ/NFP questions whose plan turned to correction with the card
  T+P+sO identical format and text, but the lens block is taken from another training question (seed 0 permutation)
The skill paragraph describing the extra fields is identical in T+P+O and T+P+sO. Fold 0 is development.
Replicates: each condition is revised twice by independent agent runs (rv_* and rv2_*, identical inputs), so that a
difference between conditions can be compared with the run-to-run spread. Same base prompt for all (rv_base_prompt.md).

## E3: card case sheets with tentative hypotheses (written 2026-10-25 14:25 KST)
Start: same as E1 (t0 cards, t0 training rollout). The harness writes data/card_sheets.md: per card, every training
question it was attached to (label, Plain vs card rating, outcome, grader reason excerpt) and rule-made TENTATIVE
revision hypotheses. E3-text: hypotheses from ratings and grader reasons only. E3-latent: the same plus the Oracle-lens
correction plan without/with the card per case and lens-based hypotheses (e.g. a normal question that still passed but
whose plan turned to correcting with the card = risk candidate; an FPQ that failed with an unchanged plan = card may not
register). For every card the base agent must record modify / keep / check-later with a one-line reason
(cards/decisions.md); changes are judged by actual ratings, never by lens signals. One card per question, same selector
(gpt-6-luna on every question), one revision pass, 2 replicates, evaluated on dev100 and fold-0 test 242.

## E7: before/after feedback without the grader's reason (written 2026-10-25 15:45 KST, before any E7 result)
Question: does the hidden-state block add more when the text feedback carries no LLM-written explanation (MCE style:
answer + gold + score)? Same start as E1 (t0 cards, t0 training rollout, 104 carded rows), same base agent, one revision
pass, same selector, evaluated on dev100 and fold-0 test 242 with Plain alongside.
  N      E1's T rows with judge_reason_for_card_answer removed and the full card answer added (card_answer); label,
         reference correction, Plain/card ratings and outcome unchanged. Prompt e7_base_prompt.md (one field line
         changed from rv_base_prompt.md).
  N+P+O  N + the probe and lens blocks and the SKILL.md of E1's T+P+O (byte-identical inputs).
2 replicates (e7a_*, e7b_*). Read-out: per role, mean over replicates of (N+P+O - N), compared with E1's mean
(T+P+O - T) over rv/rv2 and with the run-to-run spread of each condition. Hidden information "adds more without the
reason" only if the E7 difference is positive on both dev100 and test and larger than the E1 difference on both.
Also reported: N vs T (what the grader's reason is worth) and every arm vs Plain and GEPA (67.6 on test). Fold 0 is
development; no decision about the final method is taken on test alone.

## E8: E1 conditions revised with reasoning effort high (written 2026-10-25 15:50 KST, after E1 rv2 and E3 results)
Seen before writing: E1 rv/rv2 and E3 r1/r2 (dev100 and test). Two observations motivate E8: (1) the same condition
differs by up to 5.9 macro points between replicates; (2) the medium-effort agents ran 3-9 commands in about a minute
and printed the probe or lens fields in at most 1-2 commands, so the hidden fields were barely read.
E8 repeats T, T+P and T+P+O with identical inputs (E1's untouched data files and SKILL.md) and prompt, with the base
agent's reasoning effort high instead of medium, 2 replicates (e8a_*, e8b_*), same selector, dev100 and fold-0 test.
Read-out: per role, mean over replicates of (T+P - T) and (T+P+O - T), compared with the replicate spread; the number of
agent commands that read the hidden fields. Fold 0 is development; no decision about the final method on test alone.

## E8 amendment (written 2026-10-25 16:00 KST): stopped before evaluation
The codex rollouts (~/.codex/sessions) show that every tool output is cut to about 10,000 tokens: `cat data/train.json`
reached the model as about 42,000 of 212,000 characters (about 20 of 156 rows; the per-card summary at the end was
visible). The agents' own filtered scripts printed ratings and grader reasons but almost never the probe or lens fields.
This holds for E1, E3, E7 and E8 alike (E8 high effort: lens fields seen for the same ~20 rows). E8 was stopped while
the selector ran (no answers generated); its six revised workspaces are kept unevaluated. E7 continues as requested.

## E9: E1 conditions with every training row in the agent prompt (written 2026-10-25 16:00 KST, before any E9 result)
Same as E1 (T, T+P, T+P+O; E1's untouched inputs; medium effort; same selector; dev100 and fold-0 test 242, Plain
alongside) except the delivery: the agent prompt is rv_base_prompt.md followed by the full train.json (one row per line)
and feedback_summary.json, so all 156 rows and their probe/lens fields reach the model (smoke test: an agent given the
inline copy answered the last row id and a middle row's lens values correctly). 2 replicates (e9a_*, e9b_*).
Read-out: per role, mean over replicates of (T+P - T) and (T+P+O - T), against the replicate spread, and against E1.
Fold 0 is development; no decision about the final method on test alone.

## E10: optimization-history memory, one more iteration of E9 (written 2026-10-25 16:08 KST, before any E9 result)
Continues each of E9's six memories (T, T+P, T+P+O x 2 replicates) for one more revision. Training rollout of the E9
memory (card choice on every question, Oracle lens read on carded prompts), the condition's feedback rows (ba_next) and
data/history.md: per training question, previous version (t0 cards, t0 rollout) vs current (E9 memory) card and
rating; T+P+O additionally the lens correction plan without a card and with each version's card, typed both / score
only / plan only / neither (T and T+P: score changed / same). All files inlined in the prompt (e10_base_prompt.md: E9's
prompt with one history paragraph). Same selector; dev100 and fold-0 test 242, Plain alongside.
Read-out: per role, (E10 - its own E9 parent) per condition, mean over replicates, against the replicate spread; and
T+P+O vs T after E10. Fold 0 is development; no decision about the final method on test alone.

## E11: same diagnosis procedure, output-based vs hidden-state cause tags (written 2026-10-25 16:18 KST, before any E9-E11 result)
Start as E9 (t1 cards; E1's untouched rows, all files inlined in the prompt; medium effort). The harness writes
data/diagnosis.md listing the same 66 training questions in both arms (every failed question and every normal question
its card put at risk), grouped by card, each with one cause tag. D_T (on T rows): tags from ratings only (card-broke,
both-failed, at-risk, no-card-failed; FPQ weak-correction = rated 2-3, ignored-premise = rated 0-1). D_H (on T+P+O
rows): tags from the Plain-state probe and Oracle-lens plan (turned-on, already-on, at-risk, plan-off-failed; FPQ
knows-not-said = p_fpq >= 0.5, not-detected). SKILL.md gains the same three-step procedure in both arms (narrow cards
that broke normal questions; "knows" FPQ -> state the correction plainly; "missing" FPQ -> give the fact), naming the
arm's tags, and asks for cards/decisions.md. 2 replicates (e11a_*, e11b_*). Same selector; dev100 and fold-0 test 242.
Read-out: per role, mean over replicates of (D_H - D_T), against the replicate spread; also each vs E9's T and T+P+O.
Fold 0 is development; no decision about the final method on test alone.

## E12: E11 without hedged guidance (written 2026-10-25 16:52 KST, after E7 results, before any E9/E11 result)
Motivation: the guidance told the agent that the lens content is "often unrelated, use it as a hint" and to "confirm
every change against the ratings", so ratings dominated. E12 = E11 (same rows, same 66-question diagnosis, same
procedure) with: both arms, the procedure's last line "The ratings show what happened; the tags show why, so let the
tags decide which fix to make."; D_H only, the lens paragraph states the plan direction's reliability measured on the
training questions only (FRAME on the reading vs on the answer: 122/156 = 78% without a card, 76/104 = 73% with the t0
card) and "The ratings show what happened; these fields show why.", and the hedge sentence leaves the tag vocabulary.
2 replicates (e12a_*, e12b_*), evaluated after E11 on the same servers. Read-out: E12 D_H - E11 D_H and E12 D_H - E12 D_T
per role, against the replicate spread. Fold 0 is development; no decision about the final method on test alone.

## E12 amendment (written 2026-10-25 17:20 KST): stopped before evaluation
Delivery and wording variants are paused in favour of testing whether the hidden state measured WITH a context
separates which context helps a given question (next section). E12's four revised workspaces are kept unevaluated.

## E9 result note (written 2026-10-25 17:29 KST)
E9 (all rows inlined) test: T 68.8/68.1, T+P 69.3/68.9, T+P+O 67.3/69.1; dev100 T 64.4/68.8, T+P 63.8/68.8, T+P+O
67.5/61.2. No condition differs from T beyond the replicate spread; full delivery did not raise E1's level either.

## E13: does the hidden state WITH a context tell which context helps a training question? (written 2026-10-25 17:29 KST, before any E13 data)
Data: training questions only (156). Pairs (question, context) = no context (Plain) plus every distinct context the
question received in the training rollouts t0, v0, v1, v1x, v2x, r1x, r2x and E10's rollouts of the six E9 memories
(answers are greedy and cached by (system, question), so identical pairs are merged). Outcome: Luna rating of that
answer, pass = rating >= 4. Hidden state: layers 20-60 (step 4) at the last prompt token with the context attached
(latent_harness /hidden; first checked against the stored Plain states). Lens: FRAME on the Oracle readout with that
context, where it was read (READ_CARDED rollouts); comparisons use the pairs where every predictor's inputs exist.
Predictors of pass(question, context), all logistic regressions, evaluated with 5-fold cross-validation grouped by
question (all contexts of one question in the same fold):
  T      label, context = none, the context's net pass effect on the OTHER training questions (leave-question-out),
         TF-IDF similarity between question and context
  T+P    T + out-of-fold probe score from the with-context state (per-layer standardised logistic regression, mean of
         11 layers, C = 0.003, trained on the other folds' pairs) and its difference from the same question's Plain score
  T+P+O  T+P + lens plan with the context, plan without, and their interactions with the label
Primary metric: within-question pairwise accuracy (pairs of contexts of the same question with different pass
outcomes; share ordered correctly; 50% = chance). Secondary: AUC for "context fixes a Plain failure" and "context
breaks a Plain pass". The internal state helps only if T+P - T has a 95% interval (bootstrap over questions) above 0;
the lens adds only if T+P+O - T+P does. Only training data are used; no dev or test item enters E13.

## E14: probe gate on context use (written 2026-10-25 17:49 KST, after training-only estimates, before any dev/test computation)
Rule: after the selector picks a card, drop it (answer without context) when the Plain-state probe gives p_fpq < 0.3.
The threshold 0.3 is fixed from the training rollout of t0 (training macro 61.5 -> 65.4; 0.4 and 0.5 gave +3.2).
p_fpq comes from the train-only probe (items.json; cross-fit on training, fixed for dev/test). No new generation: a
dropped card means the stored Plain answer and rating are used. Applied unchanged to every memory already evaluated
on dev100 and fold-0 test (t0, E1, E3, E7, E9, E11 when available). Success: the mean gain over memories is positive
on dev100 and on test, and positive for t0 on both. Also reported: per memory, and the lens+probe veto (plan turned
on by the card and p_fpq < 0.5) once with-card lens reads exist for dev/test. This uses the hidden state at test time
(one probe read before answering); it is a separate claim from the optimization-feedback experiments.

## E14 result (written 2026-10-25 17:50 KST): criterion not met
Probe gate (drop the card when p_fpq < 0.3) over 23 evaluated memories: mean gain dev100 -2.66 (positive in 1 of 23),
test +0.20 (positive in 10 of 23); t0 dev100 72.5 -> 68.1, test 71.2 -> 73.7. The training estimate (+3.9) did not
carry over: dev100 is 80% FPQ, and dropping cards on low-p_fpq FPQs loses fixes. Not pursued further.

## E11 result (written 2026-10-25 18:27 KST)
test: D_T 69.6/68.8, D_H 71.9/72.4; paired over items and both replicates D_H - D_T = +2.96 (95% item bootstrap
-0.01 to +6.16). dev100: D_T 69.4/63.1, D_H 68.1/64.4; difference +0.00 (-4.38 to +4.69). Positive on test only;
not established. The revised card texts were as similar across arms as within an arm.

## E13 result (written 2026-10-25 18:27 KST): both pre-registered criteria met
640 (question, context) pairs on the 156 training questions (41 distinct contexts incl. none; t0 and the six E9
memories, all with lens reads); extracted Plain states match the stored ones (per-layer cosine >= 0.9997).
Within-question pairwise accuracy (358 comparable pairs; 50% = chance): T 48.9%, T+P 61.5%, T+P+O 69.6%.
T+P - T +12.6 points (95% bootstrap over questions +1.0 to +22.9); T+P+O - T+P +8.1 (+2.1 to +14.3).
AUC for a context breaking a Plain pass: T 0.35, T+P 0.58, T+P+O 0.73; for fixing a Plain failure: 0.59, 0.53, 0.66.

## E13 correction (written 2026-10-25 18:39 KST): the earlier result had an outcome leak
The T feature "net pass effect of the context on other questions" was computed from all questions, including the
held-out fold. Now it uses training-fold outcomes only (the probe was already nested correctly). The leaky output is
kept as results/data/v4/e13_leaky.json. Corrected (reproduces an independent audit exactly):
pairwise T 46.6%, T+P 55.6%, T+P+O 64.8%; T+P - T +8.9 (-4.4 to +21.3): criterion NOT met; T+P+O - T+P +9.2
(+2.5 to +15.7): criterion met. Top-scored context per question, mean pass: 57.7 / 60.9 / 64.7%.
Only 4 of 484 held-out carded pairs had a context absent from the training fold, so this shows prediction for already
observed contexts on other questions, not for new contexts. The lens feature is a correction-word detector on the
with-context reading: it shows that the reaction after attaching a context helps predict the outcome, not a cause.

## E10 result (written 2026-10-25 19:07 KST)
dev100: T 65.6/65.6, T+P 66.2/65.0, T+P+O 65.6/63.1. test: T 69.8/67.6, T+P 64.4/71.6, T+P+O 69.9/68.0. The history
memory gave no gain for any arm, and no arm improved on its E9 parent beyond the replicate spread.

## E15: choosing among candidate contexts by the with-context reaction, on held-out questions (written 2026-10-25 19:17 KST, before any E15 read)
Pool: for every dev100 and fold-0 test question, no context plus every distinct context that one of the card-format
memories evaluated so far attached to it (t0, rv_*, rv2_*, e3*, e7*, e9*, e10*, e11*); their answers and Luna ratings
already exist, so no answer is generated or judged. New reads per (question, context): the with-context hidden state
(latent_harness /hidden) and the Oracle-lens reading (serve27b_lens, "gen": false; reader settings unchanged).
Predictors are trained on the 640 E13 training pairs only and must not use the label (unknown at test time):
  T      no-context flag, the context's net pass effect on training questions (0 if the context never occurred in
         training), TF-IDF question-context similarity
  T+P    T + with-context pass-probe score and its difference from the same question's no-context score, + p_fpq
  T+P+O  T+P + lens plan with the context, plan without, and both times p_fpq
Rule: per question, answer with the candidate of highest predicted pass. Read-out: macro of the chosen answers on
dev100 first, then test, for T, T+P, T+P+O, against (a) the mean of the single memories, (b) t0, (c) a uniformly random
candidate (expected value), and (d) the best candidate per question (upper bound, uses outcomes; reported as such).
Success: T+P+O > T on dev100 and on test (95% interval over questions above 0 on at least one, positive on both).
This tests context choice at test time with a candidate pool; it is not yet an optimization-loop result.
E15 amendment (19:21 KST, before any analysis; only a 2-question read smoke test was run): a lens read costs about 4 s
per pair on one server, so the pool is restricted to the first replicate of each condition (t0, rv_T, rv_TP, rv_TPO,
e3t_r1, e3l_r1, e7a_N, e7a_NPO, e9a_T, e9a_TP, e9a_TPO, e10a_T, e10a_TP, e10a_TPO, e11a_D_T, e11a_D_H): 652 dev100 and
1702 test pairs. The single-memory comparison (a) uses these 16 memories.

## E15 result (written 2026-10-25 20:39 KST): criterion not met
Bug fixed before analysis: dev/test states of contexts that also occurred in training (and of no context) had been
skipped because the state file name did not include the role; re-extracted per role. Choice per question among the
candidates of the 16 memories (macro / FPQ / NFP):
  dev100 (752 candidates, 69 contexts unseen in training): T 71.25/67.5/75.0, T+P 67.5/65.0/70.0, T+P+O 67.5/65.0/70.0;
    single memories mean 67.34, t0 72.5, random candidate 67.0, best candidate (upper bound) 83.1;
    T+P+O - T -3.75 (95% bootstrap -10.7 to +1.8).
  test (1944 candidates, 84 unseen contexts): T 71.19, T+P 70.14, T+P+O 71.42; single memories mean 69.54, t0 71.22,
    random 69.76, upper bound 86.85; T+P+O - T +0.23 (-6.4 to +7.2).
The with-context reaction did not choose better contexts than the text features on held-out questions, where most
candidates came from revised memories whose contexts were not in the training pairs.

## E16 (B): do reactions predict the outcome of NEW contexts on the SAME training questions? (written 2026-10-25 20:45 KST, before any E16 computation)
Scenario: the outcomes of the old contexts (no context and t0's cards) are known on the 156 training questions; the
agent then writes new contexts (here: the cards of the six E9 memories that are not t0 cards, rolled out in E10 with
lens reads). Predictors are fit on the old pairs only and score the new pairs before their outcomes are used.
  T      label, no-context flag, TF-IDF question-context similarity (the new context's own effect is unknown)
  T+P    T + with-context pass-probe score (trained on old pairs) and its difference from the question's no-context score
  T+P+O  T+P + lens plan with the context, plan without, interactions with the label
Metrics on the new pairs: AUC for "breaks a Plain pass" (NFP and FPQ separately reported), AUC for "fixes a Plain
failure", within-question pairwise accuracy among {no context, new contexts}; 95% bootstrap over questions.
Decision: if T+P+O - T on the NFP-break AUC has a 95% interval above 0, the internal-reaction memory is built on top;
otherwise Cancer-Myth C3 is reported as negative and work moves to MedCalc.

## E16 result (written 2026-10-25 20:47 KST): criterion met
260 old pairs (no context + t0 cards), 362 new pairs (25 contexts of the E9 memories absent from t0) on the same
156 training questions; 149 NFP-break cases (32 broken), 150 fix cases (55 fixed). AUC on the new pairs:
  NFP break: T 0.31, T+P 0.73, T+P+O 0.74; T+P+O - T +0.44 (95% +0.23 to +0.60)  -> pre-registered criterion met.
  any break 0.29 / 0.70 / 0.72; fix 0.54 / 0.74 / 0.75; within-question pairwise 44.5 / 71.5 / 73.5%.
Post-hoc check (added after seeing the result, not pre-registered): a stronger output baseline that also knows the
question's Plain rating and the old card's rating (Tout): NFP break 0.43 -> 0.74 with P+O (+0.31, +0.10 to +0.49),
any break 0.53 -> 0.72 (+0.19, +0.05 to +0.33), fix 0.63 -> 0.74 (+0.12, +0.04 to +0.20).
Scope: the new contexts are revisions of t0 cards on the same questions (the training-feedback setting); E15 showed
the signal does not carry to new questions.

## E17: two-pass revision with a pre-rollout check from internal reactions vs from outputs (written 2026-10-25 21:07 KST, before any E17 run)
Start: t1 (t0 cards, t0 training rollout). Selector everywhere: Qwen3.6-27B itself (vLLM, number-only reply, greedy).
Pass 1 (same for both arms): MCE-format rows inlined (question, label, reference correction, card, Plain and card
ratings, outcome, first 800 characters of the card answer; full answers in data/card_answers.jsonl), no judge reason;
3 independent drafts d1-d3 (gpt-6-luna medium).
Check (no answers generated): the Qwen selector assigns each draft's cards on the 156 training questions; the
with-context states and Oracle-lens readings of those pairs are read on the HF servers. Two predictors of pass are fit
on all earlier training pairs with outcomes (no context, t0, the six E9 memories): TXT = outputs only (label, Plain
rating, the t0 card's rating on the question, no-context flag, TF-IDF similarity); HID = TXT + with-context probe score
and its change from no context + lens plan with/without the card and their label interactions. Report per card: NFP
questions that pass without context but are predicted to fail, FPQ questions predicted to stay failing, FPQ predicted
to be fixed, with probabilities; HID additionally shows the observed reaction (plan without -> with, probe change).
Pass 2: each draft branches into TXT and HID with the same prompt (e17_pass2_prompt.md) and its arm's report; the agent
revises and records per-card decisions. 6 second-pass memories.
Evaluation (all with the same vLLM generation and Luna judge; not comparable with the earlier HF numbers): Plain, t0,
d1-d3, and the six pass-2 memories on dev100 and fold-0 test.
Read-out: per role, mean over drafts of (HID2 - TXT2) and of NFP pass in HID2 vs TXT2; each pass 2 vs its draft.
Expected: a small gain on NFP for HID2 (1-3 macro points), FPQ about equal; with 3 drafts the spread may cover it.
Success: HID2 - TXT2 > 0 on dev100 and on test (mean over drafts).
If not: (1) check whether the agents acted on the flags (decisions.md, changed flagged cards); (2) acted but no gain ->
use the predictor to screen candidates (rollout budget) instead of as feedback; (3) did not act -> force the
per-flag decision; (4) none works -> report Cancer-Myth C3 as negative and move to MedCalc.

## E16 correction (written 2026-10-25 21:12 KST): the post-hoc output baseline was invalid
The learned output baseline (Tout) could not learn "new card vs old card rating": in its fit set the old card's rating
was either the pair's own outcome (first version, a leak) or absent. Using the t0 card's rating on the question
directly as the risk score gives NFP-break AUC 0.69 on the 95 cases with a t0 card (an independent audit: 0.718 vs
0.739 with internal signals added, interval including 0). So internal signals are NOT shown to beat what the outputs
already say about NFP breaks; the pre-registered comparison against T stands but T is a weak baseline. E17 tests the
hypothesis; its TXT predictor is fit on t0 + E9 pairs, where the old card's rating is a learnable feature for new cards.

## E18: internal-observation experience memory in the MCE loop (written 2026-10-25 21:56 KST, before any E18 run)
Definition: in the same MCE optimisation procedure, the base agent's output-experience memory is extended with Qwen's
internal observations and their change history. Arms: OUT (output memory) and INT (output memory + internal fields).
Both: start from an empty memory (iteration 1 = Plain records), free context format (get_context may return any text or
nothing; LLM calls inside it go to the answering model), Qwen3.6-27B selects/answers (vLLM), Luna Well judge scores,
80 training items per iteration (40 FPQ / 40 NFP, same for both arms, new seeded sample each iteration), dev100 for
selection, fold-0 test at the end. Records per question x actual context x version: context excerpt, change, rating,
answer preview (800 chars), answer behaviour (fixed Luna summary, both arms). INT adds: probe p_fpq (Plain) and pass
score with the context, Oracle-lens reading (raw) and its fixed Luna interpretation (plan, correction plan, target,
change vs no context / previous version, quotes), answer-vs-reading relation. Readings are HF re-runs of the vLLM
input. Views given to the agent: per-question timeline, per-context summary, retrieved past cases; each item tagged
observed / retrieved / unknown. Same SKILL procedure in both arms; INT's SKILL only adds the field definitions. Meta
agent (iteration >= 2) sees only dev scores and earlier SKILLs, run in an isolated folder.
Pilot (this entry): iteration 1 only, one run per arm, no meta. Checks, in order: engine (token ids, behaviour
agreement, cache keys), 30 interpretation samples shown to the user, delivery (records keyed to the same input and
score, prompt not cut), use (agent cites internal fields; edits differ between arms), dev100 (reference only).
Main run (after the pilot): 3 iterations x 2 arms x 2 replicates with the meta agent. Success: INT's dev-selected
memory beats OUT's on test by >= 1 point with INT's edit records showing decisions that cite internal fields. This tests
whether adding internal information on the base-agent side improves the whole loop; effects via the meta agent's
SKILL (through different dev scores) are not separated.
E18 amendment (22:04 KST, before any E18 reading): the Oracle lens reads four blocks, 36, 44, 52, 60 (chosen in advance,
evenly spaced inside the reader's trained range 20-60), instead of 52 and 60; adapter, reader prompt (concepts_raw),
scale, sampling and length unchanged. Four-block readings are stored apart (lens4_*.jsonl, own server cache) and never
mixed with the earlier two-block readings; iteration 1 re-reads the Plain prompts with four blocks.
E18 pilot note (22:27 KST): iteration 1 delivered the records (OUT 50k, INT 60k characters inlined) but both agents wrote
one general instruction attached to every question and cited only question ids (no internal field); dev100 OUT 64.4
(FPQ 68.8 / NFP 60.0), INT 64.4 (63.7 / 65.0), Plain (vLLM) 59.4. Pre-agreed response to "delivered but not used": one
procedure change, identical in both arms - step 5 asks for one line per listed question with its change (or no change)
and the memory fields relied on. Applied from iteration 2 of the pilot.
E18 pilot iteration 2 (22:46 KST): with the per-question decision lines, INT cited internal fields in 45 of 48 lines;
dev100 OUT 61.3 (FPQ 62.5 / NFP 60.0), INT 68.1 (56.2 / 80.0); one run each, not a result.

## E19: sentence ablation of one context - do internal changes point in the direction of the real score change? (written 2026-10-25 22:46 KST, before any E19 read)
Context: the pilot OUT run's version-1 instruction (attached to every question), split into sentences; variants = full
and each single-sentence removal. Items: the 80 training questions of pilot iteration 2. Per variant x question:
A = Oracle-lens reading (four blocks) interpreted by the fixed Luna prompt: correction_plan yes rate; A' = unclear or
other-content rate (reported, not merged); B = FPQ-discrimination probe on the with-context state (probe fit on Plain
states of training items, 5-fold cross-fit by question); C = pass-prediction probe on the with-context state;
D = real outcome: vLLM answer, Luna Well rating (pass >= 4), answer behaviour. Removal effect = variant - full, per
label (FPQ / NFP). Read-out: across the sentences, the rank correlation of each of A, B, C with D (NFP pass change and
FPQ pass change); across (sentence, question) pairs, whether the sign of A / B / C change separates outcome flips
(AUC). Wording: a sentence is a "revision candidate", never a "cause". Claim tested: internal reactions can narrow
revision candidates before all answers are generated; cost of A-C vs D is timed. Not a test of information absent
from outputs.

## E19 result (written 2026-10-25 23:42 KST)
9 sentences x 10 variants x 80 training questions. Full context: NFP pass 47.5, FPQ 62.5. Largest removal effect: S2
(correct an assumed inevitable outcome) - lens correction_plan -0.40 NFP / -0.38 FPQ, real pass +0.20 NFP / -0.15 FPQ
(a non-selective trade-off sentence). Rank correlation over the 9 sentences with the real pass change: NFP A -0.50,
B -0.45, C +0.47; FPQ A +0.40, B +0.59, C +0.17 (directions as expected; n = 9). Per (sentence, question) flips:
AUC 0.47-0.64 (outcome noise of ~8% per item from generation and judging is mixed in). Cost: states + probes 6 min,
four-block lens reads 40 min on two servers, real generation + judging 44 min: with four blocks the lens is not cheaper;
the probe is. Side checks the same evening: vLLM batch-invariant mode (TRITON_ATTN) was not deterministic for this model
(same concurrency 30/40 identical, different concurrency 2/40) and 2-6x slower - not used; Luna re-judging the same
answer keeps pass/fail 96.8% (84.4% on items near the threshold); the Plain pass probe does not flag flip-prone items
(AUC 0.41).

## E18 main result (written 2026-10-26 00:58 KST): criterion not met
3 iterations x {OUT, INT} x 2 replicates, 66 min wall time. Plain on fold-0 test (vLLM, same judge): 64.1 (FPQ 48.7 /
NFP 79.6; 193 FPQ, 49 NFP, so 1 NFP item = 1.0 macro point). Dev100 macro by version v0..v3 (v0 = Plain 59.4):
OUT_a 63.7 / 70.6 / 64.4, OUT_b 60.6 / 53.1 / 59.4, INT_a 64.4 / 59.4 / 59.4, INT_b 63.1 / 66.9 / 63.7.
Test of the dev-selected version: OUT_a v2 69.1 (62.7 / 75.5), OUT_b v1 59.8 (80.8 / 38.8), INT_a v1 66.3 (65.3 / 67.3),
INT_b v2 63.7 (60.1 / 67.3). Arm means OUT 64.5, INT 65.0 (+0.5 < 1 point): criterion not met; the replicate spread
inside an arm (9.3 and 2.6 points) is larger than the arm difference. Use: internal fields cited in decision lines
INT_a 35/39 then 51/62, INT_b 3/36 then 3/59, OUT 0.
Observations (not pre-registered): (1) question-specific memorisation in both arms - INT_a v2/v3 and OUT_b v3 wrote
regular-expression triggers keyed to single training questions; none matched a dev100 question (context on 0/100, dev =
Plain); the per-question decision line added after the pilot pushes the agent toward this. (2) INT_a cited internal
fields mostly next to the output that already showed the same thing ("rating 1/complies, correction_plan no"); no
decision line chose a different change because of an internal field. (3) INT_b v2 wrote a Qwen classifier inside
get_context (context on 53% of test FPQ, 29% of test NFP). (4) In the INT records, for questions answered again in the
next version (78 FPQ, 78 NFP pairs), grouped-CV AUC for passing next time: FPQ output 0.58, internal 0.65, both 0.60;
NFP output 0.78, internal 0.79, both 0.81 (n small; same direction as E13/E16, not significant). (5) The meta agent saw
only dev scores, so it could not see the train-up / dev-down pattern of memorising versions.

## E20: context-level internal feedback (sentence table) + loop fixes for both arms (written 2026-10-26 00:59 KST, before any E20 run)
Question: does internal information help the loop when it is given per context component (sentence) instead of per
question? Arms: OUT2 (output memory, as E18 OUT) and TAB (OUT2 + data/sentence_table.md). No per-question internal
fields in either arm. Sentence table (TAB only, from iteration 2): for each distinct context of the current version
attached to >= 3 of the iteration's 80 training questions, every single-sentence removal; with-context states
(latent_harness) read by B = FPQ-discrimination probe (Plain states, 5-fold cross-fit by question) and C = pass probe;
per sentence the mean change (removal - full) on NFP and FPQ and the 3 largest per-question C changes. No generation,
no Oracle lens. Fixes identical in both arms: (a) SKILL step 3-5: groups are classes of questions; rules keyed to the
wording of single training questions are not allowed; decisions.md has one line per group and, when an earlier
context is kept or edited, one line per earlier sentence (keep / change / remove, evidence). (b) Meta agent sees each
version's training score (the 80 items it answered) next to its dev100 score, with the MCE overfitting check.
Everything else as E18: empty start, vLLM Qwen3.6-27B answers, Luna Well judge, seeds 1000+k, dev100 selection, fold-0
test. Runs OUT2_a, TAB_a, OUT2_b, TAB_b (folders versions/e18_<run>, since the E18 loop code is reused).
Read-outs: (1) memorisation check - every version attaches context on dev100 (> 0) in both arms; (2) use - TAB
decision lines cite the table; (3) success - TAB's test mean >= OUT2's + 1 point; also reported against Plain test 64.1
and E18. With 2 replicates and E18's 9-point replicate spread, a 1-point difference is not separable from agent
variance: this is a mechanism check, not a confirmatory test.

## E20 result (written 2026-10-26 01:53 KST): criterion not met; the table was barely delivered
Wall time 53 min (01:00-01:53 KST). Dev100 macro v1 / v2 / v3 (v0 59.4), context on dev100 at v2 / v3:
OUT2_a 69.4 / 66.2 / 60.6 (100 / 100), OUT2_b 68.8 / 70.6 / 64.4 (100 / 100), TAB_a 64.4 / 59.4 / 59.4 (0 / 0),
TAB_b 64.4 / 60.6 / 61.9 (8 / 23). Test of the dev-selected version (Plain 64.1): OUT2_a v1 67.6 (69.9 / 65.3),
OUT2_b v2 68.8 (56.0 / 81.6), TAB_a v1 66.1 (68.9 / 63.3), TAB_b v1 59.0 (75.1 / 42.9). Means OUT2 68.2, TAB 62.6.
Read-outs: (1) memorisation check failed in TAB_a (v2: 8, v3: 15 regular expressions named as classes but each fit to
one training question; context on 0 dev questions); passed in both OUT2 runs. (2) Use: the table was delivered once
per TAB run (iteration 2); TAB_a cited it twice, calling probe changes "diagnostic only, mixed"; TAB_b did not cite
it. In iteration 3 neither TAB run had a context on >= 3 training questions (TAB_a regexes, TAB_b a Qwen gate that
writes a per-question correction), so no table was made. (3) Success criterion not met; moreover both TAB runs selected
v1, written before any table, so the test difference is iteration-1 agent variance (at iteration 1 the inputs differ only by
TAB's table-definition paragraph and 2-4% of prompt lines, the wording of
the behaviour summaries, which differs as much between the two OUT2 runs: test 67.6, 66.1, 59.0 for the three v1 tested), not a table effect.
Observations (not pre-registered): both TAB meta agents downgraded the table in the SKILL ("diagnostic clues, not
proof that a rule works"; "noisy hypotheses, not measured pass rates"), and both TAB runs moved to narrow attachment
after their v1 lost NFP on dev; "prefer the smallest change" also appears in OUT2_a's meta SKILL, so it is not
TAB-specific. The OUT2 runs kept broad context. OUT2 is the first arm in E18/E20 whose two replicates both beat Plain
on test (+3.5, +4.7); with two replicates this is not separable from chance. Design flaw found: a per-sentence table
needs a context shared by several questions, so it disappears when the agent writes per-question contexts.

## E21: failure-type counts with a probe-based internal split (written 2026-10-26 03:21 KST, before any E21 run)
Question: does internal information help when it is given as aggregated failure types with their measured reliability,
in a form that exists for any context format? Arms: OUT3 and FTY, 3 replicates each (a, b, c), E20 procedure and
train-aware meta agent in both. After each rollout both arms get data/failure_types.md: per context group (a context
shared by >= 3 of the iteration's 80 training questions; all other contexts pooled; no context), FPQ and NFP counts,
and the failures by answer behaviour with question ids. FTY adds, for the FPQ failures, the split by the with-context
pass probe (no generation): possibly_silent (score >= 0.3) vs likely_unaware (< 0.3), with ids, and one reliability
line measured before this entry on the E18 INT training records only: of 139 FPQ failures, Oracle-lens reading
"plan to correct the target premise" in 37% of the >= 0.3 group (14/38) vs 6% below (6/101); AUC 0.83 (cutoff 0.3 fixed
from these records, Youden 0.286). FTY's SKILL adds a definition: possibly_silent = the model may hold the correction
but not say it (an instruction to state it may help); likely_unaware = no internal plan (knowledge or attention may be
missing). Both arms: after the base agent, if get_context attaches context to 0 of the 100 dev100 questions (texts
only, no labels or scores), the agent is asked once to rewrite it with class-level rules. Everything else as E20
(empty start, 3 iterations, seeds 1000+k, vLLM Qwen3.6-27B, Luna Well judge, dev100 selection, fold-0 test).
Servers: vLLM 8802 (GPUs 0,1), hidden states 8798 (GPUs 2,3). Read-outs: (1) use - FTY decision lines cite the split;
(2) memorisation - versions with 0 dev100 coverage after the check; (3) success - FTY test mean >= OUT3 + 1 point and
FTY's selected versions written after at least one failure-type file (iteration >= 1, so always true here). With 3
replicates and ~9-point agent spread this is still a mechanism check.
E21 amendment (03:24 KST, after a smoke test of the failure-type step on a copy of E20 OUT2_a iteration 3 only,
before any E21 run): the with-context pass probe shifts with the context (17 of 18 FPQ failures >= 0.3 under one broad
context), so the fixed 0.3 cutoff does not split. The split is now within each context group: FPQ failures above the
group's median probe score = possibly_silent, at or below = likely_unaware (no split for groups with < 3 FPQ failures).
Reliability re-measured on the same E18 INT training records with this rule: lens plan to correct the actual premise in
22% above the median (15/67) vs 4% at or below (3/70); within-group AUC 0.66-0.85 (6 groups). Nothing else changes.
E21 amendment (03:30 KST, user-approved, before any E21 result): two more replicates per arm (d, e) run in
parallel on a second vLLM server (port 8803, GPUs 4,5), same code and settings (run_e21_extra.sh); the read-outs use all
5 replicates per arm.

## E21 result (written 2026-10-26 04:40 KST): criterion not met
5 replicates per arm (a-c on vLLM 8802, d-e on 8803). Test of the dev-selected version (Plain 64.1):
OUT3 a v2 67.7, b v2 63.9, c v2 64.3, d v3 64.8, e v1 66.9 - mean 65.5 (SD 1.7); FTY a v2 67.6, b v1 66.0, c v0 64.1,
d v3 64.2, e v1 65.1 - mean 65.4 (SD 1.5). FTY - OUT3 = -0.1. Read-outs: (1) use - all FTY decision files cite the split
from iteration 1, but no decision chose a different kind of fix because of it; (2) memorisation - no version needed
the coverage rewrite (0 attempts 2); one FTY_d version wrote one regular expression; (3) criterion not met.
Observations: the within-group median split always gives half / half, so it cannot say which group is mostly
"silent" (design flaw found after iteration 1); after iteration 2, 4 of 5 FTY runs narrowed attachment (dev coverage
6-34) vs 1 of 5 OUT3, mostly following their own v1 dev NFP drop (35-40) rather than the split; the dev100 NFP set
(20 items) drives most of the within-arm spread (one FTY v1 pair differed by 5 NFP items, all invented corrections).
Side checks the same night (offline, E18 records): lens readings track the answer (FPQ passed: correction words in
48-62% of block readings, failed: 6-17%), 12% of FPQ failures show an early-block (36/44) correction lost at block 60;
state-keyed retrieval beats text-keyed retrieval (FPQ AUC 0.81 vs 0.79, NFP 0.79 vs 0.75) but not the output key
"same Plain outcome" (0.87, 0.80), and state ordering within that key equals random.

## E22: does the lens failure type change which fix works? (2 x 3, training only; written 2026-10-26 04:48 KST, before any E22 generation)
Items: every FPQ failure in the E18 INT_a / INT_b records (unique question x actual context), with the context rebuilt
from the version snapshot and checked against the stored context id (mismatches dropped). Type from the stored reading
(fixed Luna interpretation of the four-block lens reading): silent = correction_plan yes and target target_premise;
unaware = correction_plan no; other = plan yes with another target (reported apart). Fixes appended to the same
context: (a) say-it: "If the question rests on an assumption you believe is false, say so clearly in your first
sentences and explain briefly, then answer the request."; (b) knowledge: "Relevant medical fact for this question: "
+ the item's reference correction; (c) placebo: "Answer the patient's question helpfully and clearly." Each pair is
answered by vLLM Qwen3.6-27B (greedy) and rated by the Luna Well judge (pass >= 4). Read-out: pass rate per type x fix;
decision value = (a - b) in silent minus (a - b) in unaware, 2,000 question-bootstrap 95% CI; claimed only if the CI
excludes 0. Secondary: the same with the block split (36/44 correction words but not block 60). Expected: silent
gains more from (a) relative to (b) than unaware. (b) uses the reference, so it is an upper bound for knowledge, not
a deployable fix. Training questions only.

## E23: does an internal change between versions announce the next outcome when the score is flat? (written 04:48 KST, before any E23 generation)
Chain: the four versions v0-v3 of E20 OUT2_b (contexts attached to every question). All 156 training questions are
answered with each version (vLLM, Luna judge) and read with the four-block lens (fixed settings) and the fixed Luna
interpretation. Triples (v_{k-1}, v_k, v_{k+1}) of the same question with the same pass/fail at k-1 and k. Predictors
of a pass/fail flip at k+1: (1) change of the lens correction_plan (no -> yes or yes -> no) and of the block-wise
correction-word count; (2) change of the 1-5 rating below the pass line (e.g. 2 -> 3); (3) both. AUC per label with
question-grouped CV. Read-out: (1) beats (2) in AUC; reported with 2,000 question-bootstrap CIs. Generation noise (8%
flips on re-generation) is part of the outcome.

## E22 result (written 2026-10-26 04:58 KST): criterion met
81 unique FPQ failures (question x context; 0 context mismatches in the full run): silent 15, unaware 53, other 13.
Pass rate after the fix (%): silent - say-it 46.7, knowledge 66.7, placebo 20.0; unaware - say-it 20.8, knowledge
98.1, placebo 9.4; other - 53.8, 100.0, 7.7. Decision value ((say-it - knowledge) in silent minus in unaware):
+57.4 points, 95% question-bootstrap CI +24.4 to +83.0 -> pre-registered criterion met. Secondary block split
(correction words at block 36/44 but not 60, n = 9): say-it 77.8 vs 25.0 for the other 72; decision value +44.4
(+12.7 to +72.8). Reading: for unaware failures the generic say-it instruction barely beats placebo (+11 points) and the
missing fact fixes almost all; for silent failures say-it recovers about half (+27 over placebo) and the reference fact
fixes only two thirds. The lens type therefore separates failures that need knowledge from failures that need the
model to voice what it already holds. Limits: n = 15 silent; knowledge uses the item's reference (upper bound); types
come from stored E18 readings; training questions only.

## E24: lens-routed failure types in the MCE loop (written 2026-10-26 05:05 KST, before any E24 run)
Question: when the training FPQ failures are split by the lens into "correction intent" (reading plans a correction:
silent or other target) and "unaware" (no plan), does the base agent write a better context (instructions for the
first, knowledge entries for the second) than with the same failures unsplit? Arms OUT4 and LNS, 3 replicates each,
E20 procedure + train-aware meta (as E21). Both SKILLs add one sentence: the context may combine instructions and
knowledge entries (general facts that correct a kind of misconception), and get_context may select the entries that
fit a question. Both arms get data/failure_types.md with failure counts by answer behaviour and one experience line
from E22 (training only): OUT4 the pooled line ("of 81 earlier FPQ failures, an instruction to state the correction
fixed 31%, the missing fact 93%, a neutral sentence 11%"); LNS the split line ("correction intent: instruction 50%,
fact 82%, neutral 14%; unaware: instruction 21%, fact 98%, neutral 9%") and, per context group, the FPQ failures split
into intent / unaware with ids (four-block lens reading of the actual input, fixed Luna interpretation; reads only for
failed FPQs). Test-time: no lens. Read-outs: (1) LNS writes knowledge entries aimed at its unaware failures; (2) test
FPQ and macro, LNS vs OUT4 (success: LNS mean >= OUT4 + 1 point); (3) NFP kept. The intent / unaware grouping was
chosen after E22 (post hoc) and is fixed here before any E24 data.
E24 amendment (05:12 KST, user-approved GPUs 4-7, before any E24 data): 5 replicates per arm (a-e), generation on
both vLLM servers (8802, 8803), lens reads on a dedicated server (8797, GPUs 6,7).

## E23 result (written 2026-10-26 05:56 KST): criterion not met
Chain E20 OUT2_b v0-v3 on all 156 training questions. Flat triples: FPQ 102 (2 flips at k+1), NFP 82 (13 flips).
AUC (question-bootstrap 95% CI) for a flip: FPQ lens plan change 0.49 (0.46-0.52), lens correction-word change 0.65
(0.47-0.84), rating change below the pass line 0.53 (0.49-0.56); NFP 0.50 (0.35-0.64), 0.34 (0.16-0.54), 0.52
(0.43-0.61). No predictor separates flips; the internal change is not a lead signal here. Limits: v1-v3 of this chain
are similar broad instructions, so FPQ outcomes barely moved (2 flips); version-level aggregates were not tested.

## E24 result (written 2026-10-26 06:13 KST): criterion not met
Test of the dev-selected version (Plain 64.1): OUT4 a v1 65.1, b v2 67.0, c v1 65.9, d v2 64.1, e v3 65.8 - mean 65.6 (SD
1.1; FPQ 63.8, NFP 67.3); LNS a v3 65.2, b v2 65.2, c v3 65.4, d v3 64.4, e v3 66.5 - mean 65.3 (SD 0.8; FPQ 52.3, NFP
78.4). LNS - OUT4 = -0.2. Read-outs: (1) no LNS version wrote knowledge entries (OUT4_c v1 did, a keyword-selected fact
table); LNS cited the split in 1-3 decision lines per version but kept writing instructions; (2) not met; (3) LNS kept
NFP (78.4 vs 67.3) and narrowed attachment (dev coverage 1-29 at v2/v3) while OUT4's selected broad v1 versions raised
FPQ and lost NFP. The dev100 coverage rewrite was triggered 4 times (2 per arm). Interpretation: the failure type that
E22 shows to decide the right fix does not change what the base agent writes when the decision is left to the agent.

## E25: test-time choice of where to attach a learned context (written 2026-10-26 06:51 KST, before any E25 computation; user-approved test-time use)
Contexts: every dev-selected version of E18-E24 whose get_context attaches the same text to all 242 fold-0 test
questions (fixed list taken from the ctx_test files before any gated score is computed). For each, the stored answers
with the context and the stored Plain answers (vLLM, same Luna Well judge) give the gated score without new
generation: a gated question gets its with-context answer, the others their Plain answer. Gates: all (the run as
evaluated), none (Plain), probe (train-only FPQ-discrimination probe on the question's Plain state, p_fpq >= 0.5),
text (TF-IDF logistic regression on the 156 training questions and labels, >= 0.5), Qwen (the answering model asked
"Does this question rest on a false or mistaken assumption? Answer yes or no." greedy; yes = attach), label (FPQ only;
upper bound). Read-out: mean test macro over the contexts per gate; primary comparison probe vs text and probe vs all,
paired by context, with a 2,000 question-bootstrap CI of the mean difference; also FPQ / NFP. Gate AUCs on test.

## E25 result (written 2026-10-26 06:55 KST)
14 contexts (one text on all 242 test questions). Gate AUC on test: probe 0.728, text 0.723, Qwen 0.602. Attach rate
FPQ / NFP: probe 54.9 / 20.4%, text 50.8 / 20.4%, Qwen 36.8 / 16.3%. Mean test macro (FPQ / NFP) over the contexts: all
65.6 (66.9 / 64.3), none 64.1 (48.7 / 79.6), probe 68.8 (59.5 / 78.0), text 66.3 (56.7 / 75.8), Qwen 65.3 (53.5 / 77.1),
label 73.2 (66.9 / 79.6). Paired differences (question bootstrap 95% CI): probe - text +2.5 (-0.6 to +5.9), probe - all
+3.3 (-1.8 to +8.6), probe - Qwen +3.5 (-0.2 to +7.7). Probe beats text in 14 of 14 contexts and all in 11 of 14, but no
CI excludes 0 (the contexts share the same test questions, so the 14 wins are not independent). The probe gate recovers
about half of the label gate's gain over Plain (4.7 of 9.1 points), mainly by keeping NFP (78.0 vs 64.3 for all).

## E26: folds 1 and 2 (written 2026-10-26 07:16 KST, before any fold-1/2 run)
Same pipeline on Cancer-Myth folds 1 and 2 (frozen v4 splits of latent_c1c2_audit_20261024): items per fold built as
in v4_features (train / dev / test, dev100 = dev + random unused items to 100, seed 0; p_fpq from the fold's training
items only), results under results/fold<k>/ (RESULTS_SUBDIR). Per fold 3 replicates of the E21 OUT3 procedure (E20
procedure, failure counts, train-aware meta, dev100 coverage rewrite), 3 iterations, dev-selected version on the
fold's test (242 questions). Gates as E25 on each run's dev-best broad version (one text on all test questions):
all, none, probe (p_fpq >= 0.5), text (TF-IDF LR on the fold's training items), Qwen, label. Read-out: per fold and
pooled with fold 0 (E21 OUT3 for the pipeline, E25 for the gates): test macro of the selected version vs Plain, and
gated macro probe vs text vs all. vLLM 8802/8803 (fold 1), 8804/8805 (fold 2). A symlink results/src -> ../src lets
e18_loop find v4_contexts.py under RESULTS_SUBDIR (no code change).
E26 note: a version counts as broad when its get_context returns one non-empty text for every training question (decided without test information).
E26 note: test items without a stored Plain state dropped: fold1 fpq_398, fold2 fpq_751, fpq_852 (fold1 240, fold2 239 test questions).

## E27: everything latent in one arm (written 2026-10-26 08:10 KST, before any E27 run)
Arms on fold 0, 5 replicates each, 3 iterations, E21 OUT3 procedure as the base (E20 SKILL, failure counts by answer
behaviour, train-aware meta, dev100 coverage rewrite). OUT5 = that procedure unchanged. LAT = the same plus, all from
the answering model's internal state: (1) gate - the context returned by get_context is attached only when the
question's train-only FPQ probe score p_fpq >= 0.5, in training rollouts, dev100 and test alike (the SKILL says so);
(2) base agent - the FPQ failures of each context group split by the four-block lens (correction intent / unaware, as
E24) with the E22 experience line, and, for questions also answered in the previous iteration, the change of type
(e.g. unaware -> intent); (3) meta agent - per earlier version, next to the training and dev100 scores: the number of
gated questions, and among FPQ failures the intent / unaware counts and the type transitions from the previous
version. Read-out: test macro of the dev-selected version, LAT vs OUT5 (success: LAT mean >= OUT5 + 1 point), with FPQ
/ NFP; secondary: LAT vs the E25 gate-only number on similar contexts. LAT bundles gate and observations, so a win is
not attributed to one part. Servers: vLLM 8802 + 8803, lens 8796 + 8797 (failed FPQs of LAT only).
E26 amendment (08:12 KST, before any gate result): rebuilding contexts of versions whose get_context calls Qwen per
question took up to 20 min per version, so a version is now a broad candidate only if it attached context to all 100
dev100 questions and its get_context calls no model; it must still return one non-empty text for every training
question. Everything else unchanged.

## E26 result (written 2026-10-26 08:20 KST)
Pipeline alone (dev-selected version, 3 runs per fold): fold 1 62.2 / 59.8 / 63.7 (mean 61.9; Plain 63.7), fold 2 59.0 /
64.4 / 63.0 (mean 62.1; Plain 65.0); with fold 0 (E21 OUT3, Plain 64.1, mean 65.5): +1.4 / -1.8 / -2.9 vs Plain.
Gates on broad contexts (fold 1: OUT3_a/b/c v1; fold 2: OUT3_a/b v1), test macro all / none / probe / text / Qwen /
label: fold 1 61.0 / 63.7 / 64.7 / 64.2 / 63.5 / 74.6; fold 2 55.8 / 65.0 / 65.6 / 66.2 / 61.9 / 75.1; with fold 0 (E25)
the three-fold means are 60.8 / 64.3 / 66.4 / 65.6 / 63.6 / 74.3. Probe - all +3.2 / +3.7 / +9.8 (consistent); probe -
text +2.5 / +0.5 / -0.6 (not consistent); probe - Qwen +3.5 / +1.2 / +3.7 (consistent). Choosing where to attach is
what turns the learned contexts into a gain; the hidden-state probe beats the model's own yes/no check but not a text
classifier trained on the same labels.

## E28: is a "why not voiced" summary of the lens reading stable? (written 2026-10-26 08:35 KST, before any E28 call)
Cases: the 28 E22 FPQ failures with correction intent (silent + other target). Input to Luna (fixed prompt, no answer
shown): the question, the reference correction and the stored four-block lens reading of that input. Output: one of
respects_patient_choice / defers_to_clinician_or_plan / focuses_on_requested_task / hedges_with_caveat /
targets_other_claim / unclear, with a quote. Two independent, uncached calls per case. Read-out: agreement of the two
calls (and Cohen's kappa); usable for a latent-analyst agent only if agreement >= 80%. Exploratory: E22 say-it fix rate
by category.

## E28 result (written 2026-10-26 08:35 KST): stable (criterion met)
28 cases, two uncached calls: agreement 85.7%, Cohen's kappa 0.78. First-call categories: targets_other_claim 15,
focuses_on_requested_task 6, hedges_with_caveat 3, respects_patient_choice 2, defers_to_clinician_or_plan 2. E22
say-it fix rate by category (exploratory, small n): 33 / 50 / 67 / 100 / 100%. Most correction-intent failures aim the
correction at another claim; those need the actual premise named rather than a say-it instruction.

## E28b: Sol vs Luna for the lens "why not voiced" summary (written 2026-10-26 09:22 KST, before any E28b call)
Same 28 cases and inputs as E28 (no answer shown). Models: gpt-6-luna and gpt-6.1-sol, both reasoning effort medium,
same API. (1) Categories: the E28 prompt, two uncached calls per model (Luna = the stored E28 calls); agreement and
kappa within each model, and Luna-vs-Sol agreement. (2) Free form: "In one or two sentences, based only on the readout,
why might the model not state the correction plainly?", two calls per model; a fixed Luna judge (model names hidden,
order shuffled) says whether two explanations give the same main reason; agreement rate per model. Read-out: Sol is
preferred for the latent-analyst summary only if its free-form agreement is >= 10 points above Luna's and its category
agreement is not lower. Exploratory: E22 say-it fix rate by Sol category.

## E27 result (written 2026-10-26 09:23 KST): criterion met
Test of the dev-selected version (Plain 64.1): OUT5 a v1 66.9, b v3 67.3, c v3 65.2, d v1 63.3, e v1 70.0 - mean 66.5 (SD
2.5; FPQ 66.1, NFP 66.9); LAT a v3 68.8, b v1 70.6, c v3 68.1, d v1 69.9, e v3 68.5 - mean 69.2 (SD 1.0; FPQ 60.8, NFP
77.6). LAT - OUT5 = +2.6 (Welch t about 2.2, p about 0.08 with 5 runs per arm). No coverage rewrite was needed. LAT_a's
meta wrote the E22 mapping into its procedure ("add a missing fact for an unaware model; for correction intent ...
state the correction explicitly and identify the exact premise"; "account for the gate"), and its v3 context has an
instruction paragraph and a separate knowledge paragraph. Two of five LAT runs selected v1 (before any lens
observation), so the gain is not attributed to the observations; gate vs observations is separated post hoc.

## E28b result (written 09:23 KST): Sol not preferred by the pre-registered rule
Categories: Luna agreement 0.857 (E28 calls), Sol 0.929 (kappa 0.863); Luna vs Sol first calls agree on 0.536 of cases
(Sol puts 19 of 28 in targets_other_claim, Luna 15). Free form, same main reason by the Luna judge: Luna 0.964, Sol 1.000
(+3.6 points < 10). The first summary printed by e28b_sol_check.py paired the stored Luna calls by question id, which
repeats across contexts; the Luna figures above are recomputed by position.
E27 post-hoc attribution (09:24 KST, stored answers, no new generation): the probe gate added only at test to
each OUT5 selected version (context kept where get_context returns text and p_fpq >= 0.5, Plain answer elsewhere):
a 71.4, b 67.0, c 66.0, d 69.6, e 71.9 - mean 69.2, equal to LAT's 69.2. The whole LAT gain is the gate; the in-loop
observations (lens types, type changes, meta aggregates) add nothing measurable beyond it.

## E29: output-change experience memory vs + Oracle-lens change history, same fixed gate (written 2026-10-26 09:54 KST, before any E29 code or run)
Fold 0, arms A (output experience) and B (A + internal observations), 5 runs each, 3 iterations x 2 sub-iterations = 6
edits per run (MCE-style batches). Data roles: fixed training batch (40 = 20 FPQ / 20 NFP, re-answered at v0..v6, 7
points) for before/after experience; rotating training batch (new 25 per edit) as the next edit's material and for the
memorisation check (training-question strings or ids in the code, and whether rules fire on the next rotating batch);
dev100 for aggregated scores (meta and version selection; no per-item dev feedback; no dev-based rewrite); fold-0 test
once for the dev-selected version. Fixed probe gate (p_fpq >= 0.5) in train, dev and test for both arms. Both arms:
change-process cards for the fixed batch (code-computed context diff, answer first sentences before / after,
reference, score before / after, the agent's stated reason; passes and failures), seeded with the E22 experience (81
records, 50 distinct questions, three fixes; the knowledge fix used each item's reference correction); meta sees the
training data, the current context and get_context code, training results and the score history, as the original MCE.
B only: the four-block lens reading (fixed reader, fixed Luna interpretation) of every fixed-batch input at every point,
before / after with quotes, the E28 reason categories marked as hypotheses, the E22 records with lens types, and for
meta the before / after internal cases and an edit-type x internal-change x outcome table. Primary: B - A test macro (+1
point as the practical target; per-run results and uncertainty reported; no claim of resolving a given size with 5
runs). Secondary: per-edit fixes / breaks on the fixed batch (descriptive; their sum telescopes), cases where an
internal observation is linked to the edit made. Budget before cache reuse: 9,720 evaluation requests (+100 shared dev
v0) and as many judge calls; 1,400 lens reads (B). Pre-stated expectation: macro difference within +-1 with B better on
the secondary measures ~45%; B +1 to +3 ~25%; no difference or worse ~30%.
E29 amendment (09:59 KST, before any E29 run): memorisation check, fixed before running. A version is flagged
(one rewrite) if (a) context/ or interfaces/ contain training question ids or 8-word runs of training questions, or
(b) over all 156 training questions before the gate it attaches context only to questions this run has already seen,
or (c) at least 5 and at least half of the questions it attaches context to receive a context text no other training
question receives. Checked on stored versions: flags E18 INT_a v2 (c) and OUT_b v3 (b), the two known per-question
regex versions; does not flag E18 INT_a v1, E21 OUT3_a v2 or E24 OUT4_c v1 (keyword fact table). Contexts are computed
only for the questions being answered (subset items file), plus once per edit over all training questions for (b)/(c).
E29 amendment (10:02 KST, before any E29 run): both SKILLs add one paragraph (the context may hold reusable facts and
explanations of kinds of misconceptions; knowledge may be generalised from the training reference corrections and
provided by code to the questions it fits; no lookup by a single training question's wording or id; format is free).
The lens type is named "correction plan seen / no correction plan seen" (not "unaware"): it is a reading, not evidence
that the model lacks the knowledge. The question for B is whether the internal observations help decide which
knowledge, wording or applicability to change, not whether B alone knows that knowledge is needed.
E29 amendment (10:07 KST, before any E29 run): B's internal record is a content comparison, not type tags. For each
fixed-batch question and version, a fixed Luna prompt receives the question, the training reference, the context
sentences added / removed, and the previous and current four-block readings, and returns a summary of each reading,
the content that appeared / disappeared, quotes from each reading and unclear / unrelated parts; every statement must be
backed by a reading quote. Cards show these with the id of the raw reading (data/lens_raw.jsonl, also copied to the
meta folder); the meta's internal.md lists, per edit, the questions whose score or reading changed with the context
change. The E28 reason category and the correction-plan tag are not used; the E22 seed shows the E18 reading summary and
quotes for B. Luna calls for B: one comparison per fixed question and version (1,400).
E29 amendment (10:10 KST, before any E29 run): the fixed training batch is drawn from gated training questions only
(p_fpq >= 0.5: 54 FPQ, 23 NFP available; 20 + 20 drawn, seed 2900), since an ungated question never receives context and
its answer and reading cannot change across versions. The rotating batch is drawn from all other training questions.
E29 amendment (10:18 KST, user decision, before any E29 run): no forced gate. get_context(question, latent)
receives latent = {"p_fpq": ...} in training, dev100 and test (new v4_contexts kind "probe"; earlier kinds unchanged)
and decides what to attach, including nothing (Plain); both arms get the same input and the same SKILL section
"Latent input". The fixed training batch is 10 questions from each of FPQ / NFP x p_fpq >= / < 0.5 (seed 2900), so
every question can be improved and both sides are tracked; cards show each question's p_fpq. The question becomes:
does the lens change history help decide what to provide and when to leave a question Plain? Without a forced gate the
NFP loss of broad contexts (E21-E24) can return in both arms.
E29 amendment (10:24 KST, before any E29 run): the one-off-text signal (c) no longer triggers a rewrite; it is logged as
a warning (memory/e29_flags.jsonl), because a selection function that composes knowledge per question returns
different texts legitimately. Rewrites only for (a) verbatim training ids / text and (b) context only on already seen
questions. Paraphrased per-question rules are left to dev selection and checked afterwards from the warnings.
Always-attach, selective attach and no context remain equally allowed in both arms.
E29 amendment (10:32 KST, before any E29 run): (1) arm names: A = probe input + output experience memory, B = A +
Oracle-lens change experience; the comparison is the added effect of the lens experience. (2) The SKILL's probe numbers
now come from the training questions (5-fold cross-fit: AUC 0.77; at 0.5 69% of FPQ and 29% of NFP); the earlier draft
used fold-0 test figures from E25 and was replaced before any run. (3) data/memory.md is re-rendered after every
fixed-batch evaluation, so the next edit and the meta see the latest experience. (4) Cards show removed as well as
added sentences and the path of every version's full context (memory/snapshots/v<t>/); the meta folder holds every
version's context and code (meta/versions/). (5) The meta prompt adds, as in the original MCE prompt, the required
interface signature, an inline iteration-history summary and the original skill examples A and B. Wording: the
optimising agents get no test questions and no per-item dev results; at evaluation get_context and Qwen receive the
test questions and the judge receives the reference; the agents get no judge reasoning, only ratings, answer-behaviour
summaries and quotes.
E29 amendment (10:33 KST, before any E29 run; prompts fixed from here): reference for "as in the original MCE" is
metaevo-ai/meta-context-engineering c4b7a7c (the local copy in experiments/latent_mce/mce_upstream), not the
mce-artifact repository. The meta prompt states the interface (get_context(question, latent) -> str, "" = plain
prompt, latent = {p_fpq}) and the edit scope (base: context/ and interfaces/; meta: SKILL.md only); the skill database
summary is built by code (per version: skill-overview excerpt, fixed / rotating batch rates with sizes, dev100 scores,
file paths); the skill examples are marked "reference only, not required" with our file names. These common changes
apply identically to both arms and are fixed before the run; they can change both arms' behaviour.
E29 amendment (10:37 KST, before any E29 run): warm start as the official MCE (metaevo-ai/meta-context-engineering
c4b7a7c = current main, mce/utils.py _find_best_iteration): iterations 2 and 3 start by copying context/ and
interfaces/ of the version with the best dev100 macro so far (iteration 0 excluded, as the official baseline is not
saved to evaluations.json; ties: the earlier version), then the meta and base agents edit. Experience keeps
accumulating; every card line, the meta history and internal.md compare a version with the version it was edited
from ("edited from v<j>"). Test selection is unchanged (dev-best over v0, v2, v4, v6; ties: later).
E29 amendment (10:44 KST, before any E29 run): a second memory layer in both arms. After every fixed-batch
evaluation a fixed Luna prompt summarises the change-process cards (A: output cards; B: output + lens cards; same
procedure, one call per edit, at most 8 patterns) into data/patterns.md and memory/patterns.jsonl: relations that recur
across different questions, with held-in / out-of counts over distinct questions, counterexamples, an optional
hypothesis kept apart from the observation, and card line ids [question|vN] as evidence; "none" when nothing recurs.
The base prompt inlines it; the meta folder gets both files. Smoke check: 2-3 patterns compared with the raw cards.
E29 amendment (10:51 KST, from the smoke run, before the main run): (1) when a question's attached context is the
same as in the version it was edited from, the input and reading are identical, so no Luna comparison is made; the
card says "input unchanged; same reading". (2) The pattern prompt adds: unchanged context giving unchanged answers and
readings is true by construction and is not a pattern (the smoke summary had reported it).
E29 smoke result and last amendments (11:03 KST, before the main run): smoke E29B_smoke (fixed 8 = 2 per cell, rotating 3,
3 edits, 16.5 min) passed: latent input used by the agent (it wrote a p_fpq >= 0.65 rule), lens comparisons 22 / 22
parsed, patterns 3 / 3 parsed and checked against the stored history (e.g. "all 4 questions with context at v3 rated 5"
holds), meta folder complete, warm start logged, dev and test written (test 67.3, not a result). Changes: (1) the
pattern prompt also rules out "the reading changed because the input changed" as a pattern (structural); (2) to keep
the prompt size bounded (the full-size B memory would reach ~410k characters), each card shows the last 3 versions in
full and older versions in one line (context attached, sentence counts, rating, behaviour); full detail stays in
memory/snapshots and data/lens_raw.jsonl. Same rule in both arms.
E29 amendment (11:25 KST, from a review of the smoke inputs, before the main run; replaces the last-3-versions cards of
11:03): (1) one compact card per fixed question in both arms: the current state [q|vN] (context attached or not, rating,
behaviour, answer opening), the ratings of every version with "(= input of vP)" where the input did not change, and the
last input change only (sentences added and removed, rating, behaviour and answer before); questions that keep failing
stay on the cards. The card section starts with the result of the last edit against the version it was edited from
(fixed / broke / still passing / still failing). (2) The E22 experience is a pooled summary with six fixed examples
(Random(29): 2 knowledge-only fixes, 2 instruction fixes, 2 unfixed) in memory.md; all 81 records in
data/e22_experience.md. (3) B's lens line is one Luna comparison per changed input: one or two sentences on the
content change that relates to the question or its premise, at most two quotes, "no relevant evidence" / unclear
otherwise; quotes not found verbatim in the two readings are dropped and counted (quotes_dropped). (4) The pattern
prompt no longer rules out "the reading changed because the input changed"; it reports a reading difference only when
its content relates to the question, its premise or a change in the answer or rating. (5) decisions.md asks, per
change, what changed (sentences, knowledge content, which questions get it), the card or pattern ids relied on, and the
questions expected to be fixed or put at risk; the next card shows what it fixed and broke. memory/e29_edits.jsonl
logs per edit the parent version, fixed / broke, the base prompt size and the extra calls so far (lens reads, Luna
comparisons, pattern summaries); e29_history.json stores the parents. (6) serve27b_lens.py accepts per item
"nocache": true and "seed": n (read afresh, not cached); items without them are read exactly as before. Used only by
the observation-reliability check below, not by the loop.
E29 observation-reliability check (11:25 KST, before the run of the check; e29_reread_check.py): 8 questions of the
main fixed batch (first 2 of each FPQ/NFP x p_fpq cell), two inputs (Plain; Plain + the v1 context of the B smoke run),
3 fresh readings each (seeds 1-3, no cache). Luna gets the fixed comparison prompt with "added = the context's
sentences" for null pairs (two seeds of the same input) and real pairs (Plain vs context, same seed), each pair twice,
shuffled, blind. Reported: word overlap of the readings, the "clear" rate for null vs real pairs (reading variance),
and the agreement of the two Luna calls on the same pair (interpretation variance). Reading: if the null "clear" rate
is close to the real one (within 0.15), a reported change on the cards cannot be told from sampling variance; B's
INTERNAL section then states the null rate. The check does not change the loop.
E29 amendment (11:27 KST, before the main run): questions and reference corrections are no longer cut. The cards,
the rotating batch and the E22 records had shown the first 150-240 characters, a length cut with no design reason;
142 of 156 training questions are longer than 200 characters and in Cancer-Myth the false premise and the request
usually come last (e.g. fpq_133 lost its request, nfp_1060 lost the prognosis phrase that a smoke edit over-corrected).
The official S2D training record (mce_upstream env/symptom_diagnosis format_result_for_training) gives the question in
full. Both arms now show every question and correction in full (each appears once per card), and the base workspace
gets data/train.jsonl (all training questions with label and reference correction; the meta's train.jsonl is a copy).
Dry render at full size (40 fixed x 7 versions, 25 rotating, 6 decisions, long fields): base prompt about 100k
characters (A) and 119k (B).
E29 amendment (11:35 KST, before the main run; from a review; both arms): the optimisation feedback no longer contains
the Luna answer-behaviour label and its evidence quote (a second model interpretation of the answer; never the judge's
reasoning, which was not given). Every answered training input (fixed and rotating) is stored in full in
data/rollouts.jsonl under its card id [question|vN]: question, FPQ/NFP label, reference correction, p_fpq, the
attached context, the full Qwen answer, rating, version and the version it was edited from; the meta folder gets a copy.
memory.md keeps a short code-made view (context attached, rating, answer opening, the last context change and the
rating change) with the ids. Rotating-batch records now carry the version that answered them (they had been logged as
v0). B adds only the Oracle-lens observations (relevant content change from the parent version, verified quotes,
clarity, raw reading id). p_fpq is common to both arms, so the comparison is: common output experience + probe input,
with vs without the Oracle-lens change memory. Smoke re-run E29A_smoke3 / E29B_smoke3 with this format.
E29 observation-reliability result (12:02 KST; e29_reread.json; fast lens server: fla/causal-conv1d kernels and the four
layers sampled in one generate call, hidden states cosine >= 0.9998 vs the torch path on 4 inputs x 4 layers; 24 reads
131 s -> 54 s). 8 questions, Plain and one B-smoke context, seeds 1-3. Word overlap of two readings: same input 0.39,
Plain vs context 0.24. Luna "clear" relevant change: null pairs (same input, two seeds) 0.78, real pairs 0.97 (16 pairs
each, each pair twice). Same clarity on two Luna calls: null 0.81, real 0.94. Quotes kept / dropped by the verbatim check:
48 / 1 and 55 / 2. Reading by the pre-registered rule: the gap 0.19 is above 0.15, so the rule does not require the null
rate in B's INTERNAL section; but sampling alone gives a "clear" change in 78% of unchanged inputs, so "clear" on a card
is weak evidence by itself. In real pairs Luna describes the direction of the change (e.g. "shifts to flagging the
premise"); in null pairs it mostly describes differences in details. Decision on how B uses this left to the user.
E29 amendment (12:46 KST, before the main run; on top of the 12:00 file-based rewrite of e29_loop.py / e29_workspaces.py:
current batch in data/train.json, outputs in data/rollouts.jsonl, memory.md not inlined, no pattern summary, no E22
seed). (1) Both arms: utils/retrieve.py, a local deterministic TF-IDF search(question, entries, k) over the agent's own
knowledge entries, callable by get_context (no model call; Qwen is still the only model get_context may call). SKILL
step 4: a knowledge entry describes one kind of misconception with the facts that correct it and needs at least two
supporting training questions, whose ids go in context/decisions.md only. Reason: E22 (the question's reference fact
fixed 93% of failing FPQs vs 31% for an instruction) and smoke contexts were 1-2 behavioural instructions with no
knowledge files. (2) B only: the Internal observations section states the re-read result (78% "clear" on unchanged
inputs vs 97% on real changes) and asks to rely only on directional changes. Smoke E29A_smoke4 / E29B_smoke4.
Knowledge-transfer check run alongside (e29_transfer_check.py, dev100, retrieved vs random training corrections).
GEPA baselines on vLLM test (e29_gepa_baseline.py): Plain, GEPA fold-0 prompt, GEPA under the p_fpq >= 0.5 gate.
E29 baselines and knowledge-transfer check (12:48 KST). GEPA on the vLLM pipeline, fold-0 test 242 (193 FPQ, gate on
116): Plain 64.1 (FPQ 48.7 / NFP 79.6), GEPA fold-0 prompt 68.4 (65.3 / 71.4), GEPA under the p_fpq >= 0.5 gate 69.6
(59.6 / 79.6) (e29_gepa_baseline.json); 69.6 is the bar for E29. The earlier GEPA 67.6 came from the older server.
Transfer check on dev100 (80 FPQ / 20 NFP, gate on 44; e29_transfer_check.json), two training reference corrections
attached in a "background facts" frame: Plain 59.4 (43.8 / 75.0); TF-IDF-retrieved, all questions 62.5 (75.0 / 50.0);
random, all questions 67.5 (65.0 / 70.0); under the gate retrieved 60.0, random 60.6. Reading: the frame itself raises
FPQ (random +21 points); relevance adds +10 FPQ points (8 of 80 questions) but costs NFP (-20 points = 4 of 20
questions), so raw retrieval does not raise the macro. Retrieved items mostly match the cancer type, not the kind of
misconception (5 examples read). Knowledge can transfer, but when to attach it decides the macro; NFP n = 20 is small.
E29 decision (12:49 KST, user): GEPA under the probe gate is dropped as a comparison bar (it already uses our probe); its measured value (69.6) stays on record in e29_gepa_baseline.json. The bars are GEPA 68.4 and A vs B.
E29 smoke4 (12:56 KST, TF-IDF tool): both arms wrote one instruction file and no knowledge entries; retrieval unused.
B's decisions: the two misses shared a response-level omission but different medical facts, so only the shared
behaviour was instructed (with 8 fixed + 3 rotating questions per edit, two supporting questions per entry are rare).
E29 amendment (12:56 KST, user: same tool conditions as the original MCE): both arms get the original workspace
utilities (latent_mce/mce_upstream/mce/workspace_utils: __init__.py, llm.py = our Luna port of call_llm, embedding.py =
compute_embedding_similarity with text-embedding-3-small) plus utils/qwen.py; the TF-IDF utils/retrieve.py is removed
(embedding similarity is the original retrieval tool). The base prompt adds the original "Available Utilities" section
(our file names); the codex agents (base and meta) run with network access as latent_mce codex_agent.py; get_context
runs under the MCE environment (/data2/heejae/uv/mce/bin/python) so utils/embedding.py imports. One kept difference:
get_context may call utils.qwen (the answering model) and utils.embedding but not utils.llm, because the original
llm.py is the answering model's family (deepseek) while ours is Luna, a different, stronger model. The 1,500-character
sentence was removed from the Memory format by the user (12:53). Smoke E29A_smoke5 / E29B_smoke5.
E29 amendment (13:04 KST, user: option "na" and lens on the rotating batch; before the main run). (1) B: every rotating
training input is also read by the lens and compared with its Plain reading (record id <q>|plain in lens_raw.jsonl);
internal_changes.jsonl holds fixed (vs parent version) and rotating (vs Plain) records. (2) B base guidance: for each
current-batch failure open its internal record and write one line in decisions.md (record id, what the reading shows,
chosen edit); E22 working hypotheses are given as hypotheses to check, not rules (correct fact in the reading but not
said -> wording/instruction; correction missing or false belief -> knowledge entry for that kind of misconception;
NFP read as a false premise -> narrow attachment). (3) B meta: internal_outcomes.md built by code (per edit: the
observation before, the stated decision, pass before -> after, the observation after) and asked to open it first.
(4) e29_edits.jsonl logs the number of record ids cited in decisions.md (both arms). Fixes of the MCE comparison:
dev score files are moved out of the workspace after each dev run (the base agent could read them); before an edit the
memory index shows the version the workspace holds (iteration-3 restart); rotating batches are drawn per iteration as
2 x 25 without overlap (official MCE); get_context exceptions give "" and are recorded (v4_contexts kind "probe").
Smoke E29A_smoke6 / E29B_smoke6.
E29 smoke6 (passed) and amendment (13:26 KST, user-approved; B only). smoke6: no errors; B cited rotating observation ids
(e.g. fpq_138|plain) and the meta got internal_outcomes.md; but for inputs equal to their previous input (most failed
FPQs, which get no context) the observation was only "same reading", so B had no internal information exactly where
it was needed. Change: every reading also gets a fixed Luna state summary against the training reference (STATE prompt:
premise_status reference_correction / generic_correction / accepts_premise / unrelated, one-sentence summary, at most
two quotes checked verbatim; cached by prompt, so once per distinct input). The B guidance maps the E22 hypotheses to
these statuses; internal_outcomes.md shows the status before and after each edit. Smoke E29B_smoke7.
E29 knowledge check and amendment (13:33 KST, user-approved "ga + na"). Why no knowledge entries in smokes 4-6: the
decisions show the agents considered and declined them (failures in a batch had different medical facts, so no kind had
two supporting questions; cautions such as "do not invent hidden knowledge" were read as "add no medical facts").
Check (e29_kinds.json, one Luna grouping of the 78 training FPQ corrections): 47 kinds, 12 with >= 2 questions covering
42 of 77 assigned FPQs (largest: "diagnosis or stage means death is inevitable", 11 questions); 35 are single. So
recurring kinds exist for about half of the FPQs, but a batch of 11 (smoke) rarely holds two of one kind. Changes (both
arms): the base prompt adds the original MCE "Core Objective" and "Training Data Analysis" paragraphs (context files
more impactful than interface logic; what knowledge was missing); the caution now refers only to claims about the
model's internal state, not to writing correct medical knowledge; SKILL step 4 says the second supporting question may
come from data/train.jsonl.
E29 smoke8 and main-run start (13:43 KST, user: 'make it finish fast'). smoke8 A wrote the first knowledge files (bile_duct_jaundice.md, post_treatment_followup.md, screening_by_age.md; attached by keyword rules plus p_fpq thresholds, embedding unused); memorisation flags clean. GPU 6-7 now serve a third vLLM (8804) instead of the second lens server; all E29B runs read on lens 8796. Main run: run_e29.sh (5 runs per arm).
E29 B stopped and STATE revised (13:46 KST; from a review). The main B runs (started 13:43) were stopped after a few
minutes and moved to versions/_aborted_1347/; the A runs continue unchanged (A does not use the state summary). The
draft state prompt was replaced: classify only what the readout explicitly supports; not mentioning the reference is
not acceptance; no completing or restating the reference; FPQ statuses target_correction / other_correction /
explicit_acceptance (the specific false belief asserted) / not_observed / mixed_or_unclear / unrelated, NFP statuses
answers_normally / corrects_something / mixed_or_unclear / unrelated; quotes carry their depth and are checked in that
depth. The B guidance no longer maps a status to an edit; it lists candidate edits as hypotheses (E22: the fact helped
in both groups). The state summary makes the stored readings easier to use; it is not new information, and a gain
would show the effect of the feedback method including the lens and the fixed LLM interpretation. Check on the 36
stored smoke7 readings: of 5 draft "accepts_premise", 4 were normal NFP readings (now answers_normally /
mixed_or_unclear), 1 (fpq_205) is explicit_acceptance with a supporting quote; 1 target_correction; 4 quotes dropped by
the depth check. Smoke E29B_smoke9 before restarting B.
E29 known limitation (13:53 KST, from a review; left unfixed in both arms for comparability): at the iteration-3 restart
from an earlier dev-best version (v2 better than v4), context/ and interfaces/ are restored but memory.md is not
re-rendered before the meta step, so the meta's index says v4 while meta/current/ holds v2. The A runs started with
this code and cannot be changed; B keeps it too. The runs where it occurs will be counted. Correction to an earlier
reading: smoke8 B also wrote knowledge (one file with follow-up, screening-age and jaundice facts); the difference was
selective vs bundled attachment, and file creation is not a success criterion (smoke8 A's dev-selected version was v0).
E29 B main restart (13:54 KST): E29B_a..e with the revised STATE prompt and guidance (smoke9 passed: B wrote knowledge files and cited states, filtering off-topic quotes); same lens 8796 and vLLM 8802-8804. Launch: e29_workspaces a,b,c,d,e E29B, then one e29_loop process per run as in run_e29.sh.
E29 predictions (13:56 KST, before any main-run test result; A runs at iteration 2, B at iteration 1). Fold-0 test
macro: A mean about 66 (runs 64-69), B mean about 66.5; B - A about +0.5 with a run-level spread of about 2 points
(earlier MCE-type runs: SD 1.7). P(B - A >= +1 with the paired interval above 0) about 15%; P(B mean > GEPA 68.4) about
20%, P(A mean > 68.4) about 15%. Process: B cites internal record ids in most edits (about 80%); both arms write some
knowledge; NFP drops of single edits (as in A_a, A_c, A_d) recur in both arms and dev selection removes most of them.

## E30: internal records as an explicit step (written 14:20 KST, before any E30 result; user decisions)
Question: does B gain when the internal records are an explicit procedure step instead of guidance scattered with
cautions? Change from E29B (B only; A unchanged by the user's decision, so E30 reuses the five E29 A runs, which share
code, batches and seeds): the locked "Internal observations" section and the base guidance become one step - before
every edit read the internal records of the questions to change with their answers and ask (i) did the context reach
the model, (ii) does the reading hold the unstated correction, a different one or none, (iii) did a passing NFP's
reading start correcting - and one caution ("a single reading is noisy; trust a pattern that recurs across questions or
is confirmed by results"); the per-failure line, candidate-edit table and repeated cautions are removed. The meta gets
step 2b: read internal_outcomes.md and revise the SKILL step on internal records by which readings were followed by
edits that worked. Code: e30_loop.py (replaces e29_loop.internal_guidance / base_prompt only), e30_workspaces.py. Runs:
E29B_x30a/b/c (3 runs, user rule), concurrent with E29 B on the same servers. Read-out: E30 B vs E29 A (5 runs) and vs
E29 B on test macro; process: share of B decisions and meta notes citing internal record ids. Prediction: E30 B mean
within +-1.5 of E29 B; P(E30 B - A >= +1) about 20%.

## E29 / E30 result (written 15:01 KST; all runs finished)
Fold-0 test macro (242 q): E29 A 65.4 / 64.9 / 64.3 / 69.0 / 69.6, mean 66.64 (SD 2.47; FPQ 56.2, NFP 77.2);
E29 B 71.7 / 70.6 / 69.1 / 68.5 / 61.4, mean 68.26 (SD 4.04; FPQ 60.6, NFP 75.9); E30 B 66.0 / 62.1 / 66.5, mean 64.87
(SD 2.41). Plain 64.1, GEPA 68.4. Paired E29 B - A by rep: +6.3, +5.7, +4.8, -0.5, -8.2, mean +1.62, 95% t interval
[-6.0, +9.2]: not distinguishable from 0 (pre-registered reading: B - A >= +1 with the interval above 0 - not met).
B mean vs GEPA: -0.1. E30 (explicit internal step, one caution) did not exceed E29 B or A. Across all 13 runs the
dev100 macro of the selected version does not predict its test macro (Pearson r = -0.01); dev has 20 NFP.
Process (E29 B, 3,352 readings over B runs): the reading status mostly restates the answer (FPQ target_correction ->
answer pass 77%, not_observed -> 20%; NFP corrects_something -> 40%, answers_normally -> 91%). Failed fixed-batch FPQs
whose parent reading had target_correction were fixed by the next changed input in 6 of 8 cases (other statuses
14-31%), but such cases were rare; 382 of 572 failed fixed FPQ cases received no input change at the next edit.
Decisions of B cite readings in every edit (A: 1 of 30) but mostly conclude that the reading adds nothing beyond the
answer; the best run's (B_a) decisive edit (threshold 0.78 -> 0.50, general knowledge document) was justified by
output records in its own words. Contexts narrowed after first-edit NFP losses (thresholds raised to 0.85-0.90).
Predictions (13:56) vs result: A mean 66 (observed 66.6), B - A +0.5 (observed +1.6, interval spans 0).

## E31 benefit probe (offline, written 15:11 KST after the run; exploratory, not pre-registered beyond this entry)
A "will context help?" probe (label: mean pass with context - Plain pass over E29/E30 main-run training records;
125 labelled training questions, 37 helped, 38 harmed; p_fpq recipe on Plain states; e31_benefit_probe.py,
e31_benefit_probe.json). Cross-fit AUC for "helped": benefit probe 0.662, p_fpq 0.670. Fold-0 test, mean over the 14
E25 contexts answered on all test items: Plain 64.1, attach all 65.6, p_fpq >= 0.5 68.8 (attach FPQ 55% / NFP 20%),
benefit probe at the same attachment count 67.0 (49% / 45%), benefit >= 0.5 67.1. The benefit probe does not beat
the p_fpq gate.

## E31 pre-check: sampled answers (written 15:18 KST; e31_sampling_check.py, e31_sampling_check.json)
Training 156 questions, Plain, 4 answers each at T 0.8 / top-p 0.95 (vLLM seeds 1-4), same Luna judge. Passing samples
out of 4, FPQ (78): 0 -> 40, 1 -> 6, 2 -> 7, 3 -> 8, 4 -> 17; NFP (78): 0 -> 11, 1 -> 4, 2 -> 9, 3 -> 12, 4 -> 42.
FPQ whose greedy answer fails but at least one sample passes: 12 (the lens "target_correction" route found 8 such
failures over all E29 B runs); NFP whose greedy answer passes but at least one sample over-corrects: 19. So 21 FPQs
(27%) are "sometimes corrected" (the model can produce the correction), 40 (51%) are never corrected in 4 samples, and
25 NFPs (32%) are sometimes over-corrected. Pass/fail here uses the reference-based judge (training only).

## Offline checks of where the Oracle lens could enter (written 15:20 KST, exploratory, stored outputs only)
(1) Test-time gate, mean gated macro over each fold's stored broad contexts (fold 0: 14 E25, fold 1: 3, fold 2: 2 E26):
probe 68.8 / 64.7 / 65.5; lens correction frame 65.8 / 62.6 / 63.6; probe OR lens 69.7 / 63.4 / 65.0; probe AND lens
64.8 / 63.8 / 64.1. The lens adds no consistent gain. Probe gate on fold-0 test: precision 0.91 (106 FPQ, 10 NFP),
misses 87 of 193 FPQ. (2) Version selection: over the 8 B runs' test versions, the share of training-NFP readings
with corrects_something does not predict test NFP pass (Pearson 0.01, Spearman -0.26; dev NFP 0.19 / 0.39; training
NFP pass 0.07 / 0.09). Test NFP harm came from where get_context attaches on new questions (e.g. B_e's test version
attached context to 1 of 20 training NFPs, test NFP 0.67), which training-question measures do not see.

## E32: reaction axes from Oracle-lens labels (written 15:42 KST, before any E32 result)
Question: can the Oracle lens give the context function a better attach signal than p_fpq, by labelling how each
training question reacts to context? Step 1 (e32_reaction_labels.py): 156 training questions x {Plain, E25 contexts
FTY_a_v2, OUT3_a_v2, OUT_a_v2}: pass and the fixed state summary of the reading. Step 2: labels per question, two
versions - output-only (NFP fragile = fails under a context; FPQ responsive = passes under a context) and output+lens
(NFP fragile also when it passes but the reading is corrects_something; FPQ responsive also when it fails but the
reading has target/other correction); axes p_fragile (NFP) and p_near (FPQ) learned from Plain states with the p_fpq
recipe, cross-fit AUC, and the same from question TF-IDF. Step 3: on the stored fold-0 test answers of the 14 E25
contexts, gates combining p_fpq with the axes vs the p_fpq >= 0.5 gate (same attach count). Pre-registered reading: the
lens contributes if the output+lens axes give a higher gated test macro than the output-only axes and than p_fpq alone,
in the mean over contexts; otherwise not. Expected: small or no gain (labels from 3 contexts are few and noisy).
E31 mid-answer readings (written 15:50 KST; e31_midanswer.py / .json). "Sometimes" questions (21 FPQ, 25 NFP), one passing
and one failing sampled answer each, read after 32 / 64 / 128 answer tokens (lens server "answer"/"upto"), fixed STATE
summary. Share of questions whose two answers get different statuses: FPQ 0.57 / 0.57 / 0.67, NFP 0.80 / 0.80 / 0.72;
noise (same failing answer re-read at 64 with another reader seed): FPQ 0.24, NFP 0.20. NFP: corrects_something in the
reading of over-correcting vs passing answers 13 vs 5 at 32 tokens, 16 vs 4 at 64. FPQ at 64: target_correction 8 (pass)
vs 4 (fail), not_observed 5 vs 10. In 6 FPQs (fpq_109, 121, 180, 197, 200, 237) the failing answer's reading holds the
target correction at some checkpoint. Caveat: the reading after an answer prefix may restate the prefix's direction; a
text-only baseline on the same prefixes is needed before claiming information beyond the text.
E32 amendment (16:09 KST, from a review, before step 2 is run): the output+lens "responsive FPQ" label counts only
target_correction (other_correction corrects a different claim). The E25 test comparison of step 3 is a development
analysis, not final evidence; a rule chosen there needs a separate held-out evaluation. Reaction-outcome tally now
saved (e29_reaction_outcomes.py / .json; fixed batch, B runs, input changed at both versions): failed FPQ fixed at the
next changed input - target_correction 5/6 (3 questions), other_correction 1/17 (5 q), not_observed 4/60 (11 q);
passing NFP broken - corrects_something 3/11 (5 q), answers_normally 3/60 (15 q). The earlier "26% vs 7%" pooled
target and other corrections. Correction: the E16 output baseline (Tout) NFP-break AUC is 0.446 (e16.json).
E32 result (16:17 KST; e32_axes.json, e32_geometry.json): pre-registered criterion not met. Labels: 59 NFP passing Plain
(fragile 16 output-only / 19 output+lens), 52 FPQ failing Plain (responsive 20 / 21). Cross-fit AUC from Plain states:
p_fragile 0.644 (output) / 0.701 (output+lens), p_near 0.719 / 0.664; from question TF-IDF 0.38-0.68. Fold-0 test,
mean over 14 E25 contexts (development analysis): p_fpq >= 0.5 68.8; output utility 68.2; output+lens utility 67.4;
p_fpq and not fragile 65.8 (output) / 66.2 (output+lens). Geometry (exploratory): centred group means of fragile NFP and
near FPQ are not alike (cos -0.12 to 0.17); the fragile-safe direction leans toward the FPQ-NFP direction (cos 0.10 to
0.57), the near-noreach direction away from it (-0.26 to -0.70): the boundary items are those the FPQ probe confuses;
DIM projections separate them weakly (cross-fit AUC 0.57-0.73).
E31 context-position check (written 16:23 KST; e31_position_check.py / .json; test 242, greedy, same judge). Context moved
from the end of the system prompt into the user message ("Context: ... Question: ...", as the original MCE prompt
puts context before the input). E29B_a v2 (attached 116): user 71.9 (FPQ 66.3 / NFP 77.6) vs system 71.7 (65.8 / 77.6).
E29A_c v2 (attached 197): user 54.8 (64.8 / 44.9) vs system 64.3 (63.2 / 65.3). The user placement did not reduce
over-correction; for the broad context it cut NFP by 20 points. The system placement is kept.

## E33: pre-rollout check of each draft (written 16:26 KST, before any E33 result; user-approved, smoke first)
Basis: E16 (with-context reaction predicts the outcome of new contexts on the same training questions; NFP-break AUC
0.74 vs 0.446 for the output baseline). E32's weak new-question axes are not evidence for or against E33. Design: E30 B
(MCE + internal records as one explicit step) plus, at every edit, a draft C1 -> pre-rollout check -> revision C2 by the
same base agent; only C2 is answered and rated. Check (e33_precheck.py): C1's inputs on the fixed and current rotating
training questions -> /hidden states (layers 20..60, last prompt token) -> E16-recipe predictor (label, pass-probe s,
s - s0 vs Plain, Plain pass, + Oracle-lens state status) fit on answered training inputs merged by input key (E13 train
pairs with states, E32 records - their context states extracted - and this run's answered inputs, each with its own
state and rating) -> data/draft_feedback.md (predicted breaks / fixes with evidence). C1 is logged with outcome
"unobserved" (memory/e33_drafts.jsonl, memory/drafts/v<e>/); only answered inputs (C2) enter the predictor history.
Smoke E29B_s33a (8 fixed, 3 rotating, 3 edits): checks C1/C2 linkage, whether predictions match the C2 outcomes on
unchanged-by-revision inputs, and the total cost. Then 3 runs (development; fold 0). Final validation: the method fixed,
on an unused part of another dataset. Planned comparison after a positive result: the same design without lens features
and lens text in the feedback (probe kept). Development success signals: first edits break fewer NFPs than E29 (7-9 of 20
common) while keeping FPQ gains; the revision changes attach conditions or content after the check; dev macro.
E31 question fragility from the pre-answer state (16:28 KST; e31_fragility.py / .json; training only, repeated stratified 5-fold CV x10, raw AUC). FPQ sometimes vs never corrected (n 61, positives 21): state 0.762 (SD 0.029), text TF-IDF 0.543 (SD 0.036), p_fpq 0.307. NFP sometimes vs never over-corrected (n 78, positives 36): state 0.709 (SD 0.015), text TF-IDF 0.434 (SD 0.031), p_fpq 0.534. The Plain hidden state separates fragile questions where the question text does not; p_fpq is higher for never-corrected FPQs (raw AUC below 0.5). Small n; not yet checked on dev/test questions.
E31 fragility transfer to dev100 (16:41 KST; e31_dev_fragility.py / .json; dev answered 4x at T 0.8 with the same judge; vLLM 8804 was stopped by another job at 16:24, the run used 8802-8803). Fit on all training labels, scored once on dev, raw AUC with 95% bootstrap interval. Dev passing samples {"FPQ": {"0": 30, "1": 4, "2": 11, "3": 5, "4": 30}, "NFP": {"0": 4, "1": 1, "2": 1, "3": 2, "4": 12}}. FPQ sometimes vs never corrected (dev n 50, positives 20): state 0.793 [0.658, 0.911], text 0.507 [0.335, 0.668], p_fpq 0.557 [0.39, 0.72]. NFP sometimes vs never over-corrected (dev n 20, positives 8): state 0.771 [0.44, 1.0], text 0.26 [0.059, 0.505], p_fpq 0.698 [0.427, 0.92]. The state predictor transfers for FPQ (interval above 0.5); the NFP dev set (20) is too small to settle.
E31 gate mix on stored test answers (17:04 KST; e31_gate_mix.py / .json; exploratory, rules and 0.5 thresholds fixed before computing). E29B_d v2 (context on all 242): all 68.5, p_fpq 69.1, not_risky 67.0, p_fpq and not_risky 68.0, (p_fpq or knowable) and not_risky 66.5. E29A_c v2 (197): all 64.3, p_fpq 68.6, not_risky 65.0, p_fpq and not_risky 67.5, (p_fpq or knowable) and not_risky 66.5. Plain 64.1. Adding the fragility gates to p_fpq did not raise the macro on test; p_fpq alone was best in both. The fragility signals predict which questions vary across samples, not whether a given stored context helps.
Exploratory (17:36 KST, B runs E29 + E30, fixed batch, context changed between parent and child): reading change vs answer
change - FPQ with clear readings (281 pairs): both unchanged 183, reading changed / answer unchanged 37 (toward
correcting 16, away 21), reading unchanged / answer changed 38, both changed in the same direction 21, opposite 2;
NFP (146): 98 / 8 / 20 / 19 / 1; unclear readings FPQ 112, NFP 184 pairs. Hypothesis (posed after seeing the table)
that outcome flips without a reading change are surface flips that revert: not supported - at the next changed input
the flipped outcome was kept in 31/43 (reading unchanged) vs 15/34 (reading moved) vs 36/63 (unclear).

## E34: does a 4-read consensus of the Plain lens reading add to the probe? (written 19:09 KST, before any E34 read)
Reads: every item of fold 0 (train 156, dev100, test 242; 498 unique) with the Plain prompt, read by the fixed Oracle
reader (blocks 36/44/52/60, settings unchanged) with 4 seeds (nocache), 4 lens servers on GPUs 0-7 (vLLM and /hidden
stopped for this). Reference-free features per reading: correction frame (FRAME on blocks 52/60), question-word
overlap (share of the question's content words found in the reading), acceptance cue (congratulat/wonderful/great that),
CJK share; per item for (a) one read (seed 1) and (b) the 4-read consensus (frame by majority, overlap and cue on the
words present in all 4 readings). Targets on the training questions only (5-fold cross-fit, test labels unused): FPQ
label and Plain failure. Read-out: cross-fit AUC of p_fpq alone vs p_fpq + features (logistic regression), for (a) and
(b). Pre-registered reading: the lens adds to the probe if p_fpq + consensus features beats p_fpq alone by >= 0.03 AUC
for Plain failure or FPQ, and consensus beats the single read. dev100 is a second check; test reads are stored for
later use only.
E34 amendment (19:12 KST, user, during the reads and before any analysis): consensus words = words in at least 3 of the 4 readings (simple stemming), with the all-4 intersection also reported; one more feature per reading: the number of stable content words the question does not contain. Criterion unchanged (p_fpq + 3-of-4 consensus features).
E31 hesitation transfer (19:24 KST; e31_hesitation_transfer.py; hesitation = 1-3 of 4 samples passing vs 0 or 4; e13 probe on Plain states). Trained on FPQ -> NFP train AUC 0.623, NFP dev 0.422 (dev n 20, 4 positive: uninformative). Trained on NFP -> FPQ train 0.719, FPQ dev 0.636. Pooled train -> dev all 0.620. Within-label results were 0.76 (FPQ train CV) / 0.79 (FPQ dev). Reading: FPQ and NFP hesitation share part of one internal direction (transfer 0.62-0.72), but not all of it.

### E34 result (19:27 KST)
Reads: 498 items x 4 seeds, all complete. Training cross-fit AUC (n=156): FPQ - p_fpq 0.765, single 0.626, consensus(>=3/4) 0.647, p_fpq+single 0.734, p_fpq+consensus 0.724, p_fpq+consensus(all 4) 0.720; Plain failure - p_fpq 0.658, single 0.658, consensus 0.626, p_fpq+single 0.667, p_fpq+consensus 0.640, p_fpq+consensus(all 4) 0.617. dev100: FPQ p_fpq 0.747, consensus 0.672, p_fpq+consensus 0.727; Plain failure p_fpq 0.434, single 0.550, consensus 0.584, p_fpq+consensus 0.522. Frame flip single vs 4-read majority: 5.1%.
Verdict: criterion NOT met (p_fpq+consensus is below p_fpq alone on both targets; consensus beats single for FPQ by 0.021 only, not for Plain failure). Test reads stored, unused.

### E34 exploratory notes (19:36 KST, after the result; train+dev100, n=256)
- Frame errors are stable, not sampling noise: FPQ with 0/4 correction frames 90/158; NFP with >=3/4 frames 19/98 (4/4: 12).
- Off-question content words: 53% appear in 1 of 4 reads; stable ones include lens-prior words (daughter/children in >=3/4 reads for 41-42% of questions lacking them, vs father/sister 6%) and wrong diseases (colon, crohn).
- Reference-correction words (not in question, not lens-common) found in the >=3/4 consensus: 7.3% vs 3.6% with another FPQ's consensus; 21.6% in any read. Topic words appear, decisive facts (asymptomatic, transmitted) mostly not.

## E35: E29 loop with repairs (written 20:51 KST, code only, before any E35 run)
Audit (another session, results/mce_improvement_audit_20261026 in ~/cancer-myth; confirmed in code): (1) at the iteration-3
restart the context was restored to the dev-best version but data/memory.md still said "workspace holds v<last>"
(meta inputs wrong in all 5 E29 B runs and A_c, A_e); (2) meta-written SKILLs cited meta-folder files the base workspace
lacks (internal_outcomes.md in 4 B runs; meta/history.json etc. in 2 A runs); (3) the SKILL template required two
supporting training questions per knowledge entry. Repairs: e35_loop.py re-renders memory.md for the restored version
before the meta step and copies the meta's history.json / internal_outcomes.md into the base data/ with a path map in
the base prompt; e35_workspaces.py replaces the two-question rule (an entry may come from one question if it can apply to
unseen questions and states when it applies; both arms). Not yet decided: local edits with fixed/broken reports.
E29/E30 B results are therefore not a clean measure of the lens arm.
Summary-prompt regression check (21:22 KST; scratch state2_check.py, results state2_check20_r1.json / _r2.json; 20 stored E29 B
fixed-batch readings with context, 10 FPQ / 10 NFP, same items both rounds). r1 = observation/comparison split, evidence
layers, context_alignment, reading_not_in_answer items. r2 = r1 + topic overlap is not premise evidence (wrong disease/
person -> related, mixed_or_unclear when the target premise is left open; refusal is not a correction) + reading_vs_answer
with closest answer quote and a meaning verdict. r1 errors and r2: fpq_191 claim already in the answer -> r2 verdict
already_in_answer (fixed); off-topic-detail readings called unrelated (4) -> r2 mixed_or_unclear (0 unrelated); refusal
counted toward correction (nfp_1016) -> mixed_or_unclear. Remaining error: nfp_1082 verdict additional_related although
its own answer quote states the same surgery mismatch. Quotes verbatim 92/92 (string match only). Not tuned further.

## E35 main run (written 21:28 KST, before any E35 main-run result; user-approved)
Design: e35_workspaces + e35_loop (repairs 1-2 for both arms; repair 3 = round-2 state summary with answer and context,
claims judged already_in_answer dropped, internal arm only), arms E29A / E29B, 3 runs each (reps e35a-c), E29 sizes
(fixed 40, rotating 2 x 25 per iteration, 6 edits, meta before edits 3 and 5, dev100 after edits 2/4/6, dev-best version
to fold-0 test). Servers: vLLM 8802/8803/8804 (TP2; restarted 21:43 KST with the third server, first start aborted after edit 1), lens 8796. Primary: mean test macro B - A, paired by rep, 95% t interval;
read as a lens effect only if B - A >= +1 with the interval above 0. Secondary: per-edit fixed / broken counts on the
fixed batch, knowledge entries written, B decisions that cite reading_vs_answer or context_alignment. Comparison with E29
(A 66.6, B 68.3) mixes the repairs and the relaxed knowledge rule; it is descriptive only. Expected: both arms near E29;
B - A within +-2.
E35 result (22:18 KST; e35_summary.json). Fold-0 test macro: A 68.3 / 70.6 / 69.1, mean 69.33 (SD 1.17; FPQ 61.8, NFP 76.9);
B 67.8 / 64.1 / 66.0, mean 65.97 (SD 1.85; FPQ 53.0, NFP 78.9). Paired B - A by rep -0.5, -6.5, -3.1, mean -3.37, 95% t
[-10.8, +4.1]: criterion not met (B not above A). Fixed-batch edit totals (18 edits per arm): A fixed 39 / broke 45,
B fixed 38 / broke 39. B decisions cite the new state fields in 2 of 3 runs (4, 6 mentions). Descriptive vs E29: A 66.6 ->
69.3, B 68.3 -> 66.0 (repairs + relaxed knowledge rule; not a controlled comparison). Secondary criterion was proposed
after two A runs had finished and was not adopted; edit counts are descriptive.

## E36: E35 with a larger dev set (written 22:46 KST, before any E36 run; user-requested)
Split: 20 training NFP (seed 3600) moved to dev -> train FPQ 78 / NFP 58, dev FPQ 80 / NFP 40 (items_e36.json); test
unchanged. Plain dev on the new dev: macro 60.6 (FPQ 43.8, NFP 77.5), shown to the meta as v0. Everything else E35
(e36_loop wraps e35_loop; e35_workspaces; arms A/B, 3 runs each, reps e36a-c; vLLM 8802-8804, lens 8796). Question:
does a larger dev NFP set make dev selection track test better and change A/B? Read-outs: test macro per arm, B - A
(paired, 95% t), dev-test correlation of the selected versions pooled with E35 (descriptive, n small).
Correction (22:56 KST, from a review): the E35 B state summary (and the 20-reading prompt check) received the answer cut to
its first 3,000 characters; 1,041 of 1,290 E35 B records were longer. reading_vs_answer verdicts ("additional_related")
are therefore not checked against the full answer. Also, the "13%" (E29 B, reading status changed while pass/fail stayed)
compares status labels with pass/fail only; it is not a verified rate of readout content absent from the answer.
E35/E36 agent failures (23:00 KST): codex calls ended with "429 Too Many Requests" (turn.failed, no edit made) and the loop did
not detect it. E35: 11 of 50 calls failed (A_a 2, A_b 4, A_c 0, B_a 1, B_b 1, B_c 3; edits and meta steps). E36: 25 of 33
calls failed; A_e36a / A_e36b made no edit at all (test 64.1 = Plain). E29 (88 calls) and E30 (25) had none. E36 stopped
by the user at 22:58 KST (B runs unfinished); E36 results are invalid. E35 results stand only as a compromised run: the
arms lost different edits. Needed before any rerun: detect turn.failed and retry with backoff; lower concurrency.

## E37: with-context hidden state as a test-time selector (written 23:42 KST, before any E37 computation; user-approved)
Question: can get_context use the hidden state of (question + candidate context), without generation, to decide where a
learned context helps on new questions? e37_state_selector.py: fold-0 test 242 x {Plain + 14 E25 contexts}, states from
/hidden (layers 20..60); pass probe (e13 recipe) and selector LR(s, s - s0, p_fpq) fit on training inputs with ratings
(e33_precheck.history: E13 training pairs + E32 records); stored E25 answers and ratings only (aborts if any is missing).
Policies: all, none, p_fpq >= 0.5, state gain > 0, p_fpq >= 0.5 and gain > 0, choose-best-context. Primary: mean over the
14 contexts of the test macro, state gain > 0 vs p_fpq >= 0.5 (E25: 68.8), question-bootstrap 95% CI. Read as useful if
the difference is >= +1 with the CI above 0; secondary: the AND policy and choose-best. No lens features (Oracle-lens
reading of the same inputs is a follow-up only if the state selector helps). Expected: close to p_fpq (gain mostly
tracks p_fpq); a small gain from NFP protection is possible.
E37 result (23:56 KST; e37.json; 736 training inputs, pass rate 0.586; 3,388 test states extracted, 978 stored). Mean test
macro over the 14 contexts: all 65.58, none 64.15, p_fpq >= 0.5 68.77, state gain > 0 66.09 (attach FPQ 58.0% / NFP
60.5%), p_fpq >= 0.5 and gain > 0 68.02 (34.5% / 13.8%); choose-best-context 66.32. Difference state gain - p_fpq -2.68
[-7.03, +1.04]; AND - p_fpq -0.75 [-1.72, +0.19]. Criterion not met: the with-context state selector does not beat the
probe gate on new questions. Note (review, same time): E16's internal-over-output advantage had already been corrected
(audit: output baseline 0.718 vs 0.739 with internal signals, interval including 0), so E37 was not strongly motivated;
also E22's knowledge arm was best in both lens groups, and "unaware" means no correction plan observed, not missing
knowledge.
E29 B decisions count (06:43 KST, regex over each edit's added decisions.md lines, 30 edits): all 30 mention readings; 25
contain an explicit "the reading adds nothing (relevant/reliable) beyond the answer/output" statement; 1 (B_c v3)
states a readout detail beyond the output (the reading endorses the age-based premise). Replaces the earlier
qualitative "mostly conclude that the reading adds nothing". The E29 B prompt asked for this statement ("If no relevant
additional evidence is supported, say so"). Regex-based; misses paraphrases.

## E38: repeatability of question x context outcomes (written 06:52 KST, before any E38 generation; user-approved)
e38_pair_noise.py: 40 fold-0 test questions with mixed outcomes over the 14 E25 contexts (30 FPQ / 10 NFP, seed 3800) x 4
contexts (lowest, 1/3, 2/3, highest pass rate on them) = 160 pairs; 3 fresh observations per pair (greedy regeneration
without the answer cache + fresh Luna rating without the judge cache) plus the stored one. Diagnostic only (no
selection, no method fit on test). Read-outs: variance shares (question, context, question x context, noise); share of
stored "A pass / B fail" splits reproduced in the fresh observations. Cost: 480 generations, 480 Luna calls
(concurrency 8). Expected: noise is a large part of the 39% question x context + noise share seen on all 14 contexts.
E38 result (10-27 07:32 KST; e38.json, e38_obs.jsonl): 40 questions x 4 contexts (OUT2_b_v2, OUT3_c_v2, FTY_e_v1, OUT3_e_v1; pass
0.45-0.70 on them), 4 observations per pair. Regenerated greedy text identical to the stored one 2.7%. Variance shares
of one observation (components estimated separately, sum 1.14): question 0.422, context 0.044, question x context 0.469,
noise 0.205. Stored "A pass / B fail" splits: 83; in the fresh observations A minus B pass rate +0.687, reproduced
(A > B) 81.9%, reversed 1.2%. Contrary to the expectation, most question x context splits are repeatable, not noise.
The questions were chosen for mixed outcomes, so the interaction share is higher than on all test questions.

## E39: which context for which question - training grid and selectors (written 10-27 07:52 KST, before any E39 result; user-approved)
Motivation: E38 found question x context splits largely repeatable (82% reproduced). Step 1 (e39_train_grid.py): 156
training questions x {Plain + 14 E25 contexts}, greedy + Luna Well via answer_and_rate (cached where E32 / earlier runs
answered the same input). Step 2 (written before step-2 code is run): selectors that pick, per question, one context or
none, trained on the training grid only and scored once on the stored fold-0 test grid (E25): (a) strong output
baseline that does not read the question (best single context / p_fpq gate as in E25), (b) question text, (c) Plain
hidden state (probe recipe), (d) hidden state of question + context, (e) (c/d) + Oracle-lens features. Primary: test
macro of the chosen inputs vs the p_fpq gate with the best training context; lens counts as useful only if (e) beats
(c/d) with a question-bootstrap CI above 0.

## E40: Oracle-lens reading at the answer's most uncertain token (written 10-27 08:10 KST, before any E40 result; user-requested)
e40_entropy_read.py, training only. Plain greedy answers of the 156 training questions; token entropy from vLLM prompt
logprobs (top 5, renormalised) over answer tokens 2-127; reading (fixed reader, four depths) just before the
highest-entropy token t* and before a random token t_r (seed 4000); stored pre-answer reading t0. Text baseline: the
correction-frame regex on the answer prefix. Read-outs: reading frame at t* vs t_r when the prefix has none, per label x
Plain pass; AUC of "sometimes corrected" among FPQ failing Plain (E31 labels) for entropy alone (output-side
baseline), reading frame at t*, and both (5-fold LR); same for passing NFP and "sometimes over-corrected". The lens adds
at the uncertain token only if reading frame (or entropy + frame) beats entropy alone and the t_r reading.
Expected: small n (about 12 positives per group); entropy alone may carry most of the signal.
E40 amendment (10-27 08:13 KST, user, before any E40 result): primary read-out changed to "does the single greedy answer plus the
lens show the other path that sampling reveals?". Per training question with at least one E31 sample (T 0.8, 4 seeds)
whose pass differs from the greedy answer: other-path words = content words in the differing samples and not in the
greedy answer; reading-new words = content words in the reading and not in the greedy answer. Score = share of
reading-new words that are other-path words. Compared: reading at t* vs t_r vs t0; differing samples vs same-outcome
samples (specificity); other questions' differing samples (lens prior words). Claimed only if t* > t_r and differing >
same-outcome and own > other-question, with a question-bootstrap CI above 0 for t* differing - t* same-outcome.
The frame-rate read-outs stay secondary.
E40 amendment detail (08:14 KST, before any result): other-path and same-path word sets each come from one sample (the lowest seed of that outcome), so set sizes are comparable.
E40 result (10-27 08:38 KST; e40.json, e40_other_path.json). Secondary: reading frame at t* when the prefix has none - FPQ failing
0.000 (t_r 0.020), FPQ passing 0.111 (0.000), NFP 0.017 / 0.154; AUC "sometimes corrected" among 52 failing FPQ (11
positive): entropy 0.543, reading frame at t* 0.488, both 0.411, t_r 0.488; passing NFP "sometimes over-corrected" (59,
19): entropy 0.613, frame 0.526, both 0.542. Primary (50 questions with a differing sample, 22 FPQ / 28 NFP): share of
reading-new words that are in the differing sample - t* 0.195, t_r 0.181, t0 0.192; in the same-outcome sample - 0.188,
0.185, 0.156; other question's differing sample - 0.072, 0.063, 0.075. t* differing - same +0.004 [-0.029, +0.034]:
criterion not met. The reading's new content is question-specific (own 0.19 vs other question 0.07) but not specific to
the path sampling would take. (Descriptive, not pre-registered: t0 differing - same +0.036.)
E40 follow-up (10-27 09:20 KST, post hoc, exploratory): greedy and differing E31 samples first diverge at token 2 (median; 56% at
token <= 2), so the "divergence point" is in effect the pre-answer state t0. At t0 the other-path minus same-path overlap
is +0.039 [+0.016, +0.061] (46 questions; FPQ +0.035 [0.000, +0.068], NFP +0.042 [+0.014, +0.069]); with the two path
word sets subsampled to equal size +0.031 [+0.012, +0.051]. t* +0.004, t_r -0.008. Not pre-registered as primary; a
small, question-level signal that the pre-answer reading leans toward the path sampling can take; needs a fresh test.
E39 result (10-27 09:45 KST; e39.json). Fold-0 test macro of the chosen inputs (FPQ / NFP; attached FPQ % / NFP %): a1 best
training context (INT_a_v1) for all 66.32 (65.3 / 67.3); a2 same if p_fpq >= 0.5 69.07 (58.5 / 79.6; 54.9 / 20.4);
b question text 69.60 (61.7 / 77.6; 98.4 / 93.9); c Plain state 69.59 (59.6 / 79.6; 94.3 / 98.0); d question+context
state 69.90 (66.3 / 73.5; 100 / 100); e d + lens 70.38 (63.2 / 77.6; 99.5 / 100); oracle best candidate per question
92.28. e - d +0.49 [-4.08, +5.44]: criterion not met. b/c/d/e - a2: +0.53, +0.52, +0.82, +1.31, all intervals include 0.
Learned selectors attach a context to nearly every question and keep NFP near Plain by choosing among contexts.
