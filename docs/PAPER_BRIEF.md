# 공저자용 안내: 연구 배경, 방법, 근거, 참고문헌

이 문서 하나로 논문 초안(특히 Method, Experimental Setup, Related Work)을 쓰기 시작할 수 있도록 정리했습니다.
작성 2026-10-28 KST. 투고 대상: AIM (마감 2026-10-30).

> **한 줄 요약.** GEPA나 MCE 같은 프롬프트·context 최적화 방법은 답 모델의 **출력과 점수만** 보고 배웁니다.
> 우리는 답 모델의 **내부 상태**(hidden state)를 읽어서 최적화 과정에 넣으면 무엇이 달라지는지 봅니다.
> 방법의 중심은 **latent MCE**입니다. 원본 MCE 루프는 그대로 두고, agent가 받는 피드백에 Oracle lens 판독을 더합니다.

---

## 0. 먼저 읽을 것 (10분)

| 순서 | 파일 | 무엇을 알 수 있나 |
|---|---|---|
| 1 | 이 문서 | 왜, 무엇을, 어떻게 |
| 2 | [`METHOD_SETUP.md`](METHOD_SETUP.md) | Method / Setup 절에 그대로 옮길 설정값과 구현 파일 |
| 3 | [`RESULTS.md`](RESULTS.md) | 지금까지 나온 숫자와 그것을 만든 실험 (결과 절은 희재님과 정리) |
| 4 | [`../README.md`](../README.md) | 코드 구조와 실행 방법 |
| 참고 | [`../experiments/latent_mce/prereg/prereg_v4_test.md`](../experiments/latent_mce/prereg/prereg_v4_test.md) | 실험마다 미리 적은 계획과 결과 기록 (E1–E40) |

---

## 1. 용어 (논문 전체에서 같은 말을 씁니다)

| 용어 | 뜻 |
|---|---|
| FPQ / NFP | false-premise question (틀린 전제가 있는 질문) / normal question (전제가 없는 정상 질문) |
| 교정 / 과잉 교정 | FPQ에서 틀린 전제를 짚는 것 / NFP에서 없는 문제를 지적하는 것 |
| macro | (FPQ 통과율 + NFP 통과율) / 2. 높을수록 좋음. Cancer-Myth의 주 지표 |
| Plain (Base) | 고정 system prompt만 쓰고 context를 붙이지 않은 답 |
| context / `get_context` | 질문마다 system prompt 끝에 붙이는 글 / 그 글을 고르는 파이썬 함수. 둘 다 MCE의 agent가 만듦 |
| skill | base agent가 따르는 작업 절차 문서. meta agent가 고침 |
| hidden state | 답 모델이 첫 답 토큰을 만들기 직전(마지막 prompt 토큰)의 층별 내부 벡터 |
| Oracle lens | hidden state를 자연어로 풀어 주는 고정 reader (LoRA). 결과 글을 "판독(reading)"이라 부름 |
| probe (`p_fpq`) | hidden state로 "이 질문이 FPQ인가"를 예측하는 선형 분류기 |
| latent MCE | 원본 MCE + agent 피드백에 Oracle lens 판독을 더한 것 (이 논문의 방법) |

---

## 2. 왜 이 연구를 시작했나

### 2.1 출발점: 출력만 보는 최적화의 한계
- **GEPA**와 **MCE**는 답 모델을 고치지 않고 prompt나 context를 고칩니다. 둘 다 답과 점수(그리고 채점 이유)를 보고
  다음 수정을 정합니다.
- 우리 문제(틀린 전제)에서 이 방식은 **상충**을 반복했습니다. 교정을 강하게 하라는 prompt는 FPQ를 고치지만 NFP를
  망칩니다. 예: Qwen3.6-27B, Cancer-Myth fold 0 test에서 GEPA prompt는 FPQ 48.7 → 65.3, NFP 79.6 → 71.4
  (`RESULTS.md`, E29 baseline). Cancer-Myth 원 논문도 GPT-4o에서 같은 현상(NFP 88 → 59)을 보고했습니다.
- 출력만 보면 "이 실패를 어떻게 고칠지"와 "어느 질문에 손대면 안 되는지"가 잘 보이지 않습니다.

### 2.2 내부 상태에는 출력에 없는 정보가 있다
- 모델은 틀린 전제를 "알고" 있어도 답에서 말하지 않는 경우가 많습니다. Qwen3-8B 점검에서 일반 힌트("전제가 틀렸다")만
  줘도 FPQ 교정이 12% → 37%로 오르고, 구체 힌트를 주면 72%까지 오릅니다. 병목은 지식보다 **언제 교정할지**였습니다.
- 답 시작 직전의 hidden state로 만든 probe는 FPQ를 텍스트 분류기보다 잘 가립니다. 이것으로 context를 **붙일지
  말지** 정하면(probe gate) NFP를 지키면서 FPQ 이득을 대부분 남깁니다 (`RESULTS.md` §1).
- 같은 질문에 답을 4번 뽑으면 FPQ의 27%는 가끔만 교정합니다. hidden state는 이런 "흔들리는 질문"을 텍스트보다
  잘 가립니다 (dev AUC 0.79 vs 0.51, E31). 관련 연구도 "모델은 알지만 말하지 않는다"는 현상을 보고합니다 (§7).

### 2.3 그래서 묻는 질문
> 최적화 agent가 답 모델의 **내부 상태**를 볼 수 있으면, 출력과 점수만 볼 때보다 더 좋은 context와 context 함수를
> 만드는가?

교수님 계획의 기여 구성(C1–C3)은 이렇습니다. 최종 문구는 결과에 맞춰 정해야 합니다.
- **C1:** 출력만 보는 최적화는 불완전하다 (상충, 선택 실패의 근거).
- **C2:** 답 모델의 내부 반응에는 출력에 없는 정보가 있다 (probe, 판독, 흔들리는 질문 분석).
- **C3:** 내부 상태를 피드백으로 받는 MCE (latent MCE). Oracle lens를 쓴다.

---

## 3. 과제와 데이터 (교수님 구성, 2026-10-28)

| Task | 데이터셋 | 주 지표 | 상태 |
|---|---|---|---|
| 1. 틀린 전제 | **Cancer-Myth** (주 실험), SynQA | macro | Cancer-Myth는 Base·MCE 완료, latent MCE 실행 준비 완료. SynQA는 데이터 확인 중 |
| 2. 의료 계산 | MedCalc-Bench | 최종 답 정확도 | 같은 설정으로 다시 돌려야 함 |
| 3. 검색 | HotpotQA, HoVer | EM·F1 / 근거 문서 전부 찾음 | GEPA 일부, MCE 코드만 |

- 모든 칸은 같은 설정이어야 합니다: Qwen3.6-27B, thinking off, greedy, 같은 분할, 같은 채점.
- 표 계획과 빈 칸 목록: `~/gepa_explainer/big_table_plan_2026-10-28.html` (서버의 설명 페이지).
- 이 저장소의 코드는 Task 1 (Cancer-Myth) 기준입니다.

**Cancer-Myth.** 환자가 묻는 암 관련 질문입니다. FPQ에는 전문가가 쓴 정답 교정(reference correction)이 있습니다.
예: "자궁경부암이라 자궁절제술을 받아야 하니 입양 기관을 추천해 달라" → 정답 교정: 초기에는 가임력을 보존하는
치료가 있다. 좋은 답은 FPQ에서는 전제를 짚고, NFP에서는 바로 도움을 줍니다.

---

## 4. 방법

### 4.1 배경: MCE (Meta Context Engineering)
두 단계 agent 구조입니다 (Ye et al., 2026).
- **meta agent:** 이전 반복의 skill과 점수를 보고 **skill**(작업 절차 문서)을 고칩니다.
- **base agent:** skill을 따라 training 결과를 읽고 `context/` 파일과 `get_context` 함수를 고칩니다.
- 반복마다 training 묶음을 답하고 채점해서 base agent에게 줍니다. validation 점수가 가장 높은 반복을 test에 씁니다.
- 우리는 공식 코드(c4b7a7c)를 **수정 없이** 씁니다 (`experiments/latent_mce/mce_upstream`). 과제만 환경으로 등록합니다.

### 4.2 latent MCE (제안 방법)
원본 MCE와 다른 점은 **agent가 받는 피드백에 판독이 더해진다**는 것 하나입니다.

```
(1) 판독 만들기 (training 질문마다 한 번, 답 생성 없음)
    질문 + 고정 system prompt → Qwen3.6-27B → 마지막 prompt 토큰의 hidden state (block 36/44/52/60)
                                            → Oracle lens → 판독 4개 (짧은 글)
(2) 피드백에 넣기
    base agent: 학습 기록마다 질문·라벨·정답 교정·답·점수 + internal_reading
    meta agent: meta_agent/train.jsonl의 질문마다 internal_reading, 이전 반복 기록의 위치 안내
(3) 나머지는 원본 MCE와 같음. get_context(question)도 같음 → test 때 내부 정보를 쓰지 않음
```
- 판독은 Luna로 요약하지 않고 원문 그대로 줍니다. 해석은 agent가 합니다.
- validation·test 질문은 lens로 읽지 않습니다. 판독은 학습 피드백일 뿐입니다.
- 구현: `src/experiment/mce_cancer_env.py` (`MCE_LATENT=lens`), 실행: `run_mce_cancer.sh`.
- 그림 초안: `~/gepa_explainer/latent_mce_method_figure_2026-10-28.html` (서버).

### 4.3 Oracle lens
- Qwen3.6-27B 위의 LoRA reader 두 개로 학습한 도구입니다 (내부 프로젝트 `oracle-lens`, 공개 논문 없음).
  - AR(activation reconstructor): 글 → 그 글 앞의 activation을 예측. 먼저 학습한 뒤 고정합니다.
  - AO(activation oracle): activation → 글. AR을 보상으로 삼아 SFT와 RL(GRPO)로 학습합니다.
- 우리는 학습된 reader(`olens_s3d_rl600`)를 고정해 씁니다. reader prompt와 설정(T 0.8, top-p 0.95, 128 토큰)은 바꾸지 않습니다.
- 관련 공개 연구: Activation Oracles (Karvonen et al.), 개선판 (Bauer et al.), LatentQA, Patchscopes (§7).

### 4.4 probe gate (별도 방법이자 비교 분석)
- hidden state(층 20–60, 11개 층)에 층별 로지스틱 회귀를 학습하고 평균합니다. training 문항만 씁니다.
- test 때 `p_fpq ≥ 0.5`인 질문에만 학습된 context를 붙입니다. 나머지는 Plain입니다.
- latent MCE에는 들어가지 않습니다. 논문에서는 "내부 상태가 붙일지 말지를 정할 수 있다"는 C2 근거로 씁니다.

---

## 5. 실험 설정 (요약, 자세한 값은 `METHOD_SETUP.md`)

| 항목 | 값 |
|---|---|
| 답 모델 | Qwen3.6-27B, vLLM 0.18.1, greedy, thinking off, 최대 2048 토큰 |
| context 위치 | system prompt 끝 `"Instructional context: …"` (user 메시지에 넣으면 NFP가 약 20점 떨어짐, E31) |
| 채점 | gpt-6-luna, Well rubric(라벨별), 1–5점, 통과 ≥ 4, 블라인드·캐시 |
| agent | gpt-6-luna, `codex` CLI, reasoning effort medium |
| MCE 설정 | 5 반복, training 묶음 50, training 100(50/50), validation 96(48/48), fold 3개 |
| 비교 | Plain, GEPA, MCE, latent MCE (+ 분석: probe gate) |
| 하드웨어 | RTX A6000 48 GB × 8. 모델 서버 하나에 GPU 2장 |

---

## 6. 절별 쓰기 안내

| 절 | 쓸 내용 | 근거 파일 |
|---|---|---|
| Introduction | §2의 흐름: 출력 기반 최적화 → 상충 → 내부 상태의 추가 정보 → latent MCE | 이 문서 §2, `RESULTS.md` §1–3 |
| Related Work | §7의 네 갈래 | §7 |
| Method | §4.1 MCE 요약 → §4.2 latent MCE (그림) → §4.3 Oracle lens → (선택) §4.4 probe gate | `METHOD_SETUP.md` §3–4, `mce_cancer_env.py` |
| Experimental Setup | 과제·데이터·지표, 답 모델, 채점, agent, MCE 설정, 기준선 | `METHOD_SETUP.md` §1–2, §4–5 |
| Results / Analysis | 큰 표 (task × 방법), probe gate, 판독 분석, 흔들리는 질문 | `RESULTS.md`, 표 계획 페이지 |
| Limitations | 편차, dev 크기, LLM 채점, vLLM 비결정성, 판독 잡음 | `METHOD_SETUP.md` §6 |

**주의**
- 설정이 다른 숫자를 한 표에 섞지 않습니다 (thinking on/off, 채점기, 예산, 분할).
- "lens가 도왔다"고 쓰지 않습니다. 현재 근거로는 "내부 상태가 붙일지 말지를 정하는 데 도왔다"까지입니다.
- 원본 MCE는 "수정 없이 사용"이라고 씁니다. 우리가 바꾼 것은 환경(과제)과 피드백 내용뿐입니다.

---

## 7. 참고문헌 (분야별)

arXiv 번호는 프로젝트 문서에서 확인한 것입니다. "(확인 필요)"는 번호를 직접 확인하지 못한 것입니다.

**A. 프롬프트·context 최적화**
- Ye et al., *Meta Context Engineering via Agentic Skill Evolution*, 2026. arXiv 2601.21557. 코드 github.com/metaevo-ai/meta-context-engineering
- Agrawal et al., *GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning*, 2025. arXiv 2507.19457 (확인 필요)
- Zhang et al., *Agentic Context Engineering (ACE)*, 2025. arXiv 2510.04618 (확인 필요)
- Zhang et al., *Rep2Skill: Representation-Guided Skill Self-Evolution for LLM Agents*, 2026. arXiv 2609.39149 — 가장 가까운 선행 연구. 내부 표현으로 agent skill을 고침 (여러 턴 과제, 같은 작은 모델이 최적화)

**B. 내부 상태를 글로 읽는 도구 (verbalizer)**
- Karvonen et al., *Activation Oracles*, 2025. arXiv 2512.15674
- Bauer et al., *Building Better Activation Oracles*, 2026. arXiv 2606.02609
- LatentQA (Pan et al., 2024), Patchscopes (Ghandeharioun et al., 2024) (확인 필요)

**C. 모델은 알지만 말하지 않는다 / probe**
- Orgad et al., *LLMs Know More Than They Show*, ICLR 2025. arXiv 2410.02707
- Burns et al., *Discovering Latent Knowledge in Language Models Without Supervision*, ICLR 2023. arXiv 2212.03827
- Li et al., *Inference-Time Intervention*, NeurIPS 2023. arXiv 2306.03341
- Ferrando et al., *Do I Know This Entity?*, ICLR 2025. arXiv 2411.14257
- Pandey et al., *LLMs Know They're Wrong and Agree Anyway*, 2026. arXiv 2604.19117
- *Detection Without Correction*, 2026. arXiv 2604.13068 · *Perfect Detection, Failed Control*, 2026. arXiv 2606.24952
- Sharma et al., *Towards Understanding Sycophancy in Language Models*, 2023. arXiv 2310.13548

**D. 틀린 전제 데이터와 과제**
- Zhu et al., *Cancer-Myth: Evaluating LLMs on Patient Questions with False Presuppositions*, ICLR 2026. arXiv 2504.11373
- *Well Actually* (Well rubric의 출처), 2026. arXiv 2608.06539
- Syn-QA² , arXiv 2403.12145 · CREPE (Yu et al., ACL 2023) · FalseQA (Hu et al., ACL 2023)
- *LLMs Struggle to Reject False Presuppositions when Stakes are High*, 2025. arXiv 2505.22354
- MedCalc-Bench (Khandekar et al., 2024), HotpotQA (Yang et al., 2018, arXiv 1809.09600), HoVer (Jiang et al., 2020, arXiv 2011.03088) (확인 필요)

더 넓은 목록과 요약: 비공개 저장소 `0914eagle/cancer-myth-internals`의 `docs/07_related_work_2026.md`,
`docs/38_research_map_and_related_work_2026-09-23.md`.

---

## 8. 어디에 무엇이 있나

| 무엇 | 어디 |
|---|---|
| 이 저장소 (논문용 코드) | github.com/0914eagle/latent-mce-paper |
| 전체 프로젝트 저장소 | github.com/JakeFRCSE/LatentGEPA (비공개; 이 변경은 branch `latent-mce-feedback`) |
| 결과 데이터 | HF 비공개 dataset `latentgpea/latent-feedback-cells-data` |
| 설명 페이지 (그림, 표 계획, 결과 페이지) | 서버 `~/gepa_explainer/` |
| Overleaf | 희재님께 받기 |

질문은 희재님께 주시면 됩니다. 숫자를 쓸 때는 항상 `RESULTS.md`의 출처(E 번호)를 함께 적어 주세요.
