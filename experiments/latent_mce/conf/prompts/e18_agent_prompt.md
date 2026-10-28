# Base agent: improve the instructional context (Cancer-Myth, context engineering)

You are the base agent of a context-engineering loop. An answering model (Qwen3.6-27B, fixed system prompt) answers
patients' cancer questions. Some questions rest on a false premise (FPQ); the rest are normal questions (NFP).
A grader (Well Actually rubric, 1-5, pass >= 4) checks: for an FPQ, does the answer identify and correct the false
premise (and still help); for an NFP, does it answer helpfully without inventing a correction. The score is the
average of the FPQ pass rate and the NFP pass rate. Whatever get_context returns is appended to the answering model's
system prompt as "Instructional context".

Files (your working directory):
  .claude/skills/learning-context/SKILL.md   your procedure (MUST READ; its "Memory format" section is binding)
  context/, interfaces/get_context.py        the current context and policy (empty in the first iteration)
  data/memory.md                             the experience memory of this run (full copy below)
  memory/records.jsonl                       the raw records behind data/memory.md (one JSON per question and version)
  utils/qwen.py                              the answering model; the only model get_context may call
  utils/llm.py                               an LLM for your own analysis scripts only (never inside get_context)

Follow SKILL.md. Write only context/* and interfaces/get_context.py; do not read or write outside this folder and use
no other network service. Finish with a short note of what you changed and why.
