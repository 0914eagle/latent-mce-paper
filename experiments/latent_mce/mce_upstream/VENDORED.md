# mce_upstream (vendored)

Source: github.com/metaevo-ai/meta-context-engineering, commit c4b7a7c (MIT, see LICENSE). The figures in `assets/`
are left out. Changed or added files (ours), everything else is byte-identical to the source commit:

| File | Change |
|---|---|
| `mce/luna.py`, `mce/codex_agent.py` (added) | gpt-6-luna calls through our OpenAI-API path; base and meta agents run as `codex exec` |
| `mce/base_agent.py`, `mce/meta_agent.py`, `mce/llm_client.py`, `mce/utils.py` | use the two files above instead of the Claude Agent SDK / OpenRouter |
| `mce/workspace_utils/llm.py`, `mce/workspace_utils/embedding.py` | Luna port of the utilities the base agent may call inside its workspace (`call_llm`, `compute_embedding_similarity`); the upstream LangChain version is kept as `workspace_utils_llm_upstream_langchain.py.txt` |

`experiments/latent_mce` uses only `mce/workspace_utils/` (copied into every agent workspace by
`src/prepare/e29_workspaces.py`, so both arms have the same tools as the original MCE). The rest is here for the
planned MCE reproduction on Symptom2Disease (`env/symptom_diagnosis/data`: train 200 / val 50 / test 212).
