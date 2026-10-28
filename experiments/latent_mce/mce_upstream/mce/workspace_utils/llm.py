"""LLM helper for the base agent's scripts (copied into each iteration workspace as utils/llm.py).

LatentGEPA (2026-10-24): the upstream version used langchain ChatOpenAI with OpenRouter (deepseek-chat-v3.1) and
structured output; this version calls gpt-6-luna through the OpenAI Responses API and asks for JSON matching the
Pydantic schema. Same public API: call_llm(prompts, schema=None), call_llm_async(prompts, schema).
The upstream file is kept as mce/workspace_utils_llm_upstream_langchain.py.txt (outside this folder, so agents do not see it).
"""
import asyncio, json, os, re, time, urllib.error, urllib.request
from pathlib import Path
from typing import List, Type, TypeVar, Union
from pydantic import BaseModel, Field

T = TypeVar('T', bound=BaseModel)
MAX_CONCURRENCY = 16
MAX_LLM_CALLS = 100
MODEL = os.environ.get("MCE_LLM_MODEL", "gpt-6-luna")
EFFORT = os.environ.get("MCE_LLM_EFFORT", "medium")


class TextResponse(BaseModel):
    """Simple text response from LLM."""
    response: str = Field(description="The LLM's response text")


def _luna(prompt: str) -> str:
    key = os.environ.get("OPENAI_API_KEY") or (Path.home() / ".config/openai_api_key").read_text().strip()
    body = json.dumps({"model": MODEL, "input": [{"role": "user", "content": prompt}], "store": False,
                       "reasoning": {"effort": EFFORT}}).encode()
    for k in range(6):
        req = urllib.request.Request("https://api.openai.com/v1/responses", data=body,
                                     headers={"Content-Type": "application/json", "Authorization": "Bearer " + key})
        try:
            with urllib.request.urlopen(req, timeout=600) as r:
                out = json.load(r)
            return "".join(c.get("text", "") for it in out.get("output", []) if it.get("type") == "message"
                           for c in it.get("content", []) if c.get("type") in ("output_text", "text")).strip()
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, ConnectionError):
            if k == 5:
                raise
            time.sleep(min(60, 5 * 2 ** k))


def _structured(prompt: str, schema: Type[T]) -> T:
    # Ask for one JSON object matching the schema; retry on parse errors
    ask = (prompt + "\n\nReturn ONLY one JSON object that matches this JSON schema (no prose, no code fence):\n"
           + json.dumps(schema.model_json_schema()))
    err = None
    for _ in range(3):
        text = _luna(ask)
        m = re.search(r"\{.*\}", text, re.S)
        try:
            return schema.model_validate_json(m.group(0) if m else text)
        except Exception as e:  # noqa: BLE001
            err = e
    raise ValueError(f"structured output failed: {err}")


async def call_llm_async(prompts: List[str], schema: Type[BaseModel]) -> List[T]:
    """Call the LLM on a batch of prompts with structured output (schema = Pydantic model)."""
    if len(prompts) > MAX_LLM_CALLS:
        raise ValueError(f"Number of prompts ({len(prompts)}) exceeds maximum allowed per batch ({MAX_LLM_CALLS})")
    sem = asyncio.Semaphore(MAX_CONCURRENCY)

    async def one(p):
        async with sem:
            return await asyncio.to_thread(_structured, p, schema)
    return await asyncio.gather(*[one(p) for p in prompts])


def call_llm(prompts: Union[str, List[str]], schema: Type[BaseModel] = None):
    """Synchronous wrapper. str -> str (or schema instance); list -> list."""
    single = isinstance(prompts, str)
    plist = [prompts] if single else prompts
    res = asyncio.run(call_llm_async(plist, schema if schema is not None else TextResponse))
    if schema is None:
        res = [r.response for r in res]
    return res[0] if single else res
