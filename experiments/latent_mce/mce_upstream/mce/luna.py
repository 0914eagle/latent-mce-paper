"""Minimal OpenAI Responses-API caller for gpt-6-luna (LatentGEPA, 2026-10-24).

Same request shape as cancer-myth results/paper_v4_qwen3/qwen3_patches.py `_openai` (model, input messages, store=False,
reasoning effort). The key is read from OPENAI_API_KEY or ~/.config/openai_api_key and is never logged.
"""
import json, os, time, urllib.error, urllib.request
from pathlib import Path

MODEL = os.environ.get("MCE_LLM_MODEL", "gpt-6-luna")
EFFORT = os.environ.get("MCE_LLM_EFFORT", "medium")


def _key():
    return os.environ.get("OPENAI_API_KEY") or (Path.home() / ".config/openai_api_key").read_text().strip()


def luna(messages, model=MODEL, effort=EFFORT, timeout=600, tries=6):
    """messages: str or list of {role, content}. Returns the output text."""
    if isinstance(messages, str):
        messages = [{"role": "user", "content": messages}]
    messages = [m for m in messages if m.get("content")]
    body = {"model": model, "input": messages, "store": False}
    if effort:
        body["reasoning"] = {"effort": effort}
    data = json.dumps(body).encode()
    for k in range(tries):
        req = urllib.request.Request("https://api.openai.com/v1/responses", data=data,
                                     headers={"Content-Type": "application/json", "Authorization": "Bearer " + _key()})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                out = json.load(r)
            texts = [c.get("text", "") for item in out.get("output", []) if item.get("type") == "message"
                     for c in item.get("content", []) if c.get("type") in ("output_text", "text")]
            return "".join(texts).strip()
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, ConnectionError) as e:
            if k == tries - 1:
                raise
            time.sleep(min(60, 5 * 2 ** k) if not (isinstance(e, urllib.error.HTTPError) and e.code == 429) else 60)
