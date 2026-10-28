"""Drop-in replacement for the Claude Agent SDK client used by MCE's base and meta agents (LatentGEPA, 2026-10-24).

The upstream code (metaevo-ai/meta-context-engineering c4b7a7c) drives its agents with claude_agent_sdk.ClaudeSDKClient;
we have no Anthropic key, so each agent turn runs as a Codex CLI session (`codex exec`) with gpt-6-luna instead.
Same calling pattern: `async with CodexSDKClient(options) as c: await c.query(p); async for m in c.receive_response(): ...`
A follow-up query (validation feedback) resumes the same Codex thread (`codex exec resume <thread_id>`).
Sandbox: workspace-write in `cwd` with network on (skills call the LLM helper). The upstream per-tool permission
handler is not enforced; `writable_check` (optional) is run after each turn and reports writes outside allowed dirs.
"""
import asyncio, json, os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, List, Optional

MODEL = os.environ.get("MCE_AGENT_MODEL", "gpt-6-luna")
EFFORT = os.environ.get("MCE_AGENT_EFFORT", "medium")
TIMEOUT = int(os.environ.get("MCE_AGENT_TIMEOUT", "3600"))


@dataclass
class ClaudeAgentOptions:
    cwd: str
    allowed_tools: List[str] = field(default_factory=list)
    setting_sources: Optional[List[str]] = None
    can_use_tool: Optional[Callable] = None


@dataclass
class CodexMessage:
    kind: str      # agent_message | command_execution | file_change | turn | error
    text: str


class CodexSDKClient:
    def __init__(self, options: ClaudeAgentOptions):
        self.options, self.thread_id, self._pending = options, None, None

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def query(self, prompt: str):
        # Start (or resume) one Codex turn; output is read by receive_response()
        cfg = ["-c", f'model="{MODEL}"', "-c", f'model_reasoning_effort="{EFFORT}"',
               "-c", "sandbox_workspace_write.network_access=true"]
        if self.thread_id is None:
            cmd = ["codex", "exec", "--skip-git-repo-check", "--json", "-C", self.options.cwd, "--sandbox", "workspace-write", *cfg, "-"]
        else:
            cmd = ["codex", "exec", "resume", "--skip-git-repo-check", "--json", *cfg, self.thread_id, "-"]
        env = dict(os.environ)
        # limit: one JSON event per line, and an event can carry a whole file the agent printed (asyncio's default 64 KB
        # line limit stopped a meta-agent turn that printed a 1 MB train.json)
        self._pending = await asyncio.create_subprocess_exec(*cmd, stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
                                                             stderr=asyncio.subprocess.PIPE, cwd=self.options.cwd, env=env,
                                                             limit=256 * 1024 * 1024)
        self._pending.stdin.write(prompt.encode()); await self._pending.stdin.drain(); self._pending.stdin.close()

    async def receive_response(self):
        # Stream Codex JSON events as CodexMessage objects until the turn ends
        proc = self._pending
        try:
            while True:
                line = await asyncio.wait_for(proc.stdout.readline(), timeout=TIMEOUT)
                if not line:
                    break
                try:
                    ev = json.loads(line)
                except json.JSONDecodeError:
                    continue
                t = ev.get("type")
                if t == "thread.started":
                    self.thread_id = ev.get("thread_id")
                elif t == "item.completed":
                    it = ev.get("item", {})
                    kind = it.get("type", "item")
                    if kind == "agent_message":
                        yield CodexMessage("agent_message", it.get("text", ""))
                    elif kind == "command_execution":
                        yield CodexMessage("command_execution", f"$ {it.get('command','')}\n(exit {it.get('exit_code')}) {str(it.get('aggregated_output',''))[:2000]}")
                    else:
                        yield CodexMessage(kind, json.dumps(it)[:2000])
                elif t in ("turn.completed", "turn.failed", "error"):
                    yield CodexMessage("turn" if t == "turn.completed" else "error", json.dumps(ev)[:2000])
        finally:
            await proc.wait()
