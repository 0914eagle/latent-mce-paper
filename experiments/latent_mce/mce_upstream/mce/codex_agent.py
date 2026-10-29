"""Drop-in replacement for the Claude Agent SDK client used by MCE's base and meta agents (LatentGEPA, 2026-10-24).

The upstream code (metaevo-ai/meta-context-engineering c4b7a7c) drives its agents with claude_agent_sdk.ClaudeSDKClient;
we have no Anthropic key, so each agent turn runs as a Codex CLI session (`codex exec`) with gpt-6-luna instead.
Same calling pattern: `async with CodexSDKClient(options) as c: await c.query(p); async for m in c.receive_response(): ...`
A follow-up query (validation feedback) resumes the same Codex thread (`codex exec resume <thread_id>`).
Sandbox: workspace-write in `cwd` with network on (skills call the LLM helper). The upstream per-tool permission
handler is not enforced; `writable_check` (optional) is run after each turn and reports writes outside allowed dirs.
A turn that ends without turn.completed (API error after Codex's own retries, crash, timeout, unreadable output) is
retried in place: the thread is resumed with a note to continue (or, before a thread exists, the prompt is sent again),
after a growing wait, up to MCE_AGENT_RETRIES times; the agent loop above never sees the failure as an exception.
"""
import asyncio, json, logging, os, tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, List, Optional

MODEL = os.environ.get("MCE_AGENT_MODEL", "gpt-6-luna")
EFFORT = os.environ.get("MCE_AGENT_EFFORT", "medium")
TIMEOUT = int(os.environ.get("MCE_AGENT_TIMEOUT", "3600"))
RETRIES = int(os.environ.get("MCE_AGENT_RETRIES", "8"))
RESUME_NOTE = "The previous turn stopped because of a connection or API error. Continue the same task from where it stopped."
logger = logging.getLogger(__name__)


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
        self.options, self.thread_id, self._pending, self._prompt = options, None, None, None

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def query(self, prompt: str):
        # Start (or resume) one Codex turn; output is read by receive_response()
        self._prompt = prompt
        await self._start(prompt)

    async def _start(self, prompt: str):
        cfg = ["-c", f'model="{MODEL}"', "-c", f'model_reasoning_effort="{EFFORT}"',
               "-c", "sandbox_workspace_write.network_access=true"]
        if self.thread_id is None:
            cmd = ["codex", "exec", "--skip-git-repo-check", "--json", "-C", self.options.cwd, "--sandbox", "workspace-write", *cfg, "-"]
        else:
            cmd = ["codex", "exec", "resume", "--skip-git-repo-check", "--json", *cfg, self.thread_id, "-"]
        env = dict(os.environ)
        # stderr to a file: an unread pipe blocks Codex once it fills
        self._stderr = tempfile.TemporaryFile()
        # limit: one JSON event per line, and an event can carry a whole file the agent printed (asyncio's default 64 KB
        # line limit stopped a meta-agent turn that printed a 1 MB train.json)
        self._pending = await asyncio.create_subprocess_exec(*cmd, stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
                                                             stderr=self._stderr, cwd=self.options.cwd, env=env,
                                                             limit=256 * 1024 * 1024)
        self._pending.stdin.write(prompt.encode()); await self._pending.stdin.drain(); self._pending.stdin.close()

    async def receive_response(self):
        # Stream Codex JSON events as CodexMessage objects until a turn completes, retrying failed turns in place
        for attempt in range(RETRIES + 1):
            done, why = False, ""
            try:
                async for kind, msg in self._read():
                    if kind == "done":
                        done = True
                    elif kind == "failed":
                        why = msg
                    else:
                        yield msg
            except Exception as e:
                why = f"{type(e).__name__}: {e}"
            if done:
                return
            self._stderr.seek(0)
            tail = self._stderr.read()[-600:].decode(errors="replace")
            if attempt == RETRIES:
                yield CodexMessage("error", f"codex turn failed {RETRIES + 1} times: {why} {tail}"[:2000])
                return
            wait = min(900, 60 * 2 ** attempt)
            logger.warning(f"codex turn failed ({why[:300]} | {tail[-300:]}); retry {attempt + 1}/{RETRIES} in {wait}s")
            yield CodexMessage("error", f"codex turn failed, retrying in {wait}s: {why}"[:2000])
            await asyncio.sleep(wait)
            await self._start(RESUME_NOTE if self.thread_id else self._prompt)

    async def _read(self):
        # One Codex process: ("msg", CodexMessage) per event, then ("done", None) or ("failed", reason)
        proc, completed, failure = self._pending, False, ""
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
                        yield "msg", CodexMessage("agent_message", it.get("text", ""))
                    elif kind == "command_execution":
                        yield "msg", CodexMessage("command_execution", f"$ {it.get('command','')}\n(exit {it.get('exit_code')}) {str(it.get('aggregated_output',''))[:2000]}")
                    else:
                        yield "msg", CodexMessage(kind, json.dumps(it)[:2000])
                elif t == "turn.completed":
                    completed = True
                    yield "msg", CodexMessage("turn", json.dumps(ev)[:2000])
                elif t in ("turn.failed", "error"):
                    failure = json.dumps(ev)[:1000]
                    yield "msg", CodexMessage("error", failure)
        finally:
            if proc.returncode is None:
                try:
                    proc.kill()
                except ProcessLookupError:
                    pass
            await proc.wait()
        yield ("done", None) if completed else ("failed", failure or f"exit {proc.returncode} without turn.completed")
