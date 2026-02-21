"""Async HTTPX client for the antislopfactory API."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable

import httpx


@dataclass
class ToolCall:
    id: str = ""
    name: str = ""
    arguments: str = ""


@dataclass
class Classification:
    label: str = ""
    probabilities: dict[str, float] = field(default_factory=dict)
    content_length: int = 0
    risk: int = 0
    final: bool = False


@dataclass
class StreamResult:
    content: str = ""
    reasoning_content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    classifications: list[Classification] = field(default_factory=list)
    final_classification: Classification | None = None
    done: bool = False


class AntiSlopClient:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip("/")
        self._timeout = httpx.Timeout(connect=15.0, read=60.0, write=60.0, pool=60.0)
        self._stream_timeout = httpx.Timeout(connect=15.0, read=None, write=60.0, pool=60.0)

    # ── simple endpoints ──────────────────────────────────────────

    async def health(self) -> dict:
        async with httpx.AsyncClient(base_url=self.base_url, timeout=self._timeout) as c:
            r = await c.get("/health")
            r.raise_for_status()
            return r.json()

    async def create_thread(self, system_prompt: str) -> str:
        async with httpx.AsyncClient(base_url=self.base_url, timeout=self._timeout) as c:
            r = await c.post("/threads", json={"system_prompt": system_prompt})
            r.raise_for_status()
            return r.json()["thread_id"]

    async def list_threads(self) -> list[dict]:
        async with httpx.AsyncClient(base_url=self.base_url, timeout=self._timeout) as c:
            r = await c.get("/threads")
            r.raise_for_status()
            return r.json()["threads"]

    async def get_messages(self, thread_id: str) -> dict:
        async with httpx.AsyncClient(base_url=self.base_url, timeout=self._timeout) as c:
            r = await c.get(f"/threads/{thread_id}/messages")
            r.raise_for_status()
            return r.json()

    async def search(self, query: str) -> list[dict]:
        async with httpx.AsyncClient(base_url=self.base_url, timeout=self._timeout) as c:
            r = await c.get("/search", params={"q": query})
            r.raise_for_status()
            return r.json().get("results", [])

    async def classify(self, text: str) -> dict:
        async with httpx.AsyncClient(base_url=self.base_url, timeout=self._timeout) as c:
            r = await c.post("/classify", json={"text": text})
            r.raise_for_status()
            return r.json()

    async def submit_tool_results(
        self, thread_id: str, results: list[dict[str, str]],
    ) -> dict:
        async with httpx.AsyncClient(base_url=self.base_url, timeout=self._timeout) as c:
            r = await c.post(
                f"/threads/{thread_id}/tool_results",
                json={"tool_results": results},
            )
            r.raise_for_status()
            return r.json()

    # ── streaming chat ────────────────────────────────────────────

    async def chat_stream(
        self,
        thread_id: str,
        message: str | None,
        tools: list[dict[str, Any]] | None = None,
        on_token: Callable[[str], None] | None = None,
        on_classification: Callable[[Classification], None] | None = None,
    ) -> StreamResult:
        body: dict[str, Any] = {"stream": True}
        if message is not None:
            body["message"] = message
        if tools:
            body["tools"] = tools

        result = StreamResult()
        tc_buf: dict[int, ToolCall] = {}

        async with httpx.AsyncClient(base_url=self.base_url, timeout=self._stream_timeout) as c:
            async with c.stream(
                "POST", f"/threads/{thread_id}/chat",
                json=body,
                headers={"Accept": "text/event-stream"},
            ) as resp:
                if resp.status_code >= 400:
                    body_bytes = await resp.aread()
                    raise httpx.HTTPStatusError(
                        f"HTTP {resp.status_code}",
                        request=resp.request,
                        response=resp,
                    )

                buf = b""
                async for chunk in resp.aiter_bytes():
                    if not chunk:
                        continue
                    buf += chunk

                    while True:
                        sep = buf.find(b"\n\n")
                        if sep == -1:
                            break
                        event_bytes = buf[:sep + 2]
                        buf = buf[sep + 2:]
                        self._process_event(
                            event_bytes, result, tc_buf, on_token, on_classification,
                        )

                if buf.strip():
                    self._process_event(
                        buf, result, tc_buf, on_token, on_classification,
                    )

        # finalize tool calls
        if tc_buf:
            result.tool_calls = [tc_buf[i] for i in sorted(tc_buf)]

        return result

    def _process_event(
        self,
        event_bytes: bytes,
        result: StreamResult,
        tc_buf: dict[int, ToolCall],
        on_token: Callable[[str], None] | None,
        on_classification: Callable[[Classification], None] | None,
    ) -> None:
        event_type = None
        data_lines: list[str] = []

        for line in event_bytes.decode("utf-8", errors="replace").split("\n"):
            if line.startswith("event:"):
                event_type = line[6:].strip()
            elif line.startswith("data:"):
                payload = line[5:].strip()
                data_lines.append(payload)

        for raw in data_lines:
            if raw == "[DONE]":
                result.done = True
                continue

            try:
                obj = json.loads(raw)
            except json.JSONDecodeError:
                continue

            if event_type == "classification":
                clf = Classification(
                    label=obj.get("label", ""),
                    probabilities=obj.get("probabilities", {}),
                    content_length=obj.get("content_length", 0),
                    risk=obj.get("risk", 0),
                    final=obj.get("final", False),
                )
                result.classifications.append(clf)
                if clf.final:
                    result.final_classification = clf
                if on_classification:
                    on_classification(clf)
            else:
                for choice in obj.get("choices") or []:
                    delta = (choice or {}).get("delta") or {}

                    content = delta.get("content")
                    if isinstance(content, str) and content:
                        result.content += content
                        if on_token:
                            on_token(content)

                    reasoning = delta.get("reasoning_content")
                    if isinstance(reasoning, str):
                        result.reasoning_content += reasoning

                    for tc in delta.get("tool_calls") or []:
                        idx = tc.get("index", 0)
                        if idx not in tc_buf:
                            tc_buf[idx] = ToolCall()
                        if tc.get("id"):
                            tc_buf[idx].id = tc["id"]
                        fn = tc.get("function") or {}
                        if fn.get("name"):
                            tc_buf[idx].name = fn["name"]
                        if fn.get("arguments"):
                            tc_buf[idx].arguments += fn["arguments"]
