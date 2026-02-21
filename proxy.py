# proxy_server.py
#
# OpenAI-compatible proxy for /v1/chat/completions (stream + non-stream)
# Single model enforced. While streaming, it parses SSE chunks token-by-token,
# classifies each emitted delta as TEXT vs TOOL (only if request.tools is defined),
# appends to a log file, and forwards the chunk unchanged to the client.
#
# Env:
#   OPENAI_API_KEY=...
#   ALLOWED_MODEL=gpt-4.1-mini
#   UPSTREAM_BASE=https://api.openai.com
#   LOG_FILE=/tmp/proxy_chunks.jsonl
#
# Run:
#   pip install fastapi uvicorn httpx
#   uvicorn proxy_server:app --host 0.0.0.0 --port 8000

import os
import json
import time
from typing import Any, Dict, Optional, Tuple, List

from dotenv import load_dotenv

load_dotenv()

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

UPSTREAM_BASE = os.environ.get("UPSTREAM_BASE", "https://api.openai.com").rstrip("/")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
ALLOWED_MODEL = os.environ.get("ALLOWED_MODEL", "local-proxy-model")
LOG_FILE = os.environ.get("LOG_FILE", "/tmp/proxy_chunks.jsonl")

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY is required")

app = FastAPI(title="OpenAI Proxy (single-model + chunk logger)", version="0.2.0")


def _now() -> int:
    return int(time.time())


def _auth_headers() -> Dict[str, str]:
    return {"Authorization": f"Bearer {OPENAI_API_KEY}"}


def _filter_hop_by_hop(headers: Dict[str, str]) -> Dict[str, str]:
    drop = {
        "host",
        "connection",
        "content-length",
        "content-type",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailer",
        "transfer-encoding",
        "upgrade",
    }
    return {k: v for k, v in headers.items() if k.lower() not in drop}


def _enforce_single_model(payload: Dict[str, Any]) -> Dict[str, Any]:
    payload["model"] = ALLOWED_MODEL
    return payload


def save_chunk_append_to_file(raw_chunk: Dict[str, Any], kind: str) -> None:
    """
    Appends one JSONL line:
      {"ts":..., "kind":"text|tool|other", "chunk":{...}}
    """
    record = {"ts": _now(), "kind": kind, "chunk": raw_chunk}
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def classify_chunk_as_tool_or_text(chunk_obj: Dict[str, Any], tools_defined: bool) -> str:
    """
    For chat.completion.chunk events:
      - If tools_defined and any choice.delta.tool_calls exists -> "tool"
      - Else if any choice.delta.content exists -> "text"
      - Else -> "other"
    """
    choices = chunk_obj.get("choices") or []
    for c in choices:
        delta = (c or {}).get("delta") or {}
        if tools_defined and delta.get("tool_calls"):
            return "tool"
        if isinstance(delta.get("content"), str) and delta["content"] != "":
            return "text"
    return "other"


def _iter_sse_events(byte_iter):
    """
    Minimal SSE parser for OpenAI-style streams.
    Yields each full SSE event (as bytes), including trailing \n\n.
    """
    buf = b""
    for chunk in byte_iter:
        if not chunk:
            continue
        buf += chunk
        while True:
            sep = buf.find(b"\n\n")
            if sep == -1:
                break
            event = buf[: sep + 2]
            buf = buf[sep + 2 :]
            yield event
    if buf:
        # last partial (shouldn't happen for well-formed SSE)
        yield buf


def _extract_data_lines(event_bytes: bytes) -> List[bytes]:
    """
    Returns list of 'data:' payloads for this event (bytes after 'data: ').
    """
    lines = event_bytes.split(b"\n")
    out = []
    for ln in lines:
        if ln.startswith(b"data:"):
            # allow "data:" or "data: "
            payload = ln[5:]
            if payload.startswith(b" "):
                payload = payload[1:]
            out.append(payload)
    return out


@app.get("/v1/models")
async def models():
    return {
        "object": "list",
        "data": [
            {
                "id": ALLOWED_MODEL,
                "object": "model",
                "created": _now(),
                "owned_by": "proxy",
            }
        ],
    }


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    payload = await request.json()
    payload = _enforce_single_model(payload)

    upstream_url = f"{UPSTREAM_BASE}/chat/completions"

    incoming_headers = dict(request.headers)
    headers = _filter_hop_by_hop(incoming_headers)
    headers.update(_auth_headers())

    stream = bool(payload.get("stream", False))
    tools_defined = bool(payload.get("tools"))  # <- key switch for tool classification

    timeout = httpx.Timeout(connect=15.0, read=None if stream else 60.0, write=60.0, pool=60.0)

    if not stream:
        async with httpx.AsyncClient(timeout=timeout) as client:
            r = await client.post(upstream_url, headers=headers, json=payload)
        return JSONResponse(status_code=r.status_code, content=r.json())

    async def sse_passthrough_and_log():
        async with httpx.AsyncClient(timeout=timeout) as client:
            async with client.stream("POST", upstream_url, headers=headers, json=payload) as r:
                if r.status_code >= 400:
                    body = await r.aread()
                    yield body
                    return

                async for event in _aiter_sse_events(r):
                    for data_payload in _extract_data_lines(event):
                        if data_payload == b"[DONE]":
                            yield event
                            return

                        try:
                            obj = json.loads(data_payload.decode("utf-8"))
                            kind = classify_chunk_as_tool_or_text(obj, tools_defined)
                            save_chunk_append_to_file(obj, kind)
                        except Exception:
                            pass

                    yield event

    return StreamingResponse(sse_passthrough_and_log(), media_type="text/event-stream")


async def _aiter_sse_events(resp: httpx.Response):
    """
    Async wrapper around SSE parsing for httpx streaming.
    """
    buf = b""
    async for chunk in resp.aiter_bytes():
        if not chunk:
            continue
        buf += chunk
        while True:
            sep = buf.find(b"\n\n")
            if sep == -1:
                break
            event = buf[: sep + 2]
            buf = buf[sep + 2 :]
            yield event
    if buf:
        yield buf