import asyncio
import json
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import httpx
import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from db import ThreadStore, genid
from util.classifier import MLPClassifier
from util.embed import embed

load_dotenv()

# ── config ────────────────────────────────────────────────────────────

MODEL_PATH = os.environ.get("MODEL_PATH", "data/safety_classifier.npz")
UPSTREAM_BASE = os.environ.get("UPSTREAM_BASE", "https://api.openai.com").rstrip("/")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
ALLOWED_MODEL = os.environ.get("ALLOWED_MODEL", "local-proxy-model")
CLASSIFY_INTERVAL = int(os.environ.get("CLASSIFY_INTERVAL", "500"))

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY is required")

# ── globals ───────────────────────────────────────────────────────────

clf: MLPClassifier | None = None
store = ThreadStore()

UNSAFE_LABELS = frozenset({
    "violence", "hate", "harassment", "sexual", "self_harm",
    "criminal", "malware", "illegal", "fraud", "harmful",
    "unethical", "unsafe_other",
})


# ── lifespan ──────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    global clf
    p = Path(MODEL_PATH)
    if p.exists():
        clf = MLPClassifier.load(str(p))
        print(f"loaded safety classifier ({clf.num_classes} classes)")
    else:
        print(f"warning: model not found at {p}")
    await store.initialize()
    print("database initialized")
    yield
    await store.close()


app = FastAPI(title="antislopfactory", lifespan=lifespan)


# ── schemas ───────────────────────────────────────────────────────────

class ClassifyRequest(BaseModel):
    text: str | list[str]


class ClassifyResult(BaseModel):
    text: str
    label: str
    probabilities: dict[str, float]


class ClassifyResponse(BaseModel):
    results: list[ClassifyResult]


class MakeThreadRequest(BaseModel):
    system_prompt: str


class MakeThreadResponse(BaseModel):
    thread_id: str


class ChatRequest(BaseModel):
    message: str
    tools: list[dict[str, Any]] | None = None
    stream: bool = True


# ── helpers ───────────────────────────────────────────────────────────

def _compute_risk(probs: dict[str, float]) -> int:
    """Map classification probabilities to a 0-10 integer risk score."""
    unsafe_sum = sum(probs.get(l, 0.0) for l in UNSAFE_LABELS)
    return min(10, int(unsafe_sum * 10))


async def _classify_text(text: str) -> dict[str, Any] | None:
    """Embed text and run safety classifier. Returns None if unavailable."""
    if clf is None or not text.strip():
        return None
    try:
        vecs = await embed(text)
        label = clf.predict(vecs)[0]
        probs = clf.predict_proba(vecs)[0]
        return {"label": label, "probabilities": probs}
    except Exception as e:
        print(f"classification error: {e}")
        return None


async def _build_upstream_messages(thread_id: str) -> list[dict[str, Any]]:
    """Reconstruct OpenAI-format messages from thread history."""
    db_messages = await store.get_messages(thread_id)
    messages: list[dict[str, Any]] = []
    for m in db_messages:
        msg: dict[str, Any] = {"role": m["role"], "content": m["content"]}
        meta = m.get("metadata")
        if isinstance(meta, dict):
            if "tool_calls" in meta:
                msg["tool_calls"] = meta["tool_calls"]
            if "tool_call_id" in meta:
                msg["tool_call_id"] = meta["tool_call_id"]
        messages.append(msg)
    return messages


def _make_classification_event(result: dict, content_length: int, final: bool) -> bytes:
    """Format a classification result as an SSE event."""
    payload = {
        "label": result["label"],
        "probabilities": result["probabilities"],
        "content_length": content_length,
        "risk": _compute_risk(result["probabilities"]),
        "final": final,
    }
    return f"event: classification\ndata: {json.dumps(payload)}\n\n".encode()


# ── SSE parsing ───────────────────────────────────────────────────────

async def _aiter_sse_events(resp: httpx.Response):
    """Yield complete SSE events (bytes including trailing \\n\\n)."""
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


def _extract_data_lines(event_bytes: bytes) -> list[bytes]:
    """Extract raw payloads after 'data:' prefixes from an SSE event."""
    out = []
    for ln in event_bytes.split(b"\n"):
        if ln.startswith(b"data:"):
            payload = ln[5:]
            if payload.startswith(b" "):
                payload = payload[1:]
            out.append(payload)
    return out


# ── routes ────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": clf is not None}


@app.get("/v1/models")
async def models():
    return {
        "object": "list",
        "data": [
            {
                "id": ALLOWED_MODEL,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "proxy",
            }
        ],
    }


@app.post("/classify", response_model=ClassifyResponse)
async def classify(req: ClassifyRequest):
    if clf is None:
        raise HTTPException(503, "model not loaded")

    texts = [req.text] if isinstance(req.text, str) else req.text
    if not texts:
        raise HTTPException(422, "text must not be empty")

    vecs = await embed(texts)
    labels = clf.predict(vecs)
    probs = clf.predict_proba(vecs)

    return ClassifyResponse(
        results=[
            ClassifyResult(text=t, label=l, probabilities=p)
            for t, l, p in zip(texts, labels, probs)
        ]
    )


# ── thread management ────────────────────────────────────────────────

@app.post("/threads", response_model=MakeThreadResponse)
async def make_thread(req: MakeThreadRequest):
    """Create a new conversation thread with a system prompt."""
    thread_id = await store.create_thread()
    await store.append_message(thread_id, "system", req.system_prompt)
    return MakeThreadResponse(thread_id=thread_id)


@app.post("/threads/{thread_id}/chat")
async def chat_message(thread_id: str, req: ChatRequest):
    """Send a user message and get a classified, streamed response."""
    thread = await store.get_thread(thread_id)
    if thread is None:
        raise HTTPException(404, "thread not found")

    # persist user message
    await store.append_message(thread_id, "user", req.message)

    # build full conversation for upstream
    messages = await _build_upstream_messages(thread_id)

    payload: dict[str, Any] = {
        "model": ALLOWED_MODEL,
        "messages": messages,
        "stream": req.stream,
    }
    if req.tools:
        payload["tools"] = req.tools

    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}"}
    tools_defined = bool(req.tools)

    # ── non-streaming path ────────────────────────────────────────
    if not req.stream:
        timeout = httpx.Timeout(connect=15.0, read=60.0, write=60.0, pool=60.0)
        async with httpx.AsyncClient(timeout=timeout) as client:
            r = await client.post(
                f"{UPSTREAM_BASE}/chat/completions", headers=headers, json=payload,
            )
        data = r.json()

        content = ""
        reasoning = ""
        tool_calls: list[Any] = []
        for choice in data.get("choices", []):
            msg = choice.get("message", {})
            if msg.get("content"):
                content += msg["content"]
            if msg.get("reasoning_content"):
                reasoning += msg["reasoning_content"]
            if msg.get("tool_calls"):
                tool_calls.extend(msg["tool_calls"])

        # classify content
        classification = await _classify_text(content) if content else None
        risk = _compute_risk(classification["probabilities"]) if classification else 0

        # persist assistant message
        meta: dict[str, Any] = {}
        if classification:
            meta["classification"] = classification
        if tool_calls:
            meta["tool_calls"] = tool_calls

        emb_array = None
        if content:
            try:
                emb = await embed(content)
                emb_array = np.array(emb[0], dtype=np.float32)
            except Exception:
                pass

        await store.append_message(
            thread_id, "assistant", content,
            reasoning_content=reasoning,
            metadata=meta or None,
            embedding=emb_array,
        )
        if risk > 0:
            await store.update_risk(thread_id, risk)

        # attach classification to response
        if classification:
            data["classification"] = {
                "label": classification["label"],
                "probabilities": classification["probabilities"],
                "risk": risk,
            }

        return JSONResponse(content=data)

    # ── streaming path ────────────────────────────────────────────
    async def stream_with_classification():
        content_buffer = ""
        reasoning_buffer = ""
        last_classified_at = 0

        timeout = httpx.Timeout(connect=15.0, read=None, write=60.0, pool=60.0)
        async with httpx.AsyncClient(timeout=timeout) as client:
            async with client.stream(
                "POST", f"{UPSTREAM_BASE}/chat/completions",
                headers=headers, json=payload,
            ) as resp:
                if resp.status_code >= 400:
                    body = await resp.aread()
                    yield body
                    return

                async for event in _aiter_sse_events(resp):
                    is_done = False

                    # parse deltas from this event
                    for data_payload in _extract_data_lines(event):
                        if data_payload.strip() == b"[DONE]":
                            is_done = True
                            break
                        try:
                            obj = json.loads(data_payload.decode("utf-8"))
                            for choice in obj.get("choices") or []:
                                delta = (choice or {}).get("delta") or {}
                                c = delta.get("content")
                                if isinstance(c, str):
                                    content_buffer += c
                                r = delta.get("reasoning_content")
                                if isinstance(r, str):
                                    reasoning_buffer += r
                        except Exception:
                            pass

                    if is_done:
                        # final classification before [DONE]
                        if content_buffer:
                            result = await _classify_text(content_buffer)
                            if result:
                                risk = _compute_risk(result["probabilities"])
                                yield _make_classification_event(
                                    result, len(content_buffer), final=True,
                                )
                                # persist
                                emb_array = None
                                try:
                                    emb = await embed(content_buffer)
                                    emb_array = np.array(emb[0], dtype=np.float32)
                                except Exception:
                                    pass
                                await store.append_message(
                                    thread_id, "assistant", content_buffer,
                                    reasoning_content=reasoning_buffer,
                                    metadata={"classification": result},
                                    embedding=emb_array,
                                )
                                await store.update_risk(thread_id, risk)
                            else:
                                # no classification, still persist
                                await store.append_message(
                                    thread_id, "assistant", content_buffer,
                                    reasoning_content=reasoning_buffer,
                                )
                        yield event
                        return

                    # forward original event
                    yield event

                    # periodic classification
                    new_chars = len(content_buffer) - last_classified_at
                    if content_buffer and new_chars >= CLASSIFY_INTERVAL:
                        result = await _classify_text(content_buffer)
                        if result:
                            last_classified_at = len(content_buffer)
                            yield _make_classification_event(
                                result, len(content_buffer), final=False,
                            )

    return StreamingResponse(
        stream_with_classification(), media_type="text/event-stream",
    )
