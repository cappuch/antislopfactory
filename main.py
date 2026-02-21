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
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from db import ThreadStore, genid
from util.classifier import MLPClassifier
from util.embed import embed
from util.grounding import ground

load_dotenv()

# ── config ────────────────────────────────────────────────────────────

MODEL_PATH = os.environ.get("MODEL_PATH", "data/safety_classifier.npz")
UPSTREAM_BASE = os.environ.get("UPSTREAM_BASE", "https://api.openai.com").rstrip("/")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
ALLOWED_MODEL = os.environ.get("ALLOWED_MODEL", "local-proxy-model")
CLASSIFY_INTERVAL = int(os.environ.get("CLASSIFY_INTERVAL", "0"))

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
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


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
    message: str | None = None
    tools: list[dict[str, Any]] | None = None
    stream: bool = True


class ToolResult(BaseModel):
    tool_call_id: str
    content: str


class ToolResultsRequest(BaseModel):
    tool_results: list[ToolResult]


class ThreadSummary(BaseModel):
    thread_id: str
    risk: int
    risk_score: float
    peak_risk: float
    first_user_message: str | None
    created_at: float
    updated_at: float


class ListThreadsResponse(BaseModel):
    threads: list[ThreadSummary]


class MessageDetail(BaseModel):
    id: int
    seq: int
    role: str
    content: str
    reasoning_content: str
    metadata: dict[str, Any] | None = None
    embedding: list[float] | None = None
    created_at: float


class ThreadMessagesResponse(BaseModel):
    thread_id: str
    risk: int
    risk_score: float
    peak_risk: float
    created_at: float
    updated_at: float
    messages: list[MessageDetail]


# ── helpers ───────────────────────────────────────────────────────────

def _msg_risk(probs: dict[str, float]) -> float:
    """Raw per-message risk from classification probabilities (0.0-10.0)."""
    unsafe_sum = sum(probs.get(l, 0.0) for l in UNSAFE_LABELS)
    return min(10.0, unsafe_sum * 10.0)


# ── cumulative thread risk formula ───────────────────────────────────
#
#   Asymmetric EMA with peak-floor ratchet.
#
#   For each classified message with raw risk r:
#
#     alpha = ALPHA_UP   if r >= R(t-1)    (fast rise)
#             ALPHA_DOWN if r <  R(t-1)    (slow decay)
#
#     R(t) = R(t-1) + alpha * (r - R(t-1))
#
#     peak  = max(peak, r)
#     floor = peak * FLOOR_RATIO
#     R(t)  = max(R(t), floor)              (can never fully wash risk)
#
#   ALPHA_UP    = 0.6   →  risky content is weighted 60%
#   ALPHA_DOWN  = 0.12  →  safe content barely nudges score down
#   FLOOR_RATIO = 0.25  →  score can never drop below 25% of peak
#
ALPHA_UP = 0.6
ALPHA_DOWN = 0.12
FLOOR_RATIO = 0.25


def _update_cumulative_risk(
    current_score: float, peak: float, msg_risk_val: float,
) -> tuple[int, float, float]:
    """Compute new cumulative thread risk.

    Returns (display_risk_int, risk_score_float, peak_float).
    """
    if msg_risk_val >= current_score:
        alpha = ALPHA_UP
    else:
        alpha = ALPHA_DOWN

    new_score = current_score + alpha * (msg_risk_val - current_score)
    new_peak = max(peak, msg_risk_val)
    floor = new_peak * FLOOR_RATIO
    new_score = max(new_score, floor)
    new_score = max(0.0, min(10.0, new_score))

    return round(new_score), new_score, new_peak


CHUNK_SIZE = 500  # chars per embedding chunk


async def _classify_text(text: str) -> dict[str, Any] | None:
    """Embed text in 500-char chunks, average the vectors, then classify."""
    if clf is None or not text.strip():
        return None
    try:
        chunks = [text[i:i + CHUNK_SIZE] for i in range(0, len(text), CHUNK_SIZE)]
        all_vecs = await embed(chunks)
        avg_vec = np.mean(np.array(all_vecs, dtype=np.float64), axis=0)
        vecs = [avg_vec.tolist()]
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
        "risk": round(_msg_risk(result["probabilities"])),
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

@app.get("/threads", response_model=ListThreadsResponse)
async def list_threads():
    """List all threads with their ID, risk, and first user message."""
    threads = await store.list_threads()
    return ListThreadsResponse(
        threads=[
            ThreadSummary(
                thread_id=t["thread_id"],
                risk=t["risk"],
                risk_score=t.get("risk_score", 0.0),
                peak_risk=t.get("peak_risk", 0.0),
                first_user_message=t["first_user_message"],
                created_at=t["created_at"],
                updated_at=t["updated_at"],
            )
            for t in threads
        ]
    )


@app.get("/threads/{thread_id}/messages", response_model=ThreadMessagesResponse)
async def get_thread_messages(thread_id: str):
    """Return all messages in a thread including embeddings and classifications."""
    thread = await store.get_thread(thread_id)
    if thread is None:
        raise HTTPException(404, "thread not found")

    db_messages = await store.get_messages(thread_id)
    messages = [
        MessageDetail(
            id=m["id"],
            seq=m["seq"],
            role=m["role"],
            content=m["content"],
            reasoning_content=m["reasoning_content"],
            metadata=m.get("metadata"),
            embedding=m["embedding"].tolist() if m.get("embedding") is not None else None,
            created_at=m["created_at"],
        )
        for m in db_messages
    ]

    return ThreadMessagesResponse(
        thread_id=thread["thread_id"],
        risk=thread["risk"],
        risk_score=thread.get("risk_score", 0.0),
        peak_risk=thread.get("peak_risk", 0.0),
        created_at=thread["created_at"],
        updated_at=thread["updated_at"],
        messages=messages,
    )


@app.post("/threads", response_model=MakeThreadResponse)
async def make_thread(req: MakeThreadRequest):
    """Create a new conversation thread with a system prompt."""
    thread_id = await store.create_thread()
    await store.append_message(thread_id, "system", req.system_prompt)
    return MakeThreadResponse(thread_id=thread_id)


@app.post("/threads/{thread_id}/tool_results")
async def submit_tool_results(thread_id: str, req: ToolResultsRequest):
    """Submit tool execution results to a thread."""
    thread = await store.get_thread(thread_id)
    if thread is None:
        raise HTTPException(404, "thread not found")
    for tr in req.tool_results:
        tool_clf = await _classify_text(tr.content)
        meta: dict[str, Any] = {"tool_call_id": tr.tool_call_id}
        if tool_clf:
            meta["classification"] = tool_clf
        await store.append_message(
            thread_id, "tool", tr.content,
            metadata=meta,
        )
        if tool_clf:
            mr = _msg_risk(tool_clf["probabilities"])
            t = await store.get_thread(thread_id)
            ri, rf, pf = _update_cumulative_risk(t["risk_score"], t["peak_risk"], mr)
            await store.update_risk(thread_id, ri, rf, pf)
    return {"ok": True}


@app.post("/threads/{thread_id}/chat")
async def chat_message(thread_id: str, req: ChatRequest):
    """Send a user message and get a classified, streamed response."""
    thread = await store.get_thread(thread_id)
    if thread is None:
        raise HTTPException(404, "thread not found")

    # persist user message (skip when continuing after tool results)
    if req.message is not None:
        user_clf = await _classify_text(req.message)
        user_meta = {"classification": user_clf} if user_clf else None
        await store.append_message(thread_id, "user", req.message, metadata=user_meta)
        # update cumulative thread risk from user message
        if user_clf:
            mr = _msg_risk(user_clf["probabilities"])
            t = await store.get_thread(thread_id)
            ri, rf, pf = _update_cumulative_risk(t["risk_score"], t["peak_risk"], mr)
            await store.update_risk(thread_id, ri, rf, pf)

    # build full conversation for upstream
    messages = await _build_upstream_messages(thread_id)

    # search grounding: inject web results into the last user message
    if req.message is not None:
        # build context from recent messages for better query generation
        recent = [m["content"] for m in messages[-4:] if m.get("content")]
        context = "\n".join(recent[-3:]) if len(recent) > 1 else ""
        grounding_block = await ground(req.message, context)
        if grounding_block:
            # append grounding to the last user message (sent to LLM, not stored)
            for m in reversed(messages):
                if m["role"] == "user":
                    m["content"] = m["content"] + "\n\n" + grounding_block
                    break

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
        if classification:
            mr = _msg_risk(classification["probabilities"])
            t = await store.get_thread(thread_id)
            risk_int, risk_f, peak_f = _update_cumulative_risk(
                t["risk_score"], t["peak_risk"], mr)
            await store.update_risk(thread_id, risk_int, risk_f, peak_f)

        # attach classification to response
        if classification:
            data["classification"] = {
                "label": classification["label"],
                "probabilities": classification["probabilities"],
                "risk": round(_msg_risk(classification["probabilities"])),
            }

        return JSONResponse(content=data)

    # ── streaming path ────────────────────────────────────────────
    async def stream_with_classification():
        content_buffer = ""
        reasoning_buffer = ""
        tool_calls_buf: dict[int, dict] = {}  # index -> accumulated tool_call
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
                                for tc in delta.get("tool_calls") or []:
                                    idx = tc.get("index", 0)
                                    if idx not in tool_calls_buf:
                                        tool_calls_buf[idx] = {
                                            "id": tc.get("id", ""),
                                            "type": tc.get("type", "function"),
                                            "function": {"name": "", "arguments": ""},
                                        }
                                    if tc.get("id"):
                                        tool_calls_buf[idx]["id"] = tc["id"]
                                    fn = tc.get("function") or {}
                                    if fn.get("name"):
                                        tool_calls_buf[idx]["function"]["name"] = fn["name"]
                                    if fn.get("arguments"):
                                        tool_calls_buf[idx]["function"]["arguments"] += fn["arguments"]
                        except Exception:
                            pass

                    if is_done:
                        accumulated_tc = (
                            [tool_calls_buf[i] for i in sorted(tool_calls_buf)]
                            if tool_calls_buf else []
                        )
                        meta: dict[str, Any] = {}
                        if accumulated_tc:
                            meta["tool_calls"] = accumulated_tc

                        if content_buffer:
                            result = await _classify_text(content_buffer)
                            if result:
                                yield _make_classification_event(
                                    result, len(content_buffer), final=True,
                                )
                                meta["classification"] = result
                                emb_array = None
                                try:
                                    emb = await embed(content_buffer)
                                    emb_array = np.array(emb[0], dtype=np.float32)
                                except Exception:
                                    pass
                                await store.append_message(
                                    thread_id, "assistant", content_buffer,
                                    reasoning_content=reasoning_buffer,
                                    metadata=meta or None,
                                    embedding=emb_array,
                                )
                                mr = _msg_risk(result["probabilities"])
                                t = await store.get_thread(thread_id)
                                ri, rf, pf = _update_cumulative_risk(
                                    t["risk_score"], t["peak_risk"], mr)
                                await store.update_risk(thread_id, ri, rf, pf)
                            else:
                                await store.append_message(
                                    thread_id, "assistant", content_buffer,
                                    reasoning_content=reasoning_buffer,
                                    metadata=meta or None,
                                )
                        elif accumulated_tc:
                            # tool-calls-only response (no content)
                            await store.append_message(
                                thread_id, "assistant", "",
                                metadata=meta,
                            )
                        yield event
                        return

                    # forward original event
                    yield event

                    # periodic classification (interval=0 means every token)
                    new_chars = len(content_buffer) - last_classified_at
                    if content_buffer and new_chars > 0 and new_chars >= CLASSIFY_INTERVAL:
                        result = await _classify_text(content_buffer)
                        if result:
                            last_classified_at = len(content_buffer)
                            yield _make_classification_event(
                                result, len(content_buffer), final=False,
                            )

    return StreamingResponse(
        stream_with_classification(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ── web search proxy ─────────────────────────────────────────────────

@app.get("/search")
async def search_proxy(q: str = Query(..., description="Search query")):
    """Search Google and return results."""
    from googlesearch import search
    loop = asyncio.get_event_loop()
    def _search():
        return [
            {"title": r.title, "url": r.url, "description": r.description}
            for r in search(q, num_results=8, advanced=True)
        ]
    results = await loop.run_in_executor(None, _search)
    return JSONResponse(content={"results": results})


# ── static frontend ──────────────────────────────────────────────────

_frontend = Path(__file__).parent / "toy_frontend"
if _frontend.is_dir():
    app.mount("/ui", StaticFiles(directory=str(_frontend), html=True), name="ui")
