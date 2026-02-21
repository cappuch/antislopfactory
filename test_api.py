#!/usr/bin/env python3
"""
Test script for the antislopfactory API.

Usage:
    # Start the server first:
    uv run uvicorn main:app --port 8000

    # Then run tests:
    uv run python test_api.py
    uv run python test_api.py --base http://localhost:9000  # custom port
"""

import argparse
import json
import sys
import time

import httpx

DEFAULT_BASE = "http://localhost:8080"
TIMEOUT = httpx.Timeout(connect=10.0, read=120.0, write=30.0, pool=30.0)

passed = 0
failed = 0


def ok(name: str, detail: str = ""):
    global passed
    passed += 1
    print(f"  \033[32mPASS\033[0m  {name}" + (f"  ({detail})" if detail else ""))


def fail(name: str, detail: str = ""):
    global failed
    failed += 1
    print(f"  \033[31mFAIL\033[0m  {name}" + (f"  ({detail})" if detail else ""))


# ── tests ─────────────────────────────────────────────────────────────

def test_health(c: httpx.Client):
    r = c.get("/health")
    if r.status_code == 200 and r.json().get("status") == "ok":
        ok("GET /health", f"model_loaded={r.json()['model_loaded']}")
    else:
        fail("GET /health", r.text[:200])


def test_models(c: httpx.Client):
    r = c.get("/v1/models")
    data = r.json()
    models = data.get("data", [])
    if r.status_code == 200 and len(models) >= 1:
        ok("GET /v1/models", f"model={models[0]['id']}")
    else:
        fail("GET /v1/models", r.text[:200])


def test_classify(c: httpx.Client):
    r = c.post("/classify", json={"text": "Hello, how are you?"})
    if r.status_code == 200:
        result = r.json()["results"][0]
        ok("POST /classify", f"label={result['label']}  top_prob={max(result['probabilities'].values()):.3f}")
    elif r.status_code == 503:
        ok("POST /classify (skipped)", "model not loaded")
    else:
        fail("POST /classify", r.text[:200])


def test_classify_batch(c: httpx.Client):
    r = c.post("/classify", json={"text": ["I love cats", "How to hack a computer"]})
    if r.status_code == 200:
        results = r.json()["results"]
        labels = [res["label"] for res in results]
        ok("POST /classify (batch)", f"labels={labels}")
    elif r.status_code == 503:
        ok("POST /classify batch (skipped)", "model not loaded")
    else:
        fail("POST /classify (batch)", r.text[:200])


def test_make_thread(c: httpx.Client) -> str | None:
    r = c.post("/threads", json={"system_prompt": "You are a helpful assistant. Keep answers brief."})
    if r.status_code == 200 and r.json().get("thread_id"):
        tid = r.json()["thread_id"]
        ok("POST /threads", f"thread_id={tid}")
        return tid
    else:
        fail("POST /threads", r.text[:200])
        return None


def test_chat_nonstream(c: httpx.Client, thread_id: str):
    r = c.post(
        f"/threads/{thread_id}/chat",
        json={"message": "What is 2+2? One word.", "stream": False},
    )
    if r.status_code != 200:
        fail("POST /chat (non-stream)", f"status={r.status_code} {r.text[:200]}")
        return

    data = r.json()
    content = ""
    for choice in data.get("choices", []):
        msg = choice.get("message", {})
        if msg.get("content"):
            content += msg["content"]

    clf = data.get("classification")
    detail = f"content={content[:80]!r}"
    if clf:
        detail += f"  label={clf['label']}  risk={clf['risk']}"
    ok("POST /chat (non-stream)", detail)


def test_chat_stream(c: httpx.Client, thread_id: str):
    """Test streaming chat with SSE parsing."""
    with c.stream(
        "POST",
        f"/threads/{thread_id}/chat",
        json={"message": "Name three primary colors. Just list them.", "stream": True},
    ) as r:
        if r.status_code != 200:
            body = r.read()
            fail("POST /chat (stream)", f"status={r.status_code} {body[:200]}")
            return

        content_tokens = 0
        classifications = []
        full_content = ""
        done = False

        buf = b""
        for chunk in r.iter_bytes():
            buf += chunk
            while True:
                sep = buf.find(b"\n\n")
                if sep == -1:
                    break
                event_bytes = buf[: sep + 2]
                buf = buf[sep + 2 :]

                # parse event type and data
                event_type = None
                data_lines = []
                for line in event_bytes.decode("utf-8", errors="replace").split("\n"):
                    if line.startswith("event:"):
                        event_type = line[len("event:"):].strip()
                    elif line.startswith("data:"):
                        data_lines.append(line[len("data:"):].strip())

                for data_str in data_lines:
                    if data_str == "[DONE]":
                        done = True
                        continue

                    try:
                        obj = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue

                    if event_type == "classification":
                        classifications.append(obj)
                    else:
                        for choice in obj.get("choices", []):
                            delta = choice.get("delta", {})
                            c_text = delta.get("content")
                            if isinstance(c_text, str) and c_text:
                                content_tokens += 1
                                full_content += c_text

        detail = f"tokens={content_tokens}  content={full_content[:80]!r}"
        detail += f"  classifications={len(classifications)}"
        if classifications:
            final = next((c for c in classifications if c.get("final")), classifications[-1])
            detail += f"  final_label={final['label']}  risk={final['risk']}"

        if done and content_tokens > 0:
            ok("POST /chat (stream)", detail)
        elif content_tokens == 0:
            fail("POST /chat (stream)", f"no content received. {detail}")
        else:
            fail("POST /chat (stream)", f"no [DONE]. {detail}")


def test_thread_not_found(c: httpx.Client):
    r = c.post(
        "/threads/nonexistent-thread-id/chat",
        json={"message": "hello", "stream": False},
    )
    if r.status_code == 404:
        ok("POST /chat (404)", "thread not found handled")
    else:
        fail("POST /chat (404)", f"expected 404, got {r.status_code}")


def test_openapi(c: httpx.Client):
    r = c.get("/openapi.json")
    if r.status_code == 200 and "paths" in r.json():
        paths = list(r.json()["paths"].keys())
        ok("GET /openapi.json", f"paths={paths}")
    else:
        fail("GET /openapi.json", r.text[:200])


# ── main ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Test antislopfactory API")
    parser.add_argument("--base", default=DEFAULT_BASE, help="Base URL of the server")
    args = parser.parse_args()

    base = args.base.rstrip("/")
    print(f"\ntesting against {base}\n")

    c = httpx.Client(base_url=base, timeout=TIMEOUT)

    # check server is up
    try:
        c.get("/health")
    except httpx.ConnectError:
        print(f"\033[31mERROR\033[0m  cannot connect to {base}")
        print("       start the server first: uv run uvicorn main:app --port 8000\n")
        sys.exit(1)

    print("── meta ────────────────────────────────")
    test_health(c)
    test_models(c)
    test_openapi(c)

    print("\n── classify ────────────────────────────")
    test_classify(c)
    test_classify_batch(c)

    print("\n── threads ─────────────────────────────")
    test_thread_not_found(c)
    thread_id = test_make_thread(c)

    if thread_id:
        print("\n── chat (non-streaming) ────────────────")
        test_chat_nonstream(c, thread_id)

        print("\n── chat (streaming) ────────────────────")
        test_chat_stream(c, thread_id)

    c.close()

    print(f"\n{'─' * 44}")
    total = passed + failed
    color = "\033[32m" if failed == 0 else "\033[31m"
    print(f"{color}{passed}/{total} passed\033[0m\n")
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
