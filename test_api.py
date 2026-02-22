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

import httpx

DEFAULT_BASE = "http://localhost:8080"
TIMEOUT = httpx.Timeout(connect=10.0, read=120.0, write=30.0, pool=30.0)


class SSEParser:
    """Incremental SSE event parser."""

    def __init__(self):
        self._buf = b""

    def feed(self, chunk: bytes) -> list[tuple[str | None, str]]:
        self._buf += chunk
        events = []
        while (sep := self._buf.find(b"\n\n")) != -1:
            frame = self._buf[:sep].decode("utf-8", errors="replace")
            self._buf = self._buf[sep + 2:]

            event_type = None
            for line in frame.split("\n"):
                if line.startswith("event:"):
                    event_type = line[6:].strip()
                elif line.startswith("data:"):
                    events.append((event_type, line[5:].strip()))
        return events


class TestRunner:
    def __init__(self, base_url: str):
        self.client = httpx.Client(base_url=base_url, timeout=TIMEOUT)
        self.passed = 0
        self.failed = 0

    def ok(self, name: str, detail: str = "") -> None:
        self.passed += 1
        print(f"  \033[32mPASS\033[0m  {name}" + (f"  ({detail})" if detail else ""))

    def fail(self, name: str, detail: str = "") -> None:
        self.failed += 1
        print(f"  \033[31mFAIL\033[0m  {name}" + (f"  ({detail})" if detail else ""))

    def close(self) -> None:
        self.client.close()

    def summary(self) -> int:
        total = self.passed + self.failed
        color = "\033[32m" if self.failed == 0 else "\033[31m"
        print(f"\n{'─' * 44}")
        print(f"{color}{self.passed}/{total} passed\033[0m\n")
        return 0 if self.failed == 0 else 1

    # ── tests ─────────────────────────────────────────────────────────

    def test_health(self) -> None:
        r = self.client.get("/health")
        if r.status_code == 200 and r.json().get("status") == "ok":
            self.ok("GET /health", f"model_loaded={r.json()['model_loaded']}")
        else:
            self.fail("GET /health", r.text[:200])

    def test_models(self) -> None:
        r = self.client.get("/v1/models")
        data = r.json()
        models = data.get("data", [])
        if r.status_code == 200 and len(models) >= 1:
            self.ok("GET /v1/models", f"model={models[0]['id']}")
        else:
            self.fail("GET /v1/models", r.text[:200])

    def test_openapi(self) -> None:
        r = self.client.get("/openapi.json")
        if r.status_code == 200 and "paths" in r.json():
            paths = list(r.json()["paths"].keys())
            self.ok("GET /openapi.json", f"paths={paths}")
        else:
            self.fail("GET /openapi.json", r.text[:200])

    def test_classify(self) -> None:
        r = self.client.post("/classify", json={"text": "Hello, how are you?"})
        if r.status_code == 200:
            result = r.json()["results"][0]
            self.ok("POST /classify", f"label={result['label']}  top_prob={max(result['probabilities'].values()):.3f}")
        elif r.status_code == 503:
            self.ok("POST /classify (skipped)", "model not loaded")
        else:
            self.fail("POST /classify", r.text[:200])

    def test_classify_batch(self) -> None:
        r = self.client.post("/classify", json={"text": ["I love cats", "How to hack a computer"]})
        if r.status_code == 200:
            results = r.json()["results"]
            labels = [res["label"] for res in results]
            self.ok("POST /classify (batch)", f"labels={labels}")
        elif r.status_code == 503:
            self.ok("POST /classify batch (skipped)", "model not loaded")
        else:
            self.fail("POST /classify (batch)", r.text[:200])

    def test_thread_not_found(self) -> None:
        r = self.client.post(
            "/threads/nonexistent-thread-id/chat",
            json={"message": "hello", "stream": False},
        )
        if r.status_code == 404:
            self.ok("POST /chat (404)", "thread not found handled")
        else:
            self.fail("POST /chat (404)", f"expected 404, got {r.status_code}")

    def test_make_thread(self) -> str | None:
        r = self.client.post("/threads", json={"system_prompt": "You are a helpful assistant. Keep answers brief."})
        if r.status_code == 200 and r.json().get("thread_id"):
            tid = r.json()["thread_id"]
            self.ok("POST /threads", f"thread_id={tid}")
            return tid
        self.fail("POST /threads", r.text[:200])
        return None

    def test_chat_nonstream(self, thread_id: str) -> None:
        r = self.client.post(
            f"/threads/{thread_id}/chat",
            json={"message": "What is 2+2? One word.", "stream": False},
        )
        if r.status_code != 200:
            self.fail("POST /chat (non-stream)", f"status={r.status_code} {r.text[:200]}")
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
        self.ok("POST /chat (non-stream)", detail)

    def test_chat_stream(self, thread_id: str) -> None:
        with self.client.stream(
            "POST",
            f"/threads/{thread_id}/chat",
            json={"message": "Name three primary colors. Just list them.", "stream": True},
        ) as r:
            if r.status_code != 200:
                body = r.read()
                self.fail("POST /chat (stream)", f"status={r.status_code} {body[:200]}")
                return

            parser = SSEParser()
            content_tokens = 0
            classifications = []
            full_content = ""
            done = False

            for chunk in r.iter_bytes():
                for event_type, data_str in parser.feed(chunk):
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
                            token = choice.get("delta", {}).get("content")
                            if isinstance(token, str) and token:
                                content_tokens += 1
                                full_content += token

            detail = f"tokens={content_tokens}  content={full_content[:80]!r}"
            detail += f"  classifications={len(classifications)}"
            if classifications:
                final = next((cl for cl in classifications if cl.get("final")), classifications[-1])
                detail += f"  final_label={final['label']}  risk={final['risk']}"

            if done and content_tokens > 0:
                self.ok("POST /chat (stream)", detail)
            elif content_tokens == 0:
                self.fail("POST /chat (stream)", f"no content received. {detail}")
            else:
                self.fail("POST /chat (stream)", f"no [DONE]. {detail}")


# ── main ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Test antislopfactory API")
    parser.add_argument("--base", default=DEFAULT_BASE, help="Base URL of the server")
    args = parser.parse_args()

    base = args.base.rstrip("/")
    print(f"\ntesting against {base}\n")

    t = TestRunner(base)

    try:
        t.client.get("/health")
    except httpx.ConnectError:
        print(f"\033[31mERROR\033[0m  cannot connect to {base}")
        print("       start the server first: uv run uvicorn main:app --port 8000\n")
        sys.exit(1)

    print("── meta ────────────────────────────────")
    t.test_health()
    t.test_models()
    t.test_openapi()

    print("\n── classify ────────────────────────────")
    t.test_classify()
    t.test_classify_batch()

    print("\n── threads ─────────────────────────────")
    t.test_thread_not_found()
    thread_id = t.test_make_thread()

    if thread_id:
        print("\n── chat (non-streaming) ────────────────")
        t.test_chat_nonstream(thread_id)

        print("\n── chat (streaming) ────────────────────")
        t.test_chat_stream(thread_id)

    t.close()
    sys.exit(t.summary())


if __name__ == "__main__":
    main()
