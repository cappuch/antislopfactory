import asyncio
import json
import os

import aiohttp
from dotenv import load_dotenv

load_dotenv()

API_URL = "https://infer.dev.takara.ai/"
API_KEY = os.environ["ds1_api_key"]

MAX_CHARS = 1000  # ~300 tokens, safely under the 512-token API limit


async def embed(inputs: str | list[str], max_chars: int = MAX_CHARS,
                session: aiohttp.ClientSession | None = None) -> list[list[float]]:
    """Get embeddings for one or more texts.

    Args:
        inputs: A single string or list of strings to embed.
        max_chars: Truncate inputs longer than this (avoids 512-token API limit).
        session: Optional shared aiohttp session for connection reuse.

    Returns:
        List of embedding vectors (list of floats).
    """
    if isinstance(inputs, str):
        inputs = [inputs]

    inputs = [t[:max_chars] for t in inputs]
    headers = {"Content-Type": "application/json", "x-api-key": API_KEY}
    payload = json.dumps({"inputs": inputs})

    own_session = session is None
    if own_session:
        session = aiohttp.ClientSession()
    try:
        async with session.post(API_URL, data=payload, headers=headers) as resp:
            body = await resp.read()
            try:
                return json.loads(body)
            except json.JSONDecodeError:
                raise RuntimeError(f"embed API returned non-JSON ({len(body)}B): {body[:300]}")
    finally:
        if own_session:
            await session.close()


def embed_sync(inputs: str | list[str], max_chars: int = MAX_CHARS) -> list[list[float]]:
    """Blocking wrapper around embed() for simple / non-async callers."""
    return asyncio.run(embed(inputs, max_chars))


if __name__ == "__main__":
    vectors = asyncio.run(embed("Hello world"))
    print(f"dims: {len(vectors[0])}")
    print(f"first 5: {vectors[0][:5]}")
