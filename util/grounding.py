"""Search grounding: generate queries with Cerebras, search with Exa, inject results."""

import asyncio
import json
import os

from cerebras.cloud.sdk import Cerebras
from dotenv import load_dotenv
from exa_py import AsyncExa

load_dotenv()

_cerebras = Cerebras(api_key=os.environ.get("CEREBRAS_API_KEY"))
_exa = AsyncExa(api_key=os.environ.get("EXA_API_KEY")) if os.environ.get("EXA_API_KEY") else None

QUERY_SCHEMA = {
    "name": "search_queries",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "needs_search": {
                "type": "boolean",
                "description": "Whether this message needs web search grounding",
            },
            "reason": {
                "type": "string",
                "description": "Brief reason for decision",
            },
            "queries": {
                "type": "array",
                "items": {"type": "string"},
                "description": "3-5 diverse search queries (empty if needs_search is false)",
            },
        },
        "required": ["needs_search", "reason", "queries"],
        "additionalProperties": False,
    },
}


def _generate_queries(user_message: str, context: str = "") -> list[str]:
    """Use Cerebras to decide if grounding is needed and generate queries."""
    prompt = f"""Decide whether this user message needs web search grounding, and if so generate 3-5 search queries.

SKIP search for: greetings, simple chitchat, math/logic, code help, creative writing, opinions, follow-ups that don't need new facts, anything the LLM can answer well from training data alone.

USE search for: current events, specific facts/stats, recent news, named entities the LLM might not know, time-sensitive info, "who/what/when" factual questions.

User message: {user_message}"""
    if context:
        prompt += f"\n\nRecent conversation context: {context}"

    try:
        resp = _cerebras.chat.completions.create(
            messages=[
                {"role": "system", "content": "You decide if a message needs web search grounding. Set needs_search=false and queries=[] to skip. Set needs_search=true with 3-5 queries to search."},
                {"role": "user", "content": prompt},
            ],
            model="gpt-oss-120b",
            max_completion_tokens=256,
            temperature=0.4,
            response_format={"type": "json_schema", "json_schema": QUERY_SCHEMA},
        )
        data = json.loads(resp.choices[0].message.content)
        if not data.get("needs_search", True):
            print(f"grounding skipped: {data.get('reason', '?')}")
            return []
        return data.get("queries", [])[:5]
    except Exception as e:
        print(f"grounding query generation error: {e}")
        return []


async def _search_exa(query: str) -> list[dict]:
    """Search a single query with Exa, return simplified results."""
    if _exa is None:
        return []
    try:
        results = await _exa.search(
            query,
            num_results=3,
            type="auto",
            contents={"text": {"max_characters": 1000}},
        )
        return [
            {"title": r.title or "", "url": r.url, "text": (r.text or "")[:800]}
            for r in results.results
        ]
    except Exception as e:
        print(f"exa search error for '{query}': {e}")
        return []


async def ground(user_message: str, context: str = "") -> str | None:
    """Generate search queries, run them concurrently on Exa, return formatted grounding block.

    Returns None if grounding is unavailable or produces no results.
    """
    if _exa is None:
        return None

    # generate queries with cerebras (sync, but very fast)
    loop = asyncio.get_event_loop()
    queries = await loop.run_in_executor(None, _generate_queries, user_message, context)
    if not queries:
        return None

    # search all queries concurrently
    search_tasks = [_search_exa(q) for q in queries]
    all_results = await asyncio.gather(*search_tasks, return_exceptions=True)

    # deduplicate by URL and format
    seen_urls = set()
    formatted = []
    for i, results in enumerate(all_results):
        if isinstance(results, Exception):
            continue
        for r in results:
            if r["url"] in seen_urls:
                continue
            seen_urls.add(r["url"])
            snippet = r["text"].strip().replace("\n", " ")[:500]
            formatted.append(f'[{r["title"]}]({r["url"]})\n{snippet}')

    if not formatted:
        return None

    block = "| GROUNDING WEB RESULTS |:\n"
    block += f"Search queries: {', '.join(queries)}\n\n"
    block += "\n\n---\n\n".join(formatted[:12])  # cap at 12 results
    return block
