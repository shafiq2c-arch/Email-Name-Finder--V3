"""DuckDuckGo fallback search client."""

import logging
from typing import List, Dict
from duckduckgo_search import DDGS

logger = logging.getLogger(__name__)


def search(query: str, num_results: int = 10) -> List[Dict]:
    """
    Synchronous DuckDuckGo search using duckduckgo-search library.

    Returns a list of dicts: [{title, snippet, url}, ...]
    Returns empty list on any error.
    """
    try:
        results = []
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=num_results):
                results.append({
                    "title":   r.get("title", ""),
                    "snippet": r.get("body", ""),
                    "url":     r.get("href", ""),
                })
        return results
    except Exception as exc:
        logger.warning(f"DuckDuckGo error for query '{query}': {exc}")
        return []


async def search_async(query: str, num_results: int = 10) -> List[Dict]:
    """Async wrapper — runs sync DDG search in a thread pool."""
    import asyncio
    loop = asyncio.get_event_loop()
    try:
        return await asyncio.wait_for(loop.run_in_executor(None, search, query, num_results), timeout=10.0)
    except asyncio.TimeoutError:
        logger.warning(f"DuckDuckGo search timed out for query '{query}'")
        return []
