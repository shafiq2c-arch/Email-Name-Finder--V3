"""SearXNG async client — primary search engine."""

import os
import logging
import httpx
import random
import asyncio
from typing import List, Dict
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

SEARXNG_BASE_URL = os.getenv("SEARXNG_BASE_URL", "https://searxngapp.app.digitalgalaxy.com:8080")
REQUEST_TIMEOUT = 25

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118.0.0.0 Safari/537.36"
]

ALL_ENGINES = ["yahoo", "duckduckgo", "bing", "google"]

async def search(query: str, num_results: int = 10, max_retries: int = 4) -> List[Dict]:
    """
    Fetch search results from SearXNG with humanoid behavior.
    """
    
    for attempt in range(max_retries):
        # Humanoid traits: 
        # 1. Random delay between requests to simulate human typing/reading speed
        if attempt > 0:
            await asyncio.sleep(random.uniform(2.0, 5.0))
        else:
            await asyncio.sleep(random.uniform(0.5, 2.0))
            
        # 2. Pick a random user agent
        headers = {
            "User-Agent": random.choice(USER_AGENTS),
            "Accept-Language": "en-US,en;q=0.9",
        }
        
        # 3. Pick engine in sequence: Yahoo, duckduckgo, bing, google
        selected_engines = ALL_ENGINES[attempt % len(ALL_ENGINES)]
        
        params = {
            "q": query,
            "format": "json",
            "language": "en",
            "engines": selected_engines,
            "pageno": 1,
        }

        try:
            async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT, headers=headers) as client:
                resp = await client.get(f"{SEARXNG_BASE_URL}/search", params=params)
                resp.raise_for_status()
                data = resp.json()

            results = []
            for item in data.get("results", [])[:num_results]:
                results.append({
                    "title":   item.get("title", ""),
                    "snippet": item.get("content", ""),
                    "url":     item.get("url", ""),
                })
            
            if results:
                # Success!
                return results
                
            # If no results but also no exception, it might be due to blocking on the selected engines.
            logger.warning(f"No results on attempt {attempt + 1}. Selected engines: {selected_engines}. Retrying...")
            
        except httpx.HTTPError as exc:
            logger.warning(f"HTTP error on attempt {attempt + 1} for query '{query}': {exc}")
        except Exception as exc:
            logger.warning(f"SearXNG error on attempt {attempt + 1} for query '{query}': {exc}")
            
    # Return empty list if all retries fail
    return []
