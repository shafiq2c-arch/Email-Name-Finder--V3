import asyncio
from services.duckduckgo_client import search_async
from services.llm_processor import process_result
from services.searxng_client import search as searxng_search

async def main():
    print("--- Testing DDG ---")
    results = await search_async("PureLogic CEO")
    print(f"DDG results count: {len(results)}")
    if results:
        print("First DDG result:", results[0])
    
    print("\n--- Testing SearXNG ---")
    s_results = await searxng_search("PureLogic CEO")
    print(f"SearXNG results count: {len(s_results)}")

    print("\n--- Testing LLM ---")
    
    r = {
        "title": "Corey Thomas - CEO - PureLogic IT Solutions",
        "snippet": "PureLogic IT Solutions is proud to have Corey Thomas as its CEO. He has led the company since 2018.",
        "url": "https://www.linkedin.com/in/coreythomas"
    }
    res = process_result(
        title=r["title"],
        snippet=r["snippet"],
        url=r["url"],
        company="PureLogic",
        designation="CEO"
    )
    print("LLM output:", res)

if __name__ == "__main__":
    asyncio.run(main())
