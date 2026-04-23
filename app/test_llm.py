import asyncio
import sys
sys.path.insert(0, '.')

async def test():
    print("Starting test...", file=sys.stderr)
    from services import searxng_client, llm_processor
    print("Searching...", file=sys.stderr)
    results = await searxng_client.search("CEO Nvidia")
    print(f"Found {len(results)} results", file=sys.stderr)
    
    if results:
        r = results[0]
        print(f"Processing: {r['title'][:50]}", file=sys.stderr)
        llm_res = llm_processor.process_result(
            title=r["title"],
            snippet=r["snippet"],
            url=r["url"],
            company="Nvidia",
            designation="CEO",
        )
        print(f"LLM Result: {llm_res}", file=sys.stderr)

if __name__ == "__main__":
    asyncio.run(test())
