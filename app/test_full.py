import asyncio
import sys
import time
sys.path.insert(0, '.')

async def test():
    from routes.search import perform_search, SearchRequest
    
    print("Creating request...", file=sys.stderr)
    req = SearchRequest(company="Nvidia", designation="CEO")
    print("Making request...", file=sys.stderr)
    
    start = time.time()
    result = await perform_search(req)
    print(f"Time: {time.time() - start:.1f}s", file=sys.stderr)
    print(f"Result: {result}", file=sys.stderr)

if __name__ == "__main__":
    asyncio.run(test())
