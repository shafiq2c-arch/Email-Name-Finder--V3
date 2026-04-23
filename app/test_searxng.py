import asyncio
import sys
sys.path.insert(0, '.')

async def test():
    print("Starting test...", file=sys.stderr)
    from services import searxng_client
    print("Imported searxng_client", file=sys.stderr)
    results = await searxng_client.search("CEO Nvidia")
    print(f"Results count: {len(results)}", file=sys.stderr)
    for r in results[:3]:
        print(f"Title: {r.get('title', 'no title')[:60]}", file=sys.stderr)

if __name__ == "__main__":
    asyncio.run(test())
