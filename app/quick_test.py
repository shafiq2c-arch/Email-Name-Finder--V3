import asyncio
import sys
import os

async def test():
    from services import searxng_client
    from services import llm_processor
    
    results = await searxng_client.search('CEO Nvidia')
    print(f'Results: {len(results)}')
    if results:
        r = results[0]
        print(f'Title: {r["title"][:50]}')
        
        llm_res = llm_processor.process_result(
            title=r['title'],
            snippet=r['snippet'],
            url=r['url'],
            company='Nvidia',
            designation='CEO',
        )
        print(f'LLM: {llm_res}')

asyncio.run(test())
