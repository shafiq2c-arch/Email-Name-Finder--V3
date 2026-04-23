"""Quick diagnostic: trace the CTO@PureLogics search pipeline step by step."""
import sys
import os
import asyncio

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

from services import query_generator, searxng_client, duckduckgo_client, llm_processor
from agents import result_filter_agent, verification_agent
from utils import confidence_score

async def debug():
    company = "PureLogics"
    designation = "CTO"
    
    # Step 1: Generate queries
    queries = query_generator.generate_queries(company, designation)
    print(f"=== STEP 1: Generated {len(queries)} queries ===")
    for i, q in enumerate(queries[:5]):
        print(f"  [{i}] {q}")
    print(f"  ... and {len(queries)-5} more\n")
    
    # Step 2: Try first 3 queries against both search engines
    for qi, query in enumerate(queries[:3]):
        print(f"=== STEP 2: Query #{qi}: {query} ===")
        
        # SearXNG
        try:
            searx = await searxng_client.search(query)
            print(f"  SearXNG: {len(searx)} results")
        except Exception as e:
            searx = []
            print(f"  SearXNG: FAILED - {e}")
        
        # DuckDuckGo
        try:
            ddg = await duckduckgo_client.search_async(query)
            print(f"  DuckDuckGo: {len(ddg)} results")
        except Exception as e:
            ddg = []
            print(f"  DuckDuckGo: FAILED - {e}")
        
        # Merge
        seen = set()
        merged = []
        for r in ddg + searx:
            if r.get("url") not in seen:
                seen.add(r.get("url"))
                merged.append(r)
        print(f"  Merged: {len(merged)} results")
        
        if not merged:
            print("  (no results to filter)\n")
            continue
        
        # Step 3: Filter
        strict = result_filter_agent.filter_results(merged, company, designation)
        relaxed = result_filter_agent.filter_results_relaxed(merged, company, designation)
        print(f"  Strict filter: {len(strict)} results")
        print(f"  Relaxed filter: {len(relaxed)} results")
        
        for r in strict[:3]:
            print(f"    STRICT: {r['title'][:80]} | {r['url'][:60]}")
        for r in relaxed[:3]:
            print(f"    RELAXED: {r['title'][:80]} | {r['url'][:60]}")
        
        # Step 4: LLM extraction on top results
        target = strict[:2] or relaxed[:2]
        for r in target:
            print(f"\n  === STEP 4: LLM on: {r['title'][:60]} ===")
            llm_res = llm_processor.process_result(
                title=r["title"], snippet=r["snippet"], url=r["url"],
                company=company, designation=designation
            )
            if llm_res:
                print(f"    Name: {llm_res['name']}")
                print(f"    company_match: {llm_res['company_match']}")
                print(f"    designation_match: {llm_res['designation_match']}")
                print(f"    current_role: {llm_res['current_role']}")
                print(f"    reasoning: {llm_res['reasoning']}")
                
                v_strict = verification_agent.verify(llm_res)
                v_relaxed = verification_agent.verify_relaxed(llm_res)
                print(f"    Strict verify: {v_strict} | Relaxed verify: {v_relaxed}")
                
                if v_strict or v_relaxed:
                    conf = confidence_score.compute_confidence(
                        company=company, company_match=llm_res["company_match"],
                        designation_match=llm_res["designation_match"],
                        current_role=llm_res["current_role"],
                        url=r["url"], snippet=r["snippet"]
                    )
                    print(f"    Confidence: {conf}")
            else:
                print("    LLM returned None (failed or unparseable)")
        print()

asyncio.run(debug())
