"""LLM Processor — calls OpenRouter API to extract person name from a search result."""

import os
import json
import logging
import re
import time
import random
from typing import Dict, List, Optional

import openai
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

_client = openai.OpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY", ""),
    base_url="https://openrouter.ai/api/v1"
)

_MODELS = [
    "anthropic/claude-3-haiku",          # Fast, cheap
    "google/gemma-3-27b-it",             # Good quality
    "meta-llama/llama-3.1-8b-instruct",  # Reliable
    "qwen/qwen3-30b-a16k",               # Good reasoning
]

# Round-robin index shared across all calls
_model_index = 0

_SYSTEM_PROMPT = """You are a precise information extraction assistant specializing in identifying company executives.

When given a search result (title, snippet, URL) along with a company name and designation,
extract the FULL NAME of the person currently holding that designation at that company.

IMPORTANT DESIGNATION MATCHING RULES:
- "CEO" and "Chief Executive Officer" are the SAME role — treat them as a match.
- "CTO" and "Chief Technology Officer" are the SAME role.
 - "CPO" and "Chief People Officer" are the SAME role — treat them as a match.
- "COO" and "Chief Operating Officer" are the SAME role.
- "MD" and "Managing Director" are the SAME role.
- "Founder" / "Co-Founder" / "Founder & CEO" are all related founder roles.
- If the snippet mentions any equivalent form of the designation, set designation_match to true.

IMPORTANT COMPANY MATCHING RULES:
- Ignore capitalization differences: "purelogics" = "PureLogics" = "Pure Logics".
- Ignore common suffixes: "PureLogics Ltd" = "PureLogics".
- If the company is clearly referenced (even spelled slightly differently), set company_match to true.

You MUST respond with a valid JSON object — no markdown, no extra text — in this exact format:
{
  "name": "<Full Name or 'Unknown'>",
  "company_match": true or false,
  "designation_match": true or false,
  "current_role": true or false,
  "reasoning": "<one sentence>"
}

Rules:
- If the snippet refers to a FORMER or EX holder, set current_role to false.
- If you cannot find a clear name, set name to "Unknown".
- Be GENEROUS in matching — if there's a reasonable chance the person holds this role, extract the name.
- Do NOT set name to "Unknown" just because you're slightly unsure.
- company_match = true if the result is clearly about the specified company (flexible matching).
- designation_match = true if any equivalent form of the specified designation is mentioned.
"""


def _build_user_prompt(title: str, snippet: str, url: str, company: str, designation: str) -> str:
    return f"""Find the current {designation} of {company}.

Company: {company}
Designation: {designation}

Search Result:
Title: {title}
Snippet: {snippet}
URL: {url}

Note: "{designation}" may appear as an abbreviation or expanded form (e.g., CEO = Chief Executive Officer).
If this result clearly identifies a person at {company} in a leadership/executive role matching "{designation}",
extract their name even if the wording is not identical.

Return ONLY the JSON object as described."""


def _parse_response(text: str) -> Optional[Dict]:
    """Extract JSON from LLM response, even if wrapped in markdown."""
    text = re.sub(r"```(?:json)?", "", text).strip().strip("`").strip()
    try:
        data = json.loads(text)
        return {
            "name":               str(data.get("name", "Unknown")).strip(),
            "company_match":      bool(data.get("company_match", False)),
            "designation_match":  bool(data.get("designation_match", False)),
            "current_role":       bool(data.get("current_role", False)),
            "reasoning":          str(data.get("reasoning", "")),
        }
    except (json.JSONDecodeError, Exception):
        return None


def _is_rate_limit_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(k in msg for k in ["rate limit", "429", "too many requests", "rate_limit"])


def _is_decommissioned_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return "decommissioned" in msg or "model_decommissioned" in msg


def process_result(
    title: str,
    snippet: str,
    url: str,
    company: str,
    designation: str,
) -> Optional[Dict]:
    """
    Call LLM with a single search result and return parsed extraction dict.
    Rotates through models on rate limit / decommission errors.
    Returns None if all models fail.
    """
    global _model_index

    user_prompt = _build_user_prompt(title, snippet, url, company, designation)

    num_models = len(_MODELS)
    if num_models == 0:
        logger.error("No models available.")
        return None

    start_index = _model_index % num_models
    _model_index = (start_index + 1) % num_models

    for i in range(num_models):
        model = _MODELS[(start_index + i) % num_models]
        try:
            response = _client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user",   "content": user_prompt},
                ],
                temperature=0.1,
                max_tokens=400,
            )
            raw = response.choices[0].message.content or ""
            result = _parse_response(raw)
            if result:
                return result

        except Exception as exc:
            if _is_decommissioned_error(exc):
                logger.warning(f"Model '{model}' is decommissioned — removing from rotation.")
                if model in _MODELS:
                    _MODELS.remove(model)
                continue

            if _is_rate_limit_error(exc):
                wait = min(2 ** i + random.uniform(0, 1), 10)
                logger.warning(
                    f"Rate limit on '{model}'. Switching in {round(wait, 1)}s "
                    f"(attempt {i+1}/{num_models})"
                )
                time.sleep(wait)
            else:
                logger.warning(f"Model '{model}' error: {exc}")

            continue

    logger.error("All OpenRouter models exhausted. Returning None.")
    return None


def process_results_progressive(
    results: List[Dict],
    company: str,
    designation: str,
    first_batch: int = 3,
    second_batch: int = 5,
) -> Optional[Dict]:
    """
    Progressive LLM extraction — your smart approach:

    Step 1: Send first `first_batch` results to LLM one by one.
            Found a name? Return immediately (fast path — saves quota).

    Step 2: Nothing found? Send next `second_batch` results.
            Found a name? Return it.

    Step 3: Still nothing? Return None → caller tries relaxed pass.

    This cuts LLM calls from 10+ down to 1-3 for most easy queries.
    """
    if not results:
        return None

    # ── Step 1: Quick scan — first N results ──
    for r in results[:first_batch]:
        result = process_result(
            title=r.get("title", ""),
            snippet=r.get("snippet", ""),
            url=r.get("url", ""),
            company=company,
            designation=designation,
        )
        if result and result.get("name", "Unknown") != "Unknown":
            logger.info(f"Found in quick scan (first {first_batch}): {result['name']}")
            return result

    # ── Step 2: Deeper scan — next N results ──
    second_slice = results[first_batch: first_batch + second_batch]
    if not second_slice:
        return None

    logger.info(f"Quick scan found nothing. Trying next {second_batch} results...")
    for r in second_slice:
        result = process_result(
            title=r.get("title", ""),
            snippet=r.get("snippet", ""),
            url=r.get("url", ""),
            company=company,
            designation=designation,
        )
        if result and result.get("name", "Unknown") != "Unknown":
            logger.info(f"Found in deep scan (next {second_batch}): {result['name']}")
            return result

    return None