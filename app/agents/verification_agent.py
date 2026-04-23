"""Verification Agent — validates LLM output before accepting a result.

Two modes:
  verify()         — strict: all 3 flags must be True  (high-confidence path)
  verify_relaxed() — lenient: company_match + name found (fallback path)
"""

from typing import Dict


def verify(llm_result: Dict) -> bool:
    """
    Strict verification — returns True only if ALL conditions are met:
    - name is a non-empty, recognised string
    - company_match is True
    - designation_match is True
    - current_role is True
    """
    if not llm_result:
        return False

    name = (llm_result.get("name") or "").strip()
    if not name or name.lower() in ("unknown", "n/a", "none", ""):
        return False

    if not llm_result.get("company_match"):
        return False

    if not llm_result.get("designation_match"):
        return False

    if not llm_result.get("current_role"):
        return False

    return True


def verify_relaxed(llm_result: Dict) -> bool:
    """
    Relaxed verification (fallback for lesser-known companies).
    Accepts the result if:
    - name is a real, non-empty string
    - company_match is True
    - At least ONE of (designation_match, current_role) is True
    This handles cases where the LLM is unsure about the designation
    wording or role currency but clearly found a person at the company.
    """
    if not llm_result:
        return False

    name = (llm_result.get("name") or "").strip()
    if not name or name.lower() in ("unknown", "n/a", "none", ""):
        return False

    if not llm_result.get("company_match"):
        return False

    # At least one of these must be True
    if not (llm_result.get("designation_match") or llm_result.get("current_role")):
        return False

    return True
