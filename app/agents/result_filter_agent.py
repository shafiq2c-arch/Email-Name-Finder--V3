"""Result Filter Agent.

Filters raw search results, keeping only those that are relevant to
the given company+designation. Uses flexible matching so that:
- "CEO" matches "Chief Executive Officer" and vice-versa
- "PureLogics" matches "Pure Logics", "purelogics", etc.
- Relaxed mode: only requires company match (used as fallback)
"""

import re
from typing import List, Dict

_EXCLUSION_PATTERNS = re.compile(
    r"\b(former|formerly|ex[- ]|previous|previously|was the|retired|ex-ceo|ex-cfo|ex-cto|ex-coo)\b",
    re.IGNORECASE,
)

# Designation alias groups — any token in a group matches any other token in that group
_DESIGNATION_ALIASES: List[List[str]] = [
    ["ceo", "chief executive officer", "chief exec"],
    ["cfo", "chief financial officer"],
    ["cto", "chief technology officer", "chief technical officer"],
    ["coo", "chief operating officer"],
    ["cmo", "chief marketing officer"],
    ["cpo", "chief product officer"],
    ["md", "managing director"],
    ["vp", "vice president"],
    ["svp", "senior vice president"],
    ["evp", "executive vice president"],
    ["president", "president & ceo", "president and ceo"],
    ["founder", "co-founder", "cofounder"],
    ["director", "executive director"],
    ["chairman", "chairperson", "chair"],
    ["head", "head of"],
    ["manager", "general manager"],
]


def _get_designation_group(designation: str) -> List[str]:
    """Return all alias tokens for this designation (including itself)."""
    d = designation.strip().lower()
    for group in _DESIGNATION_ALIASES:
        if any(d == alias or d in alias for alias in group):
            return group
    return [d]


def _normalize_company(company: str) -> List[str]:
    """
    Return multiple forms of the company name to match flexibly.
    e.g. 'PureLogics' -> ['purelogics', 'pure logics', 'pure logic']
    Also returns individual significant words (len > 3) as fallback tokens.
    """
    raw = company.strip().lower()
    variants = {raw}

    # Insert spaces before uppercase letters (CamelCase splitting)
    spaced = re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', company).strip().lower()
    variants.add(spaced)

    # Remove common suffixes for matching
    for suffix in [" inc", " llc", " ltd", " pvt", " limited", " technologies",
                   " technology", " solutions", " services", " group", " corp"]:
        cleaned = raw.replace(suffix, "").strip()
        if cleaned:
            variants.add(cleaned)
        cleaned2 = spaced.replace(suffix, "").strip()
        if cleaned2:
            variants.add(cleaned2)

    # Also include significant individual words (length > 3)
    for word in re.split(r'\s+', spaced):
        if len(word) > 3:
            variants.add(word)

    return list(variants)


def _company_in_text(text: str, company_variants: List[str]) -> bool:
    """Return True if any company variant appears in the text."""
    for v in company_variants:
        if v in text:
            return True
    return False


def _designation_in_text(text: str, designation_group: List[str]) -> bool:
    """Return True if any designation alias appears in the text."""
    for alias in designation_group:
        if alias in text:
            return True
    return False


def filter_results(
    results: List[Dict],
    company: str,
    designation: str,
) -> List[Dict]:
    """
    Strict filter: result must mention company AND designation (or an alias).
    Former/ex- results are excluded.
    """
    company_variants  = _normalize_company(company)
    designation_group = _get_designation_group(designation)

    filtered = []
    for r in results:
        text = f"{r.get('title', '')} {r.get('snippet', '')}".lower()

        if not _company_in_text(text, company_variants):
            continue
        if not _designation_in_text(text, designation_group):
            continue

        combined = f"{r.get('title', '')} {r.get('snippet', '')}"
        if _EXCLUSION_PATTERNS.search(combined):
            continue

        filtered.append(r)

    return filtered


def filter_results_relaxed(
    results: List[Dict],
    company: str,
    designation: str,
) -> List[Dict]:
    """
    Relaxed filter (fallback): only requires company mention.
    Used when strict filter returns nothing.
    Still excludes former/ex- results.
    DOES NOT require designation to appear in the text.
    """
    company_variants = _normalize_company(company)

    relaxed = []
    for r in results:
        text = f"{r.get('title', '')} {r.get('snippet', '')}".lower()

        if not _company_in_text(text, company_variants):
            continue

        combined = f"{r.get('title', '')} {r.get('snippet', '')}"
        if _EXCLUSION_PATTERNS.search(combined):
            continue

        relaxed.append(r)

    return relaxed
