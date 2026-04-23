"""Confidence Scoring Engine.

Weights
-------
- Company match      : +40 pts
- Designation match  : +30 pts
- Source trust bonus : up to +20 pts (based on domain tier)
- Snippet clarity    : +10 pts (non-empty snippet)
"""

from urllib.parse import urlparse


_DOMAIN_TRUST: list[tuple[list[str], int]] = [
    # Tier 1 — Company's own site or LinkedIn
    (["linkedin.com"],                                                            20),
    # Tier 2 — Crunchbase, Bloomberg, Forbes C-suite pages
    (["crunchbase.com", "bloomberg.com", "forbes.com"],                           15),
    # Tier 2b — Business profile / directory sites (great for SMBs)
    (["zoominfo.com", "clutch.co", "goodfirms.io", "g2.com"],                    13),
    # Tier 3 — Other business newswires / job sites
    (["reuters.com", "wsj.com", "ft.com", "businesswire.com", "prnewswire.com",
      "glassdoor.com", "indeed.com", "ambitionbox.com"],                          10),
    # Tier 4 — General news / tech media
    (["bbc.com", "cnbc.com", "techcrunch.com", "wired.com", "dawn.com",
      "geo.tv", "thenews.com.pk", "propakistani.pk"],                              8),
]


def _source_bonus(url: str, company: str) -> int:
    """Return trust bonus for the given URL."""
    try:
        domain = urlparse(url).netloc.lower().lstrip("www.")
    except Exception:
        return 5  # minimal bonus for unparseable URLs

    # Company's own domain — highest trust
    company_slug = company.lower().replace(" ", "").replace(",", "").replace(".", "")
    if company_slug in domain.replace(".", "").replace("-", ""):
        return 20

    for domains, score in _DOMAIN_TRUST:
        if any(d in domain for d in domains):
            return score

    return 5  # unknown source — low bonus


def compute_confidence(
    *,
    company: str,
    company_match: bool,
    designation_match: bool,
    current_role: bool,
    url: str,
    snippet: str,
) -> int:
    """Return an integer confidence score between 0 and 100."""
    score = 0

    if company_match:
        score += 40
    if designation_match:
        score += 30
    if snippet and len(snippet.strip()) > 20:
        score += 10

    score += _source_bonus(url, company)

    # Penalty: role is flagged as not current
    if not current_role:
        score = max(0, score - 30)

    return min(score, 100)
