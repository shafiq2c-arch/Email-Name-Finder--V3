import sys
sys.path.insert(0, ".")

from agents.result_filter_agent import (
    filter_results, filter_results_relaxed,
    _normalize_company, _get_designation_group
)

print("=== Test 1: Company name normalization ===")
variants = _normalize_company("PureLogics")
print("Variants:", variants)

print("\n=== Test 2: Designation alias groups ===")
print("CEO group:", _get_designation_group("CEO"))
print("Chief Executive Officer group:", _get_designation_group("Chief Executive Officer"))

print("\n=== Test 3: Filter agent ===")
fake_results = [
    {
        "title": "PureLogics CEO Usman Akbar Leads the Company",
        "snippet": "Usman Akbar is the Chief Executive Officer of PureLogics.",
        "url": "https://purelogics.net/team"
    },
    {
        "title": "Unrelated Company Board",
        "snippet": "John Smith is CEO of SomeOtherCorp.",
        "url": "https://example.com"
    },
    {
        "title": "Former CEO of PureLogics steps down",
        "snippet": "The ex-CEO of PureLogics stepped down last year.",
        "url": "https://example2.com"
    },
    {
        "title": "PureLogics Software Company Profile",
        "snippet": "PureLogics is a leading IT firm with great leadership.",
        "url": "https://clutch.co/purelogics"
    },
]

strict = filter_results(fake_results, "PureLogics", "CEO")
relaxed = filter_results_relaxed(fake_results, "PureLogics", "CEO")

print("Strict filter: %d/%d kept" % (len(strict), len(fake_results)))
for r in strict:
    print("  KEPT:", r["title"])

print("Relaxed filter: %d/%d kept" % (len(relaxed), len(fake_results)))
for r in relaxed:
    print("  KEPT:", r["title"])

print("\n=== Test 4: Query count ===")
from services.query_generator import generate_queries
queries = generate_queries("PureLogics", "CEO")
print("Total queries:", len(queries))
for i, q in enumerate(queries[:10], 1):
    print("  %2d. %s" % (i, q))

print("\n=== Test 5: Verification agent ===")
from agents.verification_agent import verify, verify_relaxed

good = {"name": "Usman Rana", "company_match": True, "designation_match": True, "current_role": True}
partial = {"name": "Usman Rana", "company_match": True, "designation_match": False, "current_role": True}
bad = {"name": "Unknown", "company_match": True, "designation_match": True, "current_role": True}

print("Strict  (good)   :", verify(good))
print("Strict  (partial):", verify(partial))
print("Relaxed (partial):", verify_relaxed(partial))
print("Relaxed (bad name):", verify_relaxed(bad))

print("\n=== ALL TESTS COMPLETE ===")
