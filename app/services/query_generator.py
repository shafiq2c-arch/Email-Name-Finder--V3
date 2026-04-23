from typing import List

def generate_queries(company: str, designation: str) -> List[str]:
    c = company.strip()
    d = designation.strip()
    return [
        f'"{c}" {d}',                    # 1: "company" designation
        f'{c} who is the {d}',           # 5: company who is the designation
        f'{d} at {c}',                   # 2: designation at company
        f'"{c}" {d} LinkedIn',           # 4: "company" designation LinkedIn
    ]