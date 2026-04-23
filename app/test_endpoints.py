import requests

def test_endpoints():
    print("Testing single search...")
    res = requests.post("http://localhost:8000/search", json={"company": "OpenAI", "designation": "CEO"})
    print("Status:", res.status_code)
    print("Response:", res.json())

    print("\nTesting bulk search...")
    csv_data = "company,designation\nOpenAI,CEO\nTesla,CEO\nGoogle,CEO\n"
    res2 = requests.post(
        "http://localhost:8000/bulk-search",
        files={"file": ("test.csv", csv_data.encode("utf-8"), "text/csv")}
    )
    print("Status:", res2.status_code)
    print("Response snippet:", res2.text[:200])

if __name__ == "__main__":
    test_endpoints()
