import urllib.request
import json
import sys

def test_search():
    url = 'http://localhost:8000/search'
    data = json.dumps({"company": "PureLogic", "designation": "CEO"}).encode('utf-8')
    req = urllib.request.Request(url, data=data, headers={'Content-Type': 'application/json'})
    
    try:
        with urllib.request.urlopen(req) as response:
            status = response.getcode()
            content = response.read().decode('utf-8')
            print(f"Status Code: {status}")
            try:
                print("Response JSON:")
                print(json.dumps(json.loads(content), indent=2))
            except json.JSONDecodeError:
                print(f"Response Text: {content}")
    except urllib.error.URLError as e:
        print(f"Failed to connect or HTTP error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    test_search()
