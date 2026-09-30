import requests

BASE_URL = "http://localhost:8000"

def main():
    print("Calling /metrics/slo...")
    response = requests.get(
        f"{BASE_URL}/metrics/slo",
        timeout=100,
    )
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.text}")

if __name__ == "__main__":
    main()