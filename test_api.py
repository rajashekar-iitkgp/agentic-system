import requests
import json
import time

def test_chat(message: str, session_id: str = "test-session"):
    url = "http://127.0.0.1:8000/api/v1/chat"
    payload = {
        "message": message,
        "session_id": session_id
    }
    headers = {'Content-Type': 'application/json'}
    
    print(f"\n[USER]: {message}")
    try:
        response = requests.post(url, data=json.dumps(payload), headers=headers)
        if response.status_code == 200:
            data = response.json()
            print(f"[AGENT]: {data['response']}")
            print(f"[METADATA]: Domain={data['active_domain']}, Tools={data['retrieved_tools']}")
        else:
            print(f"Error: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"Failed to connect to server: {e}")

if __name__ == "__main__":
    print("Starting client test...")
    test_chat("I want to create an invoice for $50 for customer CUST-88")
    
   