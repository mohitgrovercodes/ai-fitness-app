import requests
import json

BASE_URL = "http://localhost:8001/api/ai"

def test_chat_with_profile():
    payload = {
        "message": "What should be my focus today?",
        "user_id": "test_user_123" # This user has a profile with goal 'muscle_gain'
    }
    
    print("Sending chat message...")
    response = requests.post(f"{BASE_URL}/chat", json=payload)
    print(f"Status Code: {response.status_code}")
    res_json = response.json()
    print(f"Response: {res_json.get('data', {}).get('response', '')}")

if __name__ == "__main__":
    test_chat_with_profile()
