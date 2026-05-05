import requests
import json

def test_chat():
    url = "http://127.0.0.1:8000/api/ai/chat"
    
    # Test Case: Complex query requiring memory and parallel tools
    payload = {
        "message": "I have a knee injury. What are some good leg exercises and what should I eat to recover faster?",
        "user_id": "test_user_001",
        "context": {
            "goal": "muscle_gain",
            "injuries": ["knee injury"]
        }
    }
    
    print(f"🚀 Sending request to {url}...")
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        
        data = response.json()
        print("\n✅ Response Received:")
        print(f"Bot: {data['data']['response']}")
        print(f"\nIntents Detected: {data['data']['intents']}")
        print(f"Conversation Summary: {data['data']['summary']}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Make sure the server is running with: uvicorn app.main:app --reload")

if __name__ == "__main__":
    test_chat()
