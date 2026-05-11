import requests
import json

BASE_URL = "http://localhost:8000/api/profile"

def test_onboarding():
    profile_data = {
        "full_name": "Shubham Jadhav",
        "age": 25,
        "gender": "male",
        "height": 175.0,
        "weight": 63.0,
        "goal": "muscle_gain",
        "activity_level": "moderately_active",
        "diet_preference": "Any",
        "injuries": "None",
        "medical_conditions": "None"
    }
    
    print("Sending onboarding data...")
    response = requests.post(f"{BASE_URL}/onboarding", json=profile_data)
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")

def test_get_profile():
    print("\nGetting profile...")
    response = requests.get(f"{BASE_URL}/me")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")

if __name__ == "__main__":
    try:
        test_onboarding()
        test_get_profile()
    except Exception as e:
        print(f"Error: {e}")
