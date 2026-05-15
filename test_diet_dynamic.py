import requests
import json
import time
import os

# Apna port yahan update karein (FastAPI generally 8000 par chalta hai)
PORT = 8000  
BASE_URL = f"http://127.0.0.1:{PORT}"
URL = f"{BASE_URL}/api/ai/generate-diet"
LOGIN_URL = f"{BASE_URL}/api/auth/login"

# ==========================================
# ENTER YOUR LOGIN CREDENTIALS HERE
# ==========================================
LOGIN_EMAIL = "test@example.com"  # Apna real registered email daalein
LOGIN_PASSWORD = "password123"    # Apna password daalein
# ==========================================

# 30 Test Cases (10 for Daily, 10 for Weekly, 10 for Monthly/Long-term)
test_cases = [
    # ====================================================
    # POINT 1: DAILY PLANS (N=1)
    # Expected: Sirf 1 din ka plan, Auto-scaler active for perfect calories.
    # ====================================================
    {"id": "D1", "goal": "fat loss", "diet_type": "veg", "duration": "daily"},
    {"id": "D2", "goal": "muscle gain", "diet_type": "non-veg", "duration": "1 day"},
    {"id": "D3", "goal": "maintenance", "diet_type": "vegan", "duration": "today"},
    {"id": "D4", "goal": "lose 5kg", "diet_type": "keto", "duration": "daily"},
    {"id": "D5", "goal": "bulking", "diet_type": "pescatarian", "duration": "1 day"},
    {"id": "D6", "goal": "fat loss", "diet_type": "jain", "duration": "daily"},
    {"id": "D7", "goal": "muscle gain", "diet_type": "veg", "duration": "today"},
    {"id": "D8", "goal": "maintenance", "diet_type": "non-veg", "duration": "1 day"},
    {"id": "D9", "goal": "fat loss", "diet_type": "vegan", "duration": "daily"},
    {"id": "D10", "goal": "bulk up", "diet_type": "veg", "duration": "today"},

    # ====================================================
    # POINT 2: WEEKLY PLANS (2 <= N <= 7)
    # Expected: Exactly N unique days, daily_totals average calculated, per_day_totals included.
    # ====================================================
    {"id": "W1", "goal": "fat loss", "diet_type": "non-veg", "duration": "weekly"},
    {"id": "W2", "goal": "muscle gain", "diet_type": "veg", "duration": "7 days"},
    {"id": "W3", "goal": "maintenance", "diet_type": "vegan", "duration": "5 days"},
    {"id": "W4", "goal": "lose weight", "diet_type": "keto", "duration": "3 days"},
    {"id": "W5", "goal": "bulking", "diet_type": "pescatarian", "duration": "6 days"},
    {"id": "W6", "goal": "fat loss", "diet_type": "jain", "duration": "4 days"},
    {"id": "W7", "goal": "muscle gain", "diet_type": "non-veg", "duration": "weekly"},
    {"id": "W8", "goal": "maintenance", "diet_type": "veg", "duration": "7 days"},
    {"id": "W9", "goal": "fat loss", "diet_type": "vegan", "duration": "5 days"},
    {"id": "W10", "goal": "bulk up", "diet_type": "keto", "duration": "3 days"},

    # ====================================================
    # POINT 3: MONTHLY / REAL GYM TEMPLATE (N > 7)
    # Expected: Exactly 7-day Rotation Template (35 unique meals), 
    # Average Scaler Active, "Clean Eating" followed for fat loss.
    # ====================================================
    {"id": "M1", "goal": "fat loss", "diet_type": "veg", "duration": "monthly"},
    {"id": "M2", "goal": "muscle gain", "diet_type": "non-veg", "duration": "30 days"},
    {"id": "M3", "goal": "maintenance", "diet_type": "vegan", "duration": "4 weeks"},
    {"id": "M4", "goal": "lose 10kg", "diet_type": "keto", "duration": "45 days"},
    {"id": "M5", "goal": "bulking", "diet_type": "pescatarian", "duration": "2 months"},
    {"id": "M6", "goal": "fat loss", "diet_type": "jain", "duration": "monthly"},
    {"id": "M7", "goal": "muscle gain", "diet_type": "veg", "duration": "30 days"},
    {"id": "M8", "goal": "maintenance", "diet_type": "non-veg", "duration": "14 days"},
    {"id": "M9", "goal": "fat loss", "diet_type": "vegan", "duration": "2 weeks"},
    {"id": "M10", "goal": "bulk up", "diet_type": "veg", "duration": "monthly"},
]

def authenticate():
    print(f"🔐 Authenticating with {LOGIN_EMAIL}...")
    try:
        response = requests.post(
            LOGIN_URL, 
            data={"username": LOGIN_EMAIL, "password": LOGIN_PASSWORD},
            headers={"Content-Type": "application/x-www-form-urlencoded"}
        )
        if response.status_code == 200:
            token = response.json().get("access_token")
            print("✅ Login Successful! Token acquired.\n")
            return token
        else:
            print(f"❌ Login Failed! {response.status_code}: {response.text}")
            print("Make sure LOGIN_EMAIL and LOGIN_PASSWORD are correct at the top of the script.")
            return None
    except Exception as e:
        print(f"❌ Server not running or connection error: {e}")
        return None

def run_tests():
    print(f"🚀 Starting Dynamic Nutrition Agent Tests on {URL}")
    print("=================================================================")
    
    token = authenticate()
    if not token:
        return
        
    headers = {"Authorization": f"Bearer {token}"}
    
    os.makedirs("test_results", exist_ok=True)
    
    for tc in test_cases:
        print(f"⏳ Running Test {tc['id']}: {tc['duration'].upper()} | {tc['goal']} | {tc['diet_type']}...")
        
        payload = {
            # Removed static user_id since backend usually infers it from the JWT Token
            "goal": tc["goal"],
            "diet_type": tc["diet_type"],
            "duration": tc["duration"]
        }
        
        try:
            start_time = time.time()
            response = requests.post(URL, json=payload, headers=headers, timeout=120)
            elapsed = time.time() - start_time
            
            if response.status_code == 200:
                data = response.json()
                
                # Check status
                if data.get("status"):
                    meals = data.get("data", {}).get("meals", [])
                    meals_count = len(meals)
                    avg_cals = data.get("data", {}).get("daily_totals", {}).get("calories", 0)
                    
                    # --- CHECK 1: LAZY COPY-PASTE DETECTION ---
                    # Check if the LLM generated the exact same diet across multiple days
                    days_dict = {}
                    for m in meals:
                        d = m.get("day", "Daily")
                        if d not in days_dict:
                            days_dict[d] = set()
                        days_dict[d].add(m.get("name", "").strip().lower())
                    
                    is_lazy_copy = False
                    day_lists = list(days_dict.values())
                    if len(day_lists) > 1:
                        for i in range(len(day_lists)):
                            for j in range(i + 1, len(day_lists)):
                                if day_lists[i] == day_lists[j] and len(day_lists[i]) > 0:
                                    is_lazy_copy = True
                                    break
                    
                    copy_paste_status = "🚨 FAILED (Lazy Copy-Paste Detected!)" if is_lazy_copy else "🌟 Pass (Unique Meals/Days)"
                    
                    # --- CHECK 2: CLEAN EATING CHECK FOR FAT LOSS ---
                    bad_foods = ["nugget", "pakora", "burger", "vada", "lasagne", "pizza"]
                    has_bad_food = any(any(bf in m.get("name", "").lower() for bf in bad_foods) for m in meals)
                    clean_status = "⚠️ Warning (Contains Junk)" if tc["goal"] == "fat loss" and has_bad_food else "🥗 Clean Diet"

                    print(f"  ✅ SUCCESS ({elapsed:.1f}s) -> Meals: {meals_count} | Avg Cal: {avg_cals} | {copy_paste_status} | {clean_status}")
                    
                    # Save output to file for manual review
                    filename = f"test_results/{tc['id']}_{tc['duration'].replace(' ', '_')}_{tc['diet_type']}.json"
                    with open(filename, "w", encoding="utf-8") as f:
                        json.dump(data, f, indent=4, ensure_ascii=False)
                else:
                    print(f"  ❌ FAILED (Logic Error): {data.get('message')}")
            else:
                print(f"  ⚠️ HTTP Error {response.status_code}: {response.text}")
                
        except requests.exceptions.ConnectionError:
            print(f"  ❌ SERVER DOWN: Cannot connect to {URL}. Make sure your FastAPI server is running!")
            break
        except Exception as e:
            print(f"  ❌ ERROR: {e}")
            
        # Optional: Add delay so we don't spam OpenAI API too hard
        print("  ⏳ Waiting 5 seconds before next request to avoid OpenAI rate limits...")
        time.time()
        time.sleep(5)
        print("-" * 65)

if __name__ == "__main__":
    run_tests()
    print("\n🎉 All tests completed! Check the 'test_results' folder for JSON outputs.")
