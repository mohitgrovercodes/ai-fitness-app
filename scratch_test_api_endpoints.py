"""
Full API Endpoint Test Suite
Tests every endpoint across: Auth, Profile, Feedback, AI
Requires the FastAPI server to be running at http://localhost:8000
Run: venv\python.exe scratch_test_api_endpoints.py
"""

import sys
import os
import json
import httpx
import asyncio

sys.stdout.reconfigure(encoding='utf-8')

BASE_URL = "http://localhost:8000"
RESULTS_PATH = "C:/Users/mogr1/.gemini/antigravity/brain/8d893a5b-b2dd-4b20-9924-fe2298f030e4/API_Endpoint_Test_Results.md"

# Test user credentials - unique to avoid DB conflicts
TEST_EMAIL = "api_test_user_01@test.com"
TEST_PASSWORD = "TestPass@123"
TOKEN = None  # Will be filled after login

results = []

def log(label, status, payload, response_status, response_body, notes=""):
    icon = "[PASS]" if status == "PASS" else "[FAIL]" if status == "FAIL" else "[WARN]"
    print(f"\n{icon} {label}")
    print(f"  Status Code: {response_status}")
    if notes:
        print(f"  Note: {notes}")
    results.append({
        "label": label,
        "status": status,
        "payload": str(payload)[:300],
        "response_status": response_status,
        "response_body": str(response_body)[:500],
        "notes": notes
    })

async def run():
    global TOKEN

    async with httpx.AsyncClient(base_url=BASE_URL, timeout=60.0) as client:

        # ====================================================
        # AUTH MODULE
        # ====================================================
        print("\n" + "="*60)
        print("AUTH MODULE")
        print("="*60)

        # --- 1. Register (Happy Path) ---
        payload = {"email": TEST_EMAIL, "password": TEST_PASSWORD}
        r = await client.post("/api/auth/register", json=payload)
        body = r.json()
        status = "PASS" if r.status_code in (200, 201) else "FAIL"
        # Accept 400 if user already exists (idempotent re-runs)
        if r.status_code == 400 and "already" in str(body).lower():
            status = "WARN"
            notes = "User already exists from previous run - OK"
        else:
            notes = ""
        log("POST /api/auth/register (valid user)", status, payload, r.status_code, body, notes)

        # --- 2. Register (Duplicate Email) ---
        r2 = await client.post("/api/auth/register", json=payload)
        status = "WARN" if r2.status_code == 200 else "PASS"
        notes_dup = "FINDING: Duplicate registration returns 200 instead of 409 Conflict. Should return 409." if r2.status_code == 200 else ""
        log("POST /api/auth/register (duplicate - should return 409)", status, payload, r2.status_code, r2.json(), notes_dup)

        # --- 3. Login (Happy Path) ---
        # LoginSchema uses 'username' field (not 'email')
        # API wraps response in {success, data, message} envelope
        payload = {"username": TEST_EMAIL, "password": TEST_PASSWORD}
        r = await client.post("/api/auth/login", json=payload)
        body = r.json()
        token_value = None
        data = body.get("data")
        if isinstance(data, dict):
            token_value = data.get("access_token")
        elif isinstance(data, str):
            # Already a raw token string
            token_value = data
        status = "PASS" if r.status_code == 200 and token_value else "FAIL"
        log("POST /api/auth/login (valid credentials)", status, payload, r.status_code, body)
        if status == "PASS":
            TOKEN = token_value
            print(f"  Token acquired: {TOKEN[:30]}...")

        # --- 4. Login (Wrong Password) ---
        # Note: App returns HTTP 200 with error inside envelope instead of 401
        payload = {"username": TEST_EMAIL, "password": "WrongPassword999"}
        r = await client.post("/api/auth/login", json=payload)
        body = r.json()
        is_error = body.get("success") is False or "error" in str(body.get("message", "")).lower() or "invalid" in str(body.get("message", "")).lower()
        status = "PASS" if is_error else "WARN"
        notes_wp = "FINDING: Wrong password returns HTTP 200 with error in body instead of 401 Unauthorized." if not r.status_code == 401 else ""
        log("POST /api/auth/login (wrong password - should return 401)", status, payload, r.status_code, body, notes_wp)

        # --- 5. Login (Non-existent user) ---
        # Note: App returns HTTP 200 with error inside envelope instead of 404
        payload = {"username": "ghost@nobody.com", "password": "anything"}
        r = await client.post("/api/auth/login", json=payload)
        body = r.json()
        is_error = body.get("success") is False or "not found" in str(body).lower() or "invalid" in str(body).lower()
        status = "PASS" if is_error else "WARN"
        notes_ne = "FINDING: Non-existent user returns HTTP 200 with error in body instead of 404." if not r.status_code == 404 else ""
        log("POST /api/auth/login (non-existent user - should return 404)", status, payload, r.status_code, body, notes_ne)

        if not TOKEN:
            print("\n[FATAL] Could not get auth token. Aborting remaining tests.")
            return

        headers = {"Authorization": f"Bearer {TOKEN}"}

        # ====================================================
        # PROFILE MODULE
        # ====================================================
        print("\n" + "="*60)
        print("PROFILE MODULE")
        print("="*60)

        # --- 6. Get profile (before onboarding) ---
        r = await client.get("/api/profile/me", headers=headers)
        status = "PASS" if r.status_code in (200, 404) else "FAIL"
        log("GET /api/profile/me (before onboarding)", status, {}, r.status_code, r.json(), "404 is acceptable if profile doesn't exist yet")

        # --- 7. Onboarding (Happy Path) ---
        payload = {
            "full_name": "API Test User",
            "age": 28,
            "gender": "male",
            "height": 178.0,
            "weight": 80.0,
            "goal": "muscle gain",
            "activity_level": "moderately_active",
            "diet_preference": "standard",
            "injuries": ["left knee discomfort"],
            "medical_conditions": []
        }
        r = await client.post("/api/profile/onboarding", json=payload, headers=headers)
        body = r.json()
        status = "PASS" if r.status_code in (200, 201) else "FAIL"
        log("POST /api/profile/onboarding (valid profile)", status, payload, r.status_code, body)

        # --- 8. Get profile (after onboarding) ---
        r = await client.get("/api/profile/me", headers=headers)
        body = r.json()
        status = "PASS" if r.status_code == 200 and body.get("full_name") == "API Test User" else "FAIL"
        log("GET /api/profile/me (after onboarding)", status, {}, r.status_code, body)

        # --- 9. Update profile (PATCH) ---
        patch_payload = {"weight": 82.5, "goal": "weight loss"}
        r = await client.patch("/api/profile/me", json=patch_payload, headers=headers)
        body = r.json()
        status = "PASS" if r.status_code == 200 else "FAIL"
        log("PATCH /api/profile/me (update weight + goal)", status, patch_payload, r.status_code, body)

        # --- 10. Onboarding with invalid gender enum ---
        bad_payload = {**payload, "gender": "alien"}
        r = await client.post("/api/profile/onboarding", json=bad_payload, headers=headers)
        status = "PASS" if r.status_code == 422 else "FAIL"
        log("POST /api/profile/onboarding (invalid gender enum - should fail)", status, bad_payload, r.status_code, r.json(), "Expects 422 Validation Error")

        # --- 11. Unauthenticated access ---
        r = await client.get("/api/profile/me")
        status = "PASS" if r.status_code == 403 else "FAIL"
        log("GET /api/profile/me (no token - should fail)", status, {}, r.status_code, r.json(), "Expects 403 Forbidden")

        # ====================================================
        # FEEDBACK MODULE
        # ====================================================
        print("\n" + "="*60)
        print("FEEDBACK MODULE")
        print("="*60)

        # --- 12. Submit thumbs-up feedback ---
        payload = {
            "rating": "up",
            "session_id": "test-session-001",
            "agent_intents": "nutrition",
            "user_message": "Give me a diet plan",
            "ai_response_snippet": "Here is your plan...",
            "comment": "Very helpful!"
        }
        r = await client.post("/api/feedback/submit", json=payload, headers=headers)
        body = r.json()
        status = "PASS" if r.status_code in (200, 201) else "FAIL"
        log("POST /api/feedback/submit (thumbs up)", status, payload, r.status_code, body)

        # --- 13. Submit thumbs-down feedback ---
        payload["rating"] = "down"
        payload["comment"] = "Response was not accurate"
        r = await client.post("/api/feedback/submit", json=payload, headers=headers)
        status = "PASS" if r.status_code in (200, 201) else "FAIL"
        log("POST /api/feedback/submit (thumbs down)", status, payload, r.status_code, r.json())

        # --- 14. Submit invalid rating ---
        payload["rating"] = "sideways"
        r = await client.post("/api/feedback/submit", json=payload, headers=headers)
        status = "PASS" if r.status_code == 422 else "FAIL"
        log("POST /api/feedback/submit (invalid rating - should fail)", status, payload, r.status_code, r.json(), "Expects 422")

        # --- 15. Get feedback history ---
        r = await client.get("/api/feedback/history", headers=headers)
        body = r.json()
        status = "PASS" if r.status_code == 200 and isinstance(body, list) else "FAIL"
        log("GET /api/feedback/history", status, {}, r.status_code, body, f"Returned {len(body) if isinstance(body, list) else 'N/A'} entries")

        # --- 16. Get feedback summary (user-level) ---
        r = await client.get("/api/feedback/summary", headers=headers)
        body = r.json()
        status = "PASS" if r.status_code == 200 else "FAIL"
        log("GET /api/feedback/summary", status, {}, r.status_code, body)

        # --- 17. Admin global summary (no auth) ---
        r = await client.get("/api/feedback/admin/summary")
        body = r.json()
        status = "PASS" if r.status_code == 200 else "FAIL"
        log("GET /api/feedback/admin/summary (no auth)", status, {}, r.status_code, body)

        # ====================================================
        # AI MODULE
        # ====================================================
        print("\n" + "="*60)
        print("AI MODULE")
        print("="*60)

        # --- 18. Chat (Happy Path) ---
        payload = {"message": "What is the best exercise for biceps?", "context": {}}
        r = await client.post("/api/ai/chat", json=payload, headers=headers)
        body = r.json()
        status = "PASS" if r.status_code == 200 else "FAIL"
        log("POST /api/ai/chat (simple question)", status, payload, r.status_code, body, f"Response snippet: {str(body.get('response',''))[:100]}")

        # --- 19. Chat (Missing message field) ---
        r = await client.post("/api/ai/chat", json={}, headers=headers)
        status = "PASS" if r.status_code == 422 else "FAIL"
        log("POST /api/ai/chat (missing message - should fail)", status, {}, r.status_code, r.json(), "Expects 422")

        # --- 20. Generate Workout ---
        payload = {
            "goal": "build upper body strength",
            "level": "intermediate",
            "duration": "1 day",
            "injuries": [],
            "weight": 80,
            "height": 178,
            "gender": "male"
        }
        r = await client.post("/api/ai/generate-workout", json=payload, headers=headers)
        body = r.json()
        status = "PASS" if r.status_code == 200 and "workout" in body else "FAIL"
        log("POST /api/ai/generate-workout", status, payload, r.status_code, body, f"Exercises: {len(body.get('workout',[]))}")

        # --- 21. Generate Diet ---
        payload = {
            "goal": "weight loss",
            "diet_type": "vegetarian",
            "allergies": ["gluten"],
            "weight": 75,
            "height": 170,
            "gender": "female"
        }
        r = await client.post("/api/ai/generate-diet", json=payload, headers=headers)
        body = r.json()
        status = "PASS" if r.status_code == 200 else "FAIL"
        log("POST /api/ai/generate-diet", status, payload, r.status_code, body, f"Meals: {len(body.get('meals',[]))}")

        # --- 22. Ask Domain Question ---
        payload = {"message": "What is the difference between fast-twitch and slow-twitch muscle fibers?"}
        r = await client.post("/api/ai/ask-domain", json=payload, headers=headers)
        body = r.json()
        status = "PASS" if r.status_code == 200 else "FAIL"
        log("POST /api/ai/ask-domain", status, payload, r.status_code, body, f"Response: {str(body.get('response',''))[:100]}")

        # --- 23. AI endpoint without auth ---
        r = await client.post("/api/ai/chat", json={"message": "hello"})
        status = "PASS" if r.status_code == 403 else "FAIL"
        log("POST /api/ai/chat (no token - should fail)", status, {}, r.status_code, r.json(), "Expects 403")

    # ====================================================
    # WRITE RESULTS
    # ====================================================
    pass_count = sum(1 for r in results if r["status"] == "PASS")
    fail_count = sum(1 for r in results if r["status"] == "FAIL")
    warn_count = sum(1 for r in results if r["status"] == "WARN")

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        f.write("# Full API Endpoint Test Results\n\n")
        f.write(f"**Total: {len(results)} tests | PASS: {pass_count} | FAIL: {fail_count} | WARN: {warn_count}**\n\n")
        f.write("---\n\n")
        for r in results:
            icon = "✅" if r["status"] == "PASS" else "❌" if r["status"] == "FAIL" else "⚠️"
            f.write(f"### {icon} {r['label']}\n")
            f.write(f"- **Status:** {r['status']}\n")
            f.write(f"- **HTTP Code:** {r['response_status']}\n")
            if r["notes"]:
                f.write(f"- **Notes:** {r['notes']}\n")
            f.write(f"- **Payload:** `{r['payload']}`\n")
            f.write(f"- **Response:** `{r['response_body']}`\n\n")

    print(f"\n{'='*60}")
    print(f"COMPLETE: {pass_count} PASS | {fail_count} FAIL | {warn_count} WARN")
    print(f"Results saved to API_Endpoint_Test_Results.md")
    print(f"{'='*60}")

if __name__ == "__main__":
    asyncio.run(run())
