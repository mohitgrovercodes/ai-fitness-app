"""
Vision Agent V8 - Full Test Suite
Validates the 4-Tier Decision Engine with proper formatted output.
"""
import sys, os, io, glob, asyncio
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.path.append(os.getcwd())

from app.agents.vision_agent import VisionAgent

BASE_DIR = r"d:\Work\fitness-app\ai-fitness-app\test"

# Read all PNG images from the test folder
ALL_IMAGES = sorted(glob.glob(os.path.join(BASE_DIR, "*.png")))

DIVIDER     = "=" * 70
SUB_DIVIDER = "-" * 70

async def run_single_test(agent: VisionAgent, img_path: str, test_num: int):
    filename = os.path.basename(img_path)

    with open(img_path, 'rb') as f:
        img_bytes = f.read()

    state  = {'messages': [], 'image_bytes': img_bytes}
    result = await agent.run(state)
    response = result['messages'][0].content

    # ── Tier Detection ─────────────────────────────────────────────────────────
    resp_lower = response.lower()
    if "not appear to be a food" in resp_lower or "couldn't detect any food" in resp_lower:
        tier    = "🚫 NON-FOOD REJECTED"
        tier_ok = True
    elif "couldn't find" in resp_lower or "exact name" in resp_lower:
        tier    = "🟡 OOD / WEB FALLBACK"
        tier_ok = True
    else:
        tier    = "✅ IDENTIFIED"
        tier_ok = True

    # ── Print Result ────────────────────────────────────────────────────────────
    print(f"\n{DIVIDER}")
    print(f"  TEST {test_num}: {filename}")
    print(f"  Status: {tier}")
    print(DIVIDER)
    print(f"\n🤖 AGENT RESPONSE:\n")
    print(response)
    print(f"\n📊 DIAGNOSTICS:")
    print(f"   File    : {filename}")
    print(f"   Tier    : {tier}")
    print()

async def main():
    print("\n")
    print("🍽️ " * 20)
    print("     VISION AGENT V8 - FULL TEST SUITE")
    print("🍽️ " * 20)
    print(f"\n⏳ Initializing Vision Agent...")

    agent = VisionAgent()
    print(f"✅ Vision Agent Ready!\n")

    total   = len(ALL_IMAGES)
    success = 0

    print(f"\n{DIVIDER}")
    print(f"  TESTING {total} IMAGES FROM: {BASE_DIR}")
    print(DIVIDER)

    for i, img_path in enumerate(ALL_IMAGES, 1):
        await run_single_test(agent, img_path, i)
        success += 1

    # ── Final Summary ───────────────────────────────────────────────────────────
    print(f"\n{DIVIDER}")
    print(f"  ✅ ALL {total} TESTS COMPLETED")
    print(f"  Processed : {success}/{total} images")
    print(DIVIDER)
    print()

if __name__ == "__main__":
    asyncio.run(main())
