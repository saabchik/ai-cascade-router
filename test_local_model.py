"""
Quick diagnostic script to test local model (LM Studio) directly.
Run: python test_local_model.py
"""
import httpx
import json

# Config from config.yaml
BASE_URL = "http://localhost:1234/v1"
MODEL = "auto"  # Use whatever is loaded

def test_local_model():
    print("=" * 60)
    print("Testing Local Model (LM Studio)")
    print("=" * 60)

    client = httpx.Client(timeout=30.0)

    # Test 1: Check if server is running
    print("\n1. Checking if LM Studio server is running...")
    try:
        resp = client.get(f"{BASE_URL}/models")
        print(f"   Status: {resp.status_code}")
        if resp.status_code == 200:
            models = resp.json()
            print(f"   Available models: {json.dumps(models, indent=2, ensure_ascii=False)[:500]}")
        else:
            print(f"   Error: {resp.text[:200]}")
            return
    except Exception as e:
        print(f"   FAILED: {type(e).__name__}: {e}")
        print("   → Make sure LM Studio is running and loaded with a model!")
        return

    # Test 2: Simple generation
    print("\n2. Testing simple generation...")
    try:
        payload = {
            "model": MODEL,
            "messages": [{"role": "user", "content": "Привет! Как дела?"}],
            "temperature": 0.7,
            "max_tokens": 100
        }
        print(f"   Sending request to {BASE_URL}/chat/completions")
        print(f"   Payload: {json.dumps(payload, ensure_ascii=False)[:200]}")

        resp = client.post(f"{BASE_URL}/chat/completions", json=payload)
        print(f"   Status: {resp.status_code}")

        if resp.status_code != 200:
            print(f"   Error response: {resp.text[:500]}")
            return

        data = resp.json()
        print(f"   Response structure keys: {list(data.keys())}")

        if "choices" in data and data["choices"]:
            text = data["choices"][0]["message"]["content"]
            tokens = data.get("usage", {})
            print(f"   ✓ Success!")
            print(f"   Response: {text[:200]}")
            print(f"   Tokens: {tokens}")
        else:
            print(f"   ✗ Invalid response structure: {json.dumps(data, indent=2, ensure_ascii=False)[:500]}")

    except Exception as e:
        print(f"   FAILED: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 60)
    print("Test complete!")
    print("=" * 60)

if __name__ == "__main__":
    test_local_model()
