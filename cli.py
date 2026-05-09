"""
AI Cascade Router CLI - interactive REPL client.
Usage: python cli.py

Type your queries naturally, no quotes, no JSON, no curl.
Commands: /exit to quit, /new to reset session.
"""
import uuid
import sys
import requests

API_URL = "http://localhost:8000/route"
session_id = str(uuid.uuid4())

print("=" * 60)
print("  AI Cascade Router CLI")
print("  Type your query and press Enter.")
print("  /exit  — quit")
print("  /new   — reset session (start fresh)")
print("=" * 60)

for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    if line == "/exit":
        break
    if line == "/new":
        session_id = str(uuid.uuid4())
        print("(session reset)\n")
        continue

    try:
        resp = requests.post(API_URL, json={
            "query": line,
            "session_id": session_id
        }, timeout=120)
        data = resp.json()

        print()
        print(data.get("response", "no response"))
        source = data.get("source", "?")
        saved = data.get("tokens_saved", 0)
        time_ms = data.get("response_time_ms", 0)
        print(f"  [{source}, saved {saved} tok, {round(time_ms)}ms]")
        print()

    except KeyboardInterrupt:
        print("\n(exiting)")
        break
    except Exception as e:
        print(f"  [Error: {e}]\n")
