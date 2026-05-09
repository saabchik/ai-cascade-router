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
MAX_CTX = 8192
session_id = str(uuid.uuid4())
ctx_usage = 0

ORANGE = "\033[38;5;214m"
GREEN = "\033[38;5;40m"
RESET = "\033[0m"

def _est_tokens(text: str) -> int:
    return len(text) // 4

def _fmt_tok(n: int) -> str:
    if n >= 1000:
        return f"{n/1000:.1f}K"
    return str(n)

def _ctx_bar(used: int, total: int) -> str:
    pct = min(used / total, 1.0)
    filled = int(pct * 10)
    bar = "▓" * filled + "░" * (10 - filled)
    return f"[ctx: {bar} {_fmt_tok(used)}/{_fmt_tok(total)}]"

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
        ctx_usage = 0
        print("(session reset)\n")
        continue

    try:
        resp = requests.post(API_URL, json={
            "query": line,
            "session_id": session_id
        }, timeout=120)
        data = resp.json()

        # Update local context tracking
        ctx_usage += _est_tokens(line) + _est_tokens(data.get("response", ""))
        ctx_usage = min(ctx_usage, MAX_CTX)  # cap at max (server trims)

        print()
        print(f"{ORANGE}{data.get('response', 'no response')}{RESET}")
        source = data.get("source", "?")
        saved = data.get("tokens_saved", 0)
        time_ms = data.get("response_time_ms", 0)
        ctx_info = _ctx_bar(ctx_usage, MAX_CTX)
        print(f"  [{GREEN}{source}{RESET}, saved {saved} tok, {round(time_ms)}ms]  {ctx_info}")
        print()

    except KeyboardInterrupt:
        print("\n(exiting)")
        break
    except Exception as e:
        print(f"  [Error: {e}]\n")
