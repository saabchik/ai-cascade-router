# Session / Context Support Implementation Plan

> **For agentic workers:** Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add session/context support so multi-turn dialogs preserve message history across requests.

**Architecture:** In-memory SessionManager stores messages per session_id, auto-trims context by max_tokens, auto-cleanup by TTL. LocalEngine and CloudClient accept optional `messages` param to bypass prompt assembly for session-aware requests.

**Tech Stack:** Python dataclasses, time-based eviction, token-count estimation.

---

### Task 1: Create session/manager.py

**Files:**
- Create: `session/manager.py`
- Create: `session/__init__.py`

- [ ] **Step 1: Create `session/manager.py`**

```python
# session/manager.py
import time
import hashlib
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Session:
    id: str
    messages: list = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)


class SessionManager:
    def __init__(self, ttl_minutes: int = 30, max_context_tokens: int = 4096):
        self.ttl_seconds = ttl_minutes * 60
        self.max_context_tokens = max_context_tokens
        self._sessions: dict[str, Session] = {}

    def _hash_id(self, session_id: str) -> str:
        return hashlib.md5(session_id.encode()).hexdigest()

    def get_or_create(self, session_id: str) -> Session:
        self._cleanup_expired()
        key = self._hash_id(session_id)
        if key in self._sessions:
            self._sessions[key].updated_at = time.time()
            return self._sessions[key]
        session = Session(id=session_id)
        self._sessions[key] = session
        return session

    def add_message(self, session_id: str, role: str, content: str):
        key = self._hash_id(session_id)
        session = self._sessions.get(key)
        if not session:
            return
        session.messages.append({"role": role, "content": content})
        session.updated_at = time.time()
        self._trim_context(session)

    def get_context(self, session_id: str) -> list:
        key = self._hash_id(session_id)
        session = self._sessions.get(key)
        if not session:
            return []
        return list(session.messages)

    def _trim_context(self, session: Session):
        total = sum(len(m.get("content", "")) // 4 for m in session.messages)
        while total > self.max_context_tokens and len(session.messages) > 1:
            removed = session.messages.pop(1) if session.messages[0].get("role") == "system" else session.messages.pop(0)
            total -= len(removed.get("content", "")) // 4

    def _cleanup_expired(self):
        now = time.time()
        expired = [k for k, v in self._sessions.items() if now - v.updated_at > self.ttl_seconds]
        for k in expired:
            del self._sessions[k]

    def clear(self):
        self._sessions.clear()
```

- [ ] **Step 2: Create `session/__init__.py`**

```python
from .manager import SessionManager, Session

__all__ = ["SessionManager", "Session"]
```

---

### Task 2: Update config.yaml with session section

**Files:**
- Modify: `config.yaml`

- [ ] **Step 1: Add session config**

```yaml
# Session / Context configuration
session:
  enabled: true
  ttl_minutes: 30
  max_context_tokens: 4096
```

Add this block after the `metrics` section (before the end of file).

---

### Task 3: Update LocalEngine to accept optional messages

**Files:**
- Modify: `models/local_engine.py`

- [ ] **Step 1: Modify the `generate` method signature**

```python
    async def generate(self, prompt: str, system_prompt: Optional[str] = None, messages: Optional[list] = None) -> GenerationResult:
        """Generate response using LM Studio API with retry (v0.2)."""
        if messages is not None:
            # Use provided messages directly (session mode)
            request_messages = messages
        else:
            request_messages = []
            if system_prompt:
                request_messages.append({"role": "system", "content": system_prompt})
            request_messages.append({"role": "user", "content": prompt})
```

Replace the current `messages = []` / `if system_prompt:` block.

---

### Task 4: Update CloudClient to accept optional messages

**Files:**
- Modify: `cloud/client.py`

- [ ] **Step 1: Modify the `generate` method signature**

```python
    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2000,
        optimize: bool = True,
        messages: Optional[list] = None
    ) -> Dict[str, Any]:
```

- [ ] **Step 2: Use messages if provided, otherwise assemble from prompt**

```python
        if messages is not None:
            # Use provided messages directly (session mode)
            chat_messages = list(messages)
        else:
            chat_messages = []
            # Add system prompt to respond in the same language as the query
            import re
            lang_hint = "Please respond in the same language as the user's question." if re.search(r'[^\x00-\x7F]', prompt) else ""
            if lang_hint:
                chat_messages.append({"role": "system", "content": lang_hint})
            if system_prompt:
                chat_messages.append({"role": "system", "content": system_prompt})
            chat_messages.append({"role": "user", "content": prompt})
```

Replace the current `messages = []` block with this.

---

### Task 5: Update main.py — session support in route_request

**Files:**
- Modify: `main.py`

- [ ] **Step 1: Add import for SessionManager**

Add after `from cloud.client import CloudClient, create_cloud_client_from_config`:
```python
from session.manager import SessionManager
```

- [ ] **Step 2: Initialize SessionManager**

Add after `metrics_logger = MetricsLogger(...)`:
```python
# Session manager
session_manager = SessionManager(
    ttl_minutes=router_engine.config.get("session", {}).get("ttl_minutes", 30),
    max_context_tokens=router_engine.config.get("session", {}).get("max_context_tokens", 4096)
)
logger.info("Session manager initialized")
```

- [ ] **Step 3: Add session_id to RouteRequest**

Replace the existing `RouteRequest` class:
```python
class RouteRequest(BaseModel):
    query: str
    task_type: Optional[str] = None
    context: Optional[Dict[str, Any]] = None
    session_id: Optional[str] = None
```

- [ ] **Step 4: Add session handling in route_request**

Add after the language detection block (after `subtasks = []`):
```python
    # Session / context handling
    session = None
    session_messages = None
    if request.session_id:
        session = session_manager.get_or_create(request.session_id)
        session_manager.add_message(request.session_id, "user", request.query)
        session_messages = session_manager.get_context(request.session_id)
        # Skip cache for session-aware requests
        should_check_cache = False
    else:
        should_check_cache = True
```

- [ ] **Step 5: Wrap cache check in session guard**

Replace `cached = await cache.get_response(request.query)` with:
```python
    cached = None
    if should_check_cache:
        cached = await cache.get_response(request.query)
```

- [ ] **Step 6: Pass messages param to local_engine.generate() calls**

In the LOCAL handler, replace:
```python
result = await local_engine.generate(
    request.query,
    system_prompt=original_lang_hint if original_lang_hint else None
)
```
with:
```python
result = await local_engine.generate(
    request.query,
    system_prompt=original_lang_hint if original_lang_hint else None,
    messages=session_messages
)
```

- [ ] **Step 7: Pass messages param to _call_cloud()**

Modify the `_call_cloud` helper — add `extra_messages` parameter:
```python
    async def _call_cloud(query_text: str, russian_override: bool = False, extra_messages: Optional[list] = None):
        ...
        if extra_messages is not None:
            result, cloud_used = await _cloud_with_messages(extra_messages)
        else:
            ...
```

Actually simpler: pass messages through the existing flow. The `_call_cloud` helper already calls `cloud_client.generate(cloud_prompt, system_prompt=...)`. Change it to pass `messages=session_messages` when available.

Replace the `_call_cloud` body to:
```python
    async def _call_cloud(query_text: str, russian_override: bool = False):
        if not cloud_client:
            raise HTTPException(status_code=503, detail="Cloud client not configured")
        needs_russian = russian_override or is_russian
        cloud_prompt = query_text
        if needs_russian:
            cloud_prompt = "Ответьте обязательно на русском языке.\n\n" + query_text
        cloud_result = await cloud_client.generate(
            cloud_prompt,
            system_prompt=original_lang_hint if original_lang_hint else None,
            messages=session_messages
        )
        cloud_used = cloud_result.get("usage", {}).get("total_tokens", 0)
        if cloud_used > 0:
            update_cloud_output_avg(cloud_used)
        return cloud_result["content"], cloud_used
```

- [ ] **Step 8: Add assistant message to session after response**

Add before `# Cache response`:
```python
        # Store assistant response in session
        if session is not None:
            session_manager.add_message(request.session_id, "assistant", response_text)
```

- [ ] **Step 9: Guard cache write behind session check**

Replace `await cache.set_response(request.query, {"text": response_text})` with:
```python
        # Cache response (only for non-session requests)
        if should_check_cache:
            await cache.set_response(request.query, {"text": response_text})
```

---

### Task 6: Write tests

**Files:**
- Create: `tests/test_session.py`

- [ ] **Step 1: Write test_session.py**

```python
import pytest
import time
from session.manager import SessionManager


class TestSessionManager:
    def test_create_session(self):
        sm = SessionManager()
        session = sm.get_or_create("test-123")
        assert session.id == "test-123"
        assert session.messages == []

    def test_get_or_create_returns_same_session(self):
        sm = SessionManager()
        s1 = sm.get_or_create("abc")
        s2 = sm.get_or_create("abc")
        assert s1 is s2

    def test_add_and_get_messages(self):
        sm = SessionManager()
        sm.get_or_create("s1")
        sm.add_message("s1", "user", "Hello")
        sm.add_message("s1", "assistant", "Hi there")
        ctx = sm.get_context("s1")
        assert len(ctx) == 2
        assert ctx[0]["role"] == "user"
        assert ctx[0]["content"] == "Hello"
        assert ctx[1]["role"] == "assistant"

    def test_context_trim_by_tokens(self):
        sm = SessionManager(max_context_tokens=50)
        sm.get_or_create("s1")
        sm.add_message("s1", "system", "You are a helpful bot.")
        # Add messages that exceed max tokens
        long_msg = "word " * 60  # ~240 chars = ~60 tokens > 50
        sm.add_message("s1", "user", long_msg)
        sm.add_message("s1", "assistant", "ok")
        ctx = sm.get_context("s1")
        # Should have been trimmed
        assert len(ctx) < 4  # At least one message was removed

    def test_ttl_cleanup(self):
        sm = SessionManager(ttl_minutes=0)  # 0 TTL = immediate expiry
        sm.get_or_create("expired-session")
        # After a tiny sleep the session should be expired
        time.sleep(0.01)
        # get_or_create triggers cleanup, should create a new one
        session = sm.get_or_create("expired-session")
        assert len(sm.get_context("expired-session")) == 0

    def test_context_empty_for_unknown_session(self):
        sm = SessionManager()
        ctx = sm.get_context("nonexistent")
        assert ctx == []

    def test_session_id_hashing(self):
        sm = SessionManager()
        s1 = sm.get_or_create("user-1")
        s2 = sm.get_or_create("user-1")  # same string
        s3 = sm.get_or_create("user1")    # different string
        assert s1 is s2
        assert s1 is not s3
```

- [ ] **Step 2: Run tests to verify**

```bash
python -m pytest tests/test_session.py -v
```
Expected: 7 passed

---

### Task 7: Regression test — run all tests

**Files:** none

- [ ] **Step 1: Run full test suite**

```bash
python -m pytest tests/ -v
```
Expected: 21 passed (14 existing + 7 new)
