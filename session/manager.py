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
        while total > self.max_context_tokens and len(session.messages) > 0:
            idx = 1 if len(session.messages) > 1 and session.messages[0].get("role") == "system" else 0
            removed = session.messages.pop(idx)
            total -= len(removed.get("content", "")) // 4

    def _cleanup_expired(self):
        now = time.time()
        expired = [k for k, v in self._sessions.items() if now - v.updated_at > self.ttl_seconds]
        for k in expired:
            del self._sessions[k]

    def clear(self):
        self._sessions.clear()
