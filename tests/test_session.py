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
        long_msg = "word " * 60
        sm.add_message("s1", "user", long_msg)
        sm.add_message("s1", "assistant", "ok")
        ctx = sm.get_context("s1")
        assert len(ctx) < 4

    def test_ttl_cleanup(self):
        sm = SessionManager(ttl_minutes=0)
        sm.get_or_create("expired-session")
        time.sleep(0.01)
        session = sm.get_or_create("expired-session")
        assert len(sm.get_context("expired-session")) == 0

    def test_context_empty_for_unknown_session(self):
        sm = SessionManager()
        ctx = sm.get_context("nonexistent")
        assert ctx == []

    def test_session_id_hashing(self):
        sm = SessionManager()
        s1 = sm.get_or_create("user-1")
        s2 = sm.get_or_create("user-1")
        s3 = sm.get_or_create("user1")
        assert s1 is s2
        assert s1 is not s3
