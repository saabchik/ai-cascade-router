# main.py
"""
AI Cascade Router - Smart LLM proxy that saves 40-70% tokens.
"""
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional, Dict, Any
import os
import re
import json
from dotenv import load_dotenv
from loguru import logger

# Routers and engines
from router.engine import RouterEngine, RouteDecision
from models.local_engine import LocalEngine
from cache.manager import CacheManager
from metrics.logger import MetricsLogger
from cloud.client import CloudClient, create_cloud_client_from_config
from session.manager import SessionManager

# Load environment
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

# Initialize FastAPI
app = FastAPI(
    title="AI Cascade Router",
    version="0.7.0",
    description="Smart LLM proxy: saves up to 70% tokens by routing to local models"
)

# ======================================
# Configuration and Initialization
# ======================================

# Config path
config_path = os.path.join(os.path.dirname(__file__), "config.yaml")

# Language detection
_RUSSIAN_MARKER = "MUST respond in Russian"

def _is_russian_text(text: str) -> bool:
    """Check if text contains Cyrillic characters."""
    return bool(re.search(r'[а-яё]', text.lower()))

def get_language_hint(text: str) -> str:
    """Return system prompt to respond in the same language as the query."""
    if re.search(r'[^\x00-\x7F]', text):
        # Check if it's Russian (Cyrillic)
        if _is_russian_text(text):
            return f"You {_RUSSIAN_MARKER} (Русский язык). Это обязательно."
        return "Please respond in the same language as the user's question."
    return ""

# Cloud token tracking for baseline calculation
_cloud_output_avg = 200  # fallback when no history
_cloud_output_count = 0

def estimate_baseline_tokens(prompt: str) -> int:
    """Estimate tokens if query went to cloud (baseline without router)."""
    prompt_tokens = len(prompt) // 4  # ~4 chars per token
    return prompt_tokens + _cloud_output_avg

def update_cloud_output_avg(output_tokens: int):
    """Update rolling average of cloud output tokens."""
    global _cloud_output_avg, _cloud_output_count
    if output_tokens > 0:
        _cloud_output_count += 1
        _cloud_output_avg = (_cloud_output_avg * (_cloud_output_count - 1) + output_tokens) / _cloud_output_count

# ======================================
# Initialize Components
# ======================================

logger.info("=" * 60)
logger.info("Initializing AI Cascade Router...")
logger.info("=" * 60)

# Router engine (ML classifier + semantic cache)
router_engine = RouterEngine(config_path=config_path)
logger.info(f"Router engine loaded, config: {config_path}")

# Cache manager
cache = CacheManager(
    max_size_mb=100,
    ttl_responses=3600,
    ttl_computations=86400
)
logger.info("Cache manager initialized")

# Metrics logger
metrics_logger = MetricsLogger(router_engine.config.get("metrics"))
logger.info("Metrics logger initialized")

# Session manager
session_manager = SessionManager(
    ttl_minutes=router_engine.config.get("session", {}).get("ttl_minutes", 30),
    max_context_tokens=router_engine.config.get("session", {}).get("max_context_tokens", 8192)
)
logger.info("Session manager initialized")

# Local engine (LM Studio)
local_engine = None
try:
    local_config = router_engine.config.get("local_model", {})
    model_name = local_config.get("model_name", "qwen2.5-7b-instruct")
    local_engine = LocalEngine(
        base_url=local_config.get("base_url", "http://localhost:1234/v1"),
        model=model_name,
        timeout=local_config.get("timeout_seconds", 30)
    )
    logger.info(f"✓ Local engine (LM Studio): {model_name}")
except Exception as e:
    logger.warning(f"✗ Failed to initialize LM Studio engine: {e}")
    logger.warning("  → Load a model in LM Studio (e.g., qwen2.5-7b-instruct)")

# Cloud client (OpenRouter)
cloud_client = None
try:
    cloud_client = create_cloud_client_from_config(router_engine.config)
    logger.info(f"☁ Cloud client (OpenRouter): {cloud_client.model}")
except Exception as e:
    logger.warning(f"✗ Failed to initialize cloud client: {e}")
    logger.warning("  → Check OPENROUTER_API_KEY in .env")

logger.info("=" * 60)
logger.info("AI Cascade Router ready!")
logger.info("=" * 60)


class RouteRequest(BaseModel):
    query: str
    task_type: Optional[str] = None
    context: Optional[Dict[str, Any]] = None
    session_id: Optional[str] = None


class RouteResponse(BaseModel):
    response: str
    source: str
    confidence: float
    tokens_saved: int = 0
    tokens_saved_percent: float = 0.0
    response_time_ms: float = 0.0
    warning: Optional[str] = None


# OpenAI-compatible request format (for VS Code, Cursor, Continue, Cline, etc.)
class OpenAIMessage(BaseModel):
    role: str
    content: str

class OpenAIRequest(BaseModel):
    model: str = "cascade-router"
    messages: list[OpenAIMessage]
    stream: bool = False
    temperature: float = 0.7
    max_tokens: int = 2000

@app.post("/v1/chat/completions")
async def openai_chat_completions(request: OpenAIRequest):
    """OpenAI-compatible endpoint for integration with VS Code/Cursor/Continue/Cline."""
    import time
    start = time.time()

    user_msg = next((m.content for m in reversed(request.messages) if m.role == "user"), None)
    if not user_msg:
        raise HTTPException(status_code=400, detail="No user message found")

    original_lang_hint = get_language_hint(user_msg)
    full_messages = [{"role": m.role, "content": m.content} for m in request.messages]

    if request.stream:
        return await _openai_stream(user_msg, original_lang_hint, full_messages, request)
    else:
        return await _openai_nonstream(user_msg, original_lang_hint, full_messages, request, start)


async def _openai_nonstream(user_msg: str, original_lang_hint: str, full_messages: list, request: OpenAIRequest, start: float):
    """Non-streaming path for /v1/chat/completions."""
    route_result = router_engine.evaluate_request(user_msg, None)
    threshold = router_engine.get_threshold_for_query(user_msg)

    response_text = ""
    source = "unknown"

    if route_result.decision in (RouteDecision.LOCAL, RouteDecision.HYBRID) and local_engine:
        local_ok = False
        try:
            result = await local_engine.generate(
                user_msg,
                system_prompt=original_lang_hint if original_lang_hint else None,
                messages=full_messages
            )
            if result.confidence >= threshold:
                response_text = result.text
                source = "local"
                local_ok = True
        except Exception as e:
            logger.warning(f"OpenAI local failed: {e}, falling back to cloud")

        if not local_ok:
            cloud_resp = await cloud_client.generate(
                user_msg,
                system_prompt=original_lang_hint if original_lang_hint else None,
                messages=full_messages
            )
            response_text = cloud_resp["content"]
            source = "local->cloud" if route_result.decision == RouteDecision.LOCAL else "hybrid"
    else:
        cloud_resp = await cloud_client.generate(
            user_msg,
            system_prompt=original_lang_hint if original_lang_hint else None,
            messages=full_messages
        )
        response_text = cloud_resp["content"]
        source = "cloud"

    return {
        "id": f"chatcmpl-{int(time.time())}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": request.model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": response_text},
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": len(user_msg) // 4,
            "completion_tokens": len(response_text) // 4,
            "total_tokens": (len(user_msg) + len(response_text)) // 4
        },
        "cascade_meta": {
            "source": source,
            "response_time_ms": round((time.time() - start) * 1000, 2)
        }
    }


async def _openai_stream(user_msg: str, original_lang_hint: str, full_messages: list, request: OpenAIRequest):
    """Streaming path for /v1/chat/completions."""
    route_result = router_engine.evaluate_request(user_msg, None)
    threshold = router_engine.get_threshold_for_query(user_msg)

    use_local = route_result.decision in (RouteDecision.LOCAL, RouteDecision.HYBRID) and local_engine

    async def event_stream():
        try:
            if use_local:
                async for chunk in local_engine.generate_stream(
                    user_msg,
                    system_prompt=original_lang_hint if original_lang_hint else None,
                    messages=full_messages
                ):
                    yield f"data: {json.dumps(chunk)}\n\n"
                yield "data: [DONE]\n\n"
                return
        except Exception as e:
            logger.warning(f"Local stream failed: {e}, falling back to cloud")

        async for chunk in cloud_client.generate_stream(
            user_msg,
            system_prompt=original_lang_hint if original_lang_hint else None,
            messages=full_messages
        ):
            yield f"data: {json.dumps(chunk)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.get("/health")
async def health_check():
    local_healthy = await local_engine.health_check() if local_engine else False
    cloud_healthy = await cloud_client.health_check() if cloud_client else False
    status = "healthy" if (local_healthy or cloud_healthy) else "unhealthy"
    return {
        "status": status,
        "local_model": "available" if local_healthy else "unavailable",
        "cloud_model": "available" if cloud_healthy else "unavailable",
        "version": "0.7.0"
    }


@app.post("/route", response_model=RouteResponse)
async def route_request(request: RouteRequest):
    import time
    start_time = time.time()

    # Detect original query language
    original_lang_hint = get_language_hint(request.query)
    is_russian = _is_russian_text(request.query)
    subtasks = []

    # Session / context handling
    session = None
    session_messages = None
    should_check_cache = True
    if request.session_id:
        session = session_manager.get_or_create(request.session_id)
        session_manager.add_message(request.session_id, "user", request.query)
        session_messages = session_manager.get_context(request.session_id)
        should_check_cache = False

    # Local helper: call cloud with Russian prompt prepend if needed
    async def _call_cloud(query_text: str, russian_override: bool = False, include_session: bool = True):
        if not cloud_client:
            raise HTTPException(status_code=503, detail="Cloud client not configured")
        needs_russian = russian_override or is_russian
        cloud_prompt = query_text
        if needs_russian:
            cloud_prompt = "Ответьте обязательно на русском языке.\n\n" + query_text
        msgs = session_messages if include_session else None
        cloud_result = await cloud_client.generate(
            cloud_prompt,
            system_prompt=original_lang_hint if original_lang_hint else None,
            messages=session_messages
        )
        cloud_used = cloud_result.get("usage", {}).get("total_tokens", 0)
        if cloud_used > 0:
            update_cloud_output_avg(cloud_used)
        return cloud_result["content"], cloud_used

    # Health check
    local_available = False
    if local_engine:
        local_available = await local_engine.health_check()
        if not local_available:
            logger.warning("Local model is NOT healthy! Subtasks may fallback to cloud.")
    else:
        logger.warning("Local engine is None! All local/hybrid will fallback to cloud.")

    # Tokens tracking
    baseline_tokens = 0
    actual_cloud_tokens = 0

    # Check cache first
    cached = None
    if should_check_cache:
        cached = await cache.get_response(request.query)
    if cached:
        tokens_saved = max(0, estimate_baseline_tokens(request.query))
        return RouteResponse(
            response=cached["text"],
            source="cache",
            confidence=1.0,
            tokens_saved=int(tokens_saved),
            tokens_saved_percent=100.0,
            response_time_ms=round((time.time() - start_time) * 1000, 2)
        )

    # Evaluate routing
    if request.task_type is not None:
        route_result = router_engine.evaluate_request(request.query, request.task_type)
        cascade_result = None
    else:
        cascade_result = await router_engine.evaluate_request_cascade(request.query)
        if cascade_result.get("mode") == "single":
            route_result = cascade_result["decision"]
            cascade_result = None
        else:
            route_result = None

    try:
        response_text = ""
        source = "unknown"
        result = None
        confidence = 0.5

        if cascade_result is None:
            # Single query mode
            baseline_tokens = estimate_baseline_tokens(request.query)
            # Include session context in baseline (history would be sent to cloud too)
            if session_messages:
                for msg in session_messages:
                    baseline_tokens += len(msg.get("content", "")) // 4

            # Get domain-specific threshold for this query
            threshold = router_engine.get_threshold_for_query(request.query)

            if route_result.decision == RouteDecision.LOCAL:
                if not local_engine:
                    raise HTTPException(status_code=503, detail="Local model unavailable")

                local_ok = False
                try:
                    result = await local_engine.generate(
                        request.query,
                        system_prompt=original_lang_hint if original_lang_hint else None,
            messages=msgs
                    )
                    if result.confidence >= threshold:
                        response_text = result.text
                        source = "local"
                        local_ok = True
                except Exception as e:
                    logger.warning(f"Local model failed: {e}, falling back to cloud")

                if not local_ok:
                    response_text, cloud_used = await _call_cloud(request.query)
                    source = "local->cloud"
                    actual_cloud_tokens += cloud_used
                    result = None

            elif route_result.decision == RouteDecision.CLOUD:
                response_text, cloud_used = await _call_cloud(request.query)
                source = "cloud"
                actual_cloud_tokens += cloud_used
                result = None

            else:  # HYBRID
                if not local_engine:
                    raise HTTPException(status_code=503, detail="Local model unavailable")
                try:
                    result = await local_engine.generate(
                        request.query,
                        system_prompt="Provide a complete and accurate response."
                    )
                    if result.confidence >= threshold:
                        response_text = result.text
                        source = "local"
                    else:
                        logger.warning(f"Hybrid local confidence {result.confidence} < threshold {threshold}, falling back to cloud")
                        response_text, cloud_used = await _call_cloud(request.query)
                        source = "hybrid"
                        actual_cloud_tokens += cloud_used
                        result = None
                except Exception as e:
                    logger.warning(f"Hybrid local failed: {e}, falling back to cloud")
                    response_text, cloud_used = await _call_cloud(request.query)
                    source = "hybrid"
                    actual_cloud_tokens += cloud_used
                    result = None

        # CASCADE MODE
        elif cascade_result is not None:
            if not local_engine:
                raise HTTPException(status_code=503, detail="Local model unavailable")
            subtasks = cascade_result.get("subtasks", [])
            responses = []

            # Calculate baseline for all subtasks
            baseline_tokens = sum(estimate_baseline_tokens(st.get("action", "")) for st in subtasks)

            # For threshold, use the task_type from first subtask or default
            first_task_type = subtasks[0].get("task_type", "question") if subtasks else "question"
            threshold = router_engine.criteria.get_threshold_for_domain(first_task_type)

            logger.info(f"Cascade mode: {len(subtasks)} subtasks, baseline={baseline_tokens}, threshold={threshold}")

            if not local_available:
                logger.warning(f"Local model is down! All subtasks will go to cloud. tokens_saved will be 0.")

            for subtask_info in subtasks:
                sub_action = subtask_info.get("action", "")
                sub_route = subtask_info.get("route", "local")
                try:
                    if sub_route == "local":
                        if not local_available:
                            raise Exception("Local model not available")
                        lang_hint = original_lang_hint or get_language_hint(sub_action)
                        sub_result = await local_engine.generate(
                            sub_action,
                            system_prompt=lang_hint if lang_hint else None
                        )

                        # Check confidence - fallback to cloud if too low
                        if sub_result.confidence < threshold:
                            logger.warning(f"Local confidence {sub_result.confidence} < threshold {threshold}, falling back to cloud")
                            content, cloud_used = await _call_cloud(sub_action, include_session=False)
                            responses.append(content)
                            if cloud_used == 0:
                                cloud_used = len(content) // 4 + len(sub_action) // 4
                            actual_cloud_tokens += cloud_used
                            logger.info(f"Subtask [local→cloud]: cloud_used={cloud_used}, total={actual_cloud_tokens}")
                        else:
                            responses.append(sub_result.text)
                            logger.info(f"Subtask [{sub_route}]: local, confidence={sub_result.confidence}")

                    elif sub_route == "cloud":
                        content, cloud_used = await _call_cloud(sub_action, include_session=False)
                        responses.append(content)
                        if cloud_used == 0:
                            cloud_used = len(content) // 4 + len(sub_action) // 4
                            logger.warning(f"Cloud subtask: no usage data, estimated {cloud_used} tokens")
                        actual_cloud_tokens += cloud_used
                        logger.info(f"Subtask [{sub_route}]: cloud_used={cloud_used}, total={actual_cloud_tokens}")

                    elif sub_route == "hybrid":
                        if not local_available:
                            content, cloud_used = await _call_cloud(sub_action, include_session=False)
                            responses.append(content)
                            if cloud_used == 0:
                                cloud_used = len(content) // 4 + len(sub_action) // 4
                            actual_cloud_tokens += cloud_used
                            logger.info(f"Subtask [hybrid→cloud]: no local, cloud_used={cloud_used}")
                        else:
                            lang_hint = original_lang_hint or get_language_hint(sub_action)
                            sub_result = await local_engine.generate(
                                sub_action,
                                system_prompt=lang_hint if lang_hint else None
                            )
                            if sub_result.confidence >= threshold:
                                responses.append(sub_result.text)
                                logger.info(f"Subtask [hybrid]: local, confidence={sub_result.confidence}")
                            else:
                                logger.warning(f"Hybrid local confidence {sub_result.confidence} < threshold {threshold}, falling back to cloud")
                                content, cloud_used = await _call_cloud(sub_action, include_session=False)
                                responses.append(content)
                                if cloud_used == 0:
                                    cloud_used = len(content) // 4 + len(sub_action) // 4
                                actual_cloud_tokens += cloud_used
                                logger.info(f"Subtask [hybrid→cloud]: cloud_used={cloud_used}, total={actual_cloud_tokens}")

                except Exception as e:
                    logger.error(f"Subtask error ({sub_route}): {e}")
                    responses.append(f"[Error: {e}]")

            responses = [r for r in responses if r is not None]
            response_text = "\n\n---\n\n".join(responses) if responses else "No response from models"
            error_count = sum(1 for r in responses if r.startswith("[Error:"))
            if error_count == len(responses):
                source = "failed"
            elif error_count > 0:
                source = "cascade-partial"
            else:
                source = "cascade"

            logger.info(f"Cascade done: baseline={baseline_tokens}, actual_cloud={actual_cloud_tokens}")

        # Store assistant response in session
        if session is not None:
            session_manager.add_message(request.session_id, "assistant", response_text)

        # Cache response (skip for session-aware requests)
        if should_check_cache:
            await cache.set_response(request.query, {"text": response_text})

        # Calculate savings
        logger.info(f"Final calc: baseline={baseline_tokens}, actual_cloud={actual_cloud_tokens}")
        tokens_saved = max(0, baseline_tokens - actual_cloud_tokens)
        if baseline_tokens > 0:
            raw_percent = (1 - actual_cloud_tokens / baseline_tokens) * 100
            tokens_saved_percent = max(0.0, round(raw_percent, 2))
        else:
            tokens_saved_percent = 0.0
            if not subtasks:
                logger.warning("baseline_tokens=0! No subtasks or empty actions.")

        # Log metrics
        response_time = (time.time() - start_time) * 1000
        metrics_logger.record_request(
            decision=source,
            tokens_used=actual_cloud_tokens,
            cloud_tokens_saved=int(tokens_saved),
            response_time_ms=response_time
        )

        # Confidence
        if cascade_result is not None:
            confidence = 0.85
        elif result is not None:
            confidence = result.confidence
        else:
            confidence = 0.5

        # Warning if local model is down
        warning = None
        if not local_available:
            warning = (
                "Local model (LM Studio) is NOT available or using a REASONING model (deepseek, etc.). "
                "All subtasks went to cloud. "
                "Load a FAST model in LM Studio (qwen2.5-7b-instruct, mistral-7b) to save tokens."
            )
        elif source in ["cascade", "cascade-partial"] and tokens_saved == 0:
            warning = (
                "No token savings detected. This usually means ALL subtasks went to cloud. "
                "Check if LM Studio is running with a FAST model (not reasoning!)."
            )

        return RouteResponse(
            response=response_text,
            source=source,
            confidence=confidence,
            tokens_saved=int(tokens_saved),
            tokens_saved_percent=tokens_saved_percent,
            response_time_ms=round(response_time, 2),
            warning=warning
        )

    except Exception as e:
        logger.error(f"Routing error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics")
async def get_metrics():
    return metrics_logger.export_metrics()


@app.on_event("shutdown")
async def shutdown_event():
    if local_engine:
        await local_engine.close()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
