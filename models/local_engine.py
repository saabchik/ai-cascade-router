import httpx
import json
import re
import asyncio
from typing import Optional, Dict, Any, AsyncGenerator
from loguru import logger

UNSURE_PATTERNS = [
    (r'\b(?:I\s+)?don\'t\s+know\b', 0.35),
    (r'\bне знаю\b', 0.35),
    (r'\bcannot\s+provide\b|\bне могу предоставит', 0.30),
    (r'\bno\s+access\b|\bнет доступа\b', 0.30),
    (r'\bnot\s+sure\b|\bне уверен\b', 0.20),
    (r'\bnot\s+confiden', 0.20),
    (r'\bplease\s+consult\b|\bлучше обратиться\b', 0.25),
    (r'\byou\s+should\s+ask\b', 0.25),
    (r'\bsorry,?\s+(?:I\s+)?(?:can\'?t|cannot|don\'?t)\b', 0.20),
    (r'\bизвини,?\s+(?:я\s+)?(?:не\s+могу|не\s+знаю)\b', 0.20),
]


def estimate_confidence(text: str) -> float:
    """Heuristic confidence estimation without a second LLM call.
    
    Base confidence: 0.70. Shifts up/down based on signals in the response text.
    Returns value between 0.10 and 0.95.
    """
    score = 0.70
    text_lower = text.lower()

    # Negative: uncertainty phrases (only strongest match applies)
    max_penalty = 0
    for pattern, weight in UNSURE_PATTERNS:
        if re.search(pattern, text_lower):
            max_penalty = max(max_penalty, weight)
    score -= max_penalty

    # Negative: very short or empty
    if len(text) < 20:
        score -= 0.30
    elif len(text) < 50:
        score -= 0.10

    # Negative: ends with "?" (model is asking back, not answering)
    stripped = text.strip()
    if stripped.endswith('?') and len(stripped) < 100:
        score -= 0.15

    # Positive: code presence
    if '```' in text:
        score += 0.15
    elif any(kw in text_lower for kw in ['def ', 'class ', 'import ', 'function ']):
        score += 0.10

    # Positive: substantive answer length
    if len(text) > 1000:
        score += 0.10
    elif len(text) > 400:
        score += 0.05

    # Positive: structured (bullet, numbered lists, or markdown headings)
    if any(l.strip().startswith('- ') for l in text.split('\n')):
        score += 0.05
    elif any(re.match(r'\d+\.\s', l.strip()) for l in text.split('\n')):
        score += 0.05

    # Positive: markdown headings (###, ##, #) or bold sections
    if re.search(r'^#{1,3}\s', text, re.MULTILINE):
        score += 0.10
    elif '**' in text:
        score += 0.05

    return max(0.10, min(0.95, score))

class GenerationResult:
    def __init__(self, text: str, confidence: float, tokens_used: int = 0, uncertainty_flags: list = None):
        self.text = text
        self.confidence = confidence
        self.tokens_used = tokens_used
        self.uncertainty_flags = uncertainty_flags or []

class LocalEngine:
    def __init__(self, base_url: str, model: str, timeout: int = 30, max_retries: int = 2):
        self.base_url = base_url.rstrip('/')
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries
        self.client = httpx.AsyncClient(timeout=httpx.Timeout(timeout))

    async def generate(self, prompt: str, system_prompt: Optional[str] = None, messages: Optional[list] = None, max_tokens: Optional[int] = None) -> GenerationResult:
        """Generate response using LM Studio API with retry (v0.2)."""
        if messages is not None:
            request_messages = messages
        else:
            request_messages = []
            if system_prompt:
                request_messages.append({"role": "system", "content": system_prompt})
            request_messages.append({"role": "user", "content": prompt})

        last_error = None
        for attempt in range(self.max_retries):
            try:
                logger.debug(f"Sending request to {self.base_url}/chat/completions (attempt {attempt + 1})")

                body = {
                    "model": self.model,
                    "messages": request_messages,
                    "temperature": 0.7,
                }
                if max_tokens is not None:
                    body["max_tokens"] = max_tokens

                response = await self.client.post(
                    f"{self.base_url}/chat/completions",
                    json=body
                )

                # Log response status
                logger.debug(f"Response status: {response.status_code}")

                response.raise_for_status()

                # Try to parse JSON
                try:
                    data = response.json()
                except Exception as json_err:
                    logger.error(f"Failed to parse JSON response: {json_err}")
                    logger.error(f"Response text: {response.text[:500]}")
                    raise

                # Validate response structure
                if "choices" not in data or not data["choices"]:
                    logger.error(f"Invalid response structure - no 'choices': {data}")
                    raise ValueError("Invalid response: missing 'choices' field")

                text = data["choices"][0]["message"]["content"]
                tokens_used = data.get("usage", {}).get("total_tokens", 0)

                logger.debug(f"Generated {tokens_used} tokens, response length: {len(text)}")

                # Heuristic confidence estimation (no second LLM call)
                confidence = estimate_confidence(text)
                flags = []
                if confidence < 0.50:
                    flags.append("низкая_уверенность")
                    logger.warning(f"Low confidence ({confidence}), likely falling back to cloud")

                return GenerationResult(text=text, confidence=confidence, tokens_used=tokens_used, uncertainty_flags=flags)

            except (httpx.HTTPStatusError, httpx.TimeoutException, httpx.ConnectError) as e:
                last_error = e
                # Get detailed error info
                error_type = type(e).__name__
                error_details = str(e)
                if not error_details and hasattr(e, 'response') and e.response:
                    error_details = f"Status {e.response.status_code}: {e.response.text[:200]}"
                elif not error_details:
                    error_details = f"{error_type} (no details)"

                if attempt < self.max_retries - 1:
                    wait_time = 2 ** attempt
                    logger.warning(f"Local model error (attempt {attempt + 1}/{self.max_retries}): {error_details}, retrying in {wait_time}s...")
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(f"Local model failed after {self.max_retries} attempts: {error_details}")

            except Exception as e:
                logger.error(f"Unexpected error generating response: {type(e).__name__}: {e}")
                if attempt == self.max_retries - 1:
                    raise
                last_error = e

        # All retries failed - raise for fallback handling in main.py
        raise last_error or Exception("Local model failed after retries")

    async def health_check(self) -> bool:
        """Check if LM Studio is available."""
        try:
            response = await self.client.get(f"{self.base_url}/models")
            return response.status_code == 200
        except Exception as e:
            logger.warning(f"Health check failed: {e}")
            return False

    async def generate_stream(self, prompt: str, system_prompt: Optional[str] = None, messages: Optional[list] = None, max_tokens: Optional[int] = None) -> AsyncGenerator[Dict[str, Any], None]:
        """Generate response as SSE stream (no self-assessment)."""
        if messages is not None:
            request_messages = messages
        else:
            request_messages = []
            if system_prompt:
                request_messages.append({"role": "system", "content": system_prompt})
            request_messages.append({"role": "user", "content": prompt})

        body = {
            "model": self.model,
            "messages": request_messages,
            "temperature": 0.7,
            "stream": True
        }
        if max_tokens is not None:
            body["max_tokens"] = max_tokens

        async with self.client.stream(
            "POST",
            f"{self.base_url}/chat/completions",
            json=body
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    data = line[6:]
                    if data.strip() == "[DONE]":
                        break
                    yield json.loads(data)

    async def close(self):
        """Close the httpx client."""
        await self.client.aclose()
