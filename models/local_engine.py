import httpx
import json
import re
import asyncio
from typing import Optional, Dict, Any, AsyncGenerator
from loguru import logger

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

    async def generate(self, prompt: str, system_prompt: Optional[str] = None, messages: Optional[list] = None) -> GenerationResult:
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
                # Log request for debugging
                logger.debug(f"Sending request to {self.base_url}/chat/completions (attempt {attempt + 1})")

                response = await self.client.post(
                    f"{self.base_url}/chat/completions",
                    json={
                        "model": self.model,
                        "messages": request_messages,
                        "temperature": 0.7,
                        "max_tokens": 2000
                    }
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

                # Get confidence and uncertainty flags via self-assessment (v0.2)
                confidence, flags = await self.run_self_assessment(prompt, text)

                # Override: detect "I don't know" phrases in response
                uncertainty_phrases = [
                    "к сожалению", "не знаю", "не имею доступа", "не могу предоставить",
                    "нет доступа", "не располагаю", "не имею информации",
                    "unfortunately", "don't know", "no access", "cannot provide",
                    "не могу сказать", "не уверен", "требует уточнения"
                ]
                text_lower = text.lower()
                if any(phrase in text_lower for phrase in uncertainty_phrases):
                    logger.warning(f"Detected uncertainty phrase in response, lowering confidence to 0.3")
                    confidence = 0.3
                    flags.append("недостаточно_контекста")

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

    async def run_self_assessment(self, query: str, response: str) -> tuple[float, list]:
        """Ask the model to assess its own confidence and return flags (v0.2)."""
        assessment_prompt = f"""Query: {query}

Response: {response}

Assess your confidence and identify any issues. Respond in this exact JSON format:
{{"confidence": 0.85, "flags": ["неизвестный_термин", "недостаточно_контекста"]}}

Available flags: "неизвестный_термин", "недостаточно_контекста", "требует_фактчекинг", "двусмысленный_запрос", "может_быть_устаревшим"
If no issues, use empty array for flags: "flags": []"""

        try:
            result = await self.client.post(
                f"{self.base_url}/chat/completions",
                json={
                    "model": self.model,
                    "messages": [{"role": "user", "content": assessment_prompt}],
                    "temperature": 0.1,
                    "max_tokens": 100
                }
            )
            result.raise_for_status()
            assessment_text = result.json()["choices"][0]["message"]["content"].strip()

            # Parse JSON from response
            import json as json_module
            try:
                # Try to extract JSON
                import re
                json_match = re.search(r'\{.*\}', assessment_text, re.DOTALL)
                if json_match:
                    data = json_module.loads(json_match.group())
                    confidence = float(data.get("confidence", 0.5))
                    flags = list(data.get("flags", []))
                    return max(0.0, min(1.0, confidence)), flags
            except:
                pass

            # Fallback: extract just number
            num_match = re.search(r'(\d+\.?\d*)', assessment_text)
            if num_match:
                confidence = float(num_match.group(1))
                return max(0.0, min(1.0, confidence)), []
            return 0.5, []

        except Exception as e:
            logger.warning(f"Self-assessment failed: {e}, defaulting to 0.5 with no flags")
            return 0.5, []

    async def health_check(self) -> bool:
        """Check if LM Studio is available."""
        try:
            response = await self.client.get(f"{self.base_url}/models")
            return response.status_code == 200
        except Exception as e:
            logger.warning(f"Health check failed: {e}")
            return False

    async def generate_stream(self, prompt: str, system_prompt: Optional[str] = None, messages: Optional[list] = None) -> AsyncGenerator[Dict[str, Any], None]:
        """Generate response as SSE stream (no self-assessment)."""
        if messages is not None:
            request_messages = messages
        else:
            request_messages = []
            if system_prompt:
                request_messages.append({"role": "system", "content": system_prompt})
            request_messages.append({"role": "user", "content": prompt})

        async with self.client.stream(
            "POST",
            f"{self.base_url}/chat/completions",
            json={
                "model": self.model,
                "messages": request_messages,
                "temperature": 0.7,
                "max_tokens": 2000,
                "stream": True
            }
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
