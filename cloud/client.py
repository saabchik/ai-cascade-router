import httpx
import os
import random
from typing import Optional, Dict, Any, List
from loguru import logger
from utils.prompt_optimizer import optimize_prompt_for_cloud, estimate_tokens

DEFAULT_FREE_MODELS = [
    "meta-llama/llama-3.1-8b-instruct:free",
    "google/gemma-2-9b-it:free",
    "qwen/qwen2.5-7b-instruct:free",
    "mistralai/mistral-7b-instruct:free",
    "deepseek/deepseek-chat:free",
]

class CloudClient:
    def __init__(
        self,
        provider: str = "openrouter",
        api_key: Optional[str] = None,
        model: str = "random_free",
        base_url: str = "https://openrouter.ai/api/v1",
        free_models: Optional[List[str]] = None
    ):
        self.provider = provider
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        self.base_url = base_url.rstrip("/")
        # Only use free_models if explicitly provided and model is "random_free"
        if model == "random_free" and free_models:
            self.free_models = free_models
        else:
            self.free_models = []  # No fallback to default

        # Select random free model
        if model == "random_free":
            self.model = random.choice(self.free_models)
            logger.info(f"Selected random free model: {self.model}")
            logger.info(f"Free models list: {self.free_models}")
        else:
            self.model = model
            logger.info(f"Using specified model: {self.model}")

        if not self.api_key:
            logger.warning("No OPENROUTER_API_KEY found - cloud client will not work")

        self.client = httpx.AsyncClient(
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "HTTP-Referer": "https://github.com/ai-cascade-router",
                "X-Title": "AI Cascade Router"
            },
            timeout=30.0
        )

    async def _try_generate(self, model: str, messages: list, temperature: float, max_tokens: int) -> Dict[str, Any]:
        """Try to generate with a specific model."""
        response = await self.client.post(
            f"{self.base_url}/chat/completions",
            json={
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens
            }
        )
        response.raise_for_status()
        data = response.json()

        # Log the actual model and response for debugging
        actual_model = data.get("model", model)
        content = data["choices"][0]["message"]["content"]
        logger.info(f"Cloud response: model={actual_model}, content_preview={content[:100]}")
        logger.debug(f"Full cloud response: {content[:500]}")

        return {
            "content": content,
            "usage": data.get("usage", {}),
            "model": actual_model
        }

    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2000,
        optimize: bool = True,
        messages: Optional[list] = None
    ) -> Dict[str, Any]:
        """Generate response using OpenRouter API with 404 fallback."""
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY not configured")

        # Optimize prompts to save tokens
        if optimize:
            prompt, system_prompt = optimize_prompt_for_cloud(prompt, system_prompt)
            tokens_saved = estimate_tokens(prompt)
            logger.info(f"Prompt optimized, estimated tokens saved: {tokens_saved}")

        if messages is not None:
            chat_messages = list(messages)
        else:
            chat_messages = []
            import re
            lang_hint = "Please respond in the same language as the user's question." if re.search(r'[^\x00-\x7F]', prompt) else ""
            if lang_hint:
                chat_messages.append({"role": "system", "content": lang_hint})
            if system_prompt:
                chat_messages.append({"role": "system", "content": system_prompt})
            chat_messages.append({"role": "user", "content": prompt})

        # Debug: log the request
        logger.debug(f"Sending to cloud: model={self.model}, messages={str(chat_messages)[:200]}")

        # Try each model in order, skipping on 404/400/timeout
        # If no free_models, just use the configured model directly
        if not self.free_models:
            logger.info(f"Using single model: {self.model}")
            try:
                return await self._try_generate(self.model, chat_messages, temperature, max_tokens)
            except Exception as e:
                logger.error(f"Model {self.model} failed: {e}")
                raise Exception(f"Cloud model {self.model} failed: {e}")

        logger.info(f"Will try models in order: {self.free_models}")
        tried_models = []
        for model in self.free_models:
            try:
                logger.info(f"Trying model: {model}")
                result = await self._try_generate(model, chat_messages, temperature, max_tokens)
                self.model = model  # Update current model on success
                return result
            except (httpx.TimeoutException, httpx.ConnectError) as e:
                logger.warning(f"Model {model} timeout/connection error, trying next...")
                tried_models.append(model)
                continue
            except httpx.HTTPStatusError as e:
                if e.response.status_code in (404, 400):
                    logger.warning(f"Model {model} not available ({e.response.status_code}), trying next...")
                    tried_models.append(model)
                    continue
                logger.error(f"OpenRouter HTTP error: {e.response.status_code} - {e.response.text}")
                raise
            except Exception as e:
                logger.warning(f"Model {model} error: {e}, trying next...")
                tried_models.append(model)
                continue

        # All models failed
        raise Exception(f"All free models failed. Tried: {tried_models}")

    async def health_check(self) -> bool:
        """Check if OpenRouter is available."""
        if not self.api_key:
            return False
        try:
            response = await self.client.get(f"{self.base_url}/models")
            return response.status_code == 200
        except Exception as e:
            logger.warning(f"OpenRouter health check failed: {e}")
            return False

    async def close(self):
        await self.client.aclose()


def create_cloud_client_from_config(config: Dict[str, Any]) -> CloudClient:
    """Factory function to create cloud client from config."""
    cloud_config = config.get("cloud_model", {})
    model = cloud_config.get("model_name", "openai/gpt-4o-mini")
    free_models = cloud_config.get("free_models")

    # Only use free_models if model is "random_free"
    if model != "random_free" or not free_models:
        free_models = None

    return CloudClient(
        provider=cloud_config.get("provider", "openrouter"),
        api_key=cloud_config.get("api_key"),
        model=model,
        base_url=cloud_config.get("base_url", "https://openrouter.ai/api/v1"),
        free_models=free_models
    )