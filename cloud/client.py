import httpx
import os
from typing import Optional, Dict, Any
from loguru import logger
from utils.prompt_optimizer import optimize_prompt_for_cloud, estimate_tokens

class CloudClient:
    def __init__(
        self,
        provider: str = "openrouter",
        api_key: Optional[str] = None,
        model: str = "openai/gpt-4o-mini",
        base_url: str = "https://openrouter.ai/api/v1"
    ):
        self.provider = provider
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        self.base_url = base_url.rstrip("/")
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

    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2000,
        optimize: bool = True,
        messages: Optional[list] = None
    ) -> Dict[str, Any]:
        """Generate response using OpenRouter API."""
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY not configured")

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

        logger.debug(f"Sending to cloud: model={self.model}, messages={str(chat_messages)[:200]}")

        try:
            response = await self.client.post(
                f"{self.base_url}/chat/completions",
                json={
                    "model": self.model,
                    "messages": chat_messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens
                }
            )
            response.raise_for_status()
            data = response.json()

            actual_model = data.get("model", self.model)
            content = data["choices"][0]["message"]["content"]
            logger.info(f"Cloud response: model={actual_model}, content_preview={content[:100]}")
            logger.debug(f"Full cloud response: {content[:500]}")

            return {
                "content": content,
                "usage": data.get("usage", {}),
                "model": actual_model
            }

        except Exception as e:
            logger.error(f"Cloud model {self.model} failed: {e}")
            raise Exception(f"Cloud model {self.model} failed: {e}")

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
    return CloudClient(
        provider=cloud_config.get("provider", "openrouter"),
        api_key=cloud_config.get("api_key"),
        model=cloud_config.get("model_name", "openai/gpt-4o-mini"),
        base_url=cloud_config.get("base_url", "https://openrouter.ai/api/v1")
    )