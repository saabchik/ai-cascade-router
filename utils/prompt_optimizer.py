# utils/prompt_optimizer.py
import re
from typing import Optional

def optimize_prompt_for_cloud(
    user_prompt: str,
    system_prompt: Optional[str] = None,
    max_length: int = 2000
) -> tuple[str, Optional[str]]:
    """
    Optimize prompts to reduce token usage.
    
    Returns: (optimized_user_prompt, optimized_system_prompt)
    """
    
    # Remove extra whitespace
    user_prompt = re.sub(r'\s+', ' ', user_prompt).strip()
    if system_prompt:
        system_prompt = re.sub(r'\s+', ' ', system_prompt).strip()
    
    # Truncate if too long
    if len(user_prompt) > max_length:
        user_prompt = user_prompt[:max_length] + "..."
    
    # Simplify system prompt if present
    if system_prompt:
        # Keep only essential instructions
        essential_keywords = ["you are", "help", "answer", "question"]
        system_lower = system_prompt.lower()
        
        # If system prompt is very long, shorten it
        if len(system_prompt) > 200:
            # Keep first 150 chars which usually contain key instructions
            system_prompt = system_prompt[:150] + "..."
    
    return user_prompt, system_prompt


def estimate_tokens(text: str) -> int:
    """Rough estimate of tokens (roughly 4 chars per token)."""
    return len(text) // 4


