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


def extract_key_context(prompt: str) -> str:
    """Extract only key information from a long prompt."""
    # Remove common filler phrases
    filler_phrases = [
        "please ", "could you ", "can you ", "would you ",
        "I need you to ", "I want you to ", "help me to ",
        "would be great if ", "it would be nice if "
    ]
    
    result = prompt.lower()
    for filler in filler_phrases:
        result = result.replace(filler, "")
    
    # Capitalize first letter
    if result:
        result = result[0].upper() + result[1:]
    
    return result[:500] if len(result) > 500 else result


def estimate_tokens(text: str) -> int:
    """Rough estimate of tokens (roughly 4 chars per token)."""
    return len(text) // 4


def should_use_cloud(
    local_confidence: float,
    tokens_saved_local: int,
    estimated_cloud_tokens: int,
    cloud_cost_per_1k: float = 0.001
) -> bool:
    """
    Decide if cloud is worth the cost vs local with low confidence.
    
    Returns True if cloud should be used.
    """
    local_cost = 0
    cloud_cost = (estimated_cloud_tokens / 1000) * cloud_cost_per_1k
    
    # If local has low confidence, cloud might be worth the extra cost
    if local_confidence < 0.6:
        return True
    
    # If local is very slow, cloud might be faster (not measured here)
    
    # Default to local if confidence is good
    return False