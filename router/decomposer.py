import httpx
from typing import List, Optional
from loguru import logger
import re

class Subtask:
    def __init__(self, action: str, task_type: str, complexity: str, index: int):
        self.action = action
        self.task_type = task_type
        self.complexity = complexity
        self.index = index

class DecompositionResult:
    def __init__(self, subtasks: List[Subtask], original_query: str):
        self.subtasks = subtasks
        self.original_query = original_query

class TaskDecomposer:
    def __init__(self, base_url: str = "http://localhost:1234/v1", model: str = "qwen2.5-7b-instruct"):
        self.base_url = base_url
        self.model = model
        self.client = httpx.AsyncClient(timeout=30.0)

    async def decompose(self, query: str) -> DecompositionResult:
        """Decompose complex query into subtasks using LLM."""
        
        # Try LLM-based decomposition
        try:
            prompt = f"""You are a task decomposition assistant. Break down the user's request into smaller, independent subtasks.

For each subtask, identify:
1. What needs to be done (action)
2. What type of task it is (question/coding/analysis/planning)
3. Estimated complexity (simple/medium/complex)

User request: {query}

Respond in this format:
SUBTASKS:
1. [action] (type: question/coding/analysis/planning, complexity: simple/medium/complex)
2. [action] (type: ..., complexity: ...)"""

            response = await self.client.post(
                f"{self.base_url}/chat/completions",
                json={
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.1,
                    "max_tokens": 500
                }
            )
            response.raise_for_status()
            result_text = response.json()["choices"][0]["message"]["content"]
            
            # Parse subtasks from response
            subtasks = self._parse_subtasks(result_text, query)
            if subtasks:
                return DecompositionResult(subtasks, query)
        except Exception as e:
            logger.warning(f"LLM decomposition failed, using fallback: {e}")
        
        # Fallback: pattern-based decomposition
        return self._fallback_decompose(query)

    def _parse_subtasks(self, llm_response: str, original_query: str) -> List[Subtask]:
        """Parse subtasks from LLM response."""
        subtasks = []
        lines = llm_response.split('\n')
        
        for line in lines:
            line = line.strip()
            if not line or not line[0].isdigit():
                continue
            
            # Extract action and metadata
            action_match = re.search(r'^\d+\.\s*(.+?)\s*\(type:\s*(\w+),\s*complexity:\s*(\w+)', line)
            if action_match:
                action = action_match.group(1).strip()
                task_type = action_match.group(2)
                complexity = action_match.group(3)
                subtasks.append(Subtask(action, task_type, complexity, len(subtasks)))
        
        return subtasks

    def _fallback_decompose(self, query: str) -> DecompositionResult:
        """Fallback: split by connectors (English + Russian)."""
        connectors = [
            " and ", " then ", " also ", " plus ", " with ", ", and ",
            " и ", " затем ", " также ", " плюс ", " а также ", " и потом ", ", и "
        ]
        
        subtasks = []
        parts = [query]
        
        for conn in connectors:
            new_parts = []
            for part in parts:
                new_parts.extend(part.split(conn))
            parts = new_parts
        
        # Infer task type and complexity
        for i, part in enumerate(parts):
            part = part.strip()
            if not part:
                continue
            
            # Infer task type
            task_type = "question"
            part_lower = part.lower()
            if any(kw in part_lower for kw in ["code", "write", "function", "program"]):
                task_type = "coding"
            elif any(kw in part_lower for kw in ["analyze", "analysis", "data", "report"]):
                task_type = "analysis"
            elif any(kw in part_lower for kw in ["strategy", "plan", "recommend"]):
                task_type = "planning"
            
            # Infer complexity
            complexity = "simple"
            if any(kw in part_lower for kw in ["analyze", "strategy", "create", "develop", "compare", "evaluate"]):
                complexity = "complex"
            elif any(kw in part_lower for kw in ["explain", "describe", "summarize", "review"]):
                complexity = "medium"
            
            subtasks.append(Subtask(part, task_type, complexity, i))
        
        return DecompositionResult(subtasks if subtasks else [Subtask(query, "question", "simple", 0)], query)

    async def close(self):
        await self.client.aclose()