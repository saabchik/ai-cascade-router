from enum import Enum
from typing import Optional, Dict, Any
import re
import numpy as np
from loguru import logger
from utils.model_cache import get_embedding_model, is_embeddings_available

class Complexity(Enum):
    SIMPLE = "simple"
    MEDIUM = "medium"
    COMPLEX = "complex"

class ClassificationResult:
    def __init__(
        self,
        is_multi_step: bool,
        complexity: Complexity,
        suggested_routing: str,
        confidence: float = 0.8,
        method: str = "rule"
    ):
        self.is_multi_step = is_multi_step
        self.complexity = complexity
        self.suggested_routing = suggested_routing
        self.confidence = confidence
        self.method = method

class SemanticClassifier:
    COMPLEXITY_KEYWORDS = {
        Complexity.COMPLEX: ["analyze", "strategy", "create", "develop", "design", "compare", "evaluate", "recommend", "реши", "создай", "разработай"],
        Complexity.MEDIUM: ["explain", "describe", "summarize", "review", "объясни", "опиши", "резюмируй"],
    }
    
    MULTI_STEP_PATTERNS = [
        " and ", " then ", " also ", " plus ", " with ", ", and ", ";",
        " и ", " затем ", " также ", " плюс ", " а также ", " и потом ", ", и "
    ]
    
    TASK_TYPE_KEYWORDS = {
        "coding": ["code", "write", "function", "program", "код", "напиши", "функция"],
        "analysis": ["analyze", "analysis", "data", "report", "анализ", "данные", "отчёт"],
        "planning": ["strategy", "plan", "recommend", "стратегия", "план", "рекомендация"],
        "question": ["what", "how", "why", "when", "where", "кто", "что", "как", "почему"],
    }

    # Reference queries for ML-based classification
    REFERENCE_QUERIES = {
        Complexity.SIMPLE: [
            "What is Python?",
            "How are you?",
            "What is 2+2?",
            "Что такое Python?",
            "Как дела?",
            "Сколько будет 2+2?",
        ],
        Complexity.MEDIUM: [
            "Explain how Python decorators work",
            "Describe the difference between list and tuple",
            "Summarize this article about AI",
            "Объясни как работают декораторы в Python",
            "Опиши разницу между списком и кортежом",
            "Резюмируй эту статью про ИИ",
        ],
        Complexity.COMPLEX: [
            "Create a detailed analysis of the market strategy and provide recommendations",
            "Develop a comprehensive report on machine learning trends with predictions",
            "Design a system architecture for a scalable web application",
            "Создай подробный анализ рыночной стратегии с рекомендациями",
            "Разработай комплексный отчет о трендах машинного обучения с прогнозами",
            "Спроектируй архитектуру системы для масштабируемого веб-приложения",
        ],
    }

    def __init__(self):
        self._embedding_model = None
        self._reference_embeddings = None
        self._ml_available = False
        self._initialize_ml()

    def _initialize_ml(self):
        """Initialize ML classifier with sentence-transformers."""
        try:
            self._embedding_model = get_embedding_model()
            self._ml_available = is_embeddings_available()

            if self._ml_available:
                # Create reference embeddings
                self._reference_embeddings = {}
                for complexity, queries in self.REFERENCE_QUERIES.items():
                    embeddings = self._embedding_model.encode(queries)
                    # Average embedding for each class
                    self._reference_embeddings[complexity] = np.mean(embeddings, axis=0)

                print("[OK] ML classifier initialized with sentence-transformers")
            else:
                print("[WARN] ML classifier not available, using rule-based only")
        except Exception as e:
            print(f"[WARN] ML classifier not available: {e}, using rule-based only")
            self._ml_available = False

    def classify(self, query: str) -> ClassificationResult:
        """Classify query for routing decisions."""
        query_lower = query.lower()

        # Check for multi-step (same for both methods)
        is_multi_step = any(pattern in query_lower for pattern in self.MULTI_STEP_PATTERNS)

        # Try ML-based classification first
        if self._ml_available:
            try:
                ml_complexity, ml_confidence = self._classify_ml(query)
                routing = self._determine_routing(ml_complexity, is_multi_step)
                logger.info(f"[ML] classified: complexity={ml_complexity.value}, "
                           f"confidence={ml_confidence:.2f}, routing={routing}, "
                           f"multi_step={is_multi_step}")
                return ClassificationResult(
                    is_multi_step=is_multi_step,
                    complexity=ml_complexity,
                    suggested_routing=routing,
                    confidence=ml_confidence,
                    method="ml"
                )
            except Exception as e:
                print(f"[WARN] ML classification failed: {e}, falling back to rules")

        # Fallback to rule-based
        complexity = self._determine_complexity(query_lower)
        routing = self._determine_routing(complexity, is_multi_step)
        confidence = self._calculate_confidence(query, complexity, is_multi_step)

        logger.info(f"[RULES] classified: complexity={complexity.value}, "
                    f"confidence={confidence:.2f}, routing={routing}, "
                    f"multi_step={is_multi_step}")

        return ClassificationResult(
            is_multi_step=is_multi_step,
            complexity=complexity,
            suggested_routing=routing,
            confidence=confidence,
            method="rule"
        )

    def _classify_ml(self, query: str) -> tuple[Complexity, float]:
        """Classify using ML embeddings."""
        # Get query embedding
        query_embedding = self._embedding_model.encode(query.lower())

        # Calculate cosine similarity to each reference class
        best_complexity = Complexity.SIMPLE
        best_similarity = -1.0

        for complexity, ref_embedding in self._reference_embeddings.items():
            similarity = self._cosine_similarity(query_embedding, ref_embedding)
            if similarity > best_similarity:
                best_similarity = similarity
                best_complexity = complexity

        # Convert similarity to confidence (0.5 - 0.95 range)
        confidence = 0.5 + (best_similarity * 0.45)
        confidence = max(0.5, min(0.95, confidence))

        return best_complexity, confidence

    def _cosine_similarity(self, a, b):
        """Calculate cosine similarity between two vectors."""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    def _determine_complexity(self, query_lower: str) -> Complexity:
        """Determine query complexity based on keywords and length."""
        word_count = len(query_lower.split())
        
        # Check for complex keywords
        if any(kw in query_lower for kw in self.COMPLEXITY_KEYWORDS[Complexity.COMPLEX]):
            return Complexity.COMPLEX
        
        # Check for medium complexity keywords
        if any(kw in query_lower for kw in self.COMPLEXITY_KEYWORDS[Complexity.MEDIUM]):
            return Complexity.MEDIUM
        
        # Check length
        if word_count > 50:
            return Complexity.COMPLEX
        elif word_count > 20:
            return Complexity.MEDIUM
        
        return Complexity.SIMPLE

    def _determine_routing(self, complexity: Complexity, is_multi_step: bool) -> str:
        """Determine suggested routing based on complexity and multi-step."""
        if is_multi_step:
            return "cascade"
        
        if complexity == Complexity.COMPLEX:
            return "cloud"
        elif complexity == Complexity.MEDIUM:
            return "hybrid"
        else:
            return "local"

    def _calculate_confidence(self, query: str, complexity: Complexity, is_multi_step: bool) -> float:
        """Calculate confidence score for classification."""
        base_confidence = 0.7
        
        # Higher confidence for clear keywords
        query_lower = query.lower()
        if any(kw in query_lower for kw in self.COMPLEXITY_KEYWORDS[Complexity.COMPLEX]):
            base_confidence += 0.15
        if any(kw in query_lower for kw in self.COMPLEXITY_KEYWORDS[Complexity.MEDIUM]):
            base_confidence += 0.1
        
        # Higher confidence for clear multi-step patterns
        if is_multi_step:
            base_confidence += 0.1
        
        return min(base_confidence, 0.95)