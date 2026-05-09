# router/engine.py
import yaml
from typing import Optional, Dict, Any, List
from router.protocol import RouteRequest, RouteDecision, RouteResponse
from router.semantic import SemanticClassifier, ClassificationResult
from router.decomposer import TaskDecomposer, DecompositionResult, Subtask
from loguru import logger

class RoutingCriteria:
    def __init__(self, config: Dict[str, Any]):
        self.confidence_threshold = config.get("confidence_threshold", 0.7)
        routing_config = config.get("routing", {})

        # Domain-specific thresholds (v0.2)
        self.domain_thresholds = routing_config.get("domain_thresholds", {
            "code": 0.65,
            "analysis": 0.80,
            "chat": 0.60,
            "question": 0.55,
            "creative": 0.70,
            "technical": 0.75
        })

        self.max_length_local = routing_config.get("complexity", {}).get("max_length_for_local", 500)
        self.complex_keywords = routing_config.get("complexity", {}).get("complex_keywords", [])

        self.task_types = routing_config.get("task_types", {})
        self.local_timeout_ms = routing_config.get("urgency", {}).get("local_timeout_ms", 2000)

        history_config = routing_config.get("history", {})
        self.cache_repeated = history_config.get("cache_repeated_requests", True)

    def get_threshold_for_domain(self, domain: str) -> float:
        """Get threshold for specific domain, returns default if not found."""
        return self.domain_thresholds.get(domain, self.confidence_threshold)

class RouterEngine:
    def __init__(self, config_path: str = "config.yaml"):
        self.config = self._load_config(config_path)
        self.criteria = RoutingCriteria(self.config.get("routing", {}))
        self.local_config = self.config.get("local_model", {})
        self.cloud_config = self.config.get("cloud_model", {})
        
        # Initialize semantic classifier and decomposer
        self.semantic_classifier = SemanticClassifier()
        self.task_decomposer = TaskDecomposer(
            base_url=self.local_config.get("base_url", "http://localhost:1234/v1"),
            model=self.local_config.get("model_name", "qwen2.5-7b-instruct")
        )
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Failed to load config from {config_path}: {e}")
            return {}
    
    def _detect_domain(self, query: str) -> str:
        """Detect domain from query content."""
        query_lower = query.lower()

        # Code detection
        code_keywords = ["код", "code", "программирование", "programming", "функция", "function", "class", "debug", "debug", "syntax", "import", "def ", "print("]
        if any(kw in query_lower for kw in code_keywords):
            return "code"

        # Analysis detection
        analysis_keywords = ["анализ", "analysis", "сравни", "compare", "оцени", "evaluate", "исследуй", "investigate"]
        if any(kw in query_lower for kw in analysis_keywords):
            return "analysis"

        # Creative detection
        creative_keywords = ["напиши", "write", "сочини", "придумай", "story", "рассказ", "стих", "poem"]
        if any(kw in query_lower for kw in creative_keywords):
            return "creative"

        # Technical detection
        technical_keywords = ["настрой", "config", "установи", "install", "deploy", "server", "api", "база", "database"]
        if any(kw in query_lower for kw in technical_keywords):
            return "technical"

        # Question - default for short queries
        question_keywords = ["что", "кто", "как", "почему", "where", "when", "why", "what", "how", "?"]
        if any(kw in query_lower for kw in question_keywords):
            return "question"

        # Chat - default
        return "chat"

    def evaluate_request(self, query: str, task_type: Optional[str] = None) -> RouteResponse:
        """Evaluate request and return routing decision."""
        query_length = len(query)

        # Detect domain and get threshold (v0.2)
        domain = self._detect_domain(query)
        threshold = self.criteria.get_threshold_for_domain(domain)
        logger.info(f"Detected domain: {domain}, threshold: {threshold}")

        # Check task type routing
        if task_type and task_type in self.criteria.task_types:
            type_decision = self.criteria.task_types[task_type]
            if type_decision == "local":
                return RouteResponse(
                    decision=RouteDecision.LOCAL,
                    reason="Task type maps to local",
                    confidence=1.0
                )
            elif type_decision == "cloud_if_complex":
                if self._is_complex(query):
                    return RouteResponse(
                        decision=RouteDecision.CLOUD,
                        reason="Complex task type requires cloud",
                        confidence=1.0
                    )
        
        # Check complexity by length
        if query_length > self.criteria.max_length_local:
            return RouteResponse(
                decision=RouteDecision.CLOUD,
                reason=f"Query length {query_length} exceeds local threshold",
                confidence=1.0
            )
        
        # Check complexity by keywords
        if self._has_complex_keywords(query):
            return RouteResponse(
                decision=RouteDecision.HYBRID,
                reason="Complex keywords detected, use local preprocessing + cloud",
                confidence=0.8
            )
        
        # Default: route to local
        return RouteResponse(
            decision=RouteDecision.LOCAL,
            reason="Query is simple enough for local model",
            confidence=0.9
        )
    
    def _is_complex(self, query: str) -> bool:
        """Check if query is complex based on various heuristics."""
        word_count = len(query.split())
        return word_count > 100
    
    def _has_complex_keywords(self, query: str) -> bool:
        """Check if query contains complex keywords."""
        query_lower = query.lower()
        return any(keyword in query_lower for keyword in self.criteria.complex_keywords)
    
    def get_threshold_for_query(self, query: str) -> float:
        """Get domain-specific threshold for a query."""
        domain = self._detect_domain(query)
        return self.criteria.get_threshold_for_domain(domain)

    def get_config(self) -> Dict[str, Any]:
        return self.config
    
    async def evaluate_request_cascade(self, query: str) -> Dict[str, Any]:
        """
        Evaluate request with cascade routing - handles multi-step queries.
        
        Returns:
            {
                "mode": "single" or "cascade",
                "decision": RouteResponse (for single mode),
                "classification": ClassificationResult,
                "subtasks": List[Dict] (for cascade mode)
            }
        """
        # Classify the query
        classification = self.semantic_classifier.classify(query)
        
        # If not multi-step, use simple routing
        if not classification.is_multi_step:
            decision = self.evaluate_request(query, None)
            return {
                "mode": "single",
                "decision": decision,
                "classification": classification,
                "subtasks": []
            }
        
        # Multi-step query - decompose
        try:
            decomposition = await self.task_decomposer.decompose(query)
        except Exception as e:
            logger.warning(f"Decomposition failed: {e}, falling back to single")
            decision = self.evaluate_request(query, None)
            return {
                "mode": "single",
                "decision": decision,
                "classification": classification,
                "subtasks": []
            }
        
        # Process each subtask
        subtasks_routes = []
        for subtask in decomposition.subtasks:
            # Determine route based on complexity
            route = self._determine_route_for_subtask(subtask.complexity, classification)
            subtasks_routes.append({
                "action": subtask.action,
                "task_type": subtask.task_type,
                "complexity": subtask.complexity,
                "route": route,
                "index": subtask.index
            })
        
        return {
            "mode": "cascade",
            "classification": classification,
            "subtasks": subtasks_routes,
            "original_query": query
        }
    
    def _determine_route_for_subtask(self, decomposition_complexity: str, classification) -> str:
        """Determine routing for a subtask based on complexity."""
        comp = decomposition_complexity.lower()
        if comp == "complex":
            return "cloud"
        elif comp == "medium":
            return "hybrid"
        else:
            return "local"
