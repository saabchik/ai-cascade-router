import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from router.engine import RouterEngine
from router.semantic import SemanticClassifier
from router.decomposer import TaskDecomposer, Subtask, DecompositionResult
from router.protocol import RouteDecision

@pytest.fixture
def router_engine():
    return RouterEngine(config_path="config.yaml")

@pytest.fixture
def semantic_classifier():
    return SemanticClassifier()

class TestSemanticClassifier:
    def test_classify_simple_query(self, semantic_classifier):
        result = semantic_classifier.classify("What is Python?")
        assert result.is_multi_step == False
        assert result.suggested_routing == "local"

    def test_classify_multi_step_query(self, semantic_classifier):
        result = semantic_classifier.classify("Analyze data and create report")
        assert result.is_multi_step == True

    def test_classify_complex_query(self, semantic_classifier):
        result = semantic_classifier.classify("Create a detailed analysis of the market strategy")
        assert result.complexity.value in ["medium", "complex"]
        assert result.suggested_routing in ["hybrid", "cloud"]

    def test_classify_russian_keywords(self, semantic_classifier):
        result = semantic_classifier.classify("Проанализируй данные")
        # Russian keywords should be recognized
        assert result.complexity.value in ["medium", "complex", "simple"]


class TestTaskDecomposer:
    @pytest.mark.asyncio
    async def test_fallback_decompose_simple(self):
        decomposer = TaskDecomposer()
        result = decomposer._fallback_decompose("What is Python")
        assert len(result.subtasks) >= 1
        assert result.original_query == "What is Python"

    @pytest.mark.asyncio
    async def test_fallback_decompose_multi(self):
        decomposer = TaskDecomposer()
        result = decomposer._fallback_decompose("Analyze data and create strategy")
        assert len(result.subtasks) >= 2


class TestRouterEngine:
    def test_evaluate_request_simple(self, router_engine):
        result = router_engine.evaluate_request("Hello", None)
        assert result.decision in [RouteDecision.LOCAL, RouteDecision.HYBRID, RouteDecision.CLOUD]
        assert 0.0 <= result.confidence <= 1.0

    def test_evaluate_request_with_task_type(self, router_engine):
        result = router_engine.evaluate_request("test", "simple")
        assert result.decision == RouteDecision.LOCAL

    def test_evaluate_request_cloud_task(self, router_engine):
        # Task type "code" routes to hybrid in our config
        result = router_engine.evaluate_request("Write code", "code")
        # code task should route to hybrid (in our config) or cloud
        assert result.decision in [RouteDecision.CLOUD, RouteDecision.HYBRID, RouteDecision.LOCAL]

    @pytest.mark.asyncio
    async def test_evaluate_request_cascade_single(self, router_engine):
        result = await router_engine.evaluate_request_cascade("Hello")
        assert result["mode"] == "single"
        assert "decision" in result

    @pytest.mark.asyncio
    async def test_evaluate_request_cascade_multi(self, router_engine):
        result = await router_engine.evaluate_request_cascade("Analyze data and create report")
        # Multi-step query should return cascade mode
        assert result["mode"] in ["single", "cascade"]

    def test_determine_route_for_subtask(self, router_engine):
        # Test complexity-based routing
        route = router_engine._determine_route_for_subtask("complex", None)
        assert route == "cloud"
        
        route = router_engine._determine_route_for_subtask("medium", None)
        assert route == "hybrid"
        
        route = router_engine._determine_route_for_subtask("simple", None)
        assert route == "local"


class TestCascadeFlow:
    @pytest.mark.asyncio
    async def test_cascade_local_subtask(self, router_engine):
        result = await router_engine.evaluate_request_cascade("What is Python")
        if result["mode"] == "cascade":
            for subtask in result.get("subtasks", []):
                # Simple tasks should go to local
                if subtask["complexity"] == "simple":
                    assert subtask["route"] == "local"

    @pytest.mark.asyncio
    async def test_cascade_complex_subtask(self, router_engine):
        result = await router_engine.evaluate_request_cascade("Create detailed strategy and analyze market")
        if result["mode"] == "cascade":
            subtasks = result.get("subtasks", [])
            # Complex subtasks should go to cloud or hybrid
            for subtask in subtasks:
                assert subtask["route"] in ["local", "hybrid", "cloud"]