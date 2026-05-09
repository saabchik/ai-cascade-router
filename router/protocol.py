from dataclasses import dataclass
from enum import Enum
from typing import Optional, Dict, Any

class RouteDecision(Enum):
    LOCAL = "local"
    CLOUD = "cloud"
    HYBRID = "hybrid"

@dataclass
class RouteRequest:
    query: str
    domain: Optional[str] = None
    task_type: Optional[str] = None
    context: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "domain": self.domain,
            "task_type": self.task_type,
            "context": self.context
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RouteRequest":
        return cls(
            query=data.get("query", ""),
            domain=data.get("domain"),
            task_type=data.get("task_type"),
            context=data.get("context")
        )

@dataclass
class RouteResponse:
    decision: RouteDecision
    reason: str = ""
    confidence: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "decision": self.decision.value,
            "reason": self.reason,
            "confidence": self.confidence
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RouteResponse":
        return cls(
            decision=RouteDecision(data.get("decision", "local")),
            reason=data.get("reason", ""),
            confidence=data.get("confidence", 1.0)
        )