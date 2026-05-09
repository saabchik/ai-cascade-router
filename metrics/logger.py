# metrics/logger.py
from dataclasses import dataclass, field
from typing import Optional
from loguru import logger

@dataclass
class SessionMetrics:
    total_requests: int = 0
    local_requests: int = 0
    cloud_requests: int = 0
    hybrid_requests: int = 0
    tokens_saved: int = 0
    total_cost_usd: float = 0.0
    total_response_time_ms: float = 0.0
    local_response_times: list = field(default_factory=list)
    cloud_response_times: list = field(default_factory=list)

    @property
    def avg_response_time_ms(self) -> float:
        if self.total_requests == 0:
            return 0.0
        return self.total_response_time_ms / self.total_requests

    @property
    def delegation_rate(self) -> float:
        if self.total_requests == 0:
            return 0.0
        return (self.cloud_requests + self.hybrid_requests) / self.total_requests

class MetricsLogger:
    def __init__(self, config: Optional[dict] = None):
        self.metrics = SessionMetrics()
        self.config = config or {}
        self.enabled = self.config.get("enabled", True)

    def record_request(
        self,
        decision: str,
        tokens_used: int = 0,
        cloud_tokens_saved: int = 0,
        cost_usd: Optional[float] = None,
        response_time_ms: float = 0.0
    ):
        if not self.enabled:
            return

        # Auto-calculate cost from cloud tokens if not provided
        if cost_usd is None:
            if tokens_used > 0:
                cloud_cost_per_1k = self.config.get("cloud_cost_per_1k_tokens", 0.001)
                cost_usd = (tokens_used / 1000) * cloud_cost_per_1k
            else:
                cost_usd = 0.0

        self.metrics.total_requests += 1
        self.metrics.total_response_time_ms += response_time_ms

        if decision == "local":
            self.metrics.local_requests += 1
            self.metrics.tokens_saved += cloud_tokens_saved
            self.metrics.local_response_times.append(response_time_ms)
        elif decision == "cloud":
            self.metrics.cloud_requests += 1
            self.metrics.total_cost_usd += cost_usd
            self.metrics.cloud_response_times.append(response_time_ms)
        elif decision == "hybrid":
            self.metrics.hybrid_requests += 1
            self.metrics.total_cost_usd += cost_usd * 0.5  # Hybrid uses less cloud
            self.metrics.tokens_saved += int(cloud_tokens_saved * 0.5)
        elif decision == "cascade":
            # Cascade uses mix - count as hybrid for delegation rate
            self.metrics.hybrid_requests += 1
            self.metrics.total_cost_usd += cost_usd * 0.3  # Cascade uses less cloud than pure hybrid
            self.metrics.tokens_saved += cloud_tokens_saved

        logger.info(f"Request recorded: {decision}, tokens_saved={cloud_tokens_saved}, cost={cost_usd}")

    def get_session_metrics(self) -> SessionMetrics:
        return self.metrics

    def calculate_roi(self, maintenance_cost_usd: float = 0.01) -> float:
        """Calculate ROI: (cloud_cost_saved - maintenance_cost) / maintenance_cost * 100"""
        cloud_cost_saved = (self.metrics.tokens_saved / 1000) * 0.01  # Assume $0.01 per 1k tokens
        if maintenance_cost_usd == 0:
            maintenance_cost_usd = 0.01  # Default minimal cost
        if cloud_cost_saved == 0:
            return 0.0
        return ((cloud_cost_saved - maintenance_cost_usd) / maintenance_cost_usd) * 100

    def export_metrics(self) -> dict:
        """Export metrics as dictionary."""
        return {
            "total_requests": self.metrics.total_requests,
            "local_requests": self.metrics.local_requests,
            "cloud_requests": self.metrics.cloud_requests,
            "hybrid_requests": self.metrics.hybrid_requests,
            "tokens_saved": self.metrics.tokens_saved,
            "total_cost_usd": round(self.metrics.total_cost_usd, 4),
            "avg_response_time_ms": round(self.metrics.avg_response_time_ms, 2),
            "delegation_rate": round(self.metrics.delegation_rate, 2),
            "roi_percent": round(self.calculate_roi(), 2)
        }
