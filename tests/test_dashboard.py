import pytest
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


class TestDashboard:
    def test_dashboard_returns_html(self):
        response = client.get("/")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
        assert "AI Cascade Router" in response.text

    def test_dashboard_metrics_endpoint(self):
        response = client.get("/api/dashboard")
        assert response.status_code == 200
        data = response.json()
        assert "total_requests" in data
        assert "local_requests" in data
        assert "cloud_requests" in data
        assert "tokens_saved" in data
        assert "total_cost_usd" in data
        assert "roi_percent" in data

    def test_metrics_still_works(self):
        response = client.get("/metrics")
        assert response.status_code == 200
        data = response.json()
        assert "total_requests" in data

    def test_health_endpoint(self):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] in ["healthy", "unhealthy"]
        assert "local_model" in data
        assert "cloud_model" in data

    def test_dashboard_contains_htmx(self):
        response = client.get("/")
        assert "htmx.org" in response.text

    def test_dashboard_contains_cards(self):
        response = client.get("/")
        assert "total_requests" in response.text or "Всего запросов" in response.text

    def test_static_css_accessible(self):
        response = client.get("/static/css/style.css")
        assert response.status_code == 200
        assert "text/css" in response.headers["content-type"]

    def test_static_js_accessible(self):
        response = client.get("/static/js/dashboard.js")
        assert response.status_code == 200
        assert "javascript" in response.headers["content-type"]
