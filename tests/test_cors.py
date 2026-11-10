# backend/tests/test_cors.py
import pytest
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

@pytest.mark.parametrize("origin", [
    "http://localhost:5173",
    "http://localhost:8080",
    "https://visa-vista-review.vercel.app",
])
def test_cors_preflight(origin: str):
    response = client.options(
        "/health",
        headers={
            "Origin": origin,
            "Access-Control-Request-Method": "GET",
            "Access-Control-Request-Headers": "Authorization,Content-Type",
        },
    )
    # FastAPI's CORSMiddleware should return 200 for preflight with allowed origin
    assert response.status_code == 200
    # The Access-Control-Allow-Origin header should echo the request origin
    assert response.headers.get("access-control-allow-origin") == origin
    # Verify allowed methods and headers are present
    assert "GET" in response.headers.get("access-control-allow-methods", "")
    assert "Authorization" in response.headers.get("access-control-allow-headers", "")
    assert "Content-Type" in response.headers.get("access-control-allow-headers", "")
