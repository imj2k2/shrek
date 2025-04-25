import pytest
from fastapi.testclient import TestClient
from ui.app import app

client = TestClient(app)

def test_ping():
    resp = client.get("/ping")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}

def test_portfolio():
    resp = client.get("/portfolio")
    assert resp.status_code == 200
    assert "portfolio" in resp.json()

def test_signals():
    # Provide sample market data as expected by the agents
    sample_data = {"close": [100, 101, 102, 103, 104], "high": [105]*5, "low": [95]*5, "volume": [1000]*5}
    resp = client.post("/agents/signal", json={"data": sample_data})
    assert resp.status_code == 200
    assert "signals" in resp.json()

def test_execute_trade():
    # Provide a sample signal; adjust fields as per your executor's expectations
    sample_signal = {"symbol": "AAPL", "action": "buy", "qty": 1}
    resp = client.post("/agents/execute", json=sample_signal)
    assert resp.status_code == 200
    assert "result" in resp.json()
