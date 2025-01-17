# tests/test_api.py
import pytest
from fastapi import status
from httpx import AsyncClient

@pytest.mark.asyncio
async def test_health_check(client):
    """Test health check endpoint."""
    response = client.get("/api/health")
    assert response.status_code == status.HTTP_200_OK
    assert response.json()["status"] == "ok"

@pytest.mark.asyncio
async def test_websocket_connection(client):
    """Test WebSocket connection."""
    with client.websocket_connect("/ws") as websocket:
        data = websocket.receive_json()
        assert "type" in data
        assert data["type"] == "connection_established"