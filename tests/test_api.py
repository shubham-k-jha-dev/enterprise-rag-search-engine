import pytest
from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)

API_KEY = "Shubh@2005"
AUTH = {"X-API-Key": API_KEY}

# Health Check :-
def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    
# Workspaces :-
def test_workspaces():
    response = client.get("/workspaces")
    assert response.status_code == 200
    data = response.json()
    # check "workspaces" key exists in response
    assert "workspaces" in data
    # Check that our three default workspaces are all present
    assert "default" in data["workspaces"]
    assert "dexter" in data["workspaces"]
    assert "got" in data["workspaces"]
    
    
# Auth Tests :- 

def test_search_without_api_key():
    # Sending request with no API key should return 403 Forbidden
    response = client.post("/search", json={
        "query": "who is dexter",
        "workspace": "dexter"
    })
    assert response.status_code == 403


def test_search_with_wrong_api_key():
    # Sending request with wrong API key should also return 403
    response = client.post("/search", json={
        "query": "who is dexter",
        "workspace": "dexter"
    }, headers={"X-API-Key": "wrongkey"})
    assert response.status_code == 403
    
# Search Endpoints :- 
def test_search_returns_results():
    # Valid search with correct API key should return 200
    response = client.post("/search", json={
        "query": "who is dexter",
        "workspace": "dexter"
    }, headers=AUTH)
    assert response.status_code == 200
    data = response.json()
    # response must have these three keys
    assert "query" in data
    assert "workspace" in data
    assert "results" in data
    # results must be a list
    assert isinstance(data["results"], list)


def test_search_response_shape():
    # Each result must have text, source, chunk_id fields
    response = client.post("/search", json={
        "query": "dark passenger",
        "workspace": "dexter"
    }, headers=AUTH)
    assert response.status_code == 200
    results = response.json()["results"]
    if results:
        # check the shape of the first result
        first = results[0]
        assert "text" in first
        assert "source" in first
        assert "chunk_id" in first


def test_search_workspace_isolation():
    # Querying dexter workspace should only return dexter.txt sources
    response = client.post("/search", json={
        "query": "dark passenger dexter",
        "workspace": "dexter"
    }, headers=AUTH)
    assert response.status_code == 200
    results = response.json()["results"]
    for result in results:
        # every result source must be from dexter workspace
        assert result["source"] is None or "dexter" in result["source"]
        
# Logs Endpoint
def test_logs_returns_list():
    # /logs should return a dict with a "logs" key containing a list
    response = client.get("/logs", headers=AUTH)
    assert response.status_code == 200
    data = response.json()
    assert "logs" in data
    assert isinstance(data["logs"], list)
    