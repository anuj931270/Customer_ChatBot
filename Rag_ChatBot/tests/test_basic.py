from fastapi.testclient import TestClient

from app.database import get_recent_history, init_db, save_message
from app.main import app
from app.rag_pipeline import filter_by_threshold


def test_threshold_keeps_only_good_matches():
    results = [("doc-a", 0.80), ("doc-b", 0.36), ("doc-c", 0.20)]
    kept = filter_by_threshold(results, 0.35)
    assert [doc for doc, _ in kept] == ["doc-a", "doc-b"]


def test_threshold_can_remove_everything():
    assert filter_by_threshold([("doc-a", 0.10)], 0.35) == []


def test_history_is_saved_in_order():
    init_db()
    save_message("t1", "user", "hello")
    save_message("t1", "assistant", "hi there")
    assert get_recent_history("t1", limit=2) == [("user", "hello"), ("assistant", "hi there")]


def test_health_endpoint():
    with TestClient(app) as client:
        response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_chat_rejects_empty_question():
    with TestClient(app) as client:
        response = client.post("/chat", json={"session_id": "a", "question": ""})
    assert response.status_code == 422
