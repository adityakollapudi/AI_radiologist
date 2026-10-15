from fastapi.testclient import TestClient
from backend.app.main import app

client = TestClient(app)

def test_assistant_general_query():
    """Verify assistant handles general imaging questions in plain English."""
    payload = {
        "message": "What is the difference between CT and MRI?"
    }
    response = client.post("/assistant/chat", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "reply" in data
    assert len(data["reply"]) > 20
    assert "source" in data
    assert "suggested_queries" in data
    assert len(data["suggested_queries"]) > 0
    assert "X-Ray" in data["reply"] or "CT" in data["reply"] or "MRI" in data["reply"]

def test_assistant_contextual_scan_query():
    """Verify assistant explains findings in user-understandable terms."""
    payload = {
        "message": "Explain my scan prediction in detail",
        "context": {
            "modality": "xray",
            "prediction": "PNEUMONIA",
            "confidence": 0.945,
            "routed_to": "chest"
        }
    }
    response = client.post("/assistant/chat", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "reply" in data
    assert "PNEUMONIA" in data["reply"]
    assert "XRAY" in data["reply"] or "lung" in data["reply"].lower()

def test_assistant_gradcam_query():
    """Verify assistant explains scan highlights simply like a highlighter pen."""
    payload = {
        "message": "What do the red colors mean in the Grad-CAM heatmap?"
    }
    response = client.post("/assistant/chat", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "reply" in data
    assert "highlighter" in data["reply"].lower() or "attention" in data["reply"].lower()
    assert "colors" in data["reply"].lower() or "red" in data["reply"].lower()
