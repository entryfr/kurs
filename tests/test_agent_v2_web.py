from fastapi.testclient import TestClient

from src.ai_agent_course.web import app

client = TestClient(app)


def _fake_result() -> dict:
    return {
        "run_id": "run-123",
        "prompt_context": {
            "site_name": "Нагатинский",
            "area_ha": 8.4,
            "excavation_depth_m": 6.5,
            "latitude": 55.6721,
            "longitude": 37.6415,
        },
        "anomalies": [
            {
                "id": 0,
                "risk_class": "CRITICAL",
                "probability": 0.89,
                "utility_type": "gas",
            }
        ],
        "validation": {"issues_count": 1, "issues_by_rule": {"R01": 1}, "issues": ["R01: demo"]},
        "metrics": {
            "rms_error_m": 0.12,
            "critical_count": 1,
            "high_count": 0,
            "low_count": 0,
            "blind_zone_count": 1,
            "validation_issues_count": 1,
        },
        "artifacts": {
            "output_dir": "/tmp/out",
            "dxf_path": "/tmp/out/scheme_v2.dxf",
            "miis_xml_path": "/tmp/out/unaccounted_v2.miis.xml",
            "pdf_report_path": "/tmp/out/act_v2.pdf",
            "dxf_url": "/agent-v2-output/demo/scheme_v2.dxf",
            "miis_xml_url": "/agent-v2-output/demo/unaccounted_v2.miis.xml",
            "pdf_report_url": "/agent-v2-output/demo/act_v2.pdf",
        },
        "answer": "Тестовый ответ.",
        "mode": "fallback_no_api_key",
    }


def test_v2_index_loads() -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert "Новый AI-агент" in response.text


def test_v2_form_analyze(monkeypatch) -> None:
    monkeypatch.setattr("src.ai_agent_course.web._run_analysis", lambda **_: _fake_result())
    response = client.post(
        "/analyze",
        data={"prompt": "Проанализируй участок"},
        files={"image_file": ("mag.png", b"image-bytes", "image/png")},
    )
    assert response.status_code == 200
    assert "Тестовый ответ." in response.text
    assert "agent-v2-output" in response.text


def test_v2_api_analyze(monkeypatch) -> None:
    monkeypatch.setattr("src.ai_agent_course.web._run_analysis", lambda **_: _fake_result())
    response = client.post(
        "/api/analyze",
        data={"prompt": "Проанализируй участок"},
        files={"image_file": ("mag.png", b"image-bytes", "image/png")},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["mode"] == "fallback_no_api_key"
    assert payload["metrics"]["critical_count"] == 1


def test_v2_api_runs(monkeypatch) -> None:
    monkeypatch.setattr(
        "src.ai_agent_course.web._history_slice",
        lambda limit=20: [
            {
                "run_id": "run-42",
                "created_at": "2026-02-19T10:00:00+00:00",
                "mode": "llm_openai_compatible",
                "anomalies_count": 3,
                "critical_count": 1,
                "high_count": 1,
                "low_count": 1,
            }
        ],
    )
    response = client.get("/api/runs?limit=5")
    assert response.status_code == 200
    payload = response.json()
    assert payload["count"] == 1
    assert payload["items"][0]["run_id"] == "run-42"
