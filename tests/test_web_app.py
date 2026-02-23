import time

import pytest
from fastapi.testclient import TestClient

from src.web.app import app, job_manager


client = TestClient(app)


def _fake_summary() -> dict:
    return {
        "input_mode": "vector_layers",
        "norms_profile": "normative",
        "detected_count": 2,
        "anomalies_count": 3,
        "validation_issues": 0,
        "validation_issue_details": [],
        "validation_issues_by_rule": {},
        "validation_report": "",
        "risk_details": [],
        "segy_features": {},
        "dxf_path": "output/scheme.dxf",
        "xml_path": "output/miis.xml",
        "act_path": "output/act.docx",
    }


@pytest.fixture(autouse=True)
def _reset_jobs() -> None:
    job_manager.reset_for_tests()
    yield
    job_manager.reset_for_tests()


def test_healthz() -> None:
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_index_page_loads() -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert "Запуск геофизического пайплайна" in response.text


def test_run_form_success(monkeypatch) -> None:
    monkeypatch.setattr("src.web.app.run_pipeline", lambda **_: _fake_summary())

    response = client.post(
        "/run",
        data={
            "magnetic_grid": "data/raw/magnetic_grid.npy",
            "anomalies": "data/processed/anomalies.gpkg",
            "utilities": "data/processed/utilities.gpkg",
            "output_dir": "output",
            "anomaly_threshold": "30.0",
        },
    )
    assert response.status_code == 200
    assert "Пайплайн завершён успешно" in response.text
    assert "output/scheme.dxf" in response.text


def test_run_api_success(monkeypatch) -> None:
    monkeypatch.setattr("src.web.app.run_pipeline", lambda **_: _fake_summary())

    response = client.post(
        "/api/run",
        json={
            "magnetic_grid": "data/raw/magnetic_grid.npy",
            "anomalies": "data/processed/anomalies.gpkg",
            "utilities": "data/processed/utilities.gpkg",
            "output_dir": "output",
            "anomaly_threshold": 30.0,
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["detected_count"] == 2
    assert payload["dxf_url"] == "/output/scheme.dxf"


def test_run_api_async_success(monkeypatch) -> None:
    monkeypatch.setattr("src.web.app.run_pipeline", lambda **_: _fake_summary())

    response = client.post(
        "/api/run/async",
        json={
            "magnetic_grid": "data/raw/magnetic_grid.npy",
            "anomalies": "data/processed/anomalies.gpkg",
            "utilities": "data/processed/utilities.gpkg",
            "output_dir": "output",
            "anomaly_threshold": 30.0,
            "norms_profile": "normative",
        },
    )
    assert response.status_code == 200
    created = response.json()
    job_id = created["job_id"]

    status_payload = {}
    for _ in range(30):
        status_response = client.get(f"/api/run/{job_id}")
        assert status_response.status_code == 200
        status_payload = status_response.json()
        if status_payload["status"] == "completed":
            break
        time.sleep(0.05)

    assert status_payload["status"] == "completed"
    assert status_payload["result"]["dxf_url"] == "/output/scheme.dxf"


def test_list_runs_returns_items(monkeypatch) -> None:
    monkeypatch.setattr("src.web.app.run_pipeline", lambda **_: _fake_summary())
    create_response = client.post(
        "/api/run/async",
        json={
            "magnetic_grid": "data/raw/magnetic_grid.npy",
            "anomalies": "data/processed/anomalies.gpkg",
            "utilities": "data/processed/utilities.gpkg",
            "output_dir": "output",
            "anomaly_threshold": 30.0,
            "norms_profile": "normative",
        },
    )
    assert create_response.status_code == 200

    list_response = client.get("/api/runs?limit=5")
    assert list_response.status_code == 200
    payload = list_response.json()
    assert payload["count"] >= 1
    assert isinstance(payload["items"], list)


def test_run_form_with_uploaded_files(monkeypatch) -> None:
    captured: dict = {}

    def _fake_run_pipeline(**kwargs):
        captured.update(kwargs)
        return _fake_summary()

    monkeypatch.setattr("src.web.app.run_pipeline", _fake_run_pipeline)

    response = client.post(
        "/run",
        data={
            "magnetic_grid": "",
            "anomalies": "",
            "utilities": "",
            "miis_xml": "",
            "segy_file": "",
            "output_dir": "output",
            "anomaly_threshold": "30.0",
            "norms_profile": "normative",
        },
        files={
            "magnetic_grid_file": ("grid.npy", b"dummy-grid-bytes", "application/octet-stream"),
            "miis_xml_file": ("input.xml", b"<MIIS></MIIS>", "application/xml"),
        },
    )

    assert response.status_code == 200
    assert "Пайплайн завершён успешно" in response.text
    assert ".cache/web/uploads" in str(captured["magnetic_grid_path"])
    assert ".cache/web/uploads" in str(captured["miis_xml_path"])
