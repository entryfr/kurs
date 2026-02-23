from fastapi.testclient import TestClient

from src.web.app import app


client = TestClient(app)


def _fake_summary() -> dict:
    return {
        "detected_count": 2,
        "anomalies_count": 3,
        "validation_issues": 0,
        "dxf_path": "output/scheme.dxf",
        "xml_path": "output/miis.xml",
        "act_path": "output/act.docx",
    }


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
