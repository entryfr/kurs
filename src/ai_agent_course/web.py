from __future__ import annotations

import logging
import shutil
import threading
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from src.ai_agent_course.orchestrator import EngineeringSurveyAgent
from src.ai_agent_course.llm_client import check_llm_connectivity, resolve_llm_config
from src.ai_agent_course.secrets_loader import load_local_env

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_local_env(PROJECT_ROOT / ".env.local")
CACHE_ROOT = (PROJECT_ROOT / ".cache" / "agent_v2").resolve()
RUNS_DIR = CACHE_ROOT / "runs"
UPLOADS_DIR = CACHE_ROOT / "uploads"
HISTORY_PATH = CACHE_ROOT / "history.json"
_history_lock = threading.Lock()
RUNS_DIR.mkdir(parents=True, exist_ok=True)
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

agent = EngineeringSurveyAgent()

app = FastAPI(
    title="AI Agent V2 — Underground Utilities",
    version="2.0.0",
    description="Новая реализация: вход картинка + текстовый запрос по ТЗ.",
)
app.mount("/agent-v2-output", StaticFiles(directory=str(RUNS_DIR)), name="agent-v2-output")
templates = Jinja2Templates(directory=str(PROJECT_ROOT / "templates"))


def _read_history() -> list[dict[str, Any]]:
    if not HISTORY_PATH.exists():
        return []
    try:
        import json

        payload = json.loads(HISTORY_PATH.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            return payload
    except Exception:  # noqa: BLE001
        logger.exception("Не удалось прочитать историю запусков V2")
    return []


def _write_history(items: list[dict[str, Any]]) -> None:
    import json

    HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    HISTORY_PATH.write_text(
        json.dumps(items, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _append_history(result_payload: dict[str, Any]) -> None:
    record = {
        "run_id": result_payload.get("run_id"),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "mode": result_payload.get("mode"),
        "anomalies_count": len(result_payload.get("anomalies", [])),
        "critical_count": result_payload.get("metrics", {}).get("critical_count", 0),
        "high_count": result_payload.get("metrics", {}).get("high_count", 0),
        "low_count": result_payload.get("metrics", {}).get("low_count", 0),
    }
    with _history_lock:
        items = _read_history()
        items.insert(0, record)
        _write_history(items[:200])


def _history_slice(limit: int = 10) -> list[dict[str, Any]]:
    with _history_lock:
        return _read_history()[:limit]


def _save_upload(upload: UploadFile, target_dir: Path, prefix: str) -> Path:
    target_dir.mkdir(parents=True, exist_ok=True)
    safe_name = Path(upload.filename or f"{prefix}.bin").name
    path = target_dir / f"{prefix}_{safe_name}"
    with path.open("wb") as out:
        shutil.copyfileobj(upload.file, out)
    return path


def _artifact_url(path: Path) -> str | None:
    try:
        rel = path.resolve().relative_to(RUNS_DIR)
        return f"/agent-v2-output/{rel.as_posix()}"
    except ValueError:
        return None


def _run_analysis(
    prompt: str,
    image_path: Path,
    miis_path: Path | None = None,
    llm_provider: str | None = None,
    llm_api_key: str | None = None,
    llm_model: str | None = None,
    llm_base_url: str | None = None,
) -> dict[str, Any]:
    run_id = str(uuid.uuid4())
    output_dir = RUNS_DIR / run_id
    result = agent.run(
        image_path=image_path,
        prompt=prompt,
        output_dir=output_dir,
        miis_xml_path=miis_path,
        llm_provider=llm_provider,
        llm_api_key=llm_api_key,
        llm_model=llm_model,
        llm_base_url=llm_base_url,
    )
    payload = asdict(result)
    artifacts = payload["artifacts"]
    artifacts["dxf_url"] = _artifact_url(Path(artifacts["dxf_path"]))
    artifacts["miis_xml_url"] = _artifact_url(Path(artifacts["miis_xml_path"]))
    artifacts["pdf_report_url"] = _artifact_url(Path(artifacts["pdf_report_path"]))
    payload["run_id"] = run_id
    _append_history(payload)
    return payload


def _default_form() -> dict[str, Any]:
    return {
        "prompt": (
            'Проанализируй участок строительства ЖК "Нагатинский", '
            "координаты 55.6721°N, 37.6415°E, площадь 8.4 га, "
            "глубина котлована 6.5 м, аномалия магнитного поля: амплитуда 68 нТл, "
            "протяжённость 42 м, глубина по ВЭЗ 1.8 м."
        ),
        "llm_provider": "auto",
        "llm_model": "",
        "llm_base_url": "",
    }


def _build_context(
    form: dict[str, Any] | None = None,
    result: dict[str, Any] | None = None,
    error: str | None = None,
    llm_check: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "form": form or _default_form(),
        "result": result,
        "error": error,
        "llm_check": llm_check,
        "recent_runs": _history_slice(limit=10),
    }


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/", response_class=HTMLResponse)
def index(request: Request) -> HTMLResponse:
    context = _build_context()
    return templates.TemplateResponse(request, "agent_v2/index.html", context)


@app.post("/analyze", response_class=HTMLResponse)
def analyze_from_form(
    request: Request,
    prompt: str = Form(...),
    image_file: UploadFile = File(...),
    miis_xml_file: UploadFile | None = File(default=None),
    llm_provider: str = Form(default="auto"),
    llm_api_key: str = Form(default=""),
    llm_model: str = Form(default=""),
    llm_base_url: str = Form(default=""),
) -> HTMLResponse:
    form_data = {
        "prompt": prompt,
        "llm_provider": llm_provider,
        "llm_model": llm_model,
        "llm_base_url": llm_base_url,
    }
    try:
        upload_dir = UPLOADS_DIR / str(uuid.uuid4())
        image_path = _save_upload(image_file, upload_dir, "image")
        miis_path = None
        if miis_xml_file and miis_xml_file.filename:
            miis_path = _save_upload(miis_xml_file, upload_dir, "miis")
        result = _run_analysis(
            prompt=prompt,
            image_path=image_path,
            miis_path=miis_path,
            llm_provider=llm_provider,
            llm_api_key=llm_api_key.strip() or None,
            llm_model=llm_model.strip() or None,
            llm_base_url=llm_base_url.strip() or None,
        )
        context = _build_context(form=form_data, result=result, error=None, llm_check=None)
        return templates.TemplateResponse(request, "agent_v2/index.html", context)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Ошибка анализа в V2 форме")
        context = _build_context(form=form_data, result=None, error=str(exc), llm_check=None)
        return templates.TemplateResponse(
            request,
            "agent_v2/index.html",
            context,
            status_code=400,
        )


@app.post("/llm/check", response_class=HTMLResponse)
def llm_check_from_form(
    request: Request,
    prompt: str = Form(default=""),
    llm_provider: str = Form(default="auto"),
    llm_api_key: str = Form(default=""),
    llm_model: str = Form(default=""),
    llm_base_url: str = Form(default=""),
) -> HTMLResponse:
    form_data = {
        "prompt": prompt or _default_form()["prompt"],
        "llm_provider": llm_provider,
        "llm_model": llm_model,
        "llm_base_url": llm_base_url,
    }
    try:
        config = resolve_llm_config(
            provider=llm_provider,
            api_key=llm_api_key.strip() or None,
            model=llm_model.strip() or None,
            base_url=llm_base_url.strip() or None,
        )
        status = check_llm_connectivity(config)
        context = _build_context(form=form_data, result=None, error=None, llm_check=status)
        return templates.TemplateResponse(request, "agent_v2/index.html", context)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Ошибка проверки LLM из формы")
        context = _build_context(form=form_data, result=None, error=str(exc), llm_check=None)
        return templates.TemplateResponse(
            request,
            "agent_v2/index.html",
            context,
            status_code=400,
        )


@app.post("/api/analyze")
def analyze_api(
    prompt: str = Form(...),
    image_file: UploadFile = File(...),
    miis_xml_file: UploadFile | None = File(default=None),
    llm_provider: str = Form(default="auto"),
    llm_api_key: str = Form(default=""),
    llm_model: str = Form(default=""),
    llm_base_url: str = Form(default=""),
) -> dict[str, Any]:
    try:
        upload_dir = UPLOADS_DIR / str(uuid.uuid4())
        image_path = _save_upload(image_file, upload_dir, "image")
        miis_path = None
        if miis_xml_file and miis_xml_file.filename:
            miis_path = _save_upload(miis_xml_file, upload_dir, "miis")
        return _run_analysis(
            prompt=prompt,
            image_path=image_path,
            miis_path=miis_path,
            llm_provider=llm_provider,
            llm_api_key=llm_api_key.strip() or None,
            llm_model=llm_model.strip() or None,
            llm_base_url=llm_base_url.strip() or None,
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Ошибка анализа в V2 API")
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/llm/check")
def llm_check_api(
    llm_provider: str = Form(default="auto"),
    llm_api_key: str = Form(default=""),
    llm_model: str = Form(default=""),
    llm_base_url: str = Form(default=""),
) -> dict[str, Any]:
    config = resolve_llm_config(
        provider=llm_provider,
        api_key=llm_api_key.strip() or None,
        model=llm_model.strip() or None,
        base_url=llm_base_url.strip() or None,
    )
    return check_llm_connectivity(config)


@app.get("/api/runs")
def list_runs(limit: int = Query(default=20, ge=1, le=200)) -> dict[str, Any]:
    items = _history_slice(limit=limit)
    return {"items": items, "count": len(items)}

