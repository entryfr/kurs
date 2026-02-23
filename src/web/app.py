from __future__ import annotations

import logging
import shutil
import uuid
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

from src.pipeline.service import PROJECT_ROOT, run_pipeline
from src.web.job_manager import JobManager

logger = logging.getLogger(__name__)

DEFAULT_MAGNETIC_GRID = "data/raw/magnetic_grid.npy"
DEFAULT_ANOMALIES = "data/processed/anomalies.gpkg"
DEFAULT_UTILITIES = "data/processed/utilities.gpkg"
DEFAULT_MIIS_XML = ""
DEFAULT_SEGY = ""
DEFAULT_OUTPUT_DIR = "output"
DEFAULT_THRESHOLD = 30.0
DEFAULT_NORMS_PROFILE = "normative"

ARTIFACTS_DIR = (PROJECT_ROOT / DEFAULT_OUTPUT_DIR).resolve()
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
WEB_CACHE_DIR = (PROJECT_ROOT / ".cache" / "web").resolve()
UPLOADS_DIR = WEB_CACHE_DIR / "uploads"
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
job_manager = JobManager(history_path=WEB_CACHE_DIR / "run_history.json", max_workers=2)

app = FastAPI(
    title="Geophysical Utilities Pipeline",
    description="Веб-интерфейс запуска геофизического пайплайна.",
    version="1.0.0",
)
app.mount("/output", StaticFiles(directory=str(ARTIFACTS_DIR)), name="output")
templates = Jinja2Templates(directory=str(PROJECT_ROOT / "templates"))


class PipelineRunRequest(BaseModel):
    magnetic_grid: str = Field(default=DEFAULT_MAGNETIC_GRID)
    anomalies: str | None = Field(default=DEFAULT_ANOMALIES)
    utilities: str | None = Field(default=DEFAULT_UTILITIES)
    miis_xml: str | None = Field(default=None)
    segy_file: str | None = Field(default=None)
    output_dir: str = Field(default=DEFAULT_OUTPUT_DIR)
    anomaly_threshold: float = Field(default=DEFAULT_THRESHOLD, ge=0)
    norms_profile: str = Field(default=DEFAULT_NORMS_PROFILE, pattern="^(demo|normative|strict)$")


def _resolve_project_path(raw_path: str) -> Path:
    path = Path(raw_path.strip()).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    resolved = path.resolve()

    try:
        resolved.relative_to(PROJECT_ROOT)
    except ValueError as exc:
        raise ValueError(
            f"Путь '{raw_path}' должен находиться внутри каталога проекта: {PROJECT_ROOT}"
        ) from exc

    return resolved


def _resolve_optional_project_path(raw_path: str | None) -> Path | None:
    if raw_path is None:
        return None
    stripped = raw_path.strip()
    if not stripped:
        return None
    return _resolve_project_path(stripped)


def _validate_input_file(path: Path, param_name: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Файл '{param_name}' не найден: {path}")
    if not path.is_file():
        raise ValueError(f"Путь '{param_name}' должен указывать на файл: {path}")


def _artifact_url(path_str: str) -> str | None:
    path = Path(path_str).resolve()
    try:
        relative_to_output = path.relative_to(ARTIFACTS_DIR)
        return f"/output/{relative_to_output.as_posix()}"
    except ValueError:
        return None


def _prepare_summary(summary: dict[str, Any]) -> dict[str, Any]:
    enriched = dict(summary)
    enriched["dxf_url"] = _artifact_url(summary["dxf_path"])
    enriched["xml_url"] = _artifact_url(summary["xml_path"])
    enriched["act_url"] = _artifact_url(summary["act_path"])
    return enriched


def _save_upload(upload: UploadFile | None, run_dir: Path, prefix: str) -> str | None:
    if upload is None or not upload.filename:
        return None
    run_dir.mkdir(parents=True, exist_ok=True)
    file_name = Path(upload.filename).name
    target = run_dir / f"{prefix}_{file_name}"
    with target.open("wb") as file_out:
        shutil.copyfileobj(upload.file, file_out)
    return target.relative_to(PROJECT_ROOT).as_posix()


def _apply_upload_overrides(
    form_values: dict[str, Any],
    magnetic_grid_file: UploadFile | None = None,
    anomalies_file: UploadFile | None = None,
    utilities_file: UploadFile | None = None,
    miis_xml_file: UploadFile | None = None,
    segy_upload_file: UploadFile | None = None,
) -> dict[str, Any]:
    upload_items = [
        (magnetic_grid_file, "magnetic_grid", "magnetic"),
        (anomalies_file, "anomalies", "anomalies"),
        (utilities_file, "utilities", "utilities"),
        (miis_xml_file, "miis_xml", "miis"),
        (segy_upload_file, "segy_file", "segy"),
    ]
    if not any(item[0] and item[0].filename for item in upload_items):
        return form_values

    run_dir = UPLOADS_DIR / str(uuid.uuid4())
    result = dict(form_values)
    for upload, key, prefix in upload_items:
        saved = _save_upload(upload, run_dir, prefix)
        if saved is not None:
            result[key] = saved
    return result


def _execute(request_data: PipelineRunRequest) -> dict[str, Any]:
    magnetic_grid_path = _resolve_project_path(request_data.magnetic_grid)
    anomalies_path = _resolve_optional_project_path(request_data.anomalies)
    utilities_path = _resolve_optional_project_path(request_data.utilities)
    miis_xml_path = _resolve_optional_project_path(request_data.miis_xml)
    segy_path = _resolve_optional_project_path(request_data.segy_file)
    output_dir = _resolve_project_path(request_data.output_dir)

    _validate_input_file(magnetic_grid_path, "magnetic_grid")
    if miis_xml_path is not None:
        _validate_input_file(miis_xml_path, "miis_xml")
    else:
        if anomalies_path is None or utilities_path is None:
            raise ValueError(
                "Укажите anomalies/utilities либо передайте miis_xml для извлечения слоёв."
            )
        _validate_input_file(anomalies_path, "anomalies")
        _validate_input_file(utilities_path, "utilities")
    if segy_path is not None:
        _validate_input_file(segy_path, "segy_file")

    return run_pipeline(
        magnetic_grid_path=magnetic_grid_path,
        anomalies_path=anomalies_path if anomalies_path is not None else Path(DEFAULT_ANOMALIES),
        utilities_path=utilities_path if utilities_path is not None else Path(DEFAULT_UTILITIES),
        output_dir=output_dir,
        anomaly_threshold=request_data.anomaly_threshold,
        miis_xml_path=miis_xml_path,
        segy_path=segy_path,
        norms_profile=request_data.norms_profile,
    )


def _render_context(
    form_values: dict[str, Any],
    summary: dict[str, Any] | None = None,
    error: str | None = None,
    async_job: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "form_values": form_values,
        "summary": summary,
        "error": error,
        "async_job": async_job,
        "jobs": job_manager.list_jobs(limit=10),
    }


def _queue_async_job(payload: dict[str, Any], source: str) -> dict[str, Any]:
    run_request = PipelineRunRequest(**payload)
    created = job_manager.create_job(run_request.model_dump(), source=source)
    job_id = created["job_id"]

    def _run() -> dict[str, Any]:
        summary = _execute(run_request)
        return _prepare_summary(summary)

    job_manager.submit(job_id, _run)
    return {
        "job_id": job_id,
        "status": "queued",
        "status_url": f"/api/run/{job_id}",
    }


def _default_form_values() -> dict[str, Any]:
    return {
        "magnetic_grid": DEFAULT_MAGNETIC_GRID,
        "anomalies": DEFAULT_ANOMALIES,
        "utilities": DEFAULT_UTILITIES,
        "miis_xml": DEFAULT_MIIS_XML,
        "segy_file": DEFAULT_SEGY,
        "output_dir": DEFAULT_OUTPUT_DIR,
        "anomaly_threshold": DEFAULT_THRESHOLD,
        "norms_profile": DEFAULT_NORMS_PROFILE,
    }


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/", response_class=HTMLResponse)
def index(request: Request) -> HTMLResponse:
    context = _render_context(_default_form_values())
    return templates.TemplateResponse(request, "web/index.html", context)


@app.post("/run", response_class=HTMLResponse)
def run_from_form(
    request: Request,
    magnetic_grid: str = Form(DEFAULT_MAGNETIC_GRID),
    anomalies: str = Form(DEFAULT_ANOMALIES),
    utilities: str = Form(DEFAULT_UTILITIES),
    miis_xml: str = Form(DEFAULT_MIIS_XML),
    segy_file: str = Form(DEFAULT_SEGY),
    output_dir: str = Form(DEFAULT_OUTPUT_DIR),
    anomaly_threshold: float = Form(DEFAULT_THRESHOLD),
    norms_profile: str = Form(DEFAULT_NORMS_PROFILE),
    magnetic_grid_file: UploadFile | None = File(default=None),
    anomalies_file: UploadFile | None = File(default=None),
    utilities_file: UploadFile | None = File(default=None),
    miis_xml_file: UploadFile | None = File(default=None),
    segy_upload_file: UploadFile | None = File(default=None),
) -> HTMLResponse:
    form_values = {
        "magnetic_grid": magnetic_grid,
        "anomalies": anomalies,
        "utilities": utilities,
        "miis_xml": miis_xml,
        "segy_file": segy_file,
        "output_dir": output_dir,
        "anomaly_threshold": anomaly_threshold,
        "norms_profile": norms_profile,
    }
    form_values = _apply_upload_overrides(
        form_values,
        magnetic_grid_file=magnetic_grid_file,
        anomalies_file=anomalies_file,
        utilities_file=utilities_file,
        miis_xml_file=miis_xml_file,
        segy_upload_file=segy_upload_file,
    )

    try:
        run_request = PipelineRunRequest(**form_values)
        summary = _execute(run_request)
        context = _render_context(
            form_values=form_values, summary=_prepare_summary(summary), error=None
        )
        return templates.TemplateResponse(request, "web/index.html", context)
    except Exception as exc:
        logger.exception("Ошибка при запуске пайплайна через веб-интерфейс")
        context = _render_context(form_values=form_values, error=str(exc))
        return templates.TemplateResponse(
            request, "web/index.html", context, status_code=400
        )


@app.post("/run/async", response_class=HTMLResponse)
def run_async_from_form(
    request: Request,
    magnetic_grid: str = Form(DEFAULT_MAGNETIC_GRID),
    anomalies: str = Form(DEFAULT_ANOMALIES),
    utilities: str = Form(DEFAULT_UTILITIES),
    miis_xml: str = Form(DEFAULT_MIIS_XML),
    segy_file: str = Form(DEFAULT_SEGY),
    output_dir: str = Form(DEFAULT_OUTPUT_DIR),
    anomaly_threshold: float = Form(DEFAULT_THRESHOLD),
    norms_profile: str = Form(DEFAULT_NORMS_PROFILE),
    magnetic_grid_file: UploadFile | None = File(default=None),
    anomalies_file: UploadFile | None = File(default=None),
    utilities_file: UploadFile | None = File(default=None),
    miis_xml_file: UploadFile | None = File(default=None),
    segy_upload_file: UploadFile | None = File(default=None),
) -> HTMLResponse:
    form_values = {
        "magnetic_grid": magnetic_grid,
        "anomalies": anomalies,
        "utilities": utilities,
        "miis_xml": miis_xml,
        "segy_file": segy_file,
        "output_dir": output_dir,
        "anomaly_threshold": anomaly_threshold,
        "norms_profile": norms_profile,
    }
    form_values = _apply_upload_overrides(
        form_values,
        magnetic_grid_file=magnetic_grid_file,
        anomalies_file=anomalies_file,
        utilities_file=utilities_file,
        miis_xml_file=miis_xml_file,
        segy_upload_file=segy_upload_file,
    )

    try:
        async_job = _queue_async_job(form_values, source="web")
        context = _render_context(form_values=form_values, async_job=async_job)
        return templates.TemplateResponse(request, "web/index.html", context)
    except Exception as exc:
        logger.exception("Ошибка при асинхронном запуске пайплайна через веб-интерфейс")
        context = _render_context(form_values=form_values, error=str(exc))
        return templates.TemplateResponse(
            request, "web/index.html", context, status_code=400
        )


@app.post("/api/run")
def run_api(payload: PipelineRunRequest) -> dict[str, Any]:
    summary = _execute(payload)
    return _prepare_summary(summary)


@app.post("/api/run/async")
def run_api_async(payload: PipelineRunRequest) -> dict[str, Any]:
    return _queue_async_job(payload.model_dump(), source="api")


@app.get("/api/run/{job_id}")
def get_job(job_id: str) -> dict[str, Any]:
    job = job_manager.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    return job


@app.get("/api/runs")
def list_jobs(limit: int = Query(default=20, ge=1, le=200)) -> dict[str, Any]:
    jobs = job_manager.list_jobs(limit=limit)
    return {"items": jobs, "count": len(jobs)}
