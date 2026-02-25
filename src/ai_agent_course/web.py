from __future__ import annotations

import logging
import shutil
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from src.ai_agent_course.orchestrator import EngineeringSurveyAgent

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CACHE_ROOT = (PROJECT_ROOT / ".cache" / "agent_v2").resolve()
RUNS_DIR = CACHE_ROOT / "runs"
UPLOADS_DIR = CACHE_ROOT / "uploads"
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


def _run_analysis(prompt: str, image_path: Path, miis_path: Path | None = None) -> dict[str, Any]:
    run_id = str(uuid.uuid4())
    output_dir = RUNS_DIR / run_id
    result = agent.run(
        image_path=image_path,
        prompt=prompt,
        output_dir=output_dir,
        miis_xml_path=miis_path,
    )
    payload = asdict(result)
    artifacts = payload["artifacts"]
    artifacts["dxf_url"] = _artifact_url(Path(artifacts["dxf_path"]))
    artifacts["miis_xml_url"] = _artifact_url(Path(artifacts["miis_xml_path"]))
    artifacts["pdf_report_url"] = _artifact_url(Path(artifacts["pdf_report_path"]))
    payload["run_id"] = run_id
    return payload


def _default_form() -> dict[str, Any]:
    return {
        "prompt": (
            'Проанализируй участок строительства ЖК "Нагатинский", '
            "координаты 55.6721°N, 37.6415°E, площадь 8.4 га, "
            "глубина котлована 6.5 м, аномалия магнитного поля: амплитуда 68 нТл, "
            "протяжённость 42 м, глубина по ВЭЗ 1.8 м."
        )
    }


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/", response_class=HTMLResponse)
def index(request: Request) -> HTMLResponse:
    context = {"form": _default_form(), "result": None, "error": None}
    return templates.TemplateResponse(request, "agent_v2/index.html", context)


@app.post("/analyze", response_class=HTMLResponse)
def analyze_from_form(
    request: Request,
    prompt: str = Form(...),
    image_file: UploadFile = File(...),
    miis_xml_file: UploadFile | None = File(default=None),
) -> HTMLResponse:
    form_data = {"prompt": prompt}
    try:
        upload_dir = UPLOADS_DIR / str(uuid.uuid4())
        image_path = _save_upload(image_file, upload_dir, "image")
        miis_path = None
        if miis_xml_file and miis_xml_file.filename:
            miis_path = _save_upload(miis_xml_file, upload_dir, "miis")
        result = _run_analysis(prompt=prompt, image_path=image_path, miis_path=miis_path)
        context = {"form": form_data, "result": result, "error": None}
        return templates.TemplateResponse(request, "agent_v2/index.html", context)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Ошибка анализа в V2 форме")
        context = {"form": form_data, "result": None, "error": str(exc)}
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
) -> dict[str, Any]:
    try:
        upload_dir = UPLOADS_DIR / str(uuid.uuid4())
        image_path = _save_upload(image_file, upload_dir, "image")
        miis_path = None
        if miis_xml_file and miis_xml_file.filename:
            miis_path = _save_upload(miis_xml_file, upload_dir, "miis")
        return _run_analysis(prompt=prompt, image_path=image_path, miis_path=miis_path)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Ошибка анализа в V2 API")
        raise HTTPException(status_code=400, detail=str(exc)) from exc

