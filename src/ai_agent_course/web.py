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
from src.ai_agent_course.secrets_loader import load_local_env

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_local_env(PROJECT_ROOT / ".env.local")
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

