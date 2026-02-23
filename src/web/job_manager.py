from __future__ import annotations

import json
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class JobManager:
    def __init__(self, history_path: Path, max_workers: int = 2) -> None:
        self.history_path = history_path
        self.history_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="pipeline-job"
        )
        self._jobs: dict[str, dict[str, Any]] = {}
        self._load_history()

    def _load_history(self) -> None:
        if not self.history_path.exists():
            return
        try:
            data = json.loads(self.history_path.read_text(encoding="utf-8"))
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, dict) and "job_id" in item:
                        self._jobs[item["job_id"]] = item
        except (json.JSONDecodeError, OSError):
            # История может быть повреждена после прерывания записи — не блокируем приложение.
            self._jobs = {}

    def _persist(self) -> None:
        ordered = sorted(
            self._jobs.values(), key=lambda x: x.get("created_at", ""), reverse=True
        )
        self.history_path.write_text(
            json.dumps(ordered[:200], ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def create_job(self, request_payload: dict[str, Any], source: str) -> dict[str, Any]:
        job_id = str(uuid.uuid4())
        job = {
            "job_id": job_id,
            "source": source,
            "status": "queued",
            "created_at": _utc_now_iso(),
            "started_at": None,
            "finished_at": None,
            "error": None,
            "request": deepcopy(request_payload),
            "result": None,
        }
        with self._lock:
            self._jobs[job_id] = job
            self._persist()
            return deepcopy(job)

    def submit(self, job_id: str, run_fn: Callable[[], dict[str, Any]]) -> None:
        self._executor.submit(self._run_job, job_id, run_fn)

    def _run_job(self, job_id: str, run_fn: Callable[[], dict[str, Any]]) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            job["status"] = "running"
            job["started_at"] = _utc_now_iso()
            self._persist()

        try:
            result = run_fn()
            with self._lock:
                job = self._jobs.get(job_id)
                if job is None:
                    return
                job["status"] = "completed"
                job["result"] = deepcopy(result)
                job["finished_at"] = _utc_now_iso()
                self._persist()
        except Exception as exc:  # noqa: BLE001
            with self._lock:
                job = self._jobs.get(job_id)
                if job is None:
                    return
                job["status"] = "failed"
                job["error"] = str(exc)
                job["finished_at"] = _utc_now_iso()
                self._persist()

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        with self._lock:
            job = self._jobs.get(job_id)
            return deepcopy(job) if job else None

    def list_jobs(self, limit: int = 20) -> list[dict[str, Any]]:
        with self._lock:
            ordered = sorted(
                self._jobs.values(), key=lambda x: x.get("created_at", ""), reverse=True
            )
            return deepcopy(ordered[:limit])

    def reset_for_tests(self) -> None:
        with self._lock:
            self._jobs = {}
            if self.history_path.exists():
                self.history_path.unlink()
