from __future__ import annotations

import json
import threading
import uuid
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class ChatManager:
    def __init__(self, history_path: Path, max_sessions: int = 100) -> None:
        self.history_path = history_path
        self.max_sessions = max_sessions
        self.history_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._sessions: dict[str, dict[str, Any]] = {}
        self._load()

    def _load(self) -> None:
        if not self.history_path.exists():
            return
        try:
            data = json.loads(self.history_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            self._sessions = {}
            return

        if not isinstance(data, list):
            self._sessions = {}
            return
        for item in data:
            if isinstance(item, dict) and item.get("session_id"):
                self._sessions[item["session_id"]] = item

    def _persist(self) -> None:
        ordered = sorted(
            self._sessions.values(),
            key=lambda x: x.get("updated_at", ""),
            reverse=True,
        )
        ordered = ordered[: self.max_sessions]
        self.history_path.write_text(
            json.dumps(ordered, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _trim(self) -> None:
        ordered = sorted(
            self._sessions.items(),
            key=lambda x: x[1].get("updated_at", ""),
            reverse=True,
        )
        self._sessions = dict(ordered[: self.max_sessions])

    def create_session(self, title: str | None = None) -> dict[str, Any]:
        session_id = str(uuid.uuid4())
        session = {
            "session_id": session_id,
            "title": title or "Новый диалог",
            "created_at": _utc_now_iso(),
            "updated_at": _utc_now_iso(),
            "messages": [],
        }
        with self._lock:
            self._sessions[session_id] = session
            self._trim()
            self._persist()
            return deepcopy(session)

    def get_or_create_session(self, session_id: str | None = None) -> dict[str, Any]:
        with self._lock:
            if session_id and session_id in self._sessions:
                return deepcopy(self._sessions[session_id])
        return self.create_session()

    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        with self._lock:
            if session_id not in self._sessions:
                self._sessions[session_id] = {
                    "session_id": session_id,
                    "title": "Новый диалог",
                    "created_at": _utc_now_iso(),
                    "updated_at": _utc_now_iso(),
                    "messages": [],
                }
            session = self._sessions[session_id]
            message = {
                "role": role,
                "content": content,
                "created_at": _utc_now_iso(),
                "metadata": metadata or {},
            }
            session["messages"].append(message)
            session["updated_at"] = _utc_now_iso()

            if (
                role == "user"
                and session["title"] == "Новый диалог"
                and content.strip()
            ):
                session["title"] = content.strip()[:60]

            self._trim()
            self._persist()
            return deepcopy(session)

    def get_session(self, session_id: str) -> dict[str, Any] | None:
        with self._lock:
            session = self._sessions.get(session_id)
            return deepcopy(session) if session else None

    def list_sessions(self, limit: int = 20) -> list[dict[str, Any]]:
        with self._lock:
            ordered = sorted(
                self._sessions.values(),
                key=lambda x: x.get("updated_at", ""),
                reverse=True,
            )
            result = []
            for item in ordered[:limit]:
                result.append(
                    {
                        "session_id": item["session_id"],
                        "title": item.get("title", "Новый диалог"),
                        "created_at": item.get("created_at"),
                        "updated_at": item.get("updated_at"),
                        "messages_count": len(item.get("messages", [])),
                    }
                )
            return deepcopy(result)

    def reset_for_tests(self) -> None:
        with self._lock:
            self._sessions = {}
            if self.history_path.exists():
                self.history_path.unlink()
