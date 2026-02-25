from __future__ import annotations

import logging
from typing import Any

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import sessionmaker

from config import settings

logger = logging.getLogger(__name__)

_ENGINE_CACHE: dict[str, Engine] = {}


def resolve_database_url(database_url: str | None = None) -> str | None:
    if database_url is not None and database_url.strip():
        return database_url.strip()
    if not settings.ENABLE_DB_PERSISTENCE:
        return None
    if settings.POSTGRES_DSN:
        return settings.POSTGRES_DSN
    return None


def get_engine(database_url: str | None = None) -> Engine | None:
    resolved_url = resolve_database_url(database_url)
    if resolved_url is None:
        return None
    if resolved_url in _ENGINE_CACHE:
        return _ENGINE_CACHE[resolved_url]

    if resolved_url.startswith("sqlite"):
        engine = create_engine(resolved_url, future=True)
    else:
        engine = create_engine(
            resolved_url,
            pool_size=settings.DB_POOL_SIZE,
            max_overflow=settings.DB_MAX_OVERFLOW,
            pool_pre_ping=True,
            future=True,
        )
    _ENGINE_CACHE[resolved_url] = engine
    return engine


def build_session_factory(engine: Engine) -> sessionmaker[Any]:
    return sessionmaker(bind=engine, autoflush=False, expire_on_commit=False, future=True)


def ensure_spatial_extensions(engine: Engine) -> None:
    if engine.dialect.name != "postgresql":
        return
    extension_statements = [
        "CREATE EXTENSION IF NOT EXISTS postgis",
        "CREATE EXTENSION IF NOT EXISTS pointcloud",
        "CREATE EXTENSION IF NOT EXISTS pointcloud_postgis",
    ]
    with engine.begin() as connection:
        for statement in extension_statements:
            try:
                connection.execute(text(statement))
            except SQLAlchemyError as exc:
                logger.warning("Не удалось выполнить '%s': %s", statement, exc)
