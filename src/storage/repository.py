from __future__ import annotations

import json
import logging
import math
from typing import Any

import geopandas as gpd
from sqlalchemy import desc
from sqlalchemy.exc import SQLAlchemyError

from src.storage.database import (
    build_session_factory,
    ensure_spatial_extensions,
    get_engine,
)
from src.storage.models import AnomalyRecord, Base, PipelineRunRecord

logger = logging.getLogger(__name__)


def _json_safe_payload(payload: dict[str, Any]) -> dict[str, Any]:
    return json.loads(json.dumps(payload, default=str))


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(parsed):
        return None
    return parsed


def _serialize_run(run: PipelineRunRecord, include_anomalies: bool = False) -> dict[str, Any]:
    data: dict[str, Any] = {
        "run_uid": run.run_uid,
        "created_at": run.created_at.isoformat() if run.created_at else None,
        "norms_profile": run.norms_profile,
        "input_mode": run.input_mode,
        "anomalies_count": run.anomalies_count,
        "validation_issues": run.validation_issues,
    }
    if include_anomalies:
        data["anomalies"] = [
            {
                "anomaly_index": anomaly.anomaly_index,
                "risk_class": anomaly.risk_class,
                "risk_probability": anomaly.risk_probability,
                "utility_type": anomaly.utility_type,
                "depth_m": anomaly.depth_m,
                "in_blind_zone": anomaly.in_blind_zone,
                "corrosion_critical": anomaly.corrosion_critical,
                "geometry_wkt": anomaly.geometry_wkt,
                "pointcloud_ref": anomaly.pointcloud_ref,
            }
            for anomaly in run.anomalies
        ]
        data["payload"] = run.payload_json
    return data


def persist_pipeline_run(
    summary: dict[str, Any],
    anomalies_gdf: gpd.GeoDataFrame,
    database_url: str | None = None,
) -> dict[str, Any]:
    engine = get_engine(database_url)
    if engine is None:
        return {
            "enabled": False,
            "status": "skipped",
            "reason": "POSTGRES_DSN not configured or persistence disabled",
        }

    try:
        ensure_spatial_extensions(engine)
        Base.metadata.create_all(bind=engine)
        session_factory = build_session_factory(engine)
        run_uid = PipelineRunRecord.generate_run_uid()
        with session_factory() as session:
            run = PipelineRunRecord(
                run_uid=run_uid,
                norms_profile=str(summary.get("norms_profile", "normative")),
                input_mode=str(summary.get("input_mode", "unknown")),
                anomalies_count=int(summary.get("anomalies_count", len(anomalies_gdf))),
                validation_issues=int(summary.get("validation_issues", 0)),
                payload_json=_json_safe_payload(summary),
            )
            session.add(run)
            session.flush()

            for anomaly_index, row in anomalies_gdf.iterrows():
                geometry = row.get("geometry")
                geometry_wkt = (
                    geometry.wkt
                    if geometry is not None and hasattr(geometry, "is_empty") and not geometry.is_empty
                    else None
                )
                record = AnomalyRecord(
                    run_id=run.id,
                    anomaly_index=int(anomaly_index),
                    risk_class=str(row.get("risk_class", "LOW")),
                    risk_probability=_safe_float(row.get("risk_probability")),
                    utility_type=str(row.get("utility_type", "unknown")),
                    depth_m=_safe_float(row.get("depth")),
                    in_blind_zone=bool(row.get("in_blind_zone", False)),
                    corrosion_critical=bool(row.get("corrosion_critical", False)),
                    geometry_wkt=geometry_wkt,
                    pointcloud_ref=str(row.get("pointcloud_ref")) if row.get("pointcloud_ref") else None,
                )
                session.add(record)

            session.commit()

        return {
            "enabled": True,
            "status": "persisted",
            "run_uid": run_uid,
            "anomalies_saved": len(anomalies_gdf),
            "backend": engine.dialect.name,
        }
    except (SQLAlchemyError, ValueError, TypeError) as exc:
        logger.exception("Ошибка при сохранении результата пайплайна в БД")
        return {
            "enabled": True,
            "status": "failed",
            "error": str(exc),
            "backend": engine.dialect.name,
        }


def list_persisted_runs(limit: int = 20, database_url: str | None = None) -> list[dict[str, Any]]:
    engine = get_engine(database_url)
    if engine is None:
        return []
    Base.metadata.create_all(bind=engine)
    session_factory = build_session_factory(engine)
    with session_factory() as session:
        rows = (
            session.query(PipelineRunRecord)
            .order_by(desc(PipelineRunRecord.created_at))
            .limit(limit)
            .all()
        )
        return [_serialize_run(row, include_anomalies=False) for row in rows]


def get_persisted_run(run_uid: str, database_url: str | None = None) -> dict[str, Any] | None:
    engine = get_engine(database_url)
    if engine is None:
        return None
    Base.metadata.create_all(bind=engine)
    session_factory = build_session_factory(engine)
    with session_factory() as session:
        row = session.query(PipelineRunRecord).filter_by(run_uid=run_uid).first()
        if row is None:
            return None
        return _serialize_run(row, include_anomalies=True)
