from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


class PipelineRunRecord(Base):
    __tablename__ = "pipeline_runs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    run_uid = Column(String(64), unique=True, nullable=False, index=True)
    created_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )
    norms_profile = Column(String(32), nullable=False, default="normative")
    input_mode = Column(String(32), nullable=False, default="vector_layers")
    anomalies_count = Column(Integer, nullable=False, default=0)
    validation_issues = Column(Integer, nullable=False, default=0)
    payload_json = Column(JSON, nullable=False, default=dict)

    anomalies = relationship(
        "AnomalyRecord",
        back_populates="run",
        cascade="all, delete-orphan",
        lazy="joined",
    )

    @staticmethod
    def generate_run_uid() -> str:
        return f"run-{uuid.uuid4()}"


class AnomalyRecord(Base):
    __tablename__ = "anomaly_records"

    id = Column(Integer, primary_key=True, autoincrement=True)
    run_id = Column(Integer, ForeignKey("pipeline_runs.id"), nullable=False, index=True)
    anomaly_index = Column(Integer, nullable=False)
    risk_class = Column(String(16), nullable=False, default="LOW")
    risk_probability = Column(Float, nullable=True)
    utility_type = Column(String(64), nullable=True)
    depth_m = Column(Float, nullable=True)
    in_blind_zone = Column(Boolean, nullable=False, default=False)
    corrosion_critical = Column(Boolean, nullable=False, default=False)
    geometry_wkt = Column(Text, nullable=True)
    pointcloud_ref = Column(String(255), nullable=True)

    run = relationship("PipelineRunRecord", back_populates="anomalies")
