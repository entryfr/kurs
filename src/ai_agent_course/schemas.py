from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class PromptContext:
    site_name: str
    area_ha: float
    excavation_depth_m: float
    latitude: float | None
    longitude: float | None


@dataclass(slots=True)
class ArtifactPaths:
    output_dir: str
    dxf_path: str
    miis_xml_path: str
    pdf_report_path: str


@dataclass(slots=True)
class AgentRunResult:
    prompt_context: dict[str, Any]
    anomalies: list[dict[str, Any]]
    validation: dict[str, Any]
    metrics: dict[str, Any]
    artifacts: ArtifactPaths
    answer: str
    mode: str

