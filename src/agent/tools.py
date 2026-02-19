# src/agent/tools.py
from langchain.tools import BaseTool
from src.preprocessing.coordinate_transformer import transform_geometry
from src.detection.anomaly_detector import detect_linear_anomalies
from src.classification.risk_classifier import RiskClassifier
from src.classification.utility_type_identifier import UtilityTypeIdentifier
from src.classification.corrosion_estimator import assess_corrosion
from src.reporting.act_generator import generate_act
import numpy as np

class TransformCoordinatesTool(BaseTool):
    name = "TransformCoordinates"
    description = "Преобразует координаты из ПЗ-90.11 в МСК-50. Вход: словарь с 'geom' и 'affine_params'."

    def _run(self, geom, affine_params):
        return transform_geometry(geom, affine_params)

    async def _arun(self, geom, affine_params):
        raise NotImplementedError

class DetectAnomaliesTool(BaseTool):
    name = "DetectAnomalies"
    description = "Детектирует линейные аномалии на магнитной карте. Вход: numpy array магнитного поля."

    def _run(self, magnetic_grid):
        return detect_linear_anomalies(magnetic_grid)

    async def _arun(self, magnetic_grid):
        raise NotImplementedError

# Аналогично для остальных шагов...