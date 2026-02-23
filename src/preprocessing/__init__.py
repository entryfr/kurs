"""Модули предобработки геофизических и геопространственных данных."""

from src.preprocessing.input_loader import (
    load_anomalies_and_utilities,
    load_magnetic_grid,
    load_miis_xml,
    load_vector_layer,
)

__all__ = [
    "load_anomalies_and_utilities",
    "load_magnetic_grid",
    "load_miis_xml",
    "load_vector_layer",
]
