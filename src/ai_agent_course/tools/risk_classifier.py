from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    from catboost import CatBoostClassifier
except Exception:  # noqa: BLE001
    CatBoostClassifier = None

from sklearn.ensemble import GradientBoostingClassifier


@dataclass(slots=True)
class RiskFeatures:
    excavation_depth_m: float
    anomaly_depth_m: float
    amplitude_nt: float
    length_m: float
    utility_type_code: float


class DamageRiskClassifier:
    """P(повреждение | глубина котлована, параметры аномалии)."""

    def __init__(self) -> None:
        self._model = self._train_model()

    @staticmethod
    def _build_training_set(seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(seed)
        n = 1200  # Имитируем исторический корпус кейсов 2018-2023.
        excavation_depth = rng.uniform(1.0, 8.0, n)
        anomaly_depth = rng.uniform(0.5, 5.5, n)
        amplitude_nt = rng.uniform(10.0, 150.0, n)
        length_m = rng.uniform(8.0, 120.0, n)
        utility_code = rng.integers(0, 5, n).astype(float)

        depth_overlap = np.clip((excavation_depth - anomaly_depth) / 4.0, -1.0, 1.5)
        signal_risk = np.clip((amplitude_nt - 30.0) / 120.0, 0.0, 1.0)
        length_risk = np.clip((length_m - 10.0) / 90.0, 0.0, 1.0)
        base = 0.2 + 0.55 * depth_overlap + 0.25 * signal_risk + 0.2 * length_risk
        base += (utility_code == 2).astype(float) * 0.1  # газ
        prob = 1.0 / (1.0 + np.exp(-base))
        y = (rng.uniform(0.0, 1.0, n) < prob).astype(int)

        x = np.column_stack(
            [excavation_depth, anomaly_depth, amplitude_nt, length_m, utility_code]
        )
        return x, y

    def _train_model(self):
        x, y = self._build_training_set()
        if CatBoostClassifier is not None:
            model = CatBoostClassifier(
                depth=6,
                learning_rate=0.08,
                iterations=120,
                loss_function="Logloss",
                verbose=False,
            )
            model.fit(x, y)
            return model

        model = GradientBoostingClassifier(random_state=42)
        model.fit(x, y)
        return model

    def damage_probability(self, features: RiskFeatures) -> float:
        row = np.array(
            [
                [
                    float(features.excavation_depth_m),
                    float(features.anomaly_depth_m),
                    float(features.amplitude_nt),
                    float(features.length_m),
                    float(features.utility_type_code),
                ]
            ]
        )
        proba = self._model.predict_proba(row)[0, 1]
        return float(np.clip(proba, 0.01, 0.99))

