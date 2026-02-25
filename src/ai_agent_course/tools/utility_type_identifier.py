from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import RandomForestClassifier

UTILITY_TYPES = ["water", "sewage", "gas", "cable", "heating"]


@dataclass(slots=True)
class UtilityFeatures:
    amplitude_nt: float
    gradient_rho: float
    hyperbola_width: float
    depth_m: float
    thermal_contrast: float
    linear_extent_m: float
    rho_value: float
    magnetic_gradient: float
    seg_velocity: float
    near_road: float
    area_type: float
    age_infrastructure: float
    soil_moisture: float
    depth_to_diameter: float


class UtilityTypeIdentifierV2:
    def __init__(self) -> None:
        self._model = self._train_model()

    @staticmethod
    def _generate_training_data(seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(seed)
        n = 1200
        x = np.zeros((n, 14), dtype=float)
        y = np.zeros(n, dtype=int)

        for i in range(n):
            cls = int(rng.integers(0, len(UTILITY_TYPES)))
            y[i] = cls

            amplitude = rng.uniform(25, 140)
            gradient_rho = rng.uniform(0.1, 2.5)
            hyperbola = rng.uniform(0.4, 2.8)
            depth = rng.uniform(0.7, 4.5)
            thermal = rng.uniform(0.0, 1.0)
            linear = rng.uniform(10.0, 120.0)
            rho_val = rng.uniform(8.0, 120.0)
            mag_grad = rng.uniform(0.5, 6.0)
            seg_v = rng.uniform(0.05, 0.35)
            near_road = float(rng.integers(0, 2))
            area_type = float(rng.integers(0, 3))
            age = rng.uniform(1.0, 65.0)
            moisture = rng.uniform(0.05, 0.95)
            depth_diam = rng.uniform(1.0, 12.0)

            if cls == 0:  # water
                amplitude = rng.uniform(30, 80)
                rho_val = rng.uniform(5, 20)
                depth = rng.uniform(0.8, 2.0)
            elif cls == 1:  # sewage
                amplitude = rng.uniform(15, 55)
                hyperbola = rng.uniform(1.8, 3.2)
                rho_val = rng.uniform(5, 15)
            elif cls == 2:  # gas
                amplitude = rng.uniform(90, 145)
                thermal = rng.uniform(0.0, 0.2)
                mag_grad = rng.uniform(3.0, 7.0)
            elif cls == 3:  # cable
                hyperbola = rng.uniform(0.4, 1.2)
                gradient_rho = rng.uniform(1.0, 3.0)
                amplitude = rng.uniform(35, 95)
            elif cls == 4:  # heating
                thermal = rng.uniform(0.6, 1.0)
                rho_val = rng.uniform(10, 35)
                amplitude = rng.uniform(40, 110)

            x[i] = [
                amplitude,
                gradient_rho,
                hyperbola,
                depth,
                thermal,
                linear,
                rho_val,
                mag_grad,
                seg_v,
                near_road,
                area_type,
                age,
                moisture,
                depth_diam,
            ]
        return x, y

    def _train_model(self) -> RandomForestClassifier:
        x, y = self._generate_training_data()
        model = RandomForestClassifier(
            n_estimators=280,
            max_depth=16,
            random_state=42,
            n_jobs=-1,
        )
        model.fit(x, y)
        return model

    def predict(self, features: UtilityFeatures) -> str:
        row = np.array(
            [
                [
                    features.amplitude_nt,
                    features.gradient_rho,
                    features.hyperbola_width,
                    features.depth_m,
                    features.thermal_contrast,
                    features.linear_extent_m,
                    features.rho_value,
                    features.magnetic_gradient,
                    features.seg_velocity,
                    features.near_road,
                    features.area_type,
                    features.age_infrastructure,
                    features.soil_moisture,
                    features.depth_to_diameter,
                ]
            ],
            dtype=float,
        )
        cls = int(self._model.predict(row)[0])
        return UTILITY_TYPES[cls]

