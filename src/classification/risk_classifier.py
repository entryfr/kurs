import numpy as np
from catboost import CatBoostClassifier
from config.settings import CONSEQUENCE_WEIGHTS

class RiskClassifier:
    def __init__(self, model_path=None):
        if model_path:
            self.model = CatBoostClassifier()
            self.model.load_model(model_path)
        else:
            self.model = None  # Заглушка

    def predict_risk(self, features):
        """
        features: dict с признаками
        Возвращает класс риска: 'CRITICAL', 'HIGH', 'LOW'
        """
        if self.model:
            proba = self.model.predict_proba([features])[0]
            class_idx = np.argmax(proba)
            return ['LOW', 'HIGH', 'CRITICAL'][class_idx]
        else:
            # Логика-заглушка
            depth_diff = features.get('excavation_depth', 0) - features.get('anomaly_depth', 0)
            P = 1 / (1 + np.exp(-depth_diff * 2))
            C = CONSEQUENCE_WEIGHTS.get(features.get('utility_type', 'water'), 6)
            D = features.get('detection_factor', 1.0)
            R = P * C * (1 - D)
            if R > 15:
                return 'CRITICAL'
            elif R > 8:
                return 'HIGH'
            else:
                return 'LOW'