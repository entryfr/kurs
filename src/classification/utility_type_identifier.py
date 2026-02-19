# src/classification/utility_type_identifier.py
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

class UtilityTypeIdentifier:
    """
    Определяет тип подземной коммуникации по геофизическим признакам.
    """
    def __init__(self, model_path=None):
        if model_path:
            self.model = joblib.load(model_path)
        else:
            # Создаём заглушку (можно обучить позже)
            self.model = None
        # Список признаков в том же порядке, что и при обучении
        self.feature_names = [
            'amplitude_nt', 'gradient_rho', 'depth_vez_m', 'radar_hyperbola_w',
            'linear_extent_m', 'rho_value', 'has_thermal_anomaly', 'depth_to_diameter',
            'magnetic_gradient', 'seg_velocity', 'near_road', 'area_type',
            'age_infrastructure', 'soil_moisture'
        ]
        self.types = ['gas', 'electricity', 'heating', 'water', 'sewage', 'communication']

    def predict(self, features_dict):
        """
        features_dict: словарь с признаками (ключи должны совпадать с self.feature_names)
        Возвращает предсказанный тип (строка).
        """
        if self.model is None:
            # Заглушка: случайный тип, но с учётом некоторых правил
            # Например, если есть тепловая аномалия, исключаем газ
            if features_dict.get('has_thermal_anomaly', 0) == 1:
                possible = ['electricity', 'heating', 'water', 'sewage', 'communication']
            else:
                possible = self.types
            return np.random.choice(possible)

        # Создаём DataFrame из одного образца
        df = pd.DataFrame([features_dict])[self.feature_names]
        pred = self.model.predict(df)[0]
        return pred