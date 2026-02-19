# src/classification/utility_type_identifier.py
import joblib
import pandas as pd

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
            # Детерминированный fallback без случайности.
            # Нужен для стабильных результатов в тестах и при повторных запусках.
            has_thermal = int(features_dict.get('has_thermal_anomaly', 0) or 0)
            gradient = float(features_dict.get('magnetic_gradient', 0.0) or 0.0)
            rho_value = float(features_dict.get('rho_value', 0.0) or 0.0)
            near_road = int(features_dict.get('near_road', 0) or 0)
            linear_extent = float(features_dict.get('linear_extent_m', 0.0) or 0.0)

            # Тепловая аномалия чаще указывает на теплотрассу.
            if has_thermal == 1:
                return 'heating'

            # Высокий магнитный градиент рядом с дорогой — вероятнее кабель/электрика.
            if gradient >= 3.0 and near_road == 1:
                return 'electricity'

            # Низкое удельное сопротивление часто связано с водонасыщенными/канализационными зонами.
            if 0 < rho_value < 80:
                return 'sewage'

            if 80 <= rho_value < 150:
                return 'water'

            # Протяжённые линейные сигнатуры чаще похожи на магистральные трубопроводы.
            if linear_extent >= 30:
                return 'gas'

            return 'communication'

        # Создаём DataFrame из одного образца
        df = pd.DataFrame([features_dict])[self.feature_names]
        pred = self.model.predict(df)[0]
        return pred