# src/classification/corrosion_estimator.py
import numpy as np
from config.settings import CORROSION_COEFFS

def estimate_corrosion_rate(soil_rho, moisture, ph):
    """
    Вычисляет скорость коррозии (мм/год) по эмпирической формуле.
    soil_rho: удельное сопротивление грунта (Ом·м)
    moisture: влажность (доли единицы, 0..1)
    ph: кислотность грунта
    """
    k1, k2, k3 = CORROSION_COEFFS['k1'], CORROSION_COEFFS['k2'], CORROSION_COEFFS['k3']
    rate = k1 / soil_rho + k2 * moisture + k3 * abs(ph - 7.0)
    return rate

def estimate_remaining_life(initial_diameter, current_diameter, corrosion_rate):
    """
    Вычисляет остаточный срок службы (лет) до достижения критического износа (30% от начального диаметра).
    Если corrosion_rate == 0, возвращает бесконечность.
    """
    critical_diameter = 0.3 * initial_diameter
    if current_diameter <= critical_diameter:
        return 0.0
    if corrosion_rate <= 0:
        return float('inf')
    remaining = (current_diameter - critical_diameter) / corrosion_rate
    return max(remaining, 0.0)

def assess_corrosion(utility_data):
    """
    Основная функция для оценки коррозии по данным об объекте.
    utility_data: dict с полями:
        - soil_rho
        - moisture
        - ph
        - initial_diameter
        - current_diameter (если неизвестно, можно оценить по возрасту)
        - age (возраст, лет) – опционально
    Возвращает dict с результатами.
    """
    rate = estimate_corrosion_rate(
        utility_data['soil_rho'],
        utility_data['moisture'],
        utility_data['ph']
    )
    if 'current_diameter' not in utility_data:
        # Оцениваем текущий диаметр по возрасту и начальному диаметру
        estimated_wear = rate * utility_data.get('age', 0)
        current_diameter = max(utility_data['initial_diameter'] - estimated_wear, 0)
    else:
        current_diameter = utility_data['current_diameter']

    remaining = estimate_remaining_life(utility_data['initial_diameter'], current_diameter, rate)
    return {
        'corrosion_rate': rate,
        'current_diameter': current_diameter,
        'remaining_life': remaining,
        'critical': remaining < 5.0
    }