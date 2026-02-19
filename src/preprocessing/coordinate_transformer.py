import numpy as np
from scipy.optimize import least_squares
from pyproj import Transformer
from shapely.affinity import affine_transform
from config.settings import RMS_TOLERANCE

def fit_affine(src_points, dst_points):
    """
    src_points: массив (N,2) координат в ПЗ-90.11
    dst_points: массив (N,2) координат в МСК-50 (измеренных)
    Возвращает параметры a,b,c,d,e,f
    """
    def residuals(p, s, d):
        a,b,c, d_,e,f = p
        return np.concatenate([
            a*s[:,0] + b*s[:,1] + c - d[:,0],
            d_*s[:,0] + e*s[:,1] + f - d[:,1]
        ])
    result = least_squares(residuals, [1,0,0,0,1,0], args=(src_points, dst_points))
    rmse = np.sqrt(np.sum(result.fun**2) / (2 * len(src_points)))
    if rmse > RMS_TOLERANCE:
        raise ValueError(f"СКО {rmse:.4f} м превышает допуск {RMS_TOLERANCE} м!")
    return result.x

def transform_geometry(geom, affine_params):
    """Применить аффинное преобразование к shapely-геометрии"""
    a,b,c,d,e,f = affine_params
    return affine_transform(geom, [a, b, d, e, c, f])

def pz90_to_msk50(lon, lat):
    """Прямое преобразование ПЗ-90.11 -> МСК-50 через pyproj (как первый шаг)"""
    # Используем EPSG:4936 (ПЗ-90.11 геоцентрические) -> локальная проекция
    # Здесь нужно определить правильный EPSG для МСК-50 (например, EPSG:28406)
    transformer = Transformer.from_crs("EPSG:4936", "EPSG:28406", always_xy=True)
    return transformer.transform(lon, lat)