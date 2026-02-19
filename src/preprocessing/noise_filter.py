import pywt
import numpy as np
from scipy.ndimage import median_filter

def wavelet_denoise(data, wavelet='db4', level=3, mode='soft'):
    coeffs = pywt.wavedec2(data, wavelet, level=level)
    # Оценка порога по остаткам
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    threshold = sigma * np.sqrt(2 * np.log(data.size))
    coeffs_thresh = [coeffs[0]]
    for c in coeffs[1:]:
        # c – кортеж из трёх детализирующих коэффициентов (cH, cV, cD)
        c_th = tuple(pywt.threshold(c_i, threshold, mode=mode) for c_i in c)
        coeffs_thresh.append(c_th)
    denoised = pywt.waverec2(coeffs_thresh, wavelet)
    # Обрезаем до исходного размера (может быть на 1 пиксель больше)
    return denoised[:data.shape[0], :data.shape[1]]

def apply_filters(magnetic_grid):
    # Сначала вейвлет, потом медианный 3x3
    denoised = wavelet_denoise(magnetic_grid)
    filtered = median_filter(denoised, size=3)
    return filtered