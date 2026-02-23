
import segyio
import numpy as np

def read_segy(file_path):
    """
    Читает SEG-Y файл и возвращает массив данных и метаинформацию.
    Возвращает:
        data: numpy.ndarray формы (traces, samples)
        sample_rate: интервал дискретизации (мкс или нс)
        depths: массив глубин/времён для каждого сэмпла
    """
    with segyio.open(file_path, "r", ignore_geometry=True) as segyfile:
        # Загружаем все трассы в массив
        data = np.asarray([np.copy(tr) for tr in segyfile.trace])
        sample_rate = segyfile.bin[segyio.BinField.Interval] / 1000.0  # часто в мкс -> мс
        # Глубины/времена для каждого отсчёта
        depths = np.arange(data.shape[1]) * sample_rate
        return data, sample_rate, depths

def extract_hyperbola_parameters(radargram, threshold=0.5):
    """
    Выделяет простые параметры гиперболы из радарграммы в детерминированном виде.
    """
    if radargram.size == 0:
        return {
            'width': 0.0,
            'apex_depth': 0.0,
            'confidence': 0.0,
        }

    amplitude = np.abs(radargram)
    max_trace_idx, apex_sample_idx = np.unravel_index(np.argmax(amplitude), amplitude.shape)
    apex_profile = amplitude[:, apex_sample_idx]
    if apex_profile.max() <= 0:
        active_width = 0.0
    else:
        active_mask = apex_profile >= (apex_profile.max() * threshold)
        active_width = float(np.count_nonzero(active_mask))

    confidence = float(apex_profile.max() / (np.mean(amplitude) + 1e-6))

    return {
        'width': active_width,
        'apex_depth': float(apex_sample_idx),
        'confidence': confidence,
        'apex_trace': float(max_trace_idx),
    }


def extract_segy_features(file_path):
    """
    Извлекает признаки из SEG-Y для интеграции в классификацию.
    """
    data, sample_rate, depths = read_segy(file_path)
    params = extract_hyperbola_parameters(data)

    trace_energy = np.mean(np.abs(data), axis=1)
    if np.allclose(trace_energy, 0):
        seg_velocity = 0.0
    else:
        seg_velocity = float(np.std(trace_energy) / (np.mean(trace_energy) + 1e-6))

    apex_depth_m = float(params['apex_depth'] * sample_rate)

    return {
        'radar_hyperbola_w': float(params['width']),
        'seg_velocity': seg_velocity,
        'segy_confidence': min(params['confidence'] / 10.0, 1.0),
        'apex_depth_m': apex_depth_m,
    }