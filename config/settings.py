# config/settings.py
# ─────────────────────────────────────────────────────────────────────────────
# Центральный конфигурационный файл проекта геофизического обследования
# подземных инженерных коммуникаций
# ─────────────────────────────────────────────────────────────────────────────

# ── Геопространственные системы координат ────────────────────────────────────
CRS_INPUT   = "EPSG:4936"   # ПЗ-90.11 (геоцентрические, используется с pyproj)
CRS_LOCAL   = "EPSG:28406"  # СК-42 / Гаусс-Крюгер зона 6 (МСК-50 / Москва)
CRS_WGS84   = "EPSG:4326"   # WGS-84 — стандарт OSM/Overpass
CRS_UTM37N  = "EPSG:32637"  # UTM зона 37N — метрическая CRS для Москвы (для расчётов)

# ── Профили нормативных допусков ─────────────────────────────────────────────
NORMATIVE_PROFILES = {
    "demo": {
        "rms_tolerance": 0.18,
        "buffer_distance": 3.0,
        "description": "Демонстрационный профиль для пилотных запусков.",
    },
    "normative": {
        "rms_tolerance": 0.18,
        "buffer_distance": 2.5,
        "description": "Профиль, ориентированный на требования СП 47.",
    },
    "strict": {
        "rms_tolerance": 0.05,
        "buffer_distance": 2.0,
        "description": "Строгий внутренний контроль качества.",
    },
}

DEFAULT_NORMS_PROFILE = "normative"

# Значения по умолчанию для обратной совместимости модулей
RMS_TOLERANCE = NORMATIVE_PROFILES[DEFAULT_NORMS_PROFILE]["rms_tolerance"]
BUFFER_DISTANCE = NORMATIVE_PROFILES[DEFAULT_NORMS_PROFILE]["buffer_distance"]

# ── Весовые коэффициенты последствий (для формулы риска R = P * C * (1-D)) ──
# Чем выше — тем серьёзнее последствия повреждения коммуникации
CONSEQUENCE_WEIGHTS = {
    "gas":           10,   # Взрывопожароопасность
    "electricity":    9,   # Поражение электротоком
    "heating":        8,   # Затопление кипятком, разрушение дорожного полотна
    "water":          6,   # Размыв грунта, затопление
    "sewage":         5,   # Загрязнение грунтовых вод
    "communication":  3,   # Нарушение связи
}

# ── Коэффициенты модели коррозии ──────────────────────────────────────────────
# rate = k1/rho + k2*moisture + k3*|pH - 7|   (мм/год)
CORROSION_COEFFS = {
    "k1": 0.5,   # Коэффициент влияния удельного сопротивления грунта
    "k2": 0.3,   # Коэффициент влияния влажности
    "k3": 0.2,   # Коэффициент влияния pH
}

# ── Параметры детекции аномалий ───────────────────────────────────────────────
ANOMALY_THRESHOLD_NT   = 30.0   # Минимальная амплитуда аномалии, нТл
ANOMALY_MIN_LENGTH_PX  = 2      # Минимальная длина линейной аномалии, пиксели
BLIND_ZONE_THRESHOLD   = 50.0   # Порог шума для определения «слепой зоны», нТл
BLIND_ZONE_RADIUS      = 50.0   # Радиус поиска зон без данных, метры

# ── Параметры Overpass API ────────────────────────────────────────────────────
OVERPASS_URL            = "https://overpass-api.de/api/interpreter"
OVERPASS_TIMEOUT        = 60           # Таймаут запроса, секунды
OVERPASS_CACHE_DIR      = ".cache/osm" # Папка кеширования ответов
OVERPASS_CACHE_TTL      = 86400        # Время жизни кеша, секунды (1 сутки)

# Теги OSM, которые загружаем (тип -> список значений тега)
OSM_UTILITY_TAGS = {
    "man_made": ["pipeline"],
    "power":    ["cable", "line"],
    "utility":  ["gas", "water", "sewage", "heating", "electricity"],
}

# Маппинг OSM-тега substance -> внутренний тип проекта
OSM_SUBSTANCE_MAP = {
    "gas":          "gas",
    "water":        "water",
    "sewage":       "sewage",
    "oil":          "gas",        # Нефтепровод — приравниваем к газу по риску
    "hot_water":    "heating",
    "heating":      "heating",
    "electricity":  "electricity",
    "cable":        "electricity",
}

# ── Параметры экспорта ────────────────────────────────────────────────────────
DXF_OUTPUT_PATH  = "output/scheme.dxf"
MIIS_OUTPUT_PATH = "output/miis_output.xml"
ACT_OUTPUT_PATH  = "output/act.docx"
ACT_TEMPLATE     = "templates/act_template.jinja2"
