from src.classification.utility_type_identifier import UtilityTypeIdentifier


def test_fallback_is_deterministic_for_same_input():
    identifier = UtilityTypeIdentifier()
    features = {
        "amplitude_nt": 52.0,
        "gradient_rho": 0.6,
        "depth_vez_m": 3.0,
        "radar_hyperbola_w": 1.1,
        "linear_extent_m": 22.0,
        "rho_value": 95.0,
        "has_thermal_anomaly": 0,
        "depth_to_diameter": 5.0,
        "magnetic_gradient": 2.3,
        "seg_velocity": 0.1,
        "near_road": 0,
        "area_type": 0,
        "age_infrastructure": 18.0,
        "soil_moisture": 0.25,
    }

    first = identifier.predict(features)
    second = identifier.predict(features)
    third = identifier.predict(features)

    assert first == second == third


def test_thermal_anomaly_prioritizes_heating():
    identifier = UtilityTypeIdentifier()
    features = {
        "has_thermal_anomaly": 1,
        "magnetic_gradient": 6.0,
        "near_road": 1,
        "rho_value": 40.0,
        "linear_extent_m": 100.0,
    }

    assert identifier.predict(features) == "heating"


def test_high_gradient_near_road_maps_to_electricity():
    identifier = UtilityTypeIdentifier()
    features = {
        "has_thermal_anomaly": 0,
        "magnetic_gradient": 3.5,
        "near_road": 1,
        "rho_value": 200.0,
        "linear_extent_m": 10.0,
    }

    assert identifier.predict(features) == "electricity"
