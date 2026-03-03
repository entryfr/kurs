from __future__ import annotations

from typing import Any


def validate_rules(anomalies: list[dict[str, Any]], rms_error_m: float) -> dict[str, Any]:
    issues: list[str] = []
    by_rule: dict[str, int] = {}

    def add(rule: str, text: str) -> None:
        issues.append(f"{rule}: {text}")
        by_rule[rule] = by_rule.get(rule, 0) + 1

    if rms_error_m > 0.18:
        add("R01", f"СКО трансформации превышает 0.18 м: {rms_error_m:.3f}")

    for item in anomalies:
        aid = item.get("id")
        length_m = float(item.get("length_m", 0.0))
        amplitude = float(item.get("amplitude_nt", 0.0))
        risk_class = str(item.get("risk_class", "LOW"))
        probability = float(item.get("probability", 0.0))
        depth_diff = float(item.get("depth_diff_m", 0.0))
        depth_m = float(item.get("depth_m", 0.0))
        blind = bool(item.get("in_blind_zone", False))
        risk_index = float(item.get("risk_index", 0.0))
        utility_type = str(item.get("utility_type", "unknown"))
        recommendation = str(item.get("recommendation", ""))
        life_years = item.get("remaining_life_years")
        has_utility = bool(item.get("has_utility", False))

        if length_m <= 10.0:
            add("R02", f"Аномалия #{aid}: длина <= 10 м ({length_m:.2f})")
        if amplitude < 30.0:
            add("R03", f"Аномалия #{aid}: амплитуда < 30 нТл ({amplitude:.2f})")
        if risk_class == "CRITICAL" and abs(probability - 0.89) > 1e-6:
            add("R04", f"Аномалия #{aid}: CRITICAL должна иметь 0.89")
        if risk_class == "HIGH" and abs(probability - 0.62) > 1e-6:
            add("R05", f"Аномалия #{aid}: HIGH должна иметь 0.62")
        if risk_class == "LOW" and abs(probability - 0.12) > 1e-6:
            add("R06", f"Аномалия #{aid}: LOW должна иметь 0.12")
        if has_utility and depth_diff > 0.0 and depth_diff <= 0.5 and risk_class == "HIGH":
            add("R07", f"Аномалия #{aid}: HIGH при рассогласовании <=0.5 м")
        if blind and amplitude <= 50.0:
            add("R08", f"Аномалия #{aid}: blind-zone при амплитуде <=50 нТл")
        if risk_index > 0.7 and "Шурф" not in recommendation and depth_m <= 3.0:
            add("R09", f"Аномалия #{aid}: при R>0.7 и глубине<=3м нужен шурф")
        if 0.4 <= risk_index <= 0.7 and "ЛОМА-5" not in recommendation and depth_m <= 5.0:
            add("R10", f"Аномалия #{aid}: при 0.4<=R<=0.7 и глубине<=5м нужен ЛОМА-5")
        if utility_type == "sewage" and "Видеозондирование" not in recommendation:
            add("R11", f"Аномалия #{aid}: для канализации требуется видеозондирование")
        if life_years is not None and float(life_years) < 5.0 and not item.get("corrosion_warning"):
            add("R12", f"Аномалия #{aid}: при T_ост<5 нет обязательного предупреждения")

    return {"issues": issues, "issues_by_rule": by_rule, "issues_count": len(issues)}

