from datetime import date


def build_irrigation_advice(
    crop: str,
    growth_stage: str | None,
    soil_type: str | None,
    last_irrigation_date: date | None,
    forecast_rain_probability: float | None = None,
) -> dict[str, object]:
    inputs = {
        "crop": crop.strip() or None,
        "growth_stage": (growth_stage or "").strip() or None,
        "soil_type": (soil_type or "").strip() or None,
        "last_irrigation_date": last_irrigation_date.isoformat() if last_irrigation_date else None,
        "forecast_rain_probability": forecast_rain_probability,
    }

    missing = [
        name
        for name, value in (
            ("crop stage", inputs["growth_stage"]),
            ("soil type", inputs["soil_type"]),
            ("last irrigation date", inputs["last_irrigation_date"]),
        )
        if value is None
    ]

    reasons = [
        "No locally validated irrigation schedule is configured for this crop and region.",
        "A forecast is not observed rainfall, and this app has no soil-moisture sensor.",
        "Inspect soil moisture in the root zone and consider crop condition and local extension advice.",
    ]
    if missing:
        reasons.append("Missing information: " + ", ".join(missing) + ".")
    if forecast_rain_probability is not None:
        reasons.append(
            f"Weather forecast rain probability: {forecast_rain_probability:.0f}% "
            "(forecast only; not a field measurement)."
        )

    return {
        "action": "Check field conditions and set a farmer-chosen irrigation plan if needed.",
        "reasoning": reasons,
        "uncertainty": "High: no validated local schedule, direct soil-moisture measurement, or observed field rainfall is available.",
        "inputs": inputs,
    }
