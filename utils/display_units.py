"""Display-unit conversions for the Dash interface layer (SPEC §4.6.4).

Internal physics and orchestrator history always use SI (K, s). Charts and
metrics convert at render time according to user preferences.
"""

from __future__ import annotations

from typing import Any

KELVIN_TO_CELSIUS = 273.15

TEMPERATURE_CHOICES = ("celsius", "kelvin")
TIME_CHOICES = ("seconds", "minutes", "hours")


def normalize_display_units(raw: dict[str, Any] | None) -> dict[str, str]:
    """Return validated display-unit preferences."""
    raw = raw or {}
    temp = str(raw.get("temperature", "celsius")).lower()
    time_u = str(raw.get("time", "seconds")).lower()
    if temp not in TEMPERATURE_CHOICES:
        temp = "celsius"
    if time_u not in TIME_CHOICES:
        time_u = "seconds"
    return {"temperature": temp, "time": time_u}


def temperature_values_kelvin(values_k: list[float], units: dict[str, str] | None) -> list[float]:
    """Convert Kelvin history values for plotting."""
    prefs = normalize_display_units(units)
    if prefs["temperature"] == "kelvin":
        return [float(v) for v in values_k]
    return [float(v) - KELVIN_TO_CELSIUS for v in values_k]


def temperature_scalar_kelvin(value_k: Any, units: dict[str, str] | None) -> float | None:
    """Convert one Kelvin sample for metric tiles."""
    if value_k is None or value_k == "":
        return None
    try:
        k = float(value_k)
    except (TypeError, ValueError):
        return None
    prefs = normalize_display_units(units)
    if prefs["temperature"] == "kelvin":
        return k
    return k - KELVIN_TO_CELSIUS


def temperature_axis_label(units: dict[str, str] | None) -> str:
    prefs = normalize_display_units(units)
    return "Temperature (K)" if prefs["temperature"] == "kelvin" else "Temperature (°C)"


def temperature_unit_suffix(units: dict[str, str] | None) -> str:
    prefs = normalize_display_units(units)
    return "K" if prefs["temperature"] == "kelvin" else "°C"


def time_values_seconds(values_s: list[float], units: dict[str, str] | None) -> list[float]:
    """Convert simulation time (s) for plotting."""
    prefs = normalize_display_units(units)
    scale = _time_scale(prefs["time"])
    return [float(v) * scale for v in values_s]


def time_axis_label(units: dict[str, str] | None) -> str:
    prefs = normalize_display_units(units)
    if prefs["time"] == "minutes":
        return "Simulation time (min)"
    if prefs["time"] == "hours":
        return "Simulation time (h)"
    return "Simulation time (s)"


def format_simulation_time(time_s: Any, units: dict[str, str] | None) -> str:
    """Format simulated time for metric tiles and event log."""
    try:
        seconds = float(time_s)
    except (TypeError, ValueError):
        return "t=--"
    prefs = normalize_display_units(units)
    if prefs["time"] == "hours":
        return f"t={seconds / 3600.0:.2f} h"
    if prefs["time"] == "minutes":
        return f"t={seconds / 60.0:.1f} min"
    return f"t={seconds:.1f} s"


def helion_time_values_seconds(values_s: list[float], units: dict[str, str] | None) -> tuple[list[float], str]:
    """Helion pulses are short — offer seconds or microseconds on charts."""
    prefs = normalize_display_units(units)
    if prefs["time"] == "seconds":
        return [float(v) for v in values_s], "Time (s)"
    return [float(v) * 1e6 for v in values_s], "Time (μs)"


def _time_scale(time_mode: str) -> float:
    if time_mode == "minutes":
        return 1.0 / 60.0
    if time_mode == "hours":
        return 1.0 / 3600.0
    return 1.0
