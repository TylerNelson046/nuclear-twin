"""Tests for display-unit conversion helpers."""

from utils.display_units import (
    format_simulation_time,
    normalize_display_units,
    temperature_scalar_kelvin,
    temperature_values_kelvin,
    time_values_seconds,
)


def test_normalize_defaults() -> None:
    assert normalize_display_units(None) == {"temperature": "celsius", "time": "seconds"}


def test_temperature_celsius_conversion() -> None:
    units = {"temperature": "celsius", "time": "seconds"}
    assert temperature_values_kelvin([273.15], units) == [0.0]
    assert temperature_scalar_kelvin(373.15, units) == 100.0


def test_time_minutes() -> None:
    units = {"temperature": "kelvin", "time": "minutes"}
    assert time_values_seconds([60.0], units) == [1.0]
    assert format_simulation_time(3600.0, units) == "t=60.0 min"
