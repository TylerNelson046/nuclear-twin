"""Tests for the Helion FRC orchestrator runner (orchestrator/helion_runner.py).

Covers: run_helion_pulse, run_helion_scenario, run_helion_sweep
"""

from __future__ import annotations

import math

import pytest

from orchestrator.helion_runner import run_helion_pulse, run_helion_scenario, run_helion_sweep


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _default_raw() -> dict:
    """Valid raw inputs for a baseline custom pulse (density in m⁻³, pulse in s)."""
    return {
        "ion_temperature_kev": 20.0,
        "plasma_density_m3": 1.0e21,
        "compression_ratio": 10.0,
        "magnetic_field_t": 5.0,
        "plasma_volume_m3": 1.0,
        "pulse_duration_s": 1.0e-5,
    }


# ---------------------------------------------------------------------------
# run_helion_pulse — success path
# ---------------------------------------------------------------------------

class TestRunHelionPulse:
    def test_returns_success_for_valid_inputs(self):
        result = run_helion_pulse(_default_raw())
        assert result["success"] is True

    def test_result_contains_required_keys(self):
        result = run_helion_pulse(_default_raw())
        for key in ("t", "W_J", "T_kev", "P_fusion_w", "P_brem_w", "P_cond_w",
                    "Q", "W_initial_J", "parameters", "message"):
            assert key in result, f"Missing key: {key}"

    def test_arrays_are_lists_json_serializable(self):
        result = run_helion_pulse(_default_raw())
        assert isinstance(result["t"], list)
        assert isinstance(result["W_J"], list)
        assert isinstance(result["T_kev"], list)

    def test_q_is_float(self):
        result = run_helion_pulse(_default_raw())
        assert isinstance(result["Q"], float)

    def test_q_non_negative(self):
        result = run_helion_pulse(_default_raw())
        assert result["Q"] >= 0.0

    def test_parameters_echoed_back(self):
        result = run_helion_pulse(_default_raw())
        assert result["parameters"]["ion_temperature_kev"] == pytest.approx(20.0)
        assert result["parameters"]["compression_ratio"] == pytest.approx(10.0)

    def test_array_lengths_consistent(self):
        result = run_helion_pulse(_default_raw())
        n = len(result["t"])
        assert len(result["T_kev"]) == n
        assert len(result["P_fusion_w"]) == n
        assert len(result["P_brem_w"]) == n
        assert len(result["P_cond_w"]) == n
        assert len(result["W_J"]) == n

    def test_w_initial_positive(self):
        result = run_helion_pulse(_default_raw())
        assert result["W_initial_J"] > 0.0

    def test_density_unit_passthrough(self):
        raw = _default_raw()
        raw["plasma_density_m3"] = 5.0e21
        result = run_helion_pulse(raw)
        assert result["parameters"]["plasma_density_m3"] == pytest.approx(5.0e21)

    def test_high_compression_gives_higher_q(self):
        low = run_helion_pulse({**_default_raw(), "compression_ratio": 10.0})
        high = run_helion_pulse({**_default_raw(), "compression_ratio": 100.0})
        assert high["Q"] > low["Q"]

    def test_higher_density_gives_higher_q(self):
        low = run_helion_pulse({**_default_raw(), "plasma_density_m3": 1e21})
        high = run_helion_pulse({**_default_raw(), "plasma_density_m3": 1e22})
        assert high["Q"] > low["Q"]


class TestRunHelionPulseErrors:
    def test_invalid_temperature_below_bosch_hale_returns_failure(self):
        raw = _default_raw()
        raw["ion_temperature_kev"] = 0.1   # below min 0.5
        result = run_helion_pulse(raw)
        assert result["success"] is False
        assert "message" in result

    def test_invalid_temperature_above_bosch_hale_returns_failure(self):
        raw = _default_raw()
        raw["ion_temperature_kev"] = 200.0   # above max 190
        result = run_helion_pulse(raw)
        assert result["success"] is False

    def test_invalid_density_too_low_returns_failure(self):
        raw = _default_raw()
        raw["plasma_density_m3"] = 1e18     # below min 1e19
        result = run_helion_pulse(raw)
        assert result["success"] is False

    def test_invalid_compression_below_1_returns_failure(self):
        raw = _default_raw()
        raw["compression_ratio"] = 0.5
        result = run_helion_pulse(raw)
        assert result["success"] is False

    def test_none_value_returns_failure(self):
        raw = _default_raw()
        raw["ion_temperature_kev"] = None
        result = run_helion_pulse(raw)
        assert result["success"] is False


# ---------------------------------------------------------------------------
# run_helion_scenario
# ---------------------------------------------------------------------------

class TestRunHelionScenario:
    def test_baseline_pulse_succeeds(self):
        result = run_helion_scenario("baseline_pulse")
        assert result["success"] is True

    def test_high_compression_succeeds(self):
        result = run_helion_scenario("high_compression")
        assert result["success"] is True

    def test_sub_ignition_diagnostic_succeeds(self):
        result = run_helion_scenario("sub_ignition_diagnostic")
        assert result["success"] is True

    def test_scenario_adds_label_and_description(self):
        result = run_helion_scenario("baseline_pulse")
        assert "scenario_label" in result
        assert "scenario_description" in result
        assert len(result["scenario_label"]) > 0

    def test_unknown_scenario_returns_failure(self):
        result = run_helion_scenario("nonexistent_scenario")
        assert result["success"] is False
        assert "message" in result

    def test_baseline_q_less_than_1(self):
        result = run_helion_scenario("baseline_pulse")
        assert result["Q"] < 1.0

    def test_sub_ignition_q_very_small(self):
        result = run_helion_scenario("sub_ignition_diagnostic")
        assert result["Q"] < 0.01

    def test_scenario_result_has_arrays(self):
        result = run_helion_scenario("baseline_pulse")
        assert isinstance(result["t"], list)
        assert len(result["t"]) > 1


# ---------------------------------------------------------------------------
# run_helion_sweep
# ---------------------------------------------------------------------------

class TestRunHelionSweep:
    def _base_raw(self) -> dict:
        return _default_raw()

    def test_returns_success(self):
        result = run_helion_sweep(self._base_raw(), n_rc=5, n_density=5)
        assert result["success"] is True

    def test_q_map_shape_matches_grid(self):
        result = run_helion_sweep(self._base_raw(), n_rc=5, n_density=6)
        assert len(result["Q_map"]) == 5
        assert all(len(row) == 6 for row in result["Q_map"])

    def test_compression_ratios_length(self):
        result = run_helion_sweep(self._base_raw(), n_rc=7, n_density=4)
        assert len(result["compression_ratios"]) == 7

    def test_densities_length(self):
        result = run_helion_sweep(self._base_raw(), n_rc=4, n_density=8)
        assert len(result["densities_m3"]) == 8

    def test_ignition_mask_shape_matches_q_map(self):
        result = run_helion_sweep(self._base_raw(), n_rc=5, n_density=5)
        assert len(result["ignition_mask"]) == len(result["Q_map"])
        assert len(result["ignition_mask"][0]) == len(result["Q_map"][0])

    def test_ignition_mask_boolean_type(self):
        result = run_helion_sweep(self._base_raw(), n_rc=3, n_density=3)
        for row in result["ignition_mask"]:
            for val in row:
                assert isinstance(val, bool)

    def test_q_map_all_non_negative(self):
        result = run_helion_sweep(self._base_raw(), n_rc=5, n_density=5)
        for row in result["Q_map"]:
            for q in row:
                assert q >= 0.0

    def test_n_failed_key_present(self):
        result = run_helion_sweep(self._base_raw(), n_rc=5, n_density=5)
        assert "n_failed" in result
        assert isinstance(result["n_failed"], int)

    def test_axes_are_lists(self):
        result = run_helion_sweep(self._base_raw(), n_rc=4, n_density=4)
        assert isinstance(result["compression_ratios"], list)
        assert isinstance(result["densities_m3"], list)

    def test_ignition_region_exists_at_high_params(self):
        """Very high Rc and density should produce at least one Q > 1 grid point."""
        # Use T=1keV, B=10T, Rc goes up to 1000, n up to 1e23
        raw = {
            "ion_temperature_kev": 1.0,
            "plasma_density_m3": 1.0e21,
            "compression_ratio": 10.0,
            "magnetic_field_t": 10.0,
            "plasma_volume_m3": 1.0,
            "pulse_duration_s": 1.0e-5,
        }
        result = run_helion_sweep(raw, n_rc=10, n_density=10)
        any_ignited = any(val for row in result["ignition_mask"] for val in row)
        assert any_ignited, "Expected at least one Q > 1 in high-param sweep"

    def test_invalid_base_params_returns_failure(self):
        raw = _default_raw()
        raw["ion_temperature_kev"] = 999.0  # above Pydantic max 190
        result = run_helion_sweep(raw, n_rc=3, n_density=3)
        assert result["success"] is False
