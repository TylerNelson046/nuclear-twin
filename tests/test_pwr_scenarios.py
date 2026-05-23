"""Scenario Mode tests for predefined PWR transients."""

from __future__ import annotations

import pytest

from orchestrator.scenarios import available_pwr_scenarios, get_pwr_scenario
from orchestrator.sim_runner import run_pwr_control_step, run_pwr_scenario
from physics.pwr.thermal_hydraulics import T_IN_NOM


def test_all_required_pwr_scenarios_are_registered() -> None:
    scenarios = available_pwr_scenarios()

    assert set(scenarios) == {
        "rod_ejection",
        "boron_dilution",
        "xenon_peak_post_shutdown",
        "load_following",
    }


def test_rod_ejection_profile_inserts_500_pcm_over_0p1_seconds() -> None:
    scenario = get_pwr_scenario("rod_ejection")

    assert scenario.controls_at(0.0).rod_reactivity_pcm == pytest.approx(0.0)
    assert scenario.controls_at(0.05).rod_reactivity_pcm == pytest.approx(250.0)
    assert scenario.controls_at(0.1).rod_reactivity_pcm == pytest.approx(500.0)
    assert scenario.controls_at(10.0).rod_reactivity_pcm == pytest.approx(500.0)


def test_boron_dilution_profile_reduces_soluble_boron_linearly() -> None:
    scenario = get_pwr_scenario("boron_dilution")

    assert scenario.controls_at(0.0).boron_ppm == pytest.approx(1200.0)
    assert scenario.controls_at(2.0 * 3600.0).boron_ppm == pytest.approx(700.0)
    assert scenario.controls_at(4.0 * 3600.0).boron_ppm == pytest.approx(200.0)
    assert scenario.controls_at(6.0 * 3600.0).boron_ppm == pytest.approx(200.0)


def test_xenon_shutdown_profile_applies_full_negative_rod_worth() -> None:
    scenario = get_pwr_scenario("xenon_peak_post_shutdown")

    controls = scenario.controls_at(1.0)

    assert controls.rod_reactivity_pcm == pytest.approx(-1000.0)
    assert controls.boron_ppm == pytest.approx(0.0)
    assert controls.inlet_temperature_k == pytest.approx(T_IN_NOM)


def test_load_following_profile_ramps_rods_down_and_back_up() -> None:
    scenario = get_pwr_scenario("load_following")

    assert scenario.duration_s == pytest.approx(24.0 * 3600.0)
    assert scenario.controls_at(0.0).rod_reactivity_pcm == pytest.approx(0.0)
    assert scenario.controls_at(10.0 * 3600.0).rod_reactivity_pcm == pytest.approx(-90.0)
    assert scenario.controls_at(16.0 * 3600.0).rod_reactivity_pcm == pytest.approx(-90.0)
    assert scenario.controls_at(20.0 * 3600.0).rod_reactivity_pcm == pytest.approx(0.0)


def test_scenario_runner_resets_session_and_emits_events() -> None:
    existing_session = run_pwr_control_step(
        None,
        {
            "rod_reactivity_pcm": 100.0,
            "boron_ppm": 0.0,
            "coolant_flow_fraction": 1.0,
            "inlet_temperature_k": T_IN_NOM,
        },
    )
    assert existing_session["time_s"] > 0.0

    result = run_pwr_scenario("boron_dilution")

    assert result["status"]["ok"], result["status"]["message"]
    assert result["scenario"]["id"] == "boron_dilution"
    assert result["history"]["time_s"][0] == pytest.approx(0.0)
    assert result["time_s"] == pytest.approx(get_pwr_scenario("boron_dilution").duration_s)
    assert result["history"]["boron_reactivity_pcm"][0] < result["history"]["boron_reactivity_pcm"][-1]
    assert any(event["code"] == "scenario_started" for event in result["events"])
    assert any(event["code"].startswith("scenario_") for event in result["events"])


def test_unknown_scenario_returns_plain_language_error() -> None:
    result = run_pwr_scenario("missing")

    assert not result["status"]["ok"]
    assert "Unknown PWR scenario" in result["status"]["message"]
