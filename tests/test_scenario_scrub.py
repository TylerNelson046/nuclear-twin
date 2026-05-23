"""Tests for scenario timeline scrub utilities."""

from utils.scenario_scrub import (
    clamp_scrub_index,
    clamp_pulse_index,
    history_length,
    pulse_length,
    slice_pulse_for_scrub,
    slice_session_for_scrub,
    scrubber_panel_style,
)


def _sample_pwr_session() -> dict:
    return {
        "time_s": 20.0,
        "scenario": {"completed": True},
        "history": {
            "time_s": [0.0, 10.0, 20.0],
            "power_mw": [100.0, 110.0, 105.0],
            "power_normalized": [1.0, 1.1, 1.05],
            "fuel_temperature_k": [900.0, 910.0, 905.0],
            "coolant_temperature_k": [580.0, 585.0, 582.0],
            "total_reactivity_pcm": [0.0, 50.0, 10.0],
            "xenon_reactivity_pcm": [-100.0, -120.0, -110.0],
        },
        "metrics": {"power_mw": 105.0},
    }


def test_history_length_and_clamp() -> None:
    session = _sample_pwr_session()
    assert history_length(session) == 3
    assert clamp_scrub_index(session, 99) == 2
    assert clamp_scrub_index(session, None) == 2


def test_slice_session_for_scrub_pwr() -> None:
    session = _sample_pwr_session()
    view = slice_session_for_scrub(session, 1, reactor="pwr")
    assert len(view["history"]["time_s"]) == 2
    assert view["time_s"] == 10.0
    assert view["metrics"]["power_mw"] == 110.0


def test_slice_pulse_for_scrub() -> None:
    pulse = {
        "success": True,
        "t": [0.0, 1e-6, 2e-6],
        "T_kev": [10.0, 20.0, 30.0],
        "P_fusion_w": [1.0, 2.0, 3.0],
    }
    sliced, idx = slice_pulse_for_scrub(pulse, 1)
    assert idx == 1
    assert len(sliced["t"]) == 2
    assert sliced["T_kev"][-1] == 20.0
    assert pulse_length(pulse) == 3
    assert clamp_pulse_index(pulse, None) == 2


def test_scrubber_panel_visible_when_scenario_complete() -> None:
    session = _sample_pwr_session()
    assert scrubber_panel_style(session)["display"] == "block"
    assert scrubber_panel_style({"scenario": {"completed": True}, "history": {"time_s": [0.0]}})[
        "display"
    ] == "none"
