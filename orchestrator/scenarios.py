"""Predefined PWR transient scenario profiles for Scenario Mode.

The scenario layer belongs in the orchestrator, not the physics engine: it
translates named operator transients into time-dependent PWRControls while the
physics modules remain pure ODE/equation implementations.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from physics.pwr.engine import PWRControls
from physics.pwr.parameters import PWRParameters
from physics.pwr.thermal_hydraulics import T_IN_NOM

ControlsProfile = Callable[[float, np.ndarray | None], PWRControls]


@dataclass(frozen=True)
class PWRScenario:
    """Named transient definition for a PWR scenario run."""

    scenario_id: str
    label: str
    description: str
    duration_s: float
    n_output_points: int
    initial_parameters: PWRParameters
    max_step_s: float
    profile: ControlsProfile

    def controls_at(self, time_s: float, state: np.ndarray | None = None) -> PWRControls:
        """Return scenario-imposed boundary conditions at simulation time."""
        return self.profile(time_s, state)


def available_pwr_scenarios() -> dict[str, PWRScenario]:
    """Return all predefined Phase 1 PWR transient scenarios."""
    return {
        scenario.scenario_id: scenario
        for scenario in (
            _rod_ejection(),
            _boron_dilution(),
            _xenon_peak_post_shutdown(),
            _load_following(),
        )
    }


def get_pwr_scenario(scenario_id: str) -> PWRScenario:
    """Look up a PWR scenario by id, raising a plain ValueError if absent."""
    scenarios = available_pwr_scenarios()
    try:
        return scenarios[scenario_id]
    except KeyError as exc:
        valid = ", ".join(sorted(scenarios))
        raise ValueError(f"Unknown PWR scenario '{scenario_id}'. Choose one of: {valid}.") from exc


def pwr_scenario_options() -> list[dict[str, str]]:
    """Return Dash dropdown options for Scenario Mode."""
    return [
        {"label": scenario.label, "value": scenario.scenario_id}
        for scenario in available_pwr_scenarios().values()
    ]


def _rod_ejection() -> PWRScenario:
    """Rapid +500 pcm rod ejection over 0.1 s, then held inserted."""
    insertion_pcm = 500.0
    ramp_s = 0.1

    def _profile(time_s: float, _state: np.ndarray | None) -> PWRControls:
        fraction = min(max(time_s, 0.0) / ramp_s, 1.0)
        return PWRControls(
            rod_reactivity_pcm=insertion_pcm * fraction,
            boron_ppm=0.0,
            coolant_flow_fraction=1.0,
            inlet_temperature_k=T_IN_NOM,
        )

    return PWRScenario(
        scenario_id="rod_ejection",
        label="Rod Ejection",
        description=(
            "Rapid +500 pcm positive reactivity insertion over 0.1 s, "
            "followed by Doppler-driven self-limitation."
        ),
        duration_s=120.0,
        n_output_points=600,
        initial_parameters=PWRParameters(
            rod_reactivity_pcm=0.0,
            boron_ppm=0.0,
            coolant_flow_fraction=1.0,
            inlet_temperature_k=T_IN_NOM,
        ),
        max_step_s=0.02,
        profile=_profile,
    )


def _boron_dilution() -> PWRScenario:
    """Slow soluble-boron reduction that raises reactivity over hours."""
    start_boron_ppm = 1200.0
    end_boron_ppm = 200.0
    dilution_window_s = 4.0 * 3600.0
    duration_s = 6.0 * 3600.0

    def _profile(time_s: float, _state: np.ndarray | None) -> PWRControls:
        fraction = min(max(time_s, 0.0) / dilution_window_s, 1.0)
        boron_ppm = start_boron_ppm + fraction * (end_boron_ppm - start_boron_ppm)
        return PWRControls(
            rod_reactivity_pcm=0.0,
            boron_ppm=boron_ppm,
            coolant_flow_fraction=1.0,
            inlet_temperature_k=T_IN_NOM,
        )

    return PWRScenario(
        scenario_id="boron_dilution",
        label="Boron Dilution",
        description=(
            "Soluble boron decreases linearly from 1200 ppm to 200 ppm over "
            "4 hours, producing a gradual positive reactivity ramp."
        ),
        duration_s=duration_s,
        n_output_points=500,
        initial_parameters=PWRParameters(
            rod_reactivity_pcm=0.0,
            boron_ppm=start_boron_ppm,
            coolant_flow_fraction=1.0,
            inlet_temperature_k=T_IN_NOM,
        ),
        max_step_s=60.0,
        profile=_profile,
    )


def _xenon_peak_post_shutdown() -> PWRScenario:
    """Manual scram with low-flux iodine decay into xenon over hours."""
    scram_reactivity_pcm = -1000.0

    def _profile(_time_s: float, _state: np.ndarray | None) -> PWRControls:
        return PWRControls(
            rod_reactivity_pcm=scram_reactivity_pcm,
            boron_ppm=0.0,
            coolant_flow_fraction=1.0,
            inlet_temperature_k=T_IN_NOM,
        )

    return PWRScenario(
        scenario_id="xenon_peak_post_shutdown",
        label="Xenon Peak Post-Shutdown",
        description=(
            "Immediate manual scram using full negative rod worth, followed by "
            "I-135 decay into Xe-135 over the 6-11 hour iodine pit window."
        ),
        duration_s=18.0 * 3600.0,
        n_output_points=700,
        initial_parameters=PWRParameters(
            rod_reactivity_pcm=0.0,
            boron_ppm=0.0,
            coolant_flow_fraction=1.0,
            inlet_temperature_k=T_IN_NOM,
        ),
        max_step_s=30.0,
        profile=_profile,
    )


def _load_following() -> PWRScenario:
    """Open-loop rod schedule approximating a 24-hour daily demand cycle."""
    ramp_down_start_s = 6.0 * 3600.0
    ramp_down_end_s = 10.0 * 3600.0
    hold_low_end_s = 16.0 * 3600.0
    ramp_up_end_s = 20.0 * 3600.0
    duration_s = 24.0 * 3600.0
    low_power_rod_pcm = -90.0

    def _profile(time_s: float, _state: np.ndarray | None) -> PWRControls:
        if time_s < ramp_down_start_s:
            rod_pcm = 0.0
        elif time_s < ramp_down_end_s:
            fraction = (time_s - ramp_down_start_s) / (ramp_down_end_s - ramp_down_start_s)
            rod_pcm = fraction * low_power_rod_pcm
        elif time_s < hold_low_end_s:
            rod_pcm = low_power_rod_pcm
        elif time_s < ramp_up_end_s:
            fraction = (time_s - hold_low_end_s) / (ramp_up_end_s - hold_low_end_s)
            rod_pcm = low_power_rod_pcm * (1.0 - fraction)
        else:
            rod_pcm = 0.0

        return PWRControls(
            rod_reactivity_pcm=rod_pcm,
            boron_ppm=0.0,
            coolant_flow_fraction=1.0,
            inlet_temperature_k=T_IN_NOM,
        )

    return PWRScenario(
        scenario_id="load_following",
        label="Load Following",
        description=(
            "Predefined 24-hour rod schedule that follows a daily demand curve: "
            "100% power, down toward 50% through the day, then back toward 100%."
        ),
        duration_s=duration_s,
        n_output_points=1000,
        initial_parameters=PWRParameters(
            rod_reactivity_pcm=0.0,
            boron_ppm=0.0,
            coolant_flow_fraction=1.0,
            inlet_temperature_k=T_IN_NOM,
        ),
        max_step_s=60.0,
        profile=_profile,
    )
