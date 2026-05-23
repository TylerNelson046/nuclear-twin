"""Predefined MSR transient scenarios for Scenario Mode.

Week 11 deliverable — MSR Scenarios + UI.

Two named transient scenarios (SPEC §8 Week 11):
  pump_trip       — sudden salt flow reduction; T_salt rises; negative feedback reduces power
  load_following  — external reactivity ramp down then up; demonstrates inherent load-following

β_eff,flow vs salt velocity is not a transient — it is computed algebraically by
run_msr_beta_flow_sweep() in orchestrator/msr_runner.py.

Architecture: orchestrator layer — imports from physics/ but never from ui/.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from physics.msr.engine import MSRControls
from physics.msr.thermal import TAU_CORE_NOM

ControlsFn = Callable[[float, np.ndarray], MSRControls]


@dataclass(frozen=True)
class MSRScenario:
    """Named MSR transient definition for Scenario Mode."""

    scenario_id: str
    label: str
    description: str
    duration_s: float
    n_output_points: int
    tau_core_s: float
    tau_loop_s: float
    profile: ControlsFn

    def controls_at(self, time_s: float, state: np.ndarray) -> MSRControls:
        """Return scenario-imposed boundary conditions at simulation time."""
        return self.profile(time_s, state)


def available_msr_scenarios() -> dict[str, MSRScenario]:
    """Return all predefined Phase 3 MSR transient scenarios keyed by id."""
    return {
        s.scenario_id: s
        for s in (_pump_trip(), _load_following())
    }


def get_msr_scenario(scenario_id: str) -> MSRScenario:
    """Look up an MSR scenario by id; raise a plain ValueError if absent."""
    scenarios = available_msr_scenarios()
    try:
        return scenarios[scenario_id]
    except KeyError as exc:
        valid = ", ".join(sorted(scenarios))
        raise ValueError(
            f"Unknown MSR scenario '{scenario_id}'. Choose one of: {valid}."
        ) from exc


def msr_scenario_options() -> list[dict[str, str]]:
    """Return Dash dropdown options for the MSR Scenario Mode selector."""
    return [
        {"label": s.label, "value": s.scenario_id}
        for s in available_msr_scenarios().values()
    ]


# ---------------------------------------------------------------------------
# Scenario definitions
# ---------------------------------------------------------------------------


def _pump_trip() -> MSRScenario:
    """Sudden salt pump coastdown: flow drops from 100% to 20% over 5 s.

    Reduced heat removal causes T_salt to rise, which generates negative reactivity
    through the combined Doppler + moderator salt temperature coefficient. Power
    settles at a lower self-regulated equilibrium — a direct demonstration of the
    MSR's inherent load-following behavior without operator rod action.

    Physics validated against: SPEC §7.3 'pump trip transient' checkpoint.
    """
    ramp_end_s = 5.0
    final_flow = 0.20

    def _profile(time_s: float, _state: np.ndarray) -> MSRControls:
        if time_s <= ramp_end_s:
            fraction = 1.0 - (1.0 - final_flow) * min(time_s / ramp_end_s, 1.0)
        else:
            fraction = final_flow
        return MSRControls(external_reactivity_pcm=0.0, salt_flow_fraction=fraction)

    return MSRScenario(
        scenario_id="pump_trip",
        label="Pump Trip",
        description=(
            "Salt pump trip: flow rate drops from 100% to 20% over 5 s. "
            "Reduced advective heat removal causes T_salt to rise, generating "
            "negative reactivity via the salt temperature coefficient. Power "
            "decreases to a new self-regulated equilibrium without rod action. "
            "Demonstrates MSR inherent load-following and thermal self-regulation."
        ),
        duration_s=300.0,
        n_output_points=600,
        tau_core_s=TAU_CORE_NOM,
        tau_loop_s=20.0,
        profile=_profile,
    )


def _load_following() -> MSRScenario:
    """Smooth load following via external reactivity ramp (rod withdrawal/insertion).

    External reactivity is reduced by 50 pcm over 5 minutes (partial rod insertion),
    held at reduced power for 30 minutes, then restored over 5 minutes. The salt
    temperature coefficient assists the ramp-down by generating additional negative
    feedback as T_salt rises with the rod insertion.

    Illustrates how MSR load following differs from PWR: the combined fuel+coolant
    temperature feedback adds a faster self-limiting component to rod control.
    """
    ramp_down_start_s = 300.0
    ramp_down_end_s   = 600.0
    hold_low_end_s    = 2400.0
    ramp_up_end_s     = 2700.0
    duration_s        = 3000.0
    low_reactivity_pcm = -50.0

    def _profile(time_s: float, _state: np.ndarray) -> MSRControls:
        if time_s < ramp_down_start_s:
            rod_pcm = 0.0
        elif time_s < ramp_down_end_s:
            frac = (time_s - ramp_down_start_s) / (ramp_down_end_s - ramp_down_start_s)
            rod_pcm = frac * low_reactivity_pcm
        elif time_s < hold_low_end_s:
            rod_pcm = low_reactivity_pcm
        elif time_s < ramp_up_end_s:
            frac = (time_s - hold_low_end_s) / (ramp_up_end_s - hold_low_end_s)
            rod_pcm = low_reactivity_pcm * (1.0 - frac)
        else:
            rod_pcm = 0.0
        return MSRControls(external_reactivity_pcm=rod_pcm, salt_flow_fraction=1.0)

    return MSRScenario(
        scenario_id="load_following",
        label="Load Following",
        description=(
            "External reactivity (rod equivalent) decreases −50 pcm over 5 minutes, "
            "holds for 30 minutes at reduced power, then ramps back to zero. "
            "Salt temperature feedback contributes inherent load-following. "
            "Compare MSR transient speed against the PWR load-following scenario."
        ),
        duration_s=duration_s,
        n_output_points=800,
        tau_core_s=TAU_CORE_NOM,
        tau_loop_s=20.0,
        profile=_profile,
    )
