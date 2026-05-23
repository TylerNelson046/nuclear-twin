"""Predefined Helion FRC pulse scenarios for Scenario Mode.

Week 7 deliverable — Helion Scenarios + Parameter Sweep.

Three named scenarios covering the sub-ignition, near-ignition, and
clearly-sub-ignition regions of parameter space. Each wraps a validated
HelionParameters instance with UI-visible labels and descriptions, following
the same pattern as orchestrator/scenarios.py for PWR.

Architecture: orchestrator layer — imports from physics but not from ui/.
"""

from __future__ import annotations

from dataclasses import dataclass

from physics.helion.parameters import HelionParameters


@dataclass(frozen=True)
class HelionScenario:
    """Named Helion FRC pulse scenario."""

    scenario_id: str
    label: str
    description: str
    parameters: HelionParameters
    P_heat_w: float


def available_helion_scenarios() -> dict[str, HelionScenario]:
    """Return all predefined Phase 2 Helion pulse scenarios keyed by id."""
    return {
        scenario.scenario_id: scenario
        for scenario in (
            _baseline_pulse(),
            _high_compression(),
            _sub_ignition_diagnostic(),
        )
    }


def get_helion_scenario(scenario_id: str) -> HelionScenario:
    """Look up a Helion scenario by id, raising a plain ValueError if absent."""
    scenarios = available_helion_scenarios()
    try:
        return scenarios[scenario_id]
    except KeyError as exc:
        valid = ", ".join(sorted(scenarios))
        raise ValueError(
            f"Unknown Helion scenario '{scenario_id}'. Choose one of: {valid}."
        ) from exc


def helion_scenario_options() -> list[dict[str, str]]:
    """Return Dash dropdown options for the Helion Scenario Mode selector."""
    return [
        {"label": s.label, "value": s.scenario_id}
        for s in available_helion_scenarios().values()
    ]


def _baseline_pulse() -> HelionScenario:
    """Nominal Helion FRC operating point — sub-ignition demonstration."""
    return HelionScenario(
        scenario_id="baseline_pulse",
        label="Baseline Pulse",
        description=(
            "Nominal Helion FRC operating point: compression ratio Rᶜ = 10, "
            "pre-compression density n = 1×10²¹ m⁻³, initial temperature 20 keV. "
            "Post-compression T ≈ 93 keV. Demonstrates sub-ignition pulse dynamics "
            "with Bremsstrahlung and conduction losses dominating over fusion yield."
        ),
        parameters=HelionParameters(
            ion_temperature_kev=20.0,
            plasma_density_m3=1.0e21,
            compression_ratio=10.0,
            magnetic_field_t=5.0,
            plasma_volume_m3=1.0,
            pulse_duration_s=1.0e-5,
        ),
        P_heat_w=0.0,
    )


def _high_compression() -> HelionScenario:
    """High-compression scenario pushing toward the ignition boundary."""
    return HelionScenario(
        scenario_id="high_compression",
        label="High Compression — Ignition Approach",
        description=(
            "Aggressive compression ratio Rᶜ = 500 with elevated density "
            "n = 1×10²² m⁻³ and B = 10 T. Post-compression T ≈ 63 keV and "
            "n_final ≈ 5×10²⁴ m⁻³. Fusion power is substantial but confinement "
            "losses still dominate; Q ~ 0.09. Shows the approach toward ignition."
        ),
        parameters=HelionParameters(
            ion_temperature_kev=1.0,
            plasma_density_m3=1.0e22,
            compression_ratio=500.0,
            magnetic_field_t=10.0,
            plasma_volume_m3=0.5,
            pulse_duration_s=1.0e-5,
        ),
        P_heat_w=0.0,
    )


def _sub_ignition_diagnostic() -> HelionScenario:
    """Very low density diagnostic pulse — clearly sub-ignition regime."""
    return HelionScenario(
        scenario_id="sub_ignition_diagnostic",
        label="Sub-Ignition Diagnostic",
        description=(
            "Low pre-compression density n = 5×10¹⁹ m⁻³ with moderate compression "
            "Rᶜ = 5 and B = 3 T. Post-compression T ≈ 29 keV. Fusion power is "
            "negligible compared to loss channels; Q < 10⁻⁶. Demonstrates the "
            "deep sub-ignition regime and validates loss channel physics."
        ),
        parameters=HelionParameters(
            ion_temperature_kev=10.0,
            plasma_density_m3=5.0e19,
            compression_ratio=5.0,
            magnetic_field_t=3.0,
            plasma_volume_m3=2.0,
            pulse_duration_s=1.0e-5,
        ),
        P_heat_w=0.0,
    )
