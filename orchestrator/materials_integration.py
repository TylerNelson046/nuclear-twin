"""Thin integration adapter for the materials aging module.

Each reactor's runner calls one of these helpers once per simulation step
to advance the material state and snapshot the remaining-life projection.
Keeping the glue here avoids cluttering the long sim_runner / msr_runner /
helion_runner modules.

All return values are JSON-serializable (plain dicts of floats).
"""

from __future__ import annotations

import math
from typing import Any

from physics.materials import (
    PWRMaterialState,
    MSRMaterialState,
    HelionMaterialState,
    fresh_pwr_material_state,
    fresh_msr_material_state,
    fresh_helion_material_state,
    step_pwr_aging,
    step_msr_aging,
    step_helion_pulse_aging,
    pwr_remaining_life,
    msr_remaining_life,
    helion_remaining_life,
)


MATERIALS_HISTORY_MAX = 240  # keep ~240 hourly samples for the timeline view


# ============================================================
# PWR
# ============================================================
def initial_pwr_materials_payload() -> dict[str, Any]:
    """Fresh material state + empty history for a new PWR session."""
    state = fresh_pwr_material_state()
    return {
        "state": state.to_dict(),
        "history": {
            "time_h": [],
            "rpv_drtndt_k": [],
            "cladding_oxide_um": [],
            "control_rod_b10": [],
            "fuel_burnup": [],
        },
        "remaining_life_years": {},
    }


def advance_pwr_materials(
    payload: dict[str, Any] | None,
    dt_s: float,
    power_mw: float,
    t_fuel_k: float,
    t_coolant_k: float,
    rod_inserted_fraction: float,
    p_nominal_mw: float = 3000.0,
) -> dict[str, Any]:
    """Advance materials by ``dt_s``, snapshot remaining life, trim history."""
    if not payload:
        payload = initial_pwr_materials_payload()
    state = PWRMaterialState.from_dict(payload["state"])
    new_state = step_pwr_aging(
        state, dt_s, power_mw, t_fuel_k, t_coolant_k, rod_inserted_fraction, p_nominal_mw
    )
    life = pwr_remaining_life(
        new_state, power_mw, t_fuel_k, t_coolant_k, rod_inserted_fraction, p_nominal_mw
    )

    history = payload.get("history") or initial_pwr_materials_payload()["history"]
    history.setdefault("time_h", []).append(new_state.operating_hours)
    history.setdefault("rpv_drtndt_k", []).append(new_state.rpv_drtndt_k)
    history.setdefault("cladding_oxide_um", []).append(new_state.cladding_oxide_thickness_um)
    history.setdefault("control_rod_b10", []).append(new_state.control_rod_b10_depletion_frac)
    history.setdefault("fuel_burnup", []).append(new_state.fuel_burnup_gwd_per_mtu)
    history = _trim_history(history, MATERIALS_HISTORY_MAX)

    return {
        "state": new_state.to_dict(),
        "history": history,
        "remaining_life_years": _finite_dict(life),
    }


# ============================================================
# MSR
# ============================================================
def initial_msr_materials_payload() -> dict[str, Any]:
    state = fresh_msr_material_state()
    return {
        "state": state.to_dict(),
        "history": {
            "time_h": [],
            "corrosion_um": [],
            "tellurium_um": [],
            "tritium_g": [],
        },
        "remaining_life_years": {},
    }


def advance_msr_materials(
    payload: dict[str, Any] | None,
    dt_s: float,
    power_mw: float,
    t_salt_k: float,
    p_nominal_mw: float = 500.0,
) -> dict[str, Any]:
    if not payload:
        payload = initial_msr_materials_payload()
    state = MSRMaterialState.from_dict(payload["state"])
    new_state = step_msr_aging(state, dt_s, power_mw, t_salt_k, p_nominal_mw)
    life = msr_remaining_life(new_state, power_mw, t_salt_k, p_nominal_mw)

    history = payload.get("history") or initial_msr_materials_payload()["history"]
    history.setdefault("time_h", []).append(new_state.operating_hours)
    history.setdefault("corrosion_um", []).append(new_state.hastelloy_n_corrosion_um)
    history.setdefault("tellurium_um", []).append(new_state.tellurium_attack_depth_um)
    history.setdefault("tritium_g", []).append(new_state.tritium_inventory_g)
    history = _trim_history(history, MATERIALS_HISTORY_MAX)

    return {
        "state": new_state.to_dict(),
        "history": history,
        "remaining_life_years": _finite_dict(life),
    }


# ============================================================
# Helion
# ============================================================
def initial_helion_materials_payload() -> dict[str, Any]:
    state = fresh_helion_material_state()
    # Pre-fill remaining_life_years so the operator-view always renders the
    # full set of life cards (each shows ∞ until the first pulse fires).
    placeholder_life = {
        "first_wall_dpa_years": None,
        "thermal_cycle_years": None,
        "coil_fatigue_years": None,
        "capacitor_cycle_years": None,
        "insulator_dose_years": None,
    }
    return {
        "state": state.to_dict(),
        "history": {
            "pulses": [],
            "first_wall_dpa": [],
            "coil_fatigue": [],
            "capacitor_cycles": [],
        },
        "remaining_life_years": placeholder_life,
    }


def advance_helion_materials(
    payload: dict[str, Any] | None,
    pulse_fusion_energy_j: float,
    coil_delta_t_k: float,
    n_dhe3_reactions: float,
    pulses_per_day: float = 86400.0,
) -> dict[str, Any]:
    if not payload:
        payload = initial_helion_materials_payload()
    state = HelionMaterialState.from_dict(payload["state"])
    new_state = step_helion_pulse_aging(
        state, pulse_fusion_energy_j, coil_delta_t_k, n_dhe3_reactions
    )
    life = helion_remaining_life(
        new_state, pulse_fusion_energy_j, coil_delta_t_k, n_dhe3_reactions, pulses_per_day
    )

    history = payload.get("history") or initial_helion_materials_payload()["history"]
    history.setdefault("pulses", []).append(new_state.pulses_fired)
    history.setdefault("first_wall_dpa", []).append(new_state.first_wall_dpa)
    history.setdefault("coil_fatigue", []).append(new_state.coil_thermal_fatigue_index)
    history.setdefault("capacitor_cycles", []).append(new_state.capacitor_charge_cycles)
    history = _trim_history(history, MATERIALS_HISTORY_MAX)

    return {
        "state": new_state.to_dict(),
        "history": history,
        "remaining_life_years": _finite_dict(life),
    }


# ============================================================
# helpers
# ============================================================
def _trim_history(history: dict[str, list], max_points: int) -> dict[str, list]:
    """Trim each list in-place to the last ``max_points`` values."""
    out = {}
    for k, v in history.items():
        if len(v) > max_points:
            out[k] = v[-max_points:]
        else:
            out[k] = v
    return out


def _finite_dict(d: dict[str, float]) -> dict[str, float | None]:
    """JSON cannot represent ``math.inf`` portably; map to ``None`` instead."""
    return {k: (None if not math.isfinite(v) else float(v)) for k, v in d.items()}
