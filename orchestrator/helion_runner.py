"""Helion FRC simulation runner APIs — keeps Dash callbacks out of the physics layer.

Provides three entry points:
- run_helion_pulse()   — single pulse from raw UI inputs
- run_helion_scenario() — predefined named scenario
- run_helion_sweep()   — 2D ignition boundary sweep

All functions accept plain Python scalars/dicts, validate via Pydantic, and return
JSON-serializable dicts.  The UI never imports from physics/ directly.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from pydantic import ValidationError

from orchestrator.helion_scenarios import get_helion_scenario
from orchestrator.materials_integration import (
    advance_helion_materials,
    initial_helion_materials_payload,
)
from physics.helion.engine import integrate_helion_pulse
from physics.helion.losses import TAU_E_UNCERTAINTY_LOWER, TAU_E_UNCERTAINTY_UPPER
from physics.helion.parameters import HelionParameters
from physics.helion.sweep import ignition_boundary_sweep

# D-He3 reaction energy: 18.3 MeV = 2.93e-12 J. Used to back out the per-pulse
# reaction count from the integrated fusion energy for material-damage scaling.
_HELION_E_REACTION_J = 2.93e-12

logger = logging.getLogger(__name__)

# Default sweep grid (coarse — fast enough for on-demand UI call)
_SWEEP_RC_DEFAULT = np.logspace(0, 3, 20)   # Rc: 1 → 1000
_SWEEP_N_DEFAULT = np.logspace(19, 23, 20)  # n:  1e19 → 1e23 m⁻³


def run_helion_pulse(
    raw_inputs: dict[str, Any],
    *,
    materials_in: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run a single Helion FRC pulse from UI-supplied parameter values.

    If ``materials_in`` is provided, the returned dict includes an updated
    ``materials`` payload (state + history + remaining-life projections).
    Each call advances the machine by ONE pulse.

    Args:
        raw_inputs:   dict with keys matching HelionParameters fields plus
                      optional ``p_heat_w`` (float, default 0).
        materials_in: previous materials payload from a prior pulse, or
                      ``None`` to start fresh.

    Returns:
        JSON-serializable dict with keys:
        - success (bool)
        - message (str)
        - t (list[float])           — time array (s)
        - W_J (list[float])         — plasma energy (J)
        - T_kev (list[float])       — plasma temperature (keV)
        - P_fusion_w (list[float])  — fusion power (W)
        - P_brem_w (list[float])    — Bremsstrahlung loss (W)
        - P_cond_w (list[float])    — conduction loss (W)
        - Q (float)                 — gain factor
        - W_initial_J (float)       — initial plasma energy (J)
        - parameters (dict)         — validated parameter values
        - materials (dict)          — updated material state if materials_in given
    """
    try:
        p_heat_w = float(raw_inputs.pop("p_heat_w", 0.0))
        params = HelionParameters(**{k: float(v) for k, v in raw_inputs.items()})
    except (ValidationError, ValueError, TypeError) as exc:
        logger.warning("Helion pulse validation failed: %s", exc)
        return _error_response(str(exc))

    result = integrate_helion_pulse(params, P_heat_w=p_heat_w)

    if not result.success:
        logger.warning("Helion ODE failed: %s", result.message)
        return _error_response(result.message)

    n_ions = float(params.plasma_density_m3)
    n_e = 1.5 * n_ions
    n_points = len(result.t)

    # Integrate fusion energy over the pulse for materials accounting.
    pulse_fusion_energy_j = float(np.trapezoid(result.P_fusion, result.t))
    pulse_fusion_energy_j = max(pulse_fusion_energy_j, 0.0)
    n_reactions = pulse_fusion_energy_j / _HELION_E_REACTION_J
    # Coil thermal load proxy: peak T_keV scales the conducted heat reaching
    # the compression coil winding. 30 K nominal at T_peak = 100 keV.
    t_peak_kev = float(np.max(result.T_kev)) if n_points else 0.0
    coil_delta_t_k = 30.0 * (t_peak_kev / 100.0)

    payload = {
        "success": True,
        "message": result.message,
        "t": result.t.tolist(),
        "W_J": result.W.tolist(),
        "T_kev": result.T_kev.tolist(),
        "P_fusion_w": result.P_fusion.tolist(),
        "P_brem_w": result.P_brem.tolist(),
        "P_cond_w": result.P_cond.tolist(),
        "net_power_w": result.net_power.tolist(),
        "p_heat_w": float(result.p_heat_w),
        "sigma_v_m3_s": result.sigma_v.tolist(),
        "tau_e_s": result.tau_e.tolist(),
        "n_e_m3": [n_e] * n_points,
        "Q": float(result.Q),
        "W_initial_J": float(result.W_initial),
        "parameters": params.model_dump(),
        "pulse_fusion_energy_j": pulse_fusion_energy_j,
        "coil_delta_t_k": coil_delta_t_k,
        "n_reactions": n_reactions,
    }

    if materials_in is not None:
        payload["materials"] = advance_helion_materials(
            materials_in,
            pulse_fusion_energy_j=pulse_fusion_energy_j,
            coil_delta_t_k=coil_delta_t_k,
            n_dhe3_reactions=n_reactions,
        )
    return payload


def run_helion_scenario(
    scenario_id: str,
    *,
    materials_in: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run a predefined Helion scenario by id.

    Returns the same dict shape as run_helion_pulse(), with an additional
    ``scenario_label`` and ``scenario_description`` for UI display.
    """
    try:
        scenario = get_helion_scenario(scenario_id)
    except ValueError as exc:
        return _error_response(str(exc))

    result_dict = run_helion_pulse(
        {**scenario.parameters.model_dump(), "p_heat_w": scenario.P_heat_w},
        materials_in=materials_in,
    )
    result_dict["scenario_label"] = scenario.label
    result_dict["scenario_description"] = scenario.description
    return result_dict


def run_helion_sweep(
    raw_inputs: dict[str, Any],
    *,
    n_rc: int = 20,
    n_density: int = 20,
    include_tau_uncertainty: bool = False,
) -> dict[str, Any]:
    """Compute a 2D ignition boundary sweep using raw_inputs as the base params.

    The sweep varies compression_ratio and plasma_density_m3 over logarithmic
    grids while holding all other parameters fixed at the validated base values.

    Args:
        raw_inputs: HelionParameters fields (same as run_helion_pulse).
        n_rc:       Number of grid points on the compression_ratio axis.
        n_density:  Number of grid points on the plasma_density_m3 axis.

    Returns:
        JSON-serializable dict with keys:
        - success (bool)
        - message (str)
        - compression_ratios (list[float])
        - densities_m3 (list[float])
        - Q_map (list[list[float]])   — shape (n_rc, n_density)
        - ignition_mask (list[list[bool]])
        - n_failed (int)
        - parameters (dict)
    """
    try:
        p_heat_w = float(raw_inputs.pop("p_heat_w", 0.0))
        base_params = HelionParameters(**{k: float(v) for k, v in raw_inputs.items()})
    except (ValidationError, ValueError, TypeError) as exc:
        logger.warning("Helion sweep validation failed: %s", exc)
        return _error_response(str(exc))

    rc_array = np.logspace(0, 3, n_rc)        # 1 → 1000
    n_array = np.logspace(19, 23, n_density)   # 1e19 → 1e23 m⁻³

    sweep = ignition_boundary_sweep(rc_array, n_array, base_params, P_heat_w=p_heat_w)

    payload: dict[str, Any] = {
        "success": True,
        "message": f"Sweep complete. {sweep.n_failed} grid point(s) skipped.",
        "compression_ratios": sweep.compression_ratios.tolist(),
        "densities_m3": sweep.densities_m3.tolist(),
        "Q_map": sweep.Q_map.tolist(),
        "ignition_mask": sweep.ignition_mask.tolist(),
        "n_failed": sweep.n_failed,
        "parameters": base_params.model_dump(),
    }

    if include_tau_uncertainty:
        n_unc = min(n_rc, 12)
        n_den_unc = min(n_density, 12)
        rc_unc = np.logspace(0, 3, n_unc)
        n_unc_arr = np.logspace(19, 23, n_den_unc)
        sweep_lo = ignition_boundary_sweep(
            rc_unc, n_unc_arr, base_params, P_heat_w=p_heat_w,
            n_t_points=80, c_tau_scale=TAU_E_UNCERTAINTY_LOWER,
        )
        sweep_hi = ignition_boundary_sweep(
            rc_unc, n_unc_arr, base_params, P_heat_w=p_heat_w,
            n_t_points=80, c_tau_scale=TAU_E_UNCERTAINTY_UPPER,
        )
        payload["tau_uncertainty"] = True
        payload["compression_ratios_unc"] = sweep_lo.compression_ratios.tolist()
        payload["densities_m3_unc"] = sweep_lo.densities_m3.tolist()
        payload["Q_map_tau_lo"] = sweep_lo.Q_map.tolist()
        payload["Q_map_tau_hi"] = sweep_hi.Q_map.tolist()
        payload["message"] += (
            f" τ_E band: {sweep_lo.n_failed + sweep_hi.n_failed} extra point(s) skipped."
        )

    return payload


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _error_response(message: str) -> dict[str, Any]:
    return {"success": False, "message": message}
