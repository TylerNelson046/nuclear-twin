"""MSR simulation runner — keeps Dash callbacks out of the physics layer.

Week 11 deliverable — MSR Scenarios + UI.

Provides three entry points:
  run_msr_control_step()     — advance the MSR twin one real-time UI step
  run_msr_scenario()         — run a predefined named transient from scratch
  run_msr_beta_flow_sweep()  — compute β_eff,flow vs salt velocity (algebraic, no ODE)

All functions accept plain Python scalars/dicts, validate inputs, and return
JSON-serializable dicts suitable for dcc.Store.

DDE strategy for real-time mode (SPEC §4.6.3):
    Each control step initialises a fresh PrecursorHistory pre-filled with the
    current precursor concentrations from the session state. This is equivalent to
    assuming the reactor has been at the current (n, C₁…C₆) operating point for
    all t < t_current. The approximation is accurate when step_seconds ≪ τ_loop,
    which holds at τ_loop=20 s and a 5 s control step (25% of the loop period).
    Scenario mode uses the full method-of-steps integration inside integrate_msr.
"""

from __future__ import annotations

import logging
import math
from typing import Any

import numpy as np
from pydantic import ValidationError

from data.keepin_dnp import beta_fractions, decay_constants
from orchestrator.msr_scenarios import get_msr_scenario
from orchestrator.materials_integration import (
    advance_msr_materials,
    initial_msr_materials_payload,
)
from physics.msr.drift import compute_beta_eff_flow
from physics.msr.engine import (
    MSRControls,
    MSRModelConfig,
    build_initial_state,
    integrate_msr,
    make_critical_msr_config,
    msr_critical_base_reactivity_pcm,
)
from physics.msr.parameters import MSRParameters
from physics.msr.thermal import (
    L_CORE,
    P_NOM_MSR,
    T_SALT_INLET_NOM,
    T_SALT_NOM,
    TAU_CORE_NOM,
)
from physics.shared.kinetics import LAMBDA_PWR
from utils.event_logger import (
    EventLogBuffer,
    generate_msr_events,
    generate_msr_scenario_events,
    initial_msr_event_log,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Runner constants
# ---------------------------------------------------------------------------

MSR_STEP_SECONDS: float = 5.0          # simulation seconds advanced per UI callback
MSR_STEP_POINTS: int = 11              # output points per control step
MSR_MAX_HISTORY_POINTS: int = 400      # max stored history entries (display trim)
MSR_SCENARIO_MAX_HISTORY: int = 1000   # larger buffer for batch scenario results

# Default v_salt sweep for β_eff,flow chart: 0.01 → 10 m/s
_SWEEP_V_SALT: np.ndarray = np.logspace(-2, 1, 80)  # m/s

_BETA_ARR = np.array(beta_fractions(), dtype=np.float64)
_LAMBDA_ARR = np.array(decay_constants(), dtype=np.float64)
_BETA_EFF_STATIC = float(np.sum(_BETA_ARR))

# Loop-to-core length ratio for the sweep (fixed core geometry: L_loop / L_core ≈ 4)
_LOOP_CORE_RATIO: float = 4.0


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_msr_control_step(
    session: dict[str, Any] | None,
    raw_controls: dict[str, Any],
    *,
    step_seconds: float = MSR_STEP_SECONDS,
    time_multiplier: int = 1,
    max_history_points: int = MSR_MAX_HISTORY_POINTS,
) -> dict[str, Any]:
    """Validate controls, advance the MSR twin by one UI timestep, return session.

    Args:
        session:          Previous dcc.Store payload; None on first call.
        raw_controls:     Raw Dash control values keyed by field name.
        step_seconds:     Base simulation interval per UI callback (s).
        time_multiplier:  Acceleration factor (1, 60, or 3600). 1 = real-time.
        max_history_points: Maximum stored history entries.

    Returns:
        Updated JSON-serializable session dict.
    """
    try:
        params = _validate_msr_controls(raw_controls)
    except (ValidationError, ValueError) as exc:
        return _error_response(session, _plain_msg(exc), raw_controls)

    try:
        current = _coerce_session(session, params, raw_controls)
        config = _config_from_session(current)

        y0 = np.array(current["state"], dtype=np.float64)
        t0 = float(current["time_s"])
        effective_step = max(float(step_seconds) * max(int(time_multiplier), 1), 0.1)
        t1 = t0 + effective_step

        n_pts = max(MSR_STEP_POINTS, int(effective_step / 2) + 2)
        t_eval = np.linspace(t0, t1, n_pts)

        controls = MSRControls(
            external_reactivity_pcm=params.external_reactivity_pcm,
            salt_flow_fraction=_as_float(
                raw_controls.get("salt_flow_fraction", 1.0), "salt flow fraction"
            ),
        )

        # dt_history ≤ τ_loop/2 ensures DDE lookups always reference accepted history
        dt_hist = config.tau_loop / 2.0 if config.tau_loop > 0.0 else 10.0

        result = integrate_msr(
            y0, (t0, t1), controls, config,
            beta_arr=_BETA_ARR, lambda_arr=_LAMBDA_ARR,
            t_eval=t_eval, dt_history=dt_hist,
        )

        if not result.success:
            return _error_response(current, result.message, raw_controls)

        updated = _append_history(current, result.t, result.y, controls, config, max_history_points)
        updated["controls"] = _controls_payload(params, controls.salt_flow_fraction)
        updated["time_multiplier"] = time_multiplier
        updated["status"] = {
            "ok": True,
            "message": f"MSR advanced {effective_step:.1f} s (×{time_multiplier}).",
        }
        return _append_msr_events(current, updated, updated["controls"])

    except Exception as exc:  # pragma: no cover — final guard for UI stability
        return _error_response(session, f"MSR step failed: {exc}", raw_controls)


def run_msr_scenario(scenario_id: str) -> dict[str, Any]:
    """Reset the MSR twin and run a predefined named transient.

    Args:
        scenario_id: One of 'pump_trip', 'load_following'.

    Returns:
        JSON-serializable session dict with full history for UI rendering.
    """
    try:
        scenario = get_msr_scenario(scenario_id)
    except ValueError as exc:
        return _error_response(None, str(exc), {})

    try:
        config = make_critical_msr_config(
            tau_core=scenario.tau_core_s,
            tau_loop=scenario.tau_loop_s,
            beta_arr=_BETA_ARR,
            lambda_arr=_LAMBDA_ARR,
        )
        y0 = build_initial_state(n0=1.0, config=config, beta_arr=_BETA_ARR, lambda_arr=_LAMBDA_ARR)

        t_eval = np.linspace(0.0, scenario.duration_s, scenario.n_output_points)
        # dt_history = τ_loop/2 gives clean method-of-steps chunks
        dt_hist = scenario.tau_loop_s / 2.0 if scenario.tau_loop_s > 0 else 10.0

        result = integrate_msr(
            y0, (0.0, scenario.duration_s),
            scenario.controls_at,
            config,
            beta_arr=_BETA_ARR, lambda_arr=_LAMBDA_ARR,
            t_eval=t_eval, dt_history=dt_hist,
        )

        if not result.success:
            return _error_response(None, result.message, {})

        base_session: dict[str, Any] = {
            "time_s": 0.0,
            "state": y0.tolist(),
            "config": _config_dict(config),
            "beta_eff_flow": compute_beta_eff_flow(
                _BETA_ARR, _LAMBDA_ARR, config.tau_core, config.tau_loop
            ),
            "history": _empty_history(),
            "controls": {},
            "status": {"ok": True, "message": ""},
        }

        # Use a callable controls_at to extract per-time controls for history logging
        updated = _append_history(
            base_session, result.t, result.y,
            scenario.controls_at,
            config,
            max_history=MSR_SCENARIO_MAX_HISTORY,
        )
        # Record final controls from the last point
        final_ctrl = scenario.controls_at(float(result.t[-1]), result.y[:, -1])
        updated["controls"] = _controls_payload_from_ctrl(final_ctrl)
        updated["scenario"] = {
            "id": scenario.scenario_id,
            "label": scenario.label,
            "description": scenario.description,
            "completed": True,
        }
        updated["status"] = {
            "ok": True,
            "message": (
                f"Scenario '{scenario.label}' complete — "
                f"{scenario.duration_s:.0f} s simulated."
            ),
        }
        buffer = EventLogBuffer(initial_msr_event_log())
        buffer.extend(
            generate_msr_scenario_events(
                scenario_id=scenario.scenario_id,
                scenario_label=scenario.label,
                scenario_description=scenario.description,
                history=updated.get("history", {}),
            )
        )
        updated["events"] = buffer.to_list()
        updated["event_flags"] = {}
        return updated

    except Exception as exc:  # pragma: no cover
        return _error_response(None, f"MSR scenario failed: {exc}", {})


def run_msr_beta_flow_sweep(
    tau_core_nom: float = TAU_CORE_NOM,
    tau_loop_nom: float = 20.0,
    v_salt_arr: np.ndarray | None = None,
) -> dict[str, Any]:
    """Compute β_eff,flow vs salt velocity (algebraic, SPEC Eq. 24 — no ODE needed).

    Args:
        tau_core_nom: Core transit time at the nominal operating point (s).
        tau_loop_nom: Loop transit time at the nominal operating point (s).
        v_salt_arr:   Optional salt velocity array (m/s); defaults to logspace sweep.

    Returns:
        JSON-serializable dict:
          v_salt          (list[float]) — salt velocity sweep, m/s
          beta_eff_flow   (list[float]) — β_eff,flow at each v_salt
          beta_eff_static (float)       — static β_eff (zero-flow limit)
          v_salt_current  (float)       — nominal operating point v_salt, m/s
          beta_current    (float)       — β_eff,flow at the nominal operating point
    """
    if v_salt_arr is None:
        v_salt_arr = _SWEEP_V_SALT

    # Compute τ_core and τ_loop for each v_salt using fixed core geometry
    # τ_core = L_CORE / v_salt; ratio τ_loop/τ_core = τ_loop_nom/τ_core_nom (fixed geometry)
    ratio = tau_loop_nom / tau_core_nom if tau_core_nom > 0.0 else 4.0
    tau_core_arr = L_CORE / v_salt_arr            # s; shape (N,)
    tau_loop_arr = ratio * tau_core_arr           # s; shape (N,)

    beta_flow_arr = np.array([
        compute_beta_eff_flow(_BETA_ARR, _LAMBDA_ARR, tc, tl)
        for tc, tl in zip(tau_core_arr, tau_loop_arr)
    ])

    v_salt_current = L_CORE / tau_core_nom if tau_core_nom > 0.0 else 1.0
    beta_current = compute_beta_eff_flow(_BETA_ARR, _LAMBDA_ARR, tau_core_nom, tau_loop_nom)

    return {
        "v_salt": v_salt_arr.tolist(),
        "beta_eff_flow": beta_flow_arr.tolist(),
        "beta_eff_static": _BETA_EFF_STATIC,
        "v_salt_current": float(v_salt_current),
        "beta_current": float(beta_current),
    }


# ---------------------------------------------------------------------------
# Session management helpers
# ---------------------------------------------------------------------------


def _coerce_session(
    session: dict[str, Any] | None,
    params: MSRParameters,
    raw_controls: dict[str, Any],
) -> dict[str, Any]:
    """Return existing session or build a fresh critical MSR session."""
    if session and session.get("state") and session.get("history"):
        # If the operator changed a config field, reinitialise
        stored_cfg = session.get("config", {})
        req_tau_core = float(raw_controls.get("core_transit_time_s", TAU_CORE_NOM))
        req_tau_loop = float(raw_controls.get("loop_transit_time_s", 20.0))
        req_alpha = params.salt_temp_coeff_pcm_per_k
        if (
            abs(stored_cfg.get("tau_core", -1) - req_tau_core) > 0.01
            or abs(stored_cfg.get("tau_loop", -1) - req_tau_loop) > 0.01
            or abs(stored_cfg.get("alpha_salt_pcm_per_k", 0) - req_alpha) > 1e-6
        ):
            logger.info("MSR config changed — reinitialising session.")
            return _initial_msr_session(params, raw_controls)
        return session
    return _initial_msr_session(params, raw_controls)


def _initial_msr_session(
    params: MSRParameters,
    raw_controls: dict[str, Any],
) -> dict[str, Any]:
    """Build a fresh critical MSR session at flowing steady state."""
    tau_core = float(raw_controls.get("core_transit_time_s", TAU_CORE_NOM))
    tau_loop = float(raw_controls.get("loop_transit_time_s", 20.0))
    alpha = params.salt_temp_coeff_pcm_per_k

    config = make_critical_msr_config(
        tau_core=tau_core,
        tau_loop=tau_loop,
        alpha_salt_pcm_per_k=alpha,
        beta_arr=_BETA_ARR,
        lambda_arr=_LAMBDA_ARR,
    )
    n0 = params.initial_power_mw * 1.0e6 / P_NOM_MSR
    y0 = build_initial_state(n0=n0, config=config, beta_arr=_BETA_ARR, lambda_arr=_LAMBDA_ARR)

    beta_flow = compute_beta_eff_flow(_BETA_ARR, _LAMBDA_ARR, tau_core, tau_loop)

    return {
        "time_s": 0.0,
        "state": y0.tolist(),
        "config": _config_dict(config),
        "beta_eff_flow": float(beta_flow),
        "history": _empty_history(),
        "controls": {
            "external_reactivity_pcm": params.external_reactivity_pcm,
            "salt_flow_fraction": 1.0,
        },
        "status": {"ok": True, "message": "MSR session initialised."},
        "events": initial_msr_event_log(),
        "event_flags": {},
        "materials": initial_msr_materials_payload(),
    }


def _config_from_session(session: dict[str, Any]) -> MSRModelConfig:
    """Rehydrate MSRModelConfig from stored session config dict."""
    cfg = session.get("config", {})
    return MSRModelConfig(
        tau_core=float(cfg.get("tau_core", TAU_CORE_NOM)),
        tau_loop=float(cfg.get("tau_loop", 20.0)),
        alpha_salt_pcm_per_k=float(cfg.get("alpha_salt_pcm_per_k", -5.0)),
        T_salt_ref_k=float(cfg.get("T_salt_ref_k", T_SALT_NOM)),
        T_salt_inlet_k=float(cfg.get("T_salt_inlet_k", T_SALT_INLET_NOM)),
        P_ref=float(cfg.get("P_ref", P_NOM_MSR)),
        Lambda=float(cfg.get("Lambda", LAMBDA_PWR)),
        base_reactivity_pcm=float(cfg.get("base_reactivity_pcm", 0.0)),
    )


def _config_dict(config: MSRModelConfig) -> dict[str, float]:
    return {
        "tau_core": config.tau_core,
        "tau_loop": config.tau_loop,
        "alpha_salt_pcm_per_k": config.alpha_salt_pcm_per_k,
        "T_salt_ref_k": config.T_salt_ref_k,
        "T_salt_inlet_k": config.T_salt_inlet_k,
        "P_ref": config.P_ref,
        "Lambda": config.Lambda,
        "base_reactivity_pcm": config.base_reactivity_pcm,
    }


# ---------------------------------------------------------------------------
# History helpers
# ---------------------------------------------------------------------------


def _empty_history() -> dict[str, list[float]]:
    return {
        "time_s": [],
        "power_mw": [],
        "salt_temperature_k": [],
        "external_reactivity_pcm": [],
        "temp_reactivity_pcm": [],
        "total_reactivity_pcm": [],
        "salt_flow_fraction": [],
        "beta_eff_flow": [],
        # Delayed precursor concentrations (MSR 8-state indices 1–6).
        "precursor_C1": [],
        "precursor_C2": [],
        "precursor_C3": [],
        "precursor_C4": [],
        "precursor_C5": [],
        "precursor_C6": [],
        # Drift decomposition (SPEC Eq. 22–23); return uses C(t−τ_loop) ≈ C(t).
        "precursor_flow_out_1": [],
        "precursor_flow_out_2": [],
        "precursor_flow_out_3": [],
        "precursor_flow_out_4": [],
        "precursor_flow_out_5": [],
        "precursor_flow_out_6": [],
        "precursor_return_1": [],
        "precursor_return_2": [],
        "precursor_return_3": [],
        "precursor_return_4": [],
        "precursor_return_5": [],
        "precursor_return_6": [],
    }


def _append_history(
    session: dict[str, Any],
    times: np.ndarray,
    states: np.ndarray,
    controls: MSRControls | Any,
    config: MSRModelConfig,
    max_history: int = MSR_MAX_HISTORY_POINTS,
) -> dict[str, Any]:
    """Append ODE output points to the session history and trim if needed."""
    history = session.get("history") or _empty_history()
    for key in _empty_history():
        history.setdefault(key, [])

    start_idx = 1 if history["time_s"] else 0
    new_count = 0

    for idx in range(start_idx, len(times)):
        y = states[:, idx]
        t = float(times[idx])
        ctrl = _resolve_controls(controls, t, y)

        n = float(y[0])
        T_salt = float(y[7])
        ext_pcm = ctrl.external_reactivity_pcm if isinstance(ctrl, MSRControls) else float(getattr(ctrl, "external_reactivity_pcm", 0.0))
        rho_base = float(config.base_reactivity_pcm) * 1.0e-5
        rho_temp = config.alpha_salt_pcm_per_k * (T_salt - config.T_salt_ref_k)  # pcm
        rho_total = float(config.base_reactivity_pcm) + ext_pcm + rho_temp

        history["time_s"].append(t)
        history["power_mw"].append(n * config.P_ref / 1.0e6)
        history["salt_temperature_k"].append(T_salt)
        history["external_reactivity_pcm"].append(ext_pcm)
        history["temp_reactivity_pcm"].append(float(rho_temp))
        history["total_reactivity_pcm"].append(float(rho_total))
        flow_frac = (
            ctrl.salt_flow_fraction if isinstance(ctrl, MSRControls) else 1.0
        )
        history["salt_flow_fraction"].append(flow_frac)
        history["beta_eff_flow"].append(
            float(
                compute_beta_eff_flow(
                    _BETA_ARR, _LAMBDA_ARR, config.tau_core, config.tau_loop,
                )
            )
        )
        inv_tau = 1.0 / config.tau_core if config.tau_core > 0.0 else 0.0
        for g in range(6):
            ci = float(y[g + 1])
            history[f"precursor_C{g + 1}"].append(ci)
            if inv_tau > 0.0:
                flow_out = inv_tau * ci
                lam = float(_LAMBDA_ARR[g])
                return_est = inv_tau * ci * math.exp(-lam * config.tau_loop)
            else:
                flow_out = 0.0
                return_est = 0.0
            history[f"precursor_flow_out_{g + 1}"].append(flow_out)
            history[f"precursor_return_{g + 1}"].append(return_est)
        new_count += 1

    # Trim to max_history (keep newest)
    if max_history and len(history["time_s"]) > max_history:
        history = {k: v[-max_history:] for k, v in history.items()}

    final_y = states[:, -1]
    final_ctrl = _resolve_controls(controls, float(times[-1]), final_y)
    final_n = float(final_y[0])
    final_T = float(final_y[7])

    updated = dict(session)
    updated["time_s"] = float(times[-1])
    updated["state"] = final_y.tolist()
    updated["history"] = history
    updated["new_point_count_last_step"] = new_count
    updated["metrics"] = {
        "power_mw": final_n * config.P_ref / 1.0e6,
        "salt_temperature_k": final_T,
        "total_reactivity_pcm": (
            float(config.base_reactivity_pcm)
            + (final_ctrl.external_reactivity_pcm if isinstance(final_ctrl, MSRControls) else 0.0)
            + config.alpha_salt_pcm_per_k * (final_T - config.T_salt_ref_k)
        ),
        "beta_eff_flow": float(
            compute_beta_eff_flow(
                _BETA_ARR, _LAMBDA_ARR, config.tau_core, config.tau_loop,
            )
        ),
    }
    updated["beta_eff_flow"] = updated["metrics"]["beta_eff_flow"]

    # Advance illustrative material aging over this integration window.
    dt_total = float(times[-1] - times[0])
    if dt_total > 0.0:
        updated["materials"] = advance_msr_materials(
            session.get("materials"),
            dt_s=dt_total,
            power_mw=float(updated["metrics"]["power_mw"]),
            t_salt_k=float(final_T),
            p_nominal_mw=float(config.P_ref) / 1.0e6,
        )

    return updated


def _append_msr_events(
    previous: dict[str, Any] | None,
    session: dict[str, Any],
    controls_payload: dict[str, Any],
) -> dict[str, Any]:
    """Generate plain-language MSR events and append them to the session buffer."""
    events, flags = generate_msr_events(
        time_s=float(session.get("time_s", 0.0)),
        previous_controls=(previous or {}).get("controls"),
        current_controls=controls_payload,
        previous_metrics=(previous or {}).get("metrics"),
        current_metrics=session.get("metrics", {}),
        previous_flags=session.get("event_flags", {}),
        config=session.get("config", {}),
    )
    buffer = EventLogBuffer(session.get("events", initial_msr_event_log()))
    buffer.extend(events)
    updated = dict(session)
    updated["events"] = buffer.to_list()
    updated["event_flags"] = flags
    return updated


def _resolve_controls(
    controls: MSRControls | Any,
    time_s: float,
    state: np.ndarray,
) -> MSRControls:
    """Resolve constant or callable controls for one history sample."""
    if callable(controls):
        result = controls(time_s, state)
        return result if isinstance(result, MSRControls) else MSRControls()
    return controls


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _validate_msr_controls(raw_controls: dict[str, Any]) -> MSRParameters:
    """Build and validate MSRParameters from raw Dash values."""
    return MSRParameters(
        external_reactivity_pcm=_as_float(
            raw_controls.get("external_reactivity_pcm", 0.0), "external reactivity"
        ),
        initial_power_mw=_as_float(
            raw_controls.get("initial_power_mw", 500.0), "initial power"
        ),
        salt_temperature_k=_as_float(
            raw_controls.get("salt_temperature_k", T_SALT_NOM), "salt temperature"
        ),
        salt_velocity_m_s=_as_float(
            raw_controls.get("salt_velocity_m_s", 1.0), "salt velocity"
        ),
        core_transit_time_s=_as_float(
            raw_controls.get("core_transit_time_s", TAU_CORE_NOM), "core transit time"
        ),
        loop_transit_time_s=_as_float(
            raw_controls.get("loop_transit_time_s", 20.0), "loop transit time"
        ),
        salt_temp_coeff_pcm_per_k=_as_float(
            raw_controls.get("salt_temp_coeff_pcm_per_k", -5.0), "salt temp coefficient"
        ),
    )


def _as_float(value: Any, label: str) -> float:
    if value is None or value == "":
        raise ValueError(f"{label} is required.")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be numeric.") from exc
    if not math.isfinite(number):
        raise ValueError(f"{label} must be a finite number.")
    return number


# ---------------------------------------------------------------------------
# Payload helpers
# ---------------------------------------------------------------------------


def _controls_payload(params: MSRParameters, salt_flow_fraction: float) -> dict[str, float]:
    return {
        "external_reactivity_pcm": params.external_reactivity_pcm,
        "salt_flow_fraction": salt_flow_fraction,
        "core_transit_time_s": params.core_transit_time_s,
        "loop_transit_time_s": params.loop_transit_time_s,
        "salt_temp_coeff_pcm_per_k": params.salt_temp_coeff_pcm_per_k,
    }


def _controls_payload_from_ctrl(ctrl: MSRControls) -> dict[str, float]:
    return {
        "external_reactivity_pcm": ctrl.external_reactivity_pcm,
        "salt_flow_fraction": ctrl.salt_flow_fraction,
    }


def _error_response(
    session: dict[str, Any] | None,
    message: str,
    raw_controls: dict[str, Any],
) -> dict[str, Any]:
    base = dict(session) if session else {
        "time_s": 0.0,
        "state": [],
        "config": {},
        "beta_eff_flow": _BETA_EFF_STATIC,
        "history": _empty_history(),
        "controls": {},
    }
    base["rejected_controls"] = raw_controls
    base["status"] = {"ok": False, "message": message}
    return base


def _plain_msg(exc: Exception) -> str:
    if isinstance(exc, ValidationError):
        first = exc.errors()[0]
        field = ".".join(str(p) for p in first.get("loc", ()))
        return f"{field}: {first.get('msg', 'Invalid value')}"
    return str(exc)
