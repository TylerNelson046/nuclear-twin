"""Simulation runner APIs that keep Dash callbacks out of the physics layer.

The UI passes plain Python values into this module. The runner validates those
values, owns JSON-serializable session state, and delegates numerical work to
the reactor-specific physics engines.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from pydantic import ValidationError

from physics.pwr.engine import (
    PWRControls,
    PWRModelConfig,
    build_initial_state,
    calculate_reactivity_snapshot,
    controls_from_parameters,
    critical_base_reactivity_pcm,
    integrate_pwr,
    integrate_pwr_11state,
    make_params_array,
    neutron_population_to_flux,
    neutron_population_to_power,
    pwr_11_state_system,
)
from physics.pwr.parameters import PWRParameters
from physics.pwr.reactivity import ReferenceState
from physics.pwr.thermal_hydraulics import P_NOM, T_COOL_NOM, T_FUEL_NOM, T_IN_NOM
from orchestrator.scenarios import get_pwr_scenario
from orchestrator.materials_integration import (
    advance_pwr_materials,
    initial_pwr_materials_payload,
)
from utils.event_logger import (
    EventLogBuffer,
    EventRecord,
    generate_pwr_events,
    generate_pwr_scenario_events,
    initial_event_log,
)

PWR_STEP_SECONDS = 10.0
PWR_STEP_POINTS = 21
PWR_MAX_HISTORY_POINTS = 300

# ---------------------------------------------------------------------------
# Accelerated time mode constants (SPEC FR-05, ARCHITECTURE.md §7)
# ---------------------------------------------------------------------------

# Supported time multipliers for FR-05 (1× real-time, 60× = 1 min/s, 3600× = 1 hr/s).
VALID_TIME_MULTIPLIERS: tuple[int, ...] = (1, 60, 3600)

# Adaptive max_step thresholds for the Radau solver (seconds).
# Radau handles stiffness implicitly — max_step bounds the *largest* step the solver
# may take, not the smallest.  Correct values prevent over-stepping during a transient
# while still allowing the solver to stride through slow xenon-timescale dynamics.
_MAX_STEP_FAST_TRANSIENT = 0.5       # fast neutron burst   (|ṅ/n| > 1e-3 /s)
_MAX_STEP_THERMAL_TRANSIENT = 5.0    # thermal ramp         (|dT_fuel/dt| > 1 K/s)
_MAX_STEP_SLOW_DYNAMICS = 300.0      # steady-state / xenon (all rates below thresholds)

# For batch accelerated runs, store more points than the live dashboard limit.
PWR_ACCEL_MAX_HISTORY_POINTS = 1000


def _select_max_step(y0: np.ndarray, params: np.ndarray) -> float:
    """Return the appropriate Radau max_step (s) for the current PWR regime.

    Evaluates the ODE RHS once at the current state and classifies the dominant
    physics timescale.  Fast neutron transients need sub-second steps; slow
    xenon dynamics can tolerate 5-minute steps without accuracy loss.

    Args:
        y0:     Current 11-state vector.
        params: Flat parameter array from make_params_array().

    Returns:
        max_step in seconds: one of the _MAX_STEP_* constants.
    """
    dydt = pwr_11_state_system(0.0, y0, params)
    # Relative rate of change of neutron population (1/s)
    rn = abs(dydt[0]) / max(abs(y0[0]), 1e-10)
    # Absolute rate of fuel temperature change (K/s)
    rTf = abs(dydt[7])
    if rn > 1e-3:
        return _MAX_STEP_FAST_TRANSIENT
    if rTf > 1.0:
        return _MAX_STEP_THERMAL_TRANSIENT
    return _MAX_STEP_SLOW_DYNAMICS


def run_pwr_control_step(
    session: dict[str, Any] | None,
    raw_controls: dict[str, Any],
    *,
    step_seconds: float = PWR_STEP_SECONDS,
    time_multiplier: int = 1,
    min_output_points: int = PWR_STEP_POINTS,
    max_history_points: int | None = PWR_MAX_HISTORY_POINTS,
    history_retention: str = "tail",
) -> dict[str, Any]:
    """Validate controls, run one PWR timestep, and return UI state.

    In accelerated time mode (time_multiplier > 1) each callback covers
    ``step_seconds × time_multiplier`` seconds of simulation time.  The
    Radau solver's max_step is automatically scaled to match the dominant
    physics regime so fast transients remain accurate while slow xenon
    dynamics use large, efficient steps (SPEC FR-05).

    The returned dict is intentionally JSON-serializable so it can be stored
    in ``dcc.Store`` without relying on process-global simulation variables.

    Args:
        session:         Previous ``dcc.Store`` payload; None on first call.
        raw_controls:    Dict of raw Dash input values.
        step_seconds:    Base simulation interval per UI callback (s).
        time_multiplier:  Acceleration factor — must be in VALID_TIME_MULTIPLIERS.
                          1 = real-time, 60 = 1 min/s, 3600 = 1 hr/s.
        min_output_points: Minimum number of samples to return for the UI trace.
        max_history_points: Maximum stored history length; None keeps the full timeline.
        history_retention: "tail" keeps the newest samples; "timeline" downsamples
                           evenly so the full simulated time span remains visible.

    Returns:
        Updated session dict suitable for ``dcc.Store``.
    """
    if time_multiplier not in VALID_TIME_MULTIPLIERS:
        time_multiplier = 1

    try:
        parameters = _validate_pwr_controls(raw_controls)
    except (ValidationError, ValueError) as exc:
        return _error_response(
            session,
            _plain_validation_message(exc),
            raw_controls,
        )

    try:
        previous_controls = session.get("controls") if session else None
        previous_metrics = session.get("metrics") if session else None
        previous_flags = session.get("event_flags") if session else None
        current = _coerce_session(session, parameters)
        controls = controls_from_parameters(parameters)
        config = _config_from_session(current)
        reference_state = _reference_state_from_session(current)
        y0 = np.array(current["state"], dtype=np.float64)
        params = make_params_array(controls, reference_state, config)

        t0 = float(current["time_s"])
        effective_step = max(float(step_seconds) * time_multiplier, 0.1)
        t1 = t0 + effective_step

        # Scale output points so temporal resolution stays ~30 s regardless
        # of how large the accelerated step is.
        n_points = max(2, int(min_output_points), int(effective_step / 30) + 2)
        t_eval = np.linspace(t0, t1, n_points)

        max_step = _select_max_step(y0, params)
        result = integrate_pwr_11state(y0, (t0, t1), params, t_eval=t_eval, max_step=max_step)
        if not result.success:
            return _error_response(current, result.message, raw_controls)

        updated = _append_pwr_result(
            current,
            result.t,
            result.y,
            controls,
            config,
            reference_state,
            max_history=max_history_points,
            history_retention=history_retention,
        )
        updated["controls"] = _controls_payload(parameters)
        updated["time_multiplier"] = time_multiplier
        updated = _append_pwr_events(
            updated,
            previous_controls,
            previous_metrics,
            previous_flags,
        )
        updated["status"] = {
            "ok": True,
            "message": "PWR simulation advanced by %.1f s (×%d)." % (t1 - t0, time_multiplier),
        }
        return updated
    except Exception as exc:  # pragma: no cover - final guard for UI stability
        return _error_response(session, f"PWR step failed: {exc}", raw_controls)


def run_pwr_accelerated_batch(
    session: dict[str, Any] | None,
    raw_controls: dict[str, Any],
    duration_s: float,
    n_output_points: int = 500,
) -> dict[str, Any]:
    """Integrate a large PWR simulation window as fast as the CPU allows.

    Runs the full ``duration_s`` interval as a single ``solve_ivp`` call,
    decoupled from wall-clock time entirely.  Intended for scenario mode
    (e.g., 24-hour xenon transient, load-following profiles) where the UI
    renders a complete time-series after the batch completes rather than
    updating in real time (SPEC FR-02, FR-05).

    The adaptive max_step is selected by ``_select_max_step`` at the start
    of the run:
    - Fast transient at t=0 → 0.5 s max step (accurate capture of prompt jump)
    - Thermal ramp         → 5 s max step
    - Steady-state / xenon → 300 s max step  (~86 400 s ÷ 300 s = 288 steps
                               for a 24-hour xenon run — typically < 2 s wall time)

    Args:
        session:          Previous session dict or None.
        raw_controls:     Dict of raw Dash input values.
        duration_s:       Simulation duration to integrate (s).  Must be > 0.
        n_output_points:  Number of evenly-spaced time points returned in the
                          history (default 500, max PWR_ACCEL_MAX_HISTORY_POINTS).

    Returns:
        Session dict with full history; ``status.ok`` False on any failure.
    """
    if duration_s <= 0.0:
        return _error_response(session, "duration_s must be positive.", raw_controls)

    n_output_points = max(2, min(int(n_output_points), PWR_ACCEL_MAX_HISTORY_POINTS))

    try:
        parameters = _validate_pwr_controls(raw_controls)
    except (ValidationError, ValueError) as exc:
        return _error_response(session, _plain_validation_message(exc), raw_controls)

    try:
        previous_controls = session.get("controls") if session else None
        previous_metrics = session.get("metrics") if session else None
        previous_flags = session.get("event_flags") if session else None
        current = _coerce_session(session, parameters)
        controls = controls_from_parameters(parameters)
        config = _config_from_session(current)
        reference_state = _reference_state_from_session(current)
        y0 = np.array(current["state"], dtype=np.float64)
        params = make_params_array(controls, reference_state, config)

        t0 = float(current["time_s"])
        t1 = t0 + duration_s
        t_eval = np.linspace(t0, t1, n_output_points)

        max_step = _select_max_step(y0, params)
        result = integrate_pwr_11state(y0, (t0, t1), params, t_eval=t_eval, max_step=max_step)
        if not result.success:
            return _error_response(current, result.message, raw_controls)

        # Build a fresh history from the batch result (do not append to live history).
        fresh_session = dict(current)
        fresh_session["history"] = _empty_history()
        updated = _append_pwr_result(
            fresh_session, result.t, result.y, controls, config, reference_state,
            max_history=PWR_ACCEL_MAX_HISTORY_POINTS,
        )
        updated["controls"] = _controls_payload(parameters)
        updated["time_multiplier"] = 0   # 0 signals batch mode to the UI
        updated = _append_pwr_events(
            updated,
            previous_controls,
            previous_metrics,
            previous_flags,
        )
        updated["status"] = {
            "ok": True,
            "message": "PWR batch run: %.0f s simulated (%.0f output points)." % (
                duration_s, n_output_points
            ),
        }
        return updated
    except Exception as exc:  # pragma: no cover - final guard for UI stability
        return _error_response(session, f"PWR batch run failed: {exc}", raw_controls)


def run_pwr_scenario(scenario_id: str) -> dict[str, Any]:
    """Reset the PWR twin and run one predefined Scenario Mode transient.

    Scenario profiles impose time-dependent boundary conditions, overriding
    manual UI controls for the duration of the run. The resulting payload is the
    same JSON-serializable session shape used by live control mode.
    """
    try:
        scenario = get_pwr_scenario(scenario_id)
    except ValueError as exc:
        return _error_response(None, str(exc), {})

    try:
        current = _initial_pwr_session(scenario.initial_parameters)
        config = _config_from_session(current)
        reference_state = _reference_state_from_session(current)
        y0 = np.array(current["state"], dtype=np.float64)
        t_eval = np.linspace(0.0, scenario.duration_s, scenario.n_output_points)

        result = integrate_pwr(
            y0,
            (0.0, scenario.duration_s),
            scenario.controls_at,
            reference_state,
            config,
            t_eval=t_eval,
            max_step=scenario.max_step_s,
        )
        if not result.success:
            return _error_response(current, result.message, {})

        fresh_session = dict(current)
        fresh_session["history"] = _empty_history()
        updated = _append_pwr_result(
            fresh_session,
            result.t,
            result.y,
            scenario.controls_at,
            config,
            reference_state,
            max_history=PWR_ACCEL_MAX_HISTORY_POINTS,
        )
        final_controls = scenario.controls_at(float(result.t[-1]), result.y[:, -1])
        updated["controls"] = _controls_dict_from_pwr_controls(final_controls)
        updated["time_multiplier"] = 0
        updated["scenario"] = {
            "id": scenario.scenario_id,
            "label": scenario.label,
            "description": scenario.description,
            "completed": True,
        }

        buffer = EventLogBuffer(initial_event_log())
        buffer.extend(
            generate_pwr_scenario_events(
                scenario_id=scenario.scenario_id,
                scenario_label=scenario.label,
                scenario_description=scenario.description,
                history=updated.get("history", {}),
            )
        )
        updated["events"] = buffer.to_list()
        updated["event_flags"] = {"scenario_completed": True}
        updated["status"] = {
            "ok": True,
            "message": f"Scenario Mode complete: {scenario.label} ({scenario.duration_s:.0f} s simulated).",
        }
        return updated
    except Exception as exc:  # pragma: no cover - final guard for UI stability
        return _error_response(None, f"PWR scenario run failed: {exc}", {})


def _validate_pwr_controls(raw_controls: dict[str, Any]) -> PWRParameters:
    """Build the PWR parameter schema from raw Dash values."""
    return PWRParameters(
        rod_reactivity_pcm=_as_float(raw_controls.get("rod_reactivity_pcm"), "rod reactivity"),
        boron_ppm=_as_float(raw_controls.get("boron_ppm"), "boron concentration"),
        coolant_flow_fraction=_as_float(
            raw_controls.get("coolant_flow_fraction"),
            "coolant flow fraction",
        ),
        inlet_temperature_k=_as_float(raw_controls.get("inlet_temperature_k"), "inlet temperature"),
    )


def _as_float(value: Any, label: str) -> float:
    """Convert Dash input values into floats before Pydantic bounds checks."""
    if value is None or value == "":
        raise ValueError(f"{label} is required.")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be numeric.") from exc
    if not math.isfinite(number):
        raise ValueError(f"{label} must be a finite number.")
    return number


def _coerce_session(session: dict[str, Any] | None, parameters: PWRParameters) -> dict[str, Any]:
    """Use an existing session or create a critical PWR operating point."""
    if session and session.get("state") and session.get("history"):
        return session
    return _initial_pwr_session(parameters)


def _initial_pwr_session(parameters: PWRParameters) -> dict[str, Any]:
    """Create a critical 11-state PWR session for the first UI update."""
    controls = controls_from_parameters(parameters)
    n0 = parameters.initial_power_mw * 1.0e6 / P_NOM
    y0 = build_initial_state(
        n0=n0,
        fuel_temperature_k=parameters.fuel_temperature_k,
        coolant_temperature_k=parameters.coolant_temperature_k,
    )
    reference_state = ReferenceState(
        fuel_temperature_ref_k=parameters.fuel_temperature_k,
        coolant_temperature_ref_k=parameters.coolant_temperature_k,
    )
    base_reactivity_pcm = critical_base_reactivity_pcm(y0, controls, reference_state)
    config = PWRModelConfig(base_reactivity_pcm=base_reactivity_pcm)

    return {
        "time_s": 0.0,
        "state": y0.tolist(),
        "base_reactivity_pcm": base_reactivity_pcm,
        "reference_state": {
            "fuel_temperature_ref_k": reference_state.fuel_temperature_ref_k,
            "coolant_temperature_ref_k": reference_state.coolant_temperature_ref_k,
        },
        "history": _empty_history(),
        "controls": _controls_payload(parameters),
        "events": initial_event_log(),
        "event_flags": {},
        "materials": initial_pwr_materials_payload(),
        "status": {"ok": True, "message": "PWR session initialized."},
    }


def _config_from_session(session: dict[str, Any]) -> PWRModelConfig:
    """Rehydrate model config values stored in a Dash session payload."""
    return PWRModelConfig(base_reactivity_pcm=float(session.get("base_reactivity_pcm", 0.0)))


def _reference_state_from_session(session: dict[str, Any]) -> ReferenceState:
    """Rehydrate zero-feedback reference temperatures from a session payload."""
    payload = session.get("reference_state", {})
    return ReferenceState(
        fuel_temperature_ref_k=float(payload.get("fuel_temperature_ref_k", T_FUEL_NOM)),
        coolant_temperature_ref_k=float(payload.get("coolant_temperature_ref_k", T_COOL_NOM)),
    )


def _append_pwr_result(
    session: dict[str, Any],
    times: np.ndarray,
    states: np.ndarray,
    controls: PWRControls | Any,
    config: PWRModelConfig,
    reference_state: ReferenceState,
    max_history: int | None = PWR_MAX_HISTORY_POINTS,
    history_retention: str = "tail",
) -> dict[str, Any]:
    """Append integration samples to the stored PWR history."""
    history = session.get("history") or _empty_history()
    for key in _empty_history():
        history.setdefault(key, [])
    start_idx = 1 if history["time_s"] else 0

    history_len_before = len(history["time_s"])
    new_points_added = 0

    for idx in range(start_idx, len(times)):
        y = states[:, idx]
        sample_controls = _controls_for_sample(controls, float(times[idx]), y)
        snapshot = calculate_reactivity_snapshot(y, sample_controls, reference_state, config)
        rhs_params = make_params_array(sample_controls, reference_state, config)
        dydt = pwr_11_state_system(float(times[idx]), y, rhs_params)
        power_mw = neutron_population_to_power(float(y[0]), config) / 1.0e6
        history["time_s"].append(float(times[idx]))
        history["power_mw"].append(float(power_mw))
        history["power_normalized"].append(float(y[0] / config.reference_neutron_population))
        history["neutron_population"].append(float(y[0]))
        history["thermal_flux_n_cm2_s"].append(
            float(neutron_population_to_flux(float(y[0]), config))
        )
        history["fuel_temperature_k"].append(float(y[7]))
        history["coolant_temperature_k"].append(float(y[8]))
        history["rod_reactivity_pcm"].append(float(snapshot.rods_pcm))
        history["doppler_reactivity_pcm"].append(float(snapshot.doppler_pcm))
        history["moderator_reactivity_pcm"].append(float(snapshot.moderator_pcm))
        history["boron_reactivity_pcm"].append(float(snapshot.boron_pcm))
        history["xenon_reactivity_pcm"].append(float(snapshot.xenon_pcm))
        history["total_reactivity_pcm"].append(float(snapshot.total_pcm))
        history["iodine_concentration"].append(float(y[9]))
        history["xenon_concentration"].append(float(y[10]))
        for g in range(6):
            history[f"precursor_C{g + 1}"].append(float(y[g + 1]))
        history["boron_ppm"].append(float(sample_controls.boron_ppm))
        history["coolant_flow_fraction"].append(float(sample_controls.coolant_flow_fraction))
        history["inlet_temperature_k"].append(float(sample_controls.inlet_temperature_k))
        history["dn_dt"].append(float(dydt[0]))
        history["dT_fuel_dt"].append(float(dydt[7]))
        history["dT_coolant_dt"].append(float(dydt[8]))
        history["dI_dt"].append(float(dydt[9]))
        history["dX_dt"].append(float(dydt[10]))
        new_points_added += 1

    trimmed_history, history_rebuilt = _trim_history(history, max_history, history_retention)
    # How many points were dropped from the front so the browser can mirror the trim.
    front_trim_count = (
        0
        if history_rebuilt
        else (history_len_before + new_points_added) - len(trimmed_history["time_s"])
    )

    final_state = states[:, -1]
    final_controls = _controls_for_sample(controls, float(times[-1]), final_state)
    final_snapshot = calculate_reactivity_snapshot(final_state, final_controls, reference_state, config)
    final_power_mw = neutron_population_to_power(float(final_state[0]), config) / 1.0e6

    updated = dict(session)
    updated["time_s"] = float(times[-1])
    updated["state"] = final_state.tolist()
    updated["history"] = trimmed_history
    updated["new_point_count_last_step"] = new_points_added
    updated["front_trim_count_last_step"] = front_trim_count
    updated["history_rebuilt_last_step"] = history_rebuilt
    updated["metrics"] = {
        "power_mw": float(final_power_mw),
        "power_normalized": float(final_state[0] / config.reference_neutron_population),
        "fuel_temperature_k": float(final_state[7]),
        "coolant_temperature_k": float(final_state[8]),
        "total_reactivity_pcm": float(final_snapshot.total_pcm),
        "xenon_reactivity_pcm": float(final_snapshot.xenon_pcm),
    }

    # Advance illustrative material aging over this integration window.
    dt_total = float(times[-1] - times[0])
    if dt_total > 0.0:
        # Estimate rod insertion fraction from rod reactivity (negative = inserted).
        # 1000 pcm worth ≈ ~100% insertion in our simplified model.
        rod_pcm = float(final_controls.rod_reactivity_pcm)
        rod_inserted = max(0.0, min(1.0, -rod_pcm / 1000.0))
        updated["materials"] = advance_pwr_materials(
            session.get("materials"),
            dt_s=dt_total,
            power_mw=float(final_power_mw),
            t_fuel_k=float(final_state[7]),
            t_coolant_k=float(final_state[8]),
            rod_inserted_fraction=rod_inserted,
            p_nominal_mw=P_NOM / 1.0e6,
        )

    return updated


def _controls_for_sample(
    controls: PWRControls | Any,
    time_s: float,
    state: np.ndarray,
) -> PWRControls:
    """Resolve constant or time-dependent controls for one history sample."""
    if callable(controls):
        return controls(time_s, state)
    return controls


def _append_pwr_events(
    session: dict[str, Any],
    previous_controls: dict[str, Any] | None,
    previous_metrics: dict[str, Any] | None,
    previous_flags: dict[str, bool] | None,
) -> dict[str, Any]:
    """Generate plain-language events and append them to the session buffer."""
    events, flags = generate_pwr_events(
        time_s=float(session.get("time_s", 0.0)),
        previous_controls=previous_controls,
        current_controls=session.get("controls", {}),
        previous_metrics=previous_metrics,
        current_metrics=session.get("metrics", {}),
        previous_flags=previous_flags,
        history=session.get("history", {}),
    )
    buffer = EventLogBuffer(session.get("events", []))
    buffer.extend(events)

    updated = dict(session)
    updated["events"] = buffer.to_list()
    updated["event_flags"] = flags
    return updated


def _empty_history() -> dict[str, list[float]]:
    """Return the PWR history structure used by Plotly callbacks."""
    return {
        "time_s": [],
        "power_mw": [],
        "power_normalized": [],
        "neutron_population": [],
        "thermal_flux_n_cm2_s": [],
        "fuel_temperature_k": [],
        "coolant_temperature_k": [],
        "rod_reactivity_pcm": [],
        "doppler_reactivity_pcm": [],
        "moderator_reactivity_pcm": [],
        "boron_reactivity_pcm": [],
        "xenon_reactivity_pcm": [],
        "total_reactivity_pcm": [],
        "iodine_concentration": [],
        "xenon_concentration": [],
        # Delayed precursor concentrations (ODE state indices 1–6).
        # Not persisted in HDF5 (final-state values are saved under /pwr/state/C1–C6).
        "precursor_C1": [],
        "precursor_C2": [],
        "precursor_C3": [],
        "precursor_C4": [],
        "precursor_C5": [],
        "precursor_C6": [],
        # Operator control setpoints — captured per-sample to show scenario schedules.
        # Not persisted in HDF5; derived from PWRControls at each integration sample.
        "boron_ppm": [],
        "coolant_flow_fraction": [],
        "inlet_temperature_k": [],
        # State derivatives (1/s or K/s) for stiffness diagnostics.
        "dn_dt": [],
        "dT_fuel_dt": [],
        "dT_coolant_dt": [],
        "dI_dt": [],
        "dX_dt": [],
    }


def _trim_history(
    history: dict[str, list[float]],
    max_points: int | None = PWR_MAX_HISTORY_POINTS,
    history_retention: str = "tail",
) -> tuple[dict[str, list[float]], bool]:
    """Keep browser-side Store payloads bounded."""
    if max_points is None:
        return history, False

    point_count = len(history.get("time_s", []))
    if point_count <= max_points:
        return history, False

    if history_retention == "timeline":
        target_points = max(2, int(max_points * 0.75))
        return _downsample_history(history, target_points), True

    return {
        key: values[-max_points:]
        for key, values in history.items()
    }, False


def _downsample_history(
    history: dict[str, list[float]],
    target_points: int,
) -> dict[str, list[float]]:
    """Evenly reduce history while preserving the full simulated time span."""
    point_count = len(history.get("time_s", []))
    if point_count <= target_points:
        return history

    indices = np.unique(np.linspace(0, point_count - 1, target_points, dtype=int))
    return {
        key: [values[int(index)] for index in indices if int(index) < len(values)]
        for key, values in history.items()
    }


def _controls_payload(parameters: PWRParameters) -> dict[str, float]:
    """Return a compact, JSON-serializable controls record."""
    return {
        "rod_reactivity_pcm": parameters.rod_reactivity_pcm,
        "boron_ppm": parameters.boron_ppm,
        "coolant_flow_fraction": parameters.coolant_flow_fraction,
        "inlet_temperature_k": parameters.inlet_temperature_k,
    }


def _controls_dict_from_pwr_controls(controls: PWRControls) -> dict[str, float]:
    """Return a compact controls record from engine-level controls."""
    return {
        "rod_reactivity_pcm": controls.rod_reactivity_pcm,
        "boron_ppm": controls.boron_ppm,
        "coolant_flow_fraction": controls.coolant_flow_fraction,
        "inlet_temperature_k": controls.inlet_temperature_k,
    }


def _error_response(
    session: dict[str, Any] | None,
    message: str,
    raw_controls: dict[str, Any],
) -> dict[str, Any]:
    """Return a stable session payload with a plain-language error message."""
    response = dict(session or _initial_pwr_session(_fallback_parameters()))
    response["rejected_controls"] = raw_controls
    buffer = EventLogBuffer(response.get("events", []))
    buffer.append(
        EventRecord(
            time_s=float(response.get("time_s", 0.0)),
            message=f"Input rejected: {message}",
            severity="warning",
            code="input_rejected",
        )
    )
    response["events"] = buffer.to_list()
    response["status"] = {"ok": False, "message": message}
    return response


def _fallback_parameters() -> PWRParameters:
    """Default parameters used only when the first UI callback is invalid."""
    return PWRParameters(
        rod_reactivity_pcm=0.0,
        boron_ppm=0.0,
        coolant_flow_fraction=1.0,
        inlet_temperature_k=T_IN_NOM,
    )


def _plain_validation_message(exc: Exception) -> str:
    """Convert validation errors into text suitable for a dashboard alert."""
    if isinstance(exc, ValidationError):
        first_error = exc.errors()[0]
        field = ".".join(str(part) for part in first_error.get("loc", ()))
        return f"{field}: {first_error.get('msg', 'Invalid value')}"
    return str(exc)