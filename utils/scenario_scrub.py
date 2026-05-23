"""Slice pre-computed scenario history for timeline scrubber replay."""

from __future__ import annotations

import copy
from typing import Any


def history_length(session: dict[str, Any] | None) -> int:
    """Return the number of time samples in a session history."""
    if not session:
        return 0
    return len((session.get("history") or {}).get("time_s") or [])


def clamp_scrub_index(session: dict[str, Any], index: int | None) -> int:
    """Clamp a scrub index to valid history bounds."""
    n = history_length(session)
    if n <= 0:
        return 0
    if index is None:
        return n - 1
    return max(0, min(int(index), n - 1))


def slice_history(history: dict[str, list[Any]], end_index: int) -> dict[str, list[Any]]:
    """Return history lists truncated through end_index (inclusive)."""
    end = end_index + 1
    sliced: dict[str, list[Any]] = {}
    for key, values in history.items():
        if isinstance(values, list):
            sliced[key] = values[:end]
        else:
            sliced[key] = values
    return sliced


def pwr_metrics_at_index(history: dict[str, list[Any]], index: int) -> dict[str, float]:
    """Build PWR metric tiles from a history index."""
    return {
        "power_mw": float(history["power_mw"][index]),
        "power_normalized": float(history["power_normalized"][index]),
        "fuel_temperature_k": float(history["fuel_temperature_k"][index]),
        "coolant_temperature_k": float(history["coolant_temperature_k"][index]),
        "total_reactivity_pcm": float(history["total_reactivity_pcm"][index]),
        "xenon_reactivity_pcm": float(history["xenon_reactivity_pcm"][index]),
    }


def msr_metrics_at_index(history: dict[str, list[Any]], index: int) -> dict[str, float]:
    """Build MSR metric tiles from a history index."""
    return {
        "power_mw": float(history["power_mw"][index]),
        "salt_temperature_k": float(history["salt_temperature_k"][index]),
        "total_reactivity_pcm": float(history["total_reactivity_pcm"][index]),
        "beta_eff_flow": float(history["beta_eff_flow"][index]),
    }


def slice_session_for_scrub(
    session: dict[str, Any],
    index: int | None,
    *,
    reactor: str,
) -> dict[str, Any]:
    """Return a shallow session view with history and metrics at the scrub index."""
    idx = clamp_scrub_index(session, index)
    history = session.get("history") or {}
    if not history.get("time_s"):
        return session

    view = copy.copy(session)
    view["history"] = slice_history(history, idx)
    view["time_s"] = float(history["time_s"][idx])
    if reactor == "pwr":
        view["metrics"] = pwr_metrics_at_index(history, idx)
    elif reactor == "msr":
        view["metrics"] = msr_metrics_at_index(history, idx)
    view["scrub_index"] = idx
    return view


def scrub_time_label(time_s: float, *, index: int, total: int) -> str:
    """Format scrubber position for the UI."""
    if total <= 1:
        return f"t = {time_s:.1f} s"
    return f"t = {time_s:.1f} s  (sample {index + 1} / {total})"


def scrubber_panel_style(session: dict[str, Any] | None) -> dict[str, str]:
    """Show the scrubber panel only after a completed scenario run."""
    if session and (session.get("scenario") or {}).get("completed") and history_length(session) > 1:
        return {"display": "block"}
    return {"display": "none"}


_PULSE_SERIES_KEYS = (
    "t",
    "W_J",
    "T_kev",
    "P_fusion_w",
    "P_brem_w",
    "P_cond_w",
    "net_power_w",
    "sigma_v_m3_s",
    "tau_e_s",
    "n_e_m3",
)


def pulse_length(pulse: dict[str, Any] | None) -> int:
    """Return the number of samples in a Helion pulse result dict."""
    if not pulse or not pulse.get("success"):
        return 0
    return len(pulse.get("t") or [])


def clamp_pulse_index(pulse: dict[str, Any], index: int | None) -> int:
    """Clamp a pulse scrub index to valid bounds."""
    n = pulse_length(pulse)
    if n <= 0:
        return 0
    if index is None:
        return n - 1
    return max(0, min(int(index), n - 1))


def slice_pulse_for_scrub(pulse: dict[str, Any], index: int | None) -> tuple[dict[str, Any], int]:
    """Return pulse dict with time series truncated through index (inclusive)."""
    idx = clamp_pulse_index(pulse, index)
    if pulse_length(pulse) <= 0:
        return pulse, 0
    end = idx + 1
    sliced = dict(pulse)
    for key in _PULSE_SERIES_KEYS:
        values = pulse.get(key)
        if isinstance(values, list):
            sliced[key] = values[:end]
    return sliced, idx


def pulse_scrubber_panel_style(pulse: dict[str, Any] | None) -> dict[str, str]:
    """Show pulse scrubber after a successful multi-point integration."""
    if pulse and pulse.get("success") and pulse_length(pulse) > 1:
        return {"display": "block"}
    return {"display": "none"}


def pulse_scrub_time_label(pulse: dict[str, Any], index: int) -> str:
    """Format Helion pulse scrub position (microseconds)."""
    times = pulse.get("t") or []
    if not times:
        return "No pulse data"
    idx = max(0, min(index, len(times) - 1))
    t_us = float(times[idx]) * 1e6
    return f"t = {t_us:.2f} μs  (sample {idx + 1} / {len(times)})"
