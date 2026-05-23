"""Run paired PWR and MSR scenarios for the unified comparison view."""

from __future__ import annotations

from typing import Any

from data.keepin_dnp import beta_fractions
from orchestrator.msr_runner import run_msr_beta_flow_sweep, run_msr_scenario
from orchestrator.sim_runner import run_pwr_scenario
from physics.msr.thermal import P_NOM_MSR, TAU_CORE_NOM

PWR_BETA_STATIC = float(sum(beta_fractions()))
MSR_NOMINAL_POWER_MW = P_NOM_MSR / 1.0e6


def run_pwr_msr_comparison(
    pwr_scenario_id: str = "load_following",
    msr_scenario_id: str = "load_following",
    *,
    tau_core_nom: float = TAU_CORE_NOM,
    tau_loop_nom: float = 20.0,
) -> dict[str, Any]:
    """Execute PWR and MSR scenarios and package histories for overlay charts."""
    pwr = run_pwr_scenario(pwr_scenario_id)
    msr = run_msr_scenario(msr_scenario_id)
    pwr_ok = bool((pwr.get("status") or {}).get("ok"))
    msr_ok = bool((msr.get("status") or {}).get("ok"))

    if not pwr_ok or not msr_ok:
        messages = []
        if not pwr_ok:
            messages.append((pwr.get("status") or {}).get("message", "PWR run failed."))
        if not msr_ok:
            messages.append((msr.get("status") or {}).get("message", "MSR run failed."))
        return {
            "ok": False,
            "message": " | ".join(messages),
            "pwr": _session_snapshot(pwr),
            "msr": _session_snapshot(msr),
            "beta_sweep": None,
        }

    sweep = run_msr_beta_flow_sweep(tau_core_nom=tau_core_nom, tau_loop_nom=tau_loop_nom)
    return {
        "ok": True,
        "message": (
            f"Compared '{_scenario_label(pwr)}' (PWR) with "
            f"'{_scenario_label(msr)}' (MSR)."
        ),
        "pwr": _session_snapshot(pwr),
        "msr": _session_snapshot(msr),
        "beta_sweep": sweep,
        "pwr_beta_static": PWR_BETA_STATIC,
    }


def _scenario_label(session: dict[str, Any]) -> str:
    scenario = session.get("scenario") or {}
    return str(scenario.get("label") or "scenario")


def _session_snapshot(session: dict[str, Any]) -> dict[str, Any]:
    """Extract JSON-safe comparison fields from a twin session dict."""
    history = session.get("history") or {}
    scenario = session.get("scenario") or {}
    return {
        "label": scenario.get("label", ""),
        "description": scenario.get("description", ""),
        "duration_s": float(history.get("time_s", [0])[-1]) if history.get("time_s") else 0.0,
        "history": {
            "time_s": list(history.get("time_s", [])),
            "power_mw": list(history.get("power_mw", [])),
            "power_normalized": list(history.get("power_normalized", [])),
            "total_reactivity_pcm": list(history.get("total_reactivity_pcm", [])),
        },
    }
