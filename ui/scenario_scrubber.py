"""Reusable scenario timeline scrubber controls for Dash panels."""

from __future__ import annotations

from dash import dcc, html
import dash_bootstrap_components as dbc

from utils.scenario_scrub import scrub_time_label


def build_scenario_scrubber_card(prefix: str) -> dbc.Card:
    """Timeline scrubber shown after Scenario Mode completes (PWR / MSR)."""
    return dbc.Card(
        dbc.CardBody(
            [
                html.Div("Scenario Replay", className="eyebrow"),
                html.H6("Timeline Scrubber", className="card-title mb-2"),
                html.P(
                    "Drag to replay the pre-computed transient. Charts and metrics "
                    "update to the selected simulated time.",
                    className="small text-secondary-emphasis mb-2",
                ),
                html.Div(
                    id=f"{prefix}-scrub-time-label",
                    className="small fw-semibold text-info mb-2",
                ),
                dcc.Slider(
                    id=f"{prefix}-scrub-slider",
                    min=0,
                    max=0,
                    step=1,
                    value=0,
                    marks=None,
                    tooltip={"placement": "bottom", "always_visible": True},
                    className="scenario-scrub-slider mb-0",
                ),
            ]
        ),
        id=f"{prefix}-scrubber-panel",
        className="workstation-card mt-3 scenario-scrubber-card",
        style={"display": "none"},
    )


def build_pulse_scrubber_card(prefix: str = "helion") -> dbc.Card:
    """Timeline scrubber for one-shot pulse results (Helion)."""
    return dbc.Card(
        dbc.CardBody(
            [
                html.Div("Pulse Replay", className="eyebrow"),
                html.H6("Timeline Scrubber", className="card-title mb-2"),
                html.P(
                    "Scrub through the integrated pulse. Time axis uses microseconds.",
                    className="small text-secondary-emphasis mb-2",
                ),
                html.Div(
                    id=f"{prefix}-scrub-time-label",
                    className="small fw-semibold text-info mb-2",
                ),
                dcc.Slider(
                    id=f"{prefix}-scrub-slider",
                    min=0,
                    max=0,
                    step=1,
                    value=0,
                    marks=None,
                    tooltip={"placement": "bottom", "always_visible": True},
                    className="scenario-scrub-slider mb-0",
                ),
            ]
        ),
        id=f"{prefix}-scrubber-panel",
        className="workstation-card mt-3 scenario-scrubber-card",
        style={"display": "none"},
    )


def format_scrub_label(session: dict, index: int) -> str:
    """Label for PWR/MSR session scrubber."""
    history = session.get("history") or {}
    times = history.get("time_s") or []
    if not times:
        return "No history"
    idx = max(0, min(index, len(times) - 1))
    return scrub_time_label(float(times[idx]), index=idx, total=len(times))
