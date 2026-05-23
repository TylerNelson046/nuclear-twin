"""Unified PWR vs MSR comparison panel (Part 1 — overlay at matched scenarios)."""

from __future__ import annotations

from typing import Any

from dash import Dash, Input, Output, State, dcc, html, no_update
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from orchestrator.comparison_runner import MSR_NOMINAL_POWER_MW, run_pwr_msr_comparison
from orchestrator.msr_scenarios import msr_scenario_options
from orchestrator.scenarios import pwr_scenario_options
from physics.msr.thermal import TAU_CORE_NOM
from ui.pwr_panel import DISCLAIMER
from utils.display_units import time_axis_label, time_values_seconds

_DARK_TEMPLATE = "plotly_dark"
_PAPER_BG = "#0f172a"
_PLOT_BG = "#1e293b"
_FONT = dict(color="#e2e8f0", family="monospace")

COLORS = {
    "pwr": "#38bdf8",
    "msr": "#2dd4bf",
    "pwr_beta": "#f87171",
    "msr_beta": "#a78bfa",
}


def build_comparison_panel() -> dbc.Card:
    """Build the cross-reactor PWR vs MSR comparison workstation."""
    return dbc.Card(
        dbc.CardBody(
            [
                dcc.Store(id="compare-result-store"),
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                html.Div("Cross-Reactor Analysis", className="eyebrow"),
                                html.H3("PWR vs MSR Comparison", className="card-title mb-2"),
                                html.P(
                                    "Run matched Scenario Mode transients side by side. "
                                    "Normalized power and reactivity overlays highlight "
                                    "how precursor drift changes MSR dynamics versus a solid-fuel PWR.",
                                    className="text-secondary-emphasis",
                                ),
                            ],
                            lg=8,
                        ),
                        dbc.Col(
                            dbc.Alert(DISCLAIMER, color="warning", className="mb-0"),
                            lg=4,
                        ),
                    ],
                    className="g-3 mb-4",
                ),
                dbc.Alert(
                    "Select scenarios and run a paired comparison.",
                    id="compare-status-alert",
                    color="info",
                    className="mb-4",
                ),
                dbc.Card(
                    dbc.CardBody(
                        [
                            dbc.Row(
                                [
                                    dbc.Col(
                                        [
                                            dbc.Label("PWR scenario", html_for="compare-pwr-scenario"),
                                            dcc.Dropdown(
                                                id="compare-pwr-scenario",
                                                options=pwr_scenario_options(),
                                                value="load_following",
                                                clearable=False,
                                                className="dark-dropdown",
                                            ),
                                        ],
                                        md=5,
                                    ),
                                    dbc.Col(
                                        [
                                            dbc.Label("MSR scenario", html_for="compare-msr-scenario"),
                                            dcc.Dropdown(
                                                id="compare-msr-scenario",
                                                options=msr_scenario_options(),
                                                value="load_following",
                                                clearable=False,
                                                className="dark-dropdown",
                                            ),
                                        ],
                                        md=5,
                                    ),
                                    dbc.Col(
                                        dbc.Button(
                                            "Run Comparison",
                                            id="compare-run-btn",
                                            color="primary",
                                            className="w-100 mt-4",
                                        ),
                                        md=2,
                                    ),
                                ],
                                className="g-3 align-items-end",
                            ),
                            dbc.Row(
                                [
                                    dbc.Col(
                                        [
                                            dbc.Label("τ_core (s)", html_for="compare-tau-core"),
                                            dcc.Slider(
                                                id="compare-tau-core",
                                                min=1.0,
                                                max=50.0,
                                                step=1.0,
                                                value=float(TAU_CORE_NOM),
                                                marks={1: "1", 10: "10", 25: "25", 50: "50"},
                                            ),
                                        ],
                                        md=6,
                                    ),
                                    dbc.Col(
                                        [
                                            dbc.Label("τ_loop (s)", html_for="compare-tau-loop"),
                                            dcc.Slider(
                                                id="compare-tau-loop",
                                                min=5.0,
                                                max=100.0,
                                                step=5.0,
                                                value=20.0,
                                                marks={5: "5", 20: "20", 50: "50", 100: "100"},
                                            ),
                                        ],
                                        md=6,
                                    ),
                                ],
                                className="g-3 mt-2",
                            ),
                        ]
                    ),
                    className="workstation-card mb-4",
                ),
                html.Div(id="compare-metrics", className="mb-3"),
                dbc.Row(
                    [
                        dbc.Col(
                            _graph_card(
                                "Normalized Power",
                                "Both twins at % of respective nominal full power.",
                                "compare-power-graph",
                                _empty_power_figure(),
                            ),
                            lg=6,
                        ),
                        dbc.Col(
                            _graph_card(
                                "Total Reactivity",
                                "Net reactivity (pcm) during each transient.",
                                "compare-reactivity-graph",
                                _empty_reactivity_figure(),
                            ),
                            lg=6,
                        ),
                    ],
                    className="g-3 mb-3",
                ),
                _graph_card(
                    "β_eff,flow vs Salt Velocity",
                    "MSR flowing β (Eq. 24) with static PWR β_eff reference (no drift).",
                    "compare-beta-graph",
                    _empty_beta_figure(),
                ),
            ]
        ),
        className="comparison-panel shell-card",
    )


def register_comparison_callbacks(app: Dash) -> None:
    """Register comparison panel callbacks."""

    @app.callback(
        Output("compare-result-store", "data"),
        Output("compare-status-alert", "children"),
        Output("compare-status-alert", "color"),
        Input("compare-run-btn", "n_clicks"),
        State("compare-pwr-scenario", "value"),
        State("compare-msr-scenario", "value"),
        State("compare-tau-core", "value"),
        State("compare-tau-loop", "value"),
        prevent_initial_call=True,
    )
    def _run_comparison(
        n_clicks: int | None,
        pwr_id: str | None,
        msr_id: str | None,
        tau_core: float | None,
        tau_loop: float | None,
    ) -> tuple:
        if not n_clicks:
            return no_update, no_update, no_update
        result = run_pwr_msr_comparison(
            pwr_id or "load_following",
            msr_id or "load_following",
            tau_core_nom=float(tau_core or TAU_CORE_NOM),
            tau_loop_nom=float(tau_loop or 20.0),
        )
        color = "success" if result.get("ok") else "danger"
        return result, result.get("message", "Comparison finished."), color

    @app.callback(
        Output("compare-metrics", "children"),
        Output("compare-power-graph", "figure"),
        Output("compare-reactivity-graph", "figure"),
        Output("compare-beta-graph", "figure"),
        Input("compare-result-store", "data"),
        Input("display-units-store", "data"),
    )
    def _update_charts(
        result: dict[str, Any] | None,
        display_units: dict[str, str] | None,
    ) -> tuple:
        if not result or not result.get("ok"):
            return html.Div(), _empty_power_figure(), _empty_reactivity_figure(), _empty_beta_figure()
        return (
            _metrics_row(result),
            _power_overlay_figure(result, display_units),
            _reactivity_overlay_figure(result, display_units),
            _beta_comparison_figure(result),
        )


def _graph_card(title: str, subtitle: str, graph_id: str, figure: go.Figure) -> dbc.Card:
    return dbc.Card(
        [
            dbc.CardHeader(
                [
                    html.Div(title, className="graph-card-title"),
                    html.Small(subtitle, className="graph-card-subtitle"),
                ],
                className="graph-card-header",
            ),
            dbc.CardBody(
                dcc.Graph(id=graph_id, figure=figure, config={"displayModeBar": True, "responsive": True}),
                className="graph-card-body",
            ),
        ],
        className="workstation-card graph-card mb-0",
    )


def _base_layout(title: str = "", y_title: str = "", x_title: str = "Time") -> go.Layout:
    return go.Layout(
        template=_DARK_TEMPLATE,
        paper_bgcolor=_PAPER_BG,
        plot_bgcolor=_PLOT_BG,
        font=_FONT,
        title=dict(text=title, font=dict(size=12)) if title else None,
        xaxis_title=x_title,
        yaxis_title=y_title,
        margin=dict(l=55, r=20, t=35, b=50),
        legend=dict(orientation="h", y=1.12),
        hovermode="x unified",
    )


def _empty_power_figure() -> go.Figure:
    fig = go.Figure()
    fig.update_layout(_base_layout("Normalized Power", "% of nominal"))
    return fig


def _empty_reactivity_figure() -> go.Figure:
    fig = go.Figure()
    fig.update_layout(_base_layout("Total Reactivity", "pcm"))
    return fig


def _empty_beta_figure() -> go.Figure:
    fig = go.Figure()
    fig.update_layout(_base_layout("β_eff Comparison", "β_eff,flow"))
    fig.update_xaxes(type="log", title_text="Salt velocity (m/s)")
    return fig


def _metrics_row(result: dict[str, Any]) -> dbc.Row:
    pwr = result.get("pwr") or {}
    msr = result.get("msr") or {}
    pwr_beta = float(result.get("pwr_beta_static", 0.0))
    msr_beta = float((result.get("beta_sweep") or {}).get("beta_current", 0.0))
    return dbc.Row(
        [
            _badge("PWR scenario", pwr.get("label", "—")),
            _badge("MSR scenario", msr.get("label", "—")),
            _badge("PWR duration", f"{pwr.get('duration_s', 0):.0f} s"),
            _badge("MSR duration", f"{msr.get('duration_s', 0):.0f} s"),
            _badge("β_eff (PWR static)", f"{pwr_beta:.5f}"),
            _badge("β_eff,flow (MSR @ nominal v)", f"{msr_beta:.5f}"),
        ],
        className="g-2 metrics-row",
    )


def _badge(label: str, value: str) -> dbc.Col:
    return dbc.Col(
        dbc.Card(
            dbc.CardBody(
                [
                    html.Div(label, className="metric-label small text-secondary-emphasis"),
                    html.Div(value, className="metric-value fw-semibold"),
                ],
                className="p-2",
            ),
            className="metric-badge",
        ),
        xs=6,
        md=4,
        lg=2,
    )


def _power_overlay_figure(
    result: dict[str, Any],
    display_units: dict[str, str] | None,
) -> go.Figure:
    pwr_h = (result.get("pwr") or {}).get("history") or {}
    msr_h = (result.get("msr") or {}).get("history") or {}
    pwr_label = (result.get("pwr") or {}).get("label", "PWR")
    msr_label = (result.get("msr") or {}).get("label", "MSR")

    p_norm = pwr_h.get("power_normalized", [])
    if p_norm:
        pwr_pct = [100.0 * float(v) for v in p_norm]
    else:
        pm = pwr_h.get("power_mw", [])
        peak = max(pm) if pm else 1.0
        pwr_pct = [100.0 * float(p) / peak for p in pm]
    msr_pct = [
        100.0 * float(p) / MSR_NOMINAL_POWER_MW
        for p in msr_h.get("power_mw", [])
    ]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=time_values_seconds(pwr_h.get("time_s", []), display_units),
            y=pwr_pct,
            name=f"PWR — {pwr_label}",
            line=dict(color=COLORS["pwr"], width=2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=time_values_seconds(msr_h.get("time_s", []), display_units),
            y=msr_pct,
            name=f"MSR — {msr_label}",
            line=dict(color=COLORS["msr"], width=2, dash="dash"),
        )
    )
    fig.update_layout(
        _base_layout(
            "Normalized Power Overlay",
            "% of nominal",
            time_axis_label(display_units),
        )
    )
    return fig


def _reactivity_overlay_figure(
    result: dict[str, Any],
    display_units: dict[str, str] | None,
) -> go.Figure:
    pwr_h = (result.get("pwr") or {}).get("history") or {}
    msr_h = (result.get("msr") or {}).get("history") or {}
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=time_values_seconds(pwr_h.get("time_s", []), display_units),
            y=pwr_h.get("total_reactivity_pcm", []),
            name="PWR ρ_total",
            line=dict(color=COLORS["pwr"], width=2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=time_values_seconds(msr_h.get("time_s", []), display_units),
            y=msr_h.get("total_reactivity_pcm", []),
            name="MSR ρ_total",
            line=dict(color=COLORS["msr"], width=2, dash="dash"),
        )
    )
    fig.add_hline(y=0, line=dict(color="#475569", width=1, dash="dot"))
    fig.update_layout(
        _base_layout(
            "Reactivity Overlay",
            "pcm",
            time_axis_label(display_units),
        )
    )
    return fig


def _beta_comparison_figure(result: dict[str, Any]) -> go.Figure:
    sweep = result.get("beta_sweep") or {}
    pwr_beta = float(result.get("pwr_beta_static", 0.0))
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=sweep.get("v_salt", []),
            y=sweep.get("beta_eff_flow", []),
            name="MSR β_eff,flow",
            line=dict(color=COLORS["msr_beta"], width=2),
        )
    )
    fig.add_hline(
        y=pwr_beta,
        line=dict(color=COLORS["pwr_beta"], dash="dash", width=2),
        annotation_text=f"PWR β_eff static = {pwr_beta:.5f}",
    )
    fig.add_trace(
        go.Scatter(
            x=[sweep.get("v_salt_current", 1.0)],
            y=[sweep.get("beta_current", 0.0)],
            name="MSR nominal operating point",
            mode="markers",
            marker=dict(color=COLORS["msr"], size=12, symbol="star"),
        )
    )
    fig.update_layout(_base_layout("β_eff Comparison", "β_eff,flow", "Salt velocity (m/s)"))
    fig.update_xaxes(type="log")
    return fig
