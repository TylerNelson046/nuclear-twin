"""Helion FRC Fusion Reactor control panel and Dash callbacks.

CURRENT charts (SPEC FR-14): plasma T, power balance, W(t), ignition heatmap.
ADDITIONAL: ⟨σv⟩ and τ_E vs time; optional τ_E uncertainty band on sweep (R-02).
See docs/UI_VISUALIZATION.md for the full inventory.

Provides:
- build_helion_panel()         — full layout (controls + charts + heatmap)
- register_helion_callbacks()  — all @app.callback wrappers

Interaction model: pulse-based (not continuous like PWR).  The user sets parameters
and clicks "Run Pulse" or selects a predefined scenario.  A single synchronous
integration runs and the charts update once.  The ignition boundary heatmap is
computed on demand via a separate "Run Sweep" button.

Architecture:
- All physics is accessed through orchestrator.helion_runner, never directly.
- dcc.Store holds session state (last pulse result + last sweep result).
- No dcc.Interval — Helion pulses are one-shot, microsecond-timescale events.
- plotly_dark template + graph_objects (not express) per CLAUDE.md rules.
"""

from __future__ import annotations

from typing import Any

from dash import Dash, Input, Output, State, ctx, dcc, html, no_update
import dash_bootstrap_components as dbc
import plotly.graph_objects as go

from orchestrator.helion_runner import run_helion_pulse, run_helion_scenario, run_helion_sweep
from orchestrator.helion_scenarios import helion_scenario_options
from orchestrator.materials_integration import initial_helion_materials_payload
from physics.materials import HELION_DESIGN_LIMITS
from ui.diagrams import (
    build_damage_panel,
    build_helion_diagram,
    helion_phase_for_time,
    helion_pulse_phase_figure,
    render_helion_diagram,
    update_damage_panel_children,
)
from ui.diagrams.pwr_schematic import svg_to_img
from ui.pwr_panel import DISCLAIMER, _add_scrub_marker, _operator_tile
from ui.scenario_scrubber import build_pulse_scrubber_card
from utils.history_export import history_to_csv_text
from utils.display_units import helion_time_values_seconds
from utils.scenario_scrub import (
    clamp_pulse_index,
    pulse_length,
    pulse_scrub_time_label,
    pulse_scrubber_panel_style,
    slice_pulse_for_scrub,
)


# ---------------------------------------------------------------------------
# Styling constants
# ---------------------------------------------------------------------------

CHART_CONFIG = {
    "displayModeBar": True,
    "displaylogo": False,
    "doubleClick": "reset+autosize",
    "responsive": True,
    "scrollZoom": True,
    "toImageButtonOptions": {
        "format": "png",
        "filename": "helion_pulse",
        "height": 720,
        "width": 1280,
        "scale": 2,
    },
}

COLORS = {
    "temperature": "#f472b6",   # pink
    "fusion": "#38bdf8",        # sky-blue
    "brem": "#fb923c",          # orange
    "cond": "#a78bfa",          # purple
    "energy": "#2dd4bf",        # teal
    "sigma_v": "#facc15",       # yellow
    "tau_e": "#94a3b8",         # slate
    "net": "#4ade80",           # green
    "heat": "#fde047",          # yellow
}

_DARK_TEMPLATE = "plotly_dark"
_PAPER_BG = "#0f172a"
_PLOT_BG = "#1e293b"


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------

def build_helion_panel() -> dbc.Card:
    """Build the Phase 2 Helion FRC interactive control panel."""
    return dbc.Card(
        dbc.CardBody(
            [
                dcc.Store(id="helion-pulse-store"),
                dcc.Store(id="helion-sweep-store"),
                dcc.Store(
                    id="helion-materials-store",
                    data=initial_helion_materials_payload(),
                ),

                # ── Header ──────────────────────────────────────────────────
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                html.Div("Phase 2 Digital Twin", className="eyebrow"),
                                html.H3("Helion FRC Control Panel", className="card-title mb-2"),
                                html.P(
                                    "Configure D-He3 fusion parameters, run a microsecond-scale "
                                    "compression pulse, and map the ignition boundary.",
                                    className="card-text text-secondary-emphasis mb-0",
                                ),
                            ],
                            lg=8,
                        ),
                        dbc.Col(
                            dbc.Alert(DISCLAIMER, color="warning", className="disclaimer-alert mb-0"),
                            lg=4,
                        ),
                    ],
                    className="g-4 align-items-start mb-4",
                ),

                # ── Status alert ─────────────────────────────────────────────
                dbc.Alert(
                    "Helion session ready. Set parameters and run a pulse.",
                    id="helion-status-alert",
                    color="info",
                    className="status-alert mb-4",
                ),

                # ── Operator view: machine diagram + slow-mo phase trace ─────
                dbc.Row([
                    dbc.Col(
                        dbc.Card(
                            dbc.CardBody([
                                html.Div(
                                    id="helion-diagram-container",
                                    children=build_helion_diagram(),
                                ),
                                html.Div(
                                    id="helion-phase-label",
                                    className="helion-phase-label",
                                ),
                            ]),
                            className="operator-card",
                        ),
                        xl=7, lg=12, className="mb-3",
                    ),
                    dbc.Col(
                        [
                            dbc.Card(
                                dbc.CardBody([
                                    dcc.Graph(
                                        id="helion-phase-trace",
                                        figure=helion_pulse_phase_figure([], [], []),
                                        config={"displaylogo": False},
                                    ),
                                ]),
                                className="operator-card mb-3",
                            ),
                            html.Div(
                                id="helion-operator-tiles",
                                className="operator-tiles",
                            ),
                        ],
                        xl=5, lg=12, className="mb-3",
                    ),
                ], className="g-3 mb-3"),

                # ── Materials & remaining-life drawer ────────────────────────
                dbc.Row([
                    dbc.Col(build_damage_panel("helion"), lg=12, className="mb-3"),
                ], className="g-3 mb-4"),

                # ── Controls + Charts ─────────────────────────────────────────
                dbc.Row(
                    [
                        # Controls column
                        dbc.Col(
                            _build_controls_column(),
                            xl=3,
                            lg=4,
                            className="pe-xl-4",
                        ),
                        # Charts column
                        dbc.Col(
                            _build_charts_column(),
                            xl=9,
                            lg=8,
                        ),
                    ],
                    className="g-4",
                ),
            ]
        ),
        className="shadow-sm",
    )


def register_helion_callbacks(app: Dash) -> None:
    """Register all Helion panel Dash callbacks with the app."""
    _register_pulse_callback(app)
    _register_scenario_callback(app)
    _register_helion_operator_view_callback(app)
    _register_sweep_callback(app)
    _register_chart_update_callback(app)
    _register_heatmap_callback(app)
    _register_helion_csv_callbacks(app)


# ---------------------------------------------------------------------------
# Layout helpers — controls
# ---------------------------------------------------------------------------

def _build_controls_column() -> list:
    return [
        # ── Mode selector ────────────────────────────────────────────────────
        html.H5("Simulation Mode", className="section-label mt-2 mb-3"),
        dbc.RadioItems(
            id="helion-mode",
            options=[
                {"label": "Custom Pulse", "value": "custom"},
                {"label": "Predefined Scenario", "value": "scenario"},
            ],
            value="custom",
            inline=True,
            className="mb-3",
        ),

        # ── Scenario picker (shown in scenario mode) ──────────────────────
        html.Div(
            [
                _input_label("Scenario", "", "helion-scenario-tip",
                             "Choose a predefined operating point.",
                             html_for="helion-scenario-select"),
                dcc.Dropdown(
                    id="helion-scenario-select",
                    options=helion_scenario_options(),
                    value="baseline_pulse",
                    clearable=False,
                    className="dark-dropdown mb-3",
                ),
            ],
            id="helion-scenario-row",
            style={"display": "none"},
        ),

        # ── Custom parameter inputs ───────────────────────────────────────
        html.Div(
            id="helion-custom-inputs",
            children=[
                _input_label("Initial Temperature", "keV",
                             "helion-tip-temp",
                             "Pre-compression plasma ion temperature. Bosch-Hale valid range: 0.5–190 keV.",
                             html_for="helion-input-temp"),
                dbc.Input(id="helion-input-temp", type="number",
                          value=20.0, min=0.5, max=190.0, step=0.5,
                          className="mb-3"),

                _input_label("Plasma Density", "×10²¹ m⁻³",
                             "helion-tip-density",
                             "Pre-compression ion number density. Valid range: 1e19–1e23 m⁻³.",
                             html_for="helion-input-density"),
                dbc.Input(id="helion-input-density", type="number",
                          value=1.0, min=0.01, max=100.0, step=0.1,
                          className="mb-3"),

                _input_label("Compression Ratio Rᶜ", "",
                             "helion-tip-rc",
                             "Adiabatic compression ratio. T_final = T_initial × Rᶜ^(2/3).",
                             html_for="helion-input-rc"),
                dbc.Input(id="helion-input-rc", type="number",
                          value=10.0, min=1.0, max=1000.0, step=1.0,
                          className="mb-3"),

                _input_label("Magnetic Field", "T",
                             "helion-tip-bfield",
                             "Applied magnetic field. Sets empirical confinement time τ_E.",
                             html_for="helion-input-bfield"),
                dbc.Input(id="helion-input-bfield", type="number",
                          value=5.0, min=0.1, max=100.0, step=0.5,
                          className="mb-3"),

                _input_label("Plasma Volume", "m³",
                             "helion-tip-volume",
                             "Pre-compression plasma volume.",
                             html_for="helion-input-volume"),
                dbc.Input(id="helion-input-volume", type="number",
                          value=1.0, min=0.01, max=100.0, step=0.1,
                          className="mb-3"),

                _input_label("Pulse Duration", "μs",
                             "helion-tip-pulse",
                             "Simulated pulse length. 1–10,000 μs (0.0000001–0.01 s).",
                             html_for="helion-input-pulse"),
                dbc.Input(id="helion-input-pulse", type="number",
                          value=10.0, min=0.1, max=10000.0, step=1.0,
                          className="mb-3"),
            ],
        ),

        # ── Run buttons ───────────────────────────────────────────────────
        dbc.Button(
            "Run Pulse",
            id="helion-run-btn",
            color="primary",
            className="w-100 mb-2",
        ),
        dbc.Checklist(
            options=[{"label": " τ_E uncertainty band on sweep (0.3×–3.0×)", "value": "tau_unc"}],
            value=[],
            id="helion-tau-uncertainty",
            switch=True,
            className="mb-2",
        ),
        dbc.Button(
            "Compute Ignition Map",
            id="helion-sweep-btn",
            color="secondary",
            outline=True,
            className="w-100 mb-2",
        ),
        dbc.Button(
            "Download CSV",
            id="helion-save-csv",
            color="secondary",
            outline=True,
            className="w-100 mb-4",
        ),
        dcc.Download(id="helion-download-csv"),
        build_pulse_scrubber_card("helion"),

        # ── Metrics tiles ─────────────────────────────────────────────────
        html.H5("Last Pulse Results", className="section-label mb-3"),
        dbc.Row(
            [
                dbc.Col(_metric_card("Q Factor", "—", "helion-metric-q"), xs=6),
                dbc.Col(_metric_card("T_final", "—", "helion-metric-tfinal"), xs=6),
            ],
            className="g-2 mb-2",
        ),
        dbc.Row(
            [
                dbc.Col(_metric_card("P_fusion peak", "—", "helion-metric-pfusion"), xs=6),
                dbc.Col(_metric_card("W_initial", "—", "helion-metric-winitial"), xs=6),
            ],
            className="g-2 mb-2",
        ),
        html.Div(id="helion-pulse-summary", className="small text-secondary-emphasis mb-3"),
        html.H6("Integrated Pulse Energies", className="text-secondary-emphasis mt-3 mb-2 small"),
        dbc.Row(
            [
                dbc.Col(_metric_card("E_fusion", "—", "helion-metric-efusion"), xs=12),
            ],
            className="g-2 mb-1",
        ),
        dbc.Row(
            [
                dbc.Col(_metric_card("E_brem", "—", "helion-metric-ebrem"), xs=6),
                dbc.Col(_metric_card("E_cond", "—", "helion-metric-econd"), xs=6),
            ],
            className="g-2",
        ),
    ]


def _build_charts_column() -> list:
    return [
        dbc.Row(
            [
                dbc.Col(
                    _graph_card(
                        "Plasma Temperature",
                        "Post-compression temperature vs. time",
                        "helion-chart-temp",
                        _empty_temp_figure(),
                    ),
                    lg=6,
                ),
                dbc.Col(
                    _graph_card(
                        "Power Balance",
                        "Fusion power vs. loss channels",
                        "helion-chart-power",
                        _empty_power_figure(),
                    ),
                    lg=6,
                ),
            ],
            className="g-4 mb-4",
        ),
        dbc.Row(
            [
                dbc.Col(
                    _graph_card(
                        "Plasma Energy",
                        "Total plasma thermal energy W(t)",
                        "helion-chart-energy",
                        _empty_energy_figure(),
                    ),
                    lg=6,
                ),
                dbc.Col(
                    _graph_card(
                        "Fusion Reactivity & Confinement",
                        "Bosch-Hale ⟨σv⟩ and empirical τ_E (SPEC R-02)",
                        "helion-chart-reactivity",
                        _empty_reactivity_figure(),
                    ),
                    lg=6,
                ),
            ],
            className="g-4 mb-4",
        ),
        dbc.Row(
            [
                dbc.Col(
                    _graph_card(
                        "Instantaneous Q(t) & Brem/Fusion Ratio",
                        "Q(t) = P_fusion / losses; ratio shows Bremsstrahlung dominance below ~30 keV",
                        "helion-chart-q-time",
                        _empty_q_time_figure(),
                    ),
                    lg=6,
                ),
                dbc.Col(
                    _graph_card(
                        "Net Power dW/dt",
                        "SPEC Eq. 16: P_heat + P_fusion − P_brem − P_cond",
                        "helion-chart-net-power",
                        _empty_net_power_figure(),
                    ),
                    lg=6,
                ),
            ],
            className="g-4 mb-4",
        ),
        dbc.Row(
            [
                dbc.Col(
                    _graph_card(
                        "Compression: Pre vs Post",
                        "Adiabatic compression jump (T, n, V) from pulse parameters",
                        "helion-chart-compression",
                        _empty_compression_figure(),
                    ),
                    lg=6,
                ),
                dbc.Col(
                    _graph_card(
                        "Electron Density n_e",
                        "n_e = 1.5 × n_ions (D-He3 50/50 mix, SPEC Eq. 20)",
                        "helion-chart-n-e",
                        _empty_n_e_figure(),
                    ),
                    lg=6,
                ),
            ],
            className="g-4 mb-4",
        ),
        dbc.Row(
            [
                dbc.Col(
                    _graph_card(
                        "Ignition Boundary",
                        "Q factor map — compression ratio × density",
                        "helion-chart-heatmap",
                        _empty_heatmap_figure(),
                    ),
                    lg=12,
                ),
            ],
            className="g-4",
        ),
    ]


# ---------------------------------------------------------------------------
# Layout helpers — widgets
# ---------------------------------------------------------------------------

def _input_label(
    label: str,
    units: str,
    tooltip_id: str,
    tooltip_text: str,
    *,
    html_for: str,
) -> html.Div:
    badge = (
        dbc.Badge(units, color="secondary", className="ms-1 fw-normal")
        if units
        else None
    )
    children = [dbc.Label([label, badge], html_for=html_for, className="fw-semibold")]
    if tooltip_text:
        children.append(
            dbc.Tooltip(tooltip_text, target=tooltip_id, placement="right")
        )
    return html.Div(children, className="d-flex align-items-center mb-1")


def _metric_card(label: str, initial_value: str, card_id: str) -> dbc.Card:
    return dbc.Card(
        dbc.CardBody(
            [
                html.P(label, className="metric-label mb-0"),
                html.P(initial_value, id=card_id, className="metric-value mb-0"),
            ]
        ),
        className="metric-card",
    )


def _graph_card(title: str, subtitle: str, graph_id: str, figure: go.Figure) -> dbc.Card:
    return dbc.Card(
        [
            dbc.CardHeader(
                [
                    html.Span(title, className="fw-semibold"),
                    html.Small(subtitle, className="text-secondary ms-2"),
                ]
            ),
            dbc.CardBody(
                dcc.Graph(
                    id=graph_id,
                    figure=figure,
                    config=CHART_CONFIG,
                    className="chart-graph",
                )
            ),
        ],
        className="chart-card",
    )


# ---------------------------------------------------------------------------
# Empty figure skeletons (rendered on page load before first pulse)
# ---------------------------------------------------------------------------

def _base_layout(title: str, yaxis_title: str, xaxis_title: str = "Time (μs)") -> dict:
    return {
        "template": _DARK_TEMPLATE,
        "paper_bgcolor": _PAPER_BG,
        "plot_bgcolor": _PLOT_BG,
        "title": {"text": title, "font": {"size": 13}},
        "xaxis": {"title": xaxis_title, "gridcolor": "#334155", "zeroline": False},
        "yaxis": {"title": yaxis_title, "gridcolor": "#334155", "zeroline": False},
        "legend": {"bgcolor": "rgba(0,0,0,0)", "font": {"size": 11}},
        "margin": {"l": 60, "r": 20, "t": 40, "b": 50},
        "height": 280,
        "uirevision": "helion",
    }


def _empty_temp_figure() -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[], y=[], name="T (keV)",
                             line={"color": COLORS["temperature"], "width": 2}))
    fig.update_layout(_base_layout("Plasma Temperature", "Temperature (keV)"))
    return fig


def _empty_power_figure() -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[], y=[], name="P_fusion",
                             line={"color": COLORS["fusion"], "width": 2}))
    fig.add_trace(go.Scatter(x=[], y=[], name="P_brem",
                             line={"color": COLORS["brem"], "width": 2, "dash": "dash"}))
    fig.add_trace(go.Scatter(x=[], y=[], name="P_cond",
                             line={"color": COLORS["cond"], "width": 2, "dash": "dot"}))
    fig.update_layout(_base_layout("Power Balance", "Power (W)"))
    return fig


def _empty_energy_figure() -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[], y=[], name="W (J)",
                             line={"color": COLORS["energy"], "width": 2}))
    fig.update_layout(_base_layout("Plasma Energy W(t)", "Energy (J)"))
    return fig


def _empty_reactivity_figure() -> go.Figure:
    from plotly.subplots import make_subplots

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Scatter(x=[], y=[], name="⟨σv⟩ (m³/s)",
                   line={"color": COLORS["sigma_v"], "width": 2}),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(x=[], y=[], name="τ_E (s)",
                   line={"color": COLORS["tau_e"], "width": 2, "dash": "dot"}),
        secondary_y=True,
    )
    layout = _base_layout("Fusion Reactivity & τ_E", "⟨σv⟩ (m³/s)")
    fig.update_layout(layout)
    fig.update_yaxes(title_text="τ_E (s)", secondary_y=True, type="log")
    return fig


def _empty_compression_figure() -> go.Figure:
    fig = go.Figure()
    fig.update_layout(_base_layout("Compression", "Value", "Parameter"))
    return fig


def _compression_figure(params: dict[str, Any]) -> go.Figure:
    """Grouped bar chart comparing pre- and post-compression state."""
    rc = max(float(params.get("compression_ratio", 1.0)), 1.0)
    t_pre = float(params.get("ion_temperature_kev", 0.0))
    t_post = t_pre * rc ** (2.0 / 3.0)
    n_post = float(params.get("plasma_density_m3", 0.0))
    n_pre = n_post / rc
    v_post = float(params.get("plasma_volume_m3", 0.0))
    v_pre = v_post / rc
    fig = go.Figure(
        data=[
            go.Bar(name="Pre", x=["T (keV)", "n (×10²¹ m⁻³)", "V (m³)"], y=[t_pre, n_pre / 1e21, v_pre], marker_color="#94a3b8"),
            go.Bar(name="Post", x=["T (keV)", "n (×10²¹ m⁻³)", "V (m³)"], y=[t_post, n_post / 1e21, v_post], marker_color=COLORS["fusion"]),
        ]
    )
    fig.update_layout(
        template=_DARK_TEMPLATE,
        paper_bgcolor=_PAPER_BG,
        plot_bgcolor=_PLOT_BG,
        barmode="group",
        title="Pre- vs Post-Compression",
        yaxis_title="Value",
        height=360,
        legend=dict(orientation="h", y=1.1),
    )
    return fig


def _empty_n_e_figure() -> go.Figure:
    fig = go.Figure(
        go.Scatter(
            x=[], y=[], name="n_e (m⁻³)",
            line={"color": COLORS["cond"], "width": 2},
        )
    )
    fig.update_layout(_base_layout("Electron Density n_e", "n_e (m⁻³)"))
    fig.update_yaxes(type="log")
    return fig


def _empty_net_power_figure() -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[], y=[], name="dW/dt (W)",
                             line={"color": COLORS["net"], "width": 2}))
    fig.add_hline(y=0, line={"color": "#475569", "width": 1, "dash": "dash"})
    fig.update_layout(_base_layout("Net Power dW/dt", "Power (W)"))
    return fig


def _empty_q_time_figure() -> go.Figure:
    from plotly.subplots import make_subplots as _msp
    fig = _msp(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Scatter(x=[], y=[], name="Q(t)",
                   line={"color": COLORS["fusion"], "width": 2}),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(x=[], y=[], name="P_brem / P_fusion",
                   line={"color": COLORS["brem"], "width": 2, "dash": "dash"}),
        secondary_y=True,
    )
    layout = _base_layout("Q(t) & Bremsstrahlung Ratio", "Q(t) = P_fusion / losses")
    fig.update_layout(layout)
    fig.update_yaxes(title_text="P_brem / P_fusion", secondary_y=True)
    return fig


def _empty_heatmap_figure() -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(
        text="Click 'Compute Ignition Map' to generate",
        xref="paper", yref="paper",
        x=0.5, y=0.5,
        showarrow=False,
        font={"color": "#94a3b8", "size": 12},
    )
    fig.update_layout({
        "template": _DARK_TEMPLATE,
        "paper_bgcolor": _PAPER_BG,
        "plot_bgcolor": _PLOT_BG,
        "height": 280,
        "margin": {"l": 60, "r": 20, "t": 40, "b": 50},
        "uirevision": "helion-sweep",
    })
    return fig


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------

def _register_pulse_callback(app: Dash) -> None:
    """Run pulse on button click (custom mode only)."""

    @app.callback(
        Output("helion-pulse-store", "data"),
        Output("helion-status-alert", "children"),
        Output("helion-status-alert", "color"),
        Output("helion-materials-store", "data"),
        Input("helion-run-btn", "n_clicks"),
        State("helion-mode", "value"),
        State("helion-input-temp", "value"),
        State("helion-input-density", "value"),
        State("helion-input-rc", "value"),
        State("helion-input-bfield", "value"),
        State("helion-input-volume", "value"),
        State("helion-input-pulse", "value"),
        State("helion-materials-store", "data"),
        prevent_initial_call=True,
    )
    def _run_pulse(
        n_clicks: int | None,
        mode: str,
        temp_kev: float | None,
        density_1e21: float | None,
        rc: float | None,
        bfield_t: float | None,
        volume_m3: float | None,
        pulse_us: float | None,
        materials_in: dict[str, Any] | None,
    ) -> tuple:
        if mode != "custom":
            return no_update, no_update, no_update, no_update

        raw = {
            "ion_temperature_kev": temp_kev,
            "plasma_density_m3": (density_1e21 or 0) * 1e21,
            "compression_ratio": rc,
            "magnetic_field_t": bfield_t,
            "plasma_volume_m3": volume_m3,
            "pulse_duration_s": (pulse_us or 0) * 1e-6,
        }
        result = run_helion_pulse(raw, materials_in=materials_in)
        if result["success"]:
            q = result["Q"]
            regime = "Q > 1 — ignition achieved!" if q > 1.0 else f"Q = {q:.4g} (sub-ignition)"
            return result, f"Pulse complete. {regime}", "success", result.get("materials", no_update)
        return result, f"Pulse failed: {result['message']}", "danger", no_update


def _register_scenario_callback(app: Dash) -> None:
    """Run scenario pulse + toggle custom/scenario input visibility."""

    @app.callback(
        Output("helion-pulse-store", "data", allow_duplicate=True),
        Output("helion-status-alert", "children", allow_duplicate=True),
        Output("helion-status-alert", "color", allow_duplicate=True),
        Output("helion-scenario-row", "style"),
        Output("helion-custom-inputs", "style"),
        Output("helion-materials-store", "data", allow_duplicate=True),
        Input("helion-run-btn", "n_clicks"),
        Input("helion-mode", "value"),
        State("helion-scenario-select", "value"),
        State("helion-materials-store", "data"),
        prevent_initial_call=True,
    )
    def _run_or_toggle(
        n_clicks: int | None,
        mode: str,
        scenario_id: str | None,
        materials_in: dict[str, Any] | None,
    ) -> tuple:
        scenario_style = {"display": "block" if mode == "scenario" else "none"}
        custom_style = {"display": "block" if mode == "custom" else "none"}

        if mode != "scenario" or not n_clicks:
            return no_update, no_update, no_update, scenario_style, custom_style, no_update

        result = run_helion_scenario(scenario_id or "baseline_pulse", materials_in=materials_in)
        if result["success"]:
            q = result["Q"]
            label = result.get("scenario_label", scenario_id)
            regime = "Q > 1 — ignition!" if q > 1.0 else f"Q = {q:.4g}"
            msg = f"Scenario '{label}' complete. {regime}"
            return (
                result, msg, "success", scenario_style, custom_style,
                result.get("materials", no_update),
            )
        return (
            result, f"Scenario failed: {result['message']}", "danger",
            scenario_style, custom_style, no_update,
        )


def _register_sweep_callback(app: Dash) -> None:
    """Compute ignition boundary sweep on button click."""

    @app.callback(
        Output("helion-sweep-store", "data"),
        Output("helion-status-alert", "children", allow_duplicate=True),
        Output("helion-status-alert", "color", allow_duplicate=True),
        Input("helion-sweep-btn", "n_clicks"),
        State("helion-mode", "value"),
        State("helion-input-temp", "value"),
        State("helion-input-density", "value"),
        State("helion-input-rc", "value"),
        State("helion-input-bfield", "value"),
        State("helion-input-volume", "value"),
        State("helion-input-pulse", "value"),
        State("helion-scenario-select", "value"),
        State("helion-tau-uncertainty", "value"),
        prevent_initial_call=True,
    )
    def _run_sweep(
        n_clicks: int | None,
        mode: str,
        temp_kev: float | None,
        density_1e21: float | None,
        rc: float | None,
        bfield_t: float | None,
        volume_m3: float | None,
        pulse_us: float | None,
        scenario_id: str | None,
        tau_uncertainty: list[str] | None,
    ) -> tuple:
        if mode == "scenario":
            import orchestrator.helion_scenarios as _hs
            try:
                sc = _hs.get_helion_scenario(scenario_id or "baseline_pulse")
                base_raw = {**sc.parameters.model_dump()}
            except ValueError as exc:
                return no_update, str(exc), "danger"
        else:
            base_raw = {
                "ion_temperature_kev": temp_kev,
                "plasma_density_m3": (density_1e21 or 0) * 1e21,
                "compression_ratio": rc,
                "magnetic_field_t": bfield_t,
                "plasma_volume_m3": volume_m3,
                "pulse_duration_s": (pulse_us or 0) * 1e-6,
            }

        include_unc = bool(tau_uncertainty and "tau_unc" in tau_uncertainty)
        result = run_helion_sweep(base_raw, include_tau_uncertainty=include_unc)
        if result["success"]:
            n_ignited = sum(any(row) for row in result["ignition_mask"])
            msg = (
                f"Sweep complete ({result['n_failed']} points skipped). "
                f"{n_ignited} compression-ratio rows contain Q > 1."
            )
            return result, msg, "success"
        return result, f"Sweep failed: {result['message']}", "danger"


def _register_helion_operator_view_callback(app: Dash) -> None:
    """Update the Helion diagram, phase trace, and materials drawer.

    Reacts to either a fresh pulse-store payload (new pulse just ran) or a
    scrubber position change (user is exploring the pulse in slow-motion).
    """

    @app.callback(
        Output("helion-diagram-container", "children"),
        Output("helion-phase-trace", "figure"),
        Output("helion-phase-label", "children"),
        Output("helion-damage-bars", "children"),
        Output("helion-life-cards", "children"),
        Output("helion-life-summary", "children"),
        Output("helion-operator-tiles", "children"),
        Input("helion-pulse-store", "data"),
        Input("helion-materials-store", "data"),
        Input("helion-scrub-slider", "value"),
    )
    def update_helion_operator_view(
        pulse: dict[str, Any] | None,
        materials: dict[str, Any] | None,
        scrub_idx: int | None,
    ):
        return _build_helion_operator_view(pulse, materials, scrub_idx)


def _build_helion_operator_view(
    pulse: dict[str, Any] | None,
    materials: dict[str, Any] | None,
    scrub_idx: int | None,
):
    if not materials:
        materials = initial_helion_materials_payload()

    mat_state = materials.get("state") or {}
    mat_life = materials.get("remaining_life_years") or {}

    fw_frac = mat_state.get("first_wall_dpa", 0.0) / HELION_DESIGN_LIMITS["first_wall_dpa"]
    coil_frac = mat_state.get("coil_thermal_fatigue_index", 0.0) / HELION_DESIGN_LIMITS[
        "coil_thermal_fatigue_index"
    ]

    if not pulse or not pulse.get("success"):
        diagram_html = html.Div(
            svg_to_img(render_helion_diagram(0, 0, fw_frac, coil_frac, 0.0), alt="Helion FRC"),
            className="reactor-diagram",
        )
        phase_fig = helion_pulse_phase_figure([], [], [])
        phase_label = "Run a pulse to see the slow-motion phase view."
    else:
        t_arr = pulse["t"]
        t_total = float(t_arr[-1]) if t_arr else 0.0
        t_kev_arr = pulse["T_kev"]
        p_fus_arr = pulse["P_fusion_w"]

        # Resolve cursor: scrub_idx into a time
        if scrub_idx is not None and 0 <= scrub_idx < len(t_arr):
            t_now = float(t_arr[scrub_idx])
            t_now_kev = float(t_kev_arr[scrub_idx])
        else:
            t_now = t_total
            t_now_kev = float(t_kev_arr[-1]) if t_kev_arr else 0.0

        diagram_html = html.Div(
            svg_to_img(
                render_helion_diagram(t_now, t_total, fw_frac, coil_frac, t_now_kev),
                alt="Helion FRC",
            ),
            className="reactor-diagram",
        )
        phase_fig = helion_pulse_phase_figure(t_arr, t_kev_arr, p_fus_arr, cursor_s=t_now)
        phase_label = (
            f"Phase: {helion_phase_for_time(t_now, t_total)} "
            f"({t_now * 1e6:.2f} µs of {t_total * 1e6:.2f} µs total) — "
            f"T_ion = {t_now_kev:.1f} keV"
        )

    bars_input = [
        ("First-wall DPA", mat_state.get("first_wall_dpa", 0.0),
         HELION_DESIGN_LIMITS["first_wall_dpa"], "DPA"),
        ("Thermal cycles", mat_state.get("first_wall_thermal_cycles", 0.0),
         HELION_DESIGN_LIMITS["first_wall_thermal_cycles"], "cycles"),
        ("Coil fatigue index", mat_state.get("coil_thermal_fatigue_index", 0.0),
         HELION_DESIGN_LIMITS["coil_thermal_fatigue_index"], ""),
        ("Capacitor cycles", mat_state.get("capacitor_charge_cycles", 0.0),
         HELION_DESIGN_LIMITS["capacitor_charge_cycles"], "cycles"),
        ("Insulator dose", mat_state.get("insulator_dielectric_dose_mgy", 0.0),
         HELION_DESIGN_LIMITS["insulator_dielectric_dose_mgy"], "MGy"),
    ]
    life_input: dict[str, float] = {
        k: (float("inf") if v is None else float(v)) for k, v in mat_life.items()
    }
    life_label_map = {
        "first_wall_dpa_years": "First-wall DPA",
        "thermal_cycle_years": "Thermal cycles",
        "coil_fatigue_years": "Coil fatigue",
        "capacitor_cycle_years": "Capacitor cycles",
        "insulator_dose_years": "Insulator dose",
    }
    bars, cards, summary = update_damage_panel_children(
        bars_input, life_input, life_label_map
    )

    if pulse and pulse.get("success"):
        q = float(pulse.get("Q", 0.0))
        e_fus = float(pulse.get("pulse_fusion_energy_j", 0.0))
    else:
        q = 0.0
        e_fus = 0.0
    pulses_fired = mat_state.get("pulses_fired", 0.0)
    tiles = [
        _operator_tile("Q (gain factor)", f"{q:.3g}",
                       "Q > 1 ⇒ ignition"),
        _operator_tile("Fusion energy this pulse", f"{e_fus / 1e6:.2f} MJ", "integrated P_fusion"),
        _operator_tile("Pulses fired", f"{pulses_fired:,.0f}",
                       "drives capacitor and coil fatigue"),
        _operator_tile("Binding life", summary, ""),
    ]

    return diagram_html, phase_fig, phase_label, bars, cards, summary, tiles


def _register_chart_update_callback(app: Dash) -> None:
    """Update time-series charts and metric tiles from pulse store."""

    @app.callback(
        Output("helion-chart-temp", "figure"),
        Output("helion-chart-power", "figure"),
        Output("helion-chart-energy", "figure"),
        Output("helion-chart-reactivity", "figure"),
        Output("helion-chart-q-time", "figure"),
        Output("helion-chart-net-power", "figure"),
        Output("helion-chart-n-e", "figure"),
        Output("helion-chart-compression", "figure"),
        Output("helion-pulse-summary", "children"),
        Output("helion-metric-q", "children"),
        Output("helion-metric-tfinal", "children"),
        Output("helion-metric-pfusion", "children"),
        Output("helion-metric-winitial", "children"),
        Output("helion-metric-efusion", "children"),
        Output("helion-metric-ebrem", "children"),
        Output("helion-metric-econd", "children"),
        Output("helion-scrubber-panel", "style"),
        Output("helion-scrub-slider", "max"),
        Output("helion-scrub-slider", "value"),
        Output("helion-scrub-time-label", "children"),
        Input("helion-pulse-store", "data"),
        Input("helion-scrub-slider", "value"),
        Input("display-units-store", "data"),
        prevent_initial_call=True,
    )
    def _update_charts(
        data: dict | None,
        scrub_slider: int | None,
        display_units: dict[str, str] | None,
    ) -> tuple:
        if not data or not data.get("success"):
            return (
                *((no_update,) * 8),
                "Run a pulse to see post-compression parameters.",
                "—", "—", "—", "—", "—", "—", "—",
                {"display": "none"},
                0,
                0,
                "",
            )

        n_pts = pulse_length(data)
        if ctx.triggered_id == "helion-pulse-store":
            scrub_idx = n_pts - 1 if n_pts > 0 else 0
        else:
            scrub_idx = clamp_pulse_index(data, scrub_slider)
        view, scrub_idx = slice_pulse_for_scrub(data, scrub_idx)
        scrub_at_end = scrub_idx >= n_pts - 1

        t_plot, t_axis_title = helion_time_values_seconds(view["t"], display_units)
        T_kev = view["T_kev"]
        P_fusion = view["P_fusion_w"]
        P_brem = view["P_brem_w"]
        P_cond = view["P_cond_w"]
        W_J = view["W_J"]
        net_power = view.get("net_power_w", [])
        p_heat = float(view.get("p_heat_w", 0.0))
        sigma_v = view.get("sigma_v_m3_s", [])
        tau_e = view.get("tau_e_s", [])
        params = view.get("parameters") or data.get("parameters") or {}

        # Temperature figure
        temp_fig = go.Figure(
            go.Scatter(x=t_plot, y=T_kev, name="T (keV)",
                       line={"color": COLORS["temperature"], "width": 2, "shape": "spline"})
        )
        temp_fig.update_layout(_base_layout("Plasma Temperature", "Temperature (keV)", t_axis_title))
        if not scrub_at_end and t_plot:
            _add_scrub_marker(temp_fig, t_plot[-1])

        # Power figure
        power_traces = [
            go.Scatter(x=t_plot, y=P_fusion, name="P_fusion (W)",
                       line={"color": COLORS["fusion"], "width": 2}),
            go.Scatter(x=t_plot, y=P_brem, name="P_brem (W)",
                       line={"color": COLORS["brem"], "width": 2, "dash": "dash"}),
            go.Scatter(x=t_plot, y=P_cond, name="P_cond (W)",
                       line={"color": COLORS["cond"], "width": 2, "dash": "dot"}),
        ]
        if p_heat > 0.0:
            power_traces.append(
                go.Scatter(
                    x=t_plot,
                    y=[p_heat] * len(t_plot),
                    name="P_heat (W)",
                    line={"color": COLORS["heat"], "width": 2, "dash": "longdash"},
                )
            )
        power_fig = go.Figure(power_traces)
        power_fig.update_layout(_base_layout("Power Balance", "Power (W)", t_axis_title))
        if not scrub_at_end and t_plot:
            _add_scrub_marker(power_fig, t_plot[-1])

        # Energy figure
        energy_fig = go.Figure(
            go.Scatter(x=t_plot, y=W_J, name="W (J)",
                       line={"color": COLORS["energy"], "width": 2, "shape": "spline"})
        )
        energy_fig.update_layout(_base_layout("Plasma Energy W(t)", "Energy (J)", t_axis_title))
        if not scrub_at_end and t_plot:
            _add_scrub_marker(energy_fig, t_plot[-1])

        from plotly.subplots import make_subplots

        react_fig = make_subplots(specs=[[{"secondary_y": True}]])
        react_fig.add_trace(
            go.Scatter(
                x=t_plot, y=sigma_v, name="⟨σv⟩ (m³/s)",
                line={"color": COLORS["sigma_v"], "width": 2},
            ),
            secondary_y=False,
        )
        react_fig.add_trace(
            go.Scatter(
                x=t_plot, y=tau_e, name="τ_E (s)",
                line={"color": COLORS["tau_e"], "width": 2, "dash": "dot"},
            ),
            secondary_y=True,
        )
        react_fig.update_layout(_base_layout("Fusion Reactivity & τ_E", "⟨σv⟩ (m³/s)", t_axis_title))
        react_fig.update_yaxes(title_text="τ_E (s)", secondary_y=True, type="log")
        if not scrub_at_end and t_plot:
            _add_scrub_marker(react_fig, t_plot[-1])

        # Q(t) and brem/fusion ratio figure
        import numpy as np
        t_s = view["t"]
        q_time_fig = make_subplots(specs=[[{"secondary_y": True}]])
        q_t = [
            pf / max(pb + pc, 1e-30)
            for pf, pb, pc in zip(P_fusion, P_brem, P_cond)
        ]
        brem_ratio = [
            pb / max(pf, 1e-30)
            for pf, pb in zip(P_fusion, P_brem)
        ]
        q_time_fig.add_trace(
            go.Scatter(x=t_plot, y=q_t, name="Q(t)",
                       line={"color": COLORS["fusion"], "width": 2}),
            secondary_y=False,
        )
        q_time_fig.add_trace(
            go.Scatter(x=t_plot, y=brem_ratio, name="P_brem / P_fusion",
                       line={"color": COLORS["brem"], "width": 2, "dash": "dash"}),
            secondary_y=True,
        )
        q_time_fig.update_layout(_base_layout("Q(t) & Bremsstrahlung Ratio", "Q(t)", t_axis_title))
        q_time_fig.add_hline(y=1.0, line={"color": "white", "width": 1, "dash": "dot"},
                             annotation_text="Q = 1", secondary_y=False)
        q_time_fig.update_yaxes(title_text="P_brem / P_fusion", secondary_y=True)
        if not scrub_at_end and t_plot:
            _add_scrub_marker(q_time_fig, t_plot[-1])

        net_fig = go.Figure(
            go.Scatter(
                x=t_plot,
                y=net_power,
                name="dW/dt (W)",
                line={"color": COLORS["net"], "width": 2},
            )
        )
        net_fig.add_hline(y=0, line={"color": "#475569", "width": 1, "dash": "dash"})
        net_fig.update_layout(_base_layout("Net Power dW/dt", "Power (W)", t_axis_title))
        if not scrub_at_end and t_plot:
            _add_scrub_marker(net_fig, t_plot[-1])

        n_e = view.get("n_e_m3", [])
        n_e_fig = go.Figure(
            go.Scatter(
                x=t_plot,
                y=n_e,
                name="n_e (m⁻³)",
                line={"color": COLORS["cond"], "width": 2},
            )
        )
        n_e_fig.update_layout(_base_layout("Electron Density n_e", "n_e (m⁻³)", t_axis_title))
        n_e_fig.update_yaxes(type="log")
        if not scrub_at_end and t_plot:
            _add_scrub_marker(n_e_fig, t_plot[-1])

        compression_fig = _compression_figure(params)

        n_pre = float(params.get("plasma_density_m3", 0.0)) / max(
            float(params.get("compression_ratio", 1.0)), 1.0
        )
        summary = html.Dl(
            [
                html.Dt("Pre-compression"),
                html.Dd(
                    f"Tᵢ = {params.get('ion_temperature_kev', '—')} keV, "
                    f"nᵢ = {n_pre:.2e} m⁻³, "
                    f"Vᵢ = {float(params.get('plasma_volume_m3', 0)) / max(float(params.get('compression_ratio', 1)), 1):.2e} m³"
                ),
                html.Dt("Post-compression (pulse)"),
                html.Dd(
                    f"Rᶜ = {params.get('compression_ratio', '—')}, "
                    f"n = {params.get('plasma_density_m3', '—'):.2e} m⁻³, "
                    f"B = {params.get('magnetic_field_t', '—')} T, "
                    f"Δt = {float(params.get('pulse_duration_s', 0)) * 1e6:.1f} μs"
                ),
            ],
            className="row mb-0 small",
        )
        if data.get("scenario_label"):
            summary = html.Div([
                dbc.Alert(
                    f"Scenario: {data['scenario_label']}",
                    color="info",
                    className="py-1 mb-2 small",
                ),
                summary,
            ])

        # Integrated pulse energies via trapezoid rule
        t_arr = np.array(t_s, dtype=float)
        e_fusion = float(np.trapezoid(P_fusion, t_arr)) if len(t_arr) > 1 else 0.0
        e_brem   = float(np.trapezoid(P_brem,   t_arr)) if len(t_arr) > 1 else 0.0
        e_cond   = float(np.trapezoid(P_cond,   t_arr)) if len(t_arr) > 1 else 0.0

        def _fmt_energy(e: float) -> str:
            if abs(e) >= 1e3:
                return f"{e / 1e3:.3g} kJ"
            if abs(e) >= 1.0:
                return f"{e:.3g} J"
            return f"{e * 1e3:.3g} mJ"

        # Metric values (at scrubbed time)
        q_val = f"{data['Q']:.4g}"
        t_final = f"{T_kev[-1]:.1f} keV" if T_kev else "—"
        p_fusion_peak = f"{max(P_fusion):.3g} W" if P_fusion else "—"
        w_initial = f"{data['W_initial_J']:.3g} J"

        if n_pts > 1:
            scrub_style = pulse_scrubber_panel_style(data)
            scrub_max = n_pts - 1
            scrub_val = scrub_idx
            scrub_label = pulse_scrub_time_label(data, scrub_idx)
        else:
            scrub_style = {"display": "none"}
            scrub_max = 0
            scrub_val = 0
            scrub_label = ""

        return (
            temp_fig, power_fig, energy_fig, react_fig, q_time_fig, net_fig, n_e_fig,
            compression_fig,
            summary,
            q_val, t_final, p_fusion_peak, w_initial,
            _fmt_energy(e_fusion), _fmt_energy(e_brem), _fmt_energy(e_cond),
            scrub_style,
            scrub_max,
            scrub_val,
            scrub_label,
        )


def _helion_pulse_history(data: dict[str, Any]) -> dict[str, list[float]]:
    """Flatten pulse store payload into a time-aligned history dict for CSV export."""
    return {
        "t_s": list(data.get("t", [])),
        "T_kev": list(data.get("T_kev", [])),
        "W_J": list(data.get("W_J", [])),
        "P_fusion_w": list(data.get("P_fusion_w", [])),
        "P_brem_w": list(data.get("P_brem_w", [])),
        "P_cond_w": list(data.get("P_cond_w", [])),
        "net_power_w": list(data.get("net_power_w", [])),
        "sigma_v_m3_s": list(data.get("sigma_v_m3_s", [])),
        "tau_e_s": list(data.get("tau_e_s", [])),
        "n_e_m3": list(data.get("n_e_m3", [])),
    }


def _register_helion_csv_callbacks(app: Dash) -> None:
    @app.callback(
        Output("helion-download-csv", "data"),
        Input("helion-save-csv", "n_clicks"),
        State("helion-pulse-store", "data"),
        prevent_initial_call=True,
    )
    def _download_helion_csv(n_clicks: int | None, pulse: dict[str, Any] | None) -> Any:
        if not n_clicks or not pulse or not pulse.get("success"):
            return no_update
        history = _helion_pulse_history(pulse)
        if not history.get("t_s"):
            return no_update
        return {
            "content": history_to_csv_text(history),
            "filename": "helion_pulse.csv",
            "type": "text/csv",
        }


def _register_heatmap_callback(app: Dash) -> None:
    """Render ignition boundary heatmap from sweep store."""

    @app.callback(
        Output("helion-chart-heatmap", "figure"),
        Input("helion-sweep-store", "data"),
        prevent_initial_call=True,
    )
    def _update_heatmap(data: dict | None) -> go.Figure:
        if not data or not data.get("success"):
            return no_update

        import math
        rc = data["compression_ratios"]
        n_m3 = data["densities_m3"]
        Q_map = data["Q_map"]

        # Log₁₀(Q) for colour scale; clip to [-6, 2]
        log_Q = [
            [max(-6.0, min(2.0, math.log10(q) if q > 0 else -6.0)) for q in row]
            for row in Q_map
        ]

        # Annotate ignition boundary (Q=1 → log Q = 0) with a contour line
        fig = go.Figure()
        fig.add_trace(go.Heatmap(
            x=[math.log10(n) for n in n_m3],
            y=[math.log10(r) for r in rc],
            z=log_Q,
            colorscale=[
                [0.0, "#1e293b"],
                [0.3, "#312e81"],
                [0.5, "#7c3aed"],
                [0.7, "#db2777"],
                [1.0, "#fbbf24"],
            ],
            zmin=-6,
            zmax=2,
            colorbar={
                "title": "log₁₀(Q)",
                "tickvals": [-6, -4, -2, 0, 2],
                "ticktext": ["10⁻⁶", "10⁻⁴", "10⁻²", "Q=1", "10²"],
                "len": 0.85,
            },
            hovertemplate=(
                "log₁₀(n) = %{x:.2f}<br>"
                "log₁₀(Rᶜ) = %{y:.2f}<br>"
                "log₁₀(Q) = %{z:.2f}<extra></extra>"
            ),
        ))

        mask = data.get("ignition_mask")
        if mask:
            fig.add_trace(go.Heatmap(
                x=[math.log10(n) for n in n_m3],
                y=[math.log10(r) for r in rc],
                z=[[1.0 if cell else 0.0 for cell in row] for row in mask],
                colorscale=[[0.0, "rgba(0,0,0,0)"], [1.0, "rgba(74,222,128,0.22)"]],
                showscale=False,
                hoverinfo="skip",
                name="Ignited (Q>1)",
            ))

        # Ignition contour at Q=1 (log Q=0)
        fig.add_trace(go.Contour(
            x=[math.log10(n) for n in n_m3],
            y=[math.log10(r) for r in rc],
            z=log_Q,
            contours={"value": 0, "type": "constraint", "operation": "="},
            line={"color": "white", "width": 2, "dash": "dash"},
            showscale=False,
            name="Q = 1 (nominal τ_E)",
            hoverinfo="skip",
        ))

        if data.get("tau_uncertainty"):
            rc_u = data.get("compression_ratios_unc", rc)
            n_u = data.get("densities_m3_unc", n_m3)
            x_u = [math.log10(n) for n in n_u]
            y_u = [math.log10(r) for r in rc_u]
            for qmap, color, label in (
                (data.get("Q_map_tau_lo", []), "#22d3ee", "Q=1 (τ×0.3)"),
                (data.get("Q_map_tau_hi", []), "#f87171", "Q=1 (τ×3)"),
            ):
                if not qmap:
                    continue
                log_q_u = [
                    [max(-6.0, min(2.0, math.log10(q) if q > 0 else -6.0)) for q in row]
                    for row in qmap
                ]
                fig.add_trace(go.Contour(
                    x=x_u,
                    y=y_u,
                    z=log_q_u,
                    contours={"value": 0, "type": "constraint", "operation": "="},
                    line={"color": color, "width": 1.5, "dash": "dot"},
                    showscale=False,
                    name=label,
                    hoverinfo="skip",
                ))

        title_suffix = (
            "  (cyan/red dotted = Q=1 at τ_E×0.3 / ×3)"
            if data.get("tau_uncertainty")
            else ""
        )
        fig.update_layout({
            "template": _DARK_TEMPLATE,
            "paper_bgcolor": _PAPER_BG,
            "plot_bgcolor": _PLOT_BG,
            "title": {"text": f"Ignition Boundary  (white dashed = Q = 1){title_suffix}", "font": {"size": 13}},
            "xaxis": {
                "title": "log₁₀(n₀ / m⁻³)",
                "gridcolor": "#334155",
                "tickmode": "linear",
                "tick0": 19,
                "dtick": 1,
            },
            "yaxis": {
                "title": "log₁₀(Rᶜ)",
                "gridcolor": "#334155",
                "tickmode": "linear",
                "tick0": 0,
                "dtick": 1,
            },
            "margin": {"l": 60, "r": 20, "t": 45, "b": 55},
            "height": 280,
            "uirevision": "helion-sweep",
        })

        return fig
