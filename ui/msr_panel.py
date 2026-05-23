"""MSR control panel and Dash callbacks.

CURRENT charts (SPEC FR-18): power, salt T, reactivity breakdown, β_eff,flow vs v_salt.
ADDITIONAL: salt flow fraction vs time, β_eff,flow vs time, event log.
See docs/UI_VISUALIZATION.md for the full inventory.

Provides:
  build_msr_panel()          — full interactive layout
  register_msr_callbacks()   — all @app.callback wrappers

Layout:
  Controls column  — live operator inputs (rod reactivity, salt flow) + scenario launcher
  Charts column    — power, salt temperature, reactivity, flow, β(t), β(v) sweep

Interaction model:
  Live mode   — dcc.Interval fires every 200 ms; run_msr_control_step advances 5 s.
  Scenario    — user selects scenario and clicks "Run Scenario"; single batch run.

Architecture:
  All physics is accessed through orchestrator.msr_runner, never directly.
  dcc.Store holds session state (JSON-serializable).
  plotly_dark template + graph_objects (not express) per CLAUDE.md rules.
  Disclaimer shown on panel per SPEC SR-03.
"""

from __future__ import annotations

from typing import Any

from dash import Dash, Input, Output, State, ctx, dcc, html, no_update
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from orchestrator.msr_runner import (
    MSR_MAX_HISTORY_POINTS,
    MSR_STEP_SECONDS,
    run_msr_beta_flow_sweep,
    run_msr_control_step,
    run_msr_scenario,
)
from orchestrator.msr_scenarios import msr_scenario_options
from physics.materials import MSR_DESIGN_LIMITS
from physics.msr.thermal import M_DOT_SALT_NOM, P_NOM_MSR, T_SALT_NOM, TAU_CORE_NOM
from data.keepin_dnp import beta_fractions
from ui.diagrams import (
    build_damage_panel,
    build_msr_diagram,
    msr_loop_heatmap_figure,
    render_msr_diagram,
    update_damage_panel_children,
)
from ui.diagrams.pwr_schematic import svg_to_img
from ui.pwr_panel import DISCLAIMER, _add_scrub_marker, _event_log_items, _operator_tile
from ui.scenario_scrubber import build_scenario_scrubber_card, format_scrub_label
from utils.display_units import (
    format_simulation_time,
    temperature_axis_label,
    temperature_scalar_kelvin,
    temperature_unit_suffix,
    temperature_values_kelvin,
    time_axis_label,
    time_values_seconds,
)
from utils.history_export import history_to_csv_text
from utils.scenario_scrub import (
    clamp_scrub_index,
    history_length,
    scrubber_panel_style,
    slice_session_for_scrub,
)


# ---------------------------------------------------------------------------
# Styling constants
# ---------------------------------------------------------------------------

LIVE_INTERVAL_MS = 200
_DARK_TEMPLATE = "plotly_dark"
_PAPER_BG = "#0f172a"
_PLOT_BG = "#1e293b"
_FONT_COLOR = "#e2e8f0"

COLORS = {
    "power":        "#38bdf8",  # sky-blue
    "salt_temp":    "#f472b6",  # pink
    "rod":          "#a78bfa",  # purple
    "temp_fb":      "#fb923c",  # orange
    "total":        "#f8fafc",  # white
    "beta_curve":   "#2dd4bf",  # teal
    "beta_static":  "#94a3b8",  # slate
    "beta_point":   "#facc15",  # yellow
    "flow":         "#22d3ee",  # cyan
}

_PWR_BETA_STATIC = float(sum(beta_fractions()))

# Delayed precursor group styling (6 Keepin groups)
_PRECURSOR_COLORS = ["#f87171", "#fb923c", "#facc15", "#4ade80", "#60a5fa", "#c084fc"]
_PRECURSOR_LABELS = ["C₁", "C₂", "C₃", "C₄", "C₅", "C₆"]
_PRECURSOR_KEYS = [f"precursor_C{i}" for i in range(1, 7)]

CHART_CONFIG = {
    "displayModeBar": True,
    "displaylogo": False,
    "doubleClick": "reset+autosize",
    "responsive": True,
    "scrollZoom": True,
    "toImageButtonOptions": {
        "format": "png",
        "filename": "msr_timeseries",
        "height": 720,
        "width": 1280,
        "scale": 2,
    },
}

TIME_MULTIPLIER_OPTIONS = [
    {"label": "1× real-time", "value": 1},
    {"label": "60× accelerated", "value": 60},
    {"label": "3600× xenon timescale", "value": 3600},
]


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------


def build_msr_panel() -> dbc.Card:
    """Build the Phase 3 MSR interactive control panel."""
    return dbc.Card(
        dbc.CardBody([
            dcc.Store(id="msr-session-store"),
            dcc.Store(id="msr-run-state-store", data={"paused": False}),
            dcc.Interval(id="msr-step-interval", interval=LIVE_INTERVAL_MS, n_intervals=0),

            # ── Header ────────────────────────────────────────────────────────
            dbc.Row([
                dbc.Col([
                    html.Div("Phase 3 Digital Twin", className="eyebrow"),
                    html.H3("Molten Salt Reactor Control Panel", className="card-title mb-2"),
                    html.P(
                        "Monitor MSR point kinetics with precursor drift, salt thermal "
                        "response, and β_eff,flow reduction as a function of salt velocity.",
                        className="card-text text-secondary-emphasis mb-0",
                    ),
                ], lg=8),
                dbc.Col(
                    dbc.Alert(DISCLAIMER, color="warning", className="disclaimer-alert mb-0"),
                    lg=4,
                ),
            ], className="g-4 align-items-start mb-4"),

            # ── Status alert ──────────────────────────────────────────────────
            dbc.Alert(
                "MSR session ready. Adjust controls or select a scenario.",
                id="msr-status-alert",
                color="info",
                className="status-alert mb-4",
            ),

            # ── Operator view: loop diagram + temperature profile + tiles ────
            dbc.Row([
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody([
                            html.Div(
                                id="msr-diagram-container",
                                children=build_msr_diagram(),
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
                                    id="msr-loop-heatmap",
                                    figure=msr_loop_heatmap_figure(950, 850, 1.0),
                                    config={"displaylogo": False},
                                ),
                            ]),
                            className="operator-card mb-3",
                        ),
                        html.Div(
                            id="msr-operator-tiles",
                            className="operator-tiles",
                        ),
                    ],
                    xl=5, lg=12, className="mb-3",
                ),
            ], className="g-3 mb-3"),

            # ── Materials & remaining-life drawer ────────────────────────────
            dbc.Row([
                dbc.Col(build_damage_panel("msr"), lg=12, className="mb-3"),
            ], className="g-3 mb-4"),

            # ── Controls + Charts ─────────────────────────────────────────────
            dbc.Row([
                dbc.Col(
                    _build_controls_column(),
                    xl=3, lg=4,
                    className="control-column",
                ),
                dbc.Col(
                    _build_charts_column(),
                    xl=9, lg=8,
                    className="graph-column",
                ),
            ], className="g-4"),
        ]),
        className="msr-panel shell-card",
    )


def register_msr_callbacks(app: Dash) -> None:
    """Register all MSR panel Dash callbacks with the app factory."""
    _register_main_callback(app)
    _register_sweep_callback(app)
    _register_msr_csv_callbacks(app)
    _register_operator_view_callback(app)


def _register_operator_view_callback(app: Dash) -> None:
    @app.callback(
        Output("msr-diagram-container", "children"),
        Output("msr-loop-heatmap", "figure"),
        Output("msr-damage-bars", "children"),
        Output("msr-life-cards", "children"),
        Output("msr-life-summary", "children"),
        Output("msr-operator-tiles", "children"),
        Input("msr-session-store", "data"),
    )
    def update_msr_operator_view(session: dict[str, Any] | None):
        return _build_msr_operator_view(session)


def _build_msr_operator_view(session: dict[str, Any] | None):
    if not session:
        return (
            build_msr_diagram(),
            msr_loop_heatmap_figure(950, 850, 1.0),
            [], [], "Awaiting first integration step…", [],
        )

    metrics = session.get("metrics") or {}
    materials = session.get("materials") or {}
    mat_state = materials.get("state") or {}
    mat_life = materials.get("remaining_life_years") or {}
    controls = session.get("controls") or {}

    power_mw = float(metrics.get("power_mw", 0.0))
    t_salt = float(metrics.get("salt_temperature_k", 923.0))
    flow_fraction = float(controls.get("salt_flow_fraction", 1.0))
    beta = float(metrics.get("beta_eff_flow", 0.0))

    # Hot/cold leg split: at full flow, swing ≈ 100 K; at low flow, larger.
    swing = 100.0 / max(0.05, flow_fraction)
    t_hot = t_salt + swing / 2.0
    t_cold = t_salt - swing / 2.0

    corr_frac = mat_state.get("hastelloy_n_corrosion_um", 0.0) / MSR_DESIGN_LIMITS[
        "hastelloy_n_corrosion_um"
    ]
    hx_frac = mat_state.get("heat_exchanger_thinning_um", 0.0) / MSR_DESIGN_LIMITS[
        "heat_exchanger_thinning_um"
    ]

    diagram_html = html.Div(
        svg_to_img(render_msr_diagram(t_hot, t_cold, corr_frac, hx_frac), alt="MSR loop"),
        className="reactor-diagram",
    )
    heatmap = msr_loop_heatmap_figure(t_hot, t_cold, flow_fraction)

    bars_input = [
        ("Hastelloy-N corrosion", mat_state.get("hastelloy_n_corrosion_um", 0.0),
         MSR_DESIGN_LIMITS["hastelloy_n_corrosion_um"], "µm"),
        ("Tellurium attack depth", mat_state.get("tellurium_attack_depth_um", 0.0),
         MSR_DESIGN_LIMITS["tellurium_attack_depth_um"], "µm"),
        ("HX tube thinning", mat_state.get("heat_exchanger_thinning_um", 0.0),
         MSR_DESIGN_LIMITS["heat_exchanger_thinning_um"], "µm"),
        ("Tritium inventory", mat_state.get("tritium_inventory_g", 0.0),
         MSR_DESIGN_LIMITS["tritium_inventory_g"], "g"),
    ]
    life_input: dict[str, float] = {
        k: (float("inf") if v is None else float(v)) for k, v in mat_life.items()
    }
    life_label_map = {
        "hastelloy_corrosion_years": "Hastelloy-N corrosion",
        "tellurium_attack_years": "Tellurium attack",
        "hx_thinning_years": "HX tube thinning",
        "tritium_inventory_years": "Tritium inventory",
    }
    bars, cards, summary = update_damage_panel_children(
        bars_input, life_input, life_label_map
    )

    tiles = [
        _operator_tile("Thermal Power", f"{power_mw:.0f} MW",
                       "fuel salt is also the coolant"),
        _operator_tile("Salt Mean Temperature", f"{t_salt - 273.15:.1f} °C",
                       f"hot leg {t_hot - 273.15:.0f} °C / cold leg {t_cold - 273.15:.0f} °C"),
        _operator_tile("Flow Fraction", f"{flow_fraction:.2f}",
                       "low flow → larger temperature swing"),
        _operator_tile("β_eff (flow)", f"{beta * 1e5:.0f} pcm",
                       "delayed neutrons drifting with the salt"),
    ]

    return diagram_html, heatmap, bars, cards, summary, tiles


# ---------------------------------------------------------------------------
# Layout builders
# ---------------------------------------------------------------------------


def _build_controls_column() -> list:
    return [
        _controls_card(),
        _scenario_card(),
        build_scenario_scrubber_card("msr"),
        _config_card(),
        _event_log_card(),
    ]


def _event_log_card() -> dbc.Card:
    """Scrollable plain-language MSR event log (parity with PWR)."""
    return dbc.Card(
        [
            dbc.CardHeader(
                [
                    html.Div("Event Log Console", className="fw-semibold"),
                    html.Small(
                        "Plain-language simulation timeline",
                        className="text-secondary-emphasis",
                    ),
                ]
            ),
            dbc.ListGroup(
                id="msr-event-log",
                children=[],
                flush=True,
                className="event-log",
            ),
        ],
        className="workstation-card event-log-card mt-3",
    )


def _controls_card() -> dbc.Card:
    return dbc.Card(
        dbc.CardBody([
            html.Div("Live Boundary Conditions", className="eyebrow"),
            html.H5("Operator Inputs", className="card-title mb-3"),

            # External reactivity (rod)
            dbc.Label("External reactivity (pcm)", html_for="msr-rod-reactivity"),
            dcc.Slider(
                id="msr-rod-reactivity",
                min=-500, max=500, step=5, value=0,
                marks={-500: "-500", -250: "-250", 0: "0", 250: "+250", 500: "+500"},
                tooltip={"placement": "bottom", "always_visible": False},
                className="mb-3",
            ),

            # Salt flow fraction
            dbc.Label("Salt flow rate (fraction of nominal)", html_for="msr-flow-fraction"),
            dcc.Slider(
                id="msr-flow-fraction",
                min=0.1, max=1.5, step=0.05, value=1.0,
                marks={0.1: "0.1", 0.5: "0.5", 1.0: "1.0", 1.5: "1.5"},
                tooltip={"placement": "bottom", "always_visible": False},
                className="mb-3",
            ),

            html.Hr(className="my-3"),

            # Time multiplier
            dbc.Label("Simulation speed", html_for="msr-time-multiplier"),
            dcc.Dropdown(
                id="msr-time-multiplier",
                options=TIME_MULTIPLIER_OPTIONS,
                value=1,
                clearable=False,
                searchable=False,
                className="dark-dropdown mb-3",
            ),

            # Pause / Reset
            dbc.Row([
                dbc.Col(
                    dbc.Button(
                        "Pause",
                        id="msr-pause-toggle",
                        color="warning",
                        className="w-100",
                    ),
                    width=6,
                ),
                dbc.Col(
                    dbc.Button(
                        "Reset",
                        id="msr-reset-btn",
                        color="secondary",
                        outline=True,
                        className="w-100",
                    ),
                    width=6,
                ),
            ], className="g-2"),
        ]),
        className="workstation-card control-card",
    )


def _scenario_card() -> dbc.Card:
    return dbc.Card(
        dbc.CardBody([
            html.Div("Preset Transients", className="eyebrow"),
            html.H5("Scenario Mode", className="card-title mb-2"),
            html.P(
                "Reset the MSR twin and run a predefined transient "
                "with time-dependent boundary conditions.",
                className="small text-secondary-emphasis",
            ),
            dbc.Label("Select scenario", html_for="msr-scenario-select"),
            dcc.Dropdown(
                id="msr-scenario-select",
                options=msr_scenario_options(),
                value="pump_trip",
                clearable=False,
                className="dark-dropdown mb-3",
            ),
            dbc.Button(
                "Run Scenario",
                id="msr-start-scenario",
                color="primary",
                className="w-100 mb-3",
            ),
            dbc.Button(
                "Download CSV",
                id="msr-save-csv",
                color="secondary",
                outline=True,
                className="w-100",
            ),
            dcc.Download(id="msr-download-csv"),
        ]),
        className="workstation-card mt-3",
    )


def _config_card() -> dbc.Card:
    return dbc.Card(
        dbc.CardBody([
            html.Div("Reactor Configuration", className="eyebrow"),
            html.H5("DDE Parameters", className="card-title mb-2"),
            html.P(
                "Changes here reset the session (τ_core and τ_loop are fixed per run, "
                "per SPEC §4.6.3).",
                className="small text-secondary-emphasis mb-3",
            ),

            dbc.Label("Core transit time τ_core (s)", html_for="msr-tau-core"),
            dcc.Slider(
                id="msr-tau-core",
                min=1.0, max=50.0, step=1.0, value=float(TAU_CORE_NOM),
                marks={1: "1", 10: "10", 25: "25", 50: "50"},
                tooltip={"placement": "bottom", "always_visible": False},
                className="mb-3",
            ),

            dbc.Label("Loop transit time τ_loop (s)", html_for="msr-tau-loop"),
            dcc.Slider(
                id="msr-tau-loop",
                min=5.0, max=100.0, step=5.0, value=20.0,
                marks={5: "5", 20: "20", 50: "50", 100: "100"},
                tooltip={"placement": "bottom", "always_visible": False},
                className="mb-3",
            ),

            dbc.Label("Salt temperature coefficient (pcm/K)", html_for="msr-alpha-salt"),
            dcc.Slider(
                id="msr-alpha-salt",
                min=-15.0, max=-0.5, step=0.5, value=-5.0,
                marks={-15: "-15", -10: "-10", -5: "-5", -0.5: "-0.5"},
                tooltip={"placement": "bottom", "always_visible": False},
                className="mb-3",
            ),

            # β_eff,flow indicator
            html.Div([
                html.Span("β_eff,flow at current τ_core: ", className="small text-secondary-emphasis"),
                html.Span("—", id="msr-beta-flow-indicator", className="small fw-bold"),
            ]),
        ]),
        className="workstation-card mt-3",
    )


def _build_charts_column() -> list:
    return [
        # Metrics row
        html.Div(id="msr-metrics", className="mb-3"),

        # Power
        _graph_card(
            "Thermal Power (MWth)",
            "Real-time point kinetics power with salt feedback.",
            "msr-power-graph",
            _empty_power_figure(),
            class_name="mb-3",
        ),

        # Temperature + Reactivity
        dbc.Row([
            dbc.Col(
                _graph_card(
                    "Salt Temperature",
                    "Unified fuel-coolant temperature (K).",
                    "msr-temperature-graph",
                    _empty_temperature_figure(),
                ),
                xl=6,
            ),
            dbc.Col(
                _graph_card(
                    "Reactivity Breakdown",
                    "External (rod) + salt temperature feedback components (pcm).",
                    "msr-reactivity-graph",
                    _empty_reactivity_figure(),
                ),
                xl=6,
            ),
        ], className="g-3 mb-3"),

        dbc.Row([
            dbc.Col(
                _graph_card(
                    "Salt Flow Fraction",
                    "Operator flow setting vs simulated time.",
                    "msr-flow-graph",
                    _empty_flow_figure(),
                ),
                xl=6,
            ),
            dbc.Col(
                _graph_card(
                    "β_eff,flow vs Time",
                    "Flowing effective delayed fraction during the transient.",
                    "msr-beta-time-graph",
                    _empty_beta_time_figure(),
                ),
                xl=6,
            ),
        ], className="g-3 mb-3"),

        # β_eff,flow vs salt velocity (algebraic sweep)
        _graph_card(
            "β_eff,flow vs Salt Velocity",
            "Static algebraic sweep (SPEC Eq. 24). Current operating point highlighted.",
            "msr-beta-graph",
            _empty_beta_figure(),
            class_name="mb-3",
        ),

        dbc.Row([
            dbc.Col(
                _graph_card(
                    "Operator Control Schedule",
                    "External reactivity (pcm) and salt flow fraction vs time.",
                    "msr-control-schedule-graph",
                    _empty_control_schedule_figure(),
                ),
                xl=6,
            ),
            dbc.Col(
                _graph_card(
                    "Precursor Drift Terms (C₁–C₆)",
                    "Flow-out vs return per group (Eq. 22–23); return uses C(t−τ_loop) ≈ C(t).",
                    "msr-drift-graph",
                    _empty_drift_figure(),
                ),
                xl=6,
            ),
        ], className="g-3 mb-3"),

        # Precursor group concentrations
        _graph_card(
            "Delayed Neutron Precursor Groups (C₁–C₆)",
            "In-core precursor concentrations — shows drift effect during flow transients.",
            "msr-precursor-graph",
            _empty_precursor_figure(),
        ),
    ]


def _graph_card(
    title: str,
    subtitle: str,
    graph_id: str,
    figure: go.Figure,
    *,
    class_name: str = "",
) -> dbc.Card:
    return dbc.Card(
        [
            dbc.CardHeader([
                html.Div(title, className="graph-card-title"),
                html.Small(subtitle, className="graph-card-subtitle"),
            ], className="graph-card-header"),
            dbc.CardBody(
                dcc.Graph(
                    id=graph_id,
                    figure=figure,
                    config=CHART_CONFIG,
                    className="workstation-graph",
                ),
                className="graph-card-body",
            ),
        ],
        className=f"workstation-card graph-card {class_name}".strip(),
    )


# ---------------------------------------------------------------------------
# Callback registration
# ---------------------------------------------------------------------------


def _register_main_callback(app: Dash) -> None:

    @app.callback(
        Output("msr-session-store", "data"),
        Output("msr-status-alert", "children"),
        Output("msr-status-alert", "color"),
        Output("msr-metrics", "children"),
        Output("msr-power-graph", "figure"),
        Output("msr-temperature-graph", "figure"),
        Output("msr-reactivity-graph", "figure"),
        Output("msr-flow-graph", "figure"),
        Output("msr-beta-time-graph", "figure"),
        Output("msr-precursor-graph", "figure"),
        Output("msr-control-schedule-graph", "figure"),
        Output("msr-drift-graph", "figure"),
        Output("msr-event-log", "children"),
        Output("msr-run-state-store", "data"),
        Output("msr-step-interval", "disabled"),
        Output("msr-pause-toggle", "children"),
        Output("msr-pause-toggle", "color"),
        Output("msr-beta-flow-indicator", "children"),
        Output("msr-scrubber-panel", "style"),
        Output("msr-scrub-slider", "max"),
        Output("msr-scrub-slider", "value"),
        Output("msr-scrub-time-label", "children"),
        Input("msr-step-interval", "n_intervals"),
        Input("msr-scrub-slider", "value"),
        Input("display-units-store", "data"),
        Input("msr-rod-reactivity", "value"),
        Input("msr-flow-fraction", "value"),
        Input("msr-tau-core", "value"),
        Input("msr-tau-loop", "value"),
        Input("msr-alpha-salt", "value"),
        Input("msr-start-scenario", "n_clicks"),
        Input("msr-reset-btn", "n_clicks"),
        Input("msr-pause-toggle", "n_clicks"),
        Input("msr-time-multiplier", "value"),
        State("msr-session-store", "data"),
        State("msr-run-state-store", "data"),
        State("msr-scenario-select", "value"),
        prevent_initial_call=False,
    )
    def update_msr_panel(
        _n_intervals: int,
        scrub_slider: int | None,
        display_units: dict[str, str] | None,
        rod_reactivity_pcm: Any,
        salt_flow_fraction: Any,
        tau_core: Any,
        tau_loop: Any,
        alpha_salt: Any,
        scenario_clicks: int | None,
        reset_clicks: int | None,
        pause_clicks: int | None,
        time_multiplier: Any,
        session: dict[str, Any] | None,
        run_state: dict[str, Any] | None,
        selected_scenario: str | None,
    ) -> tuple:
        del reset_clicks, pause_clicks

        raw = {
            "external_reactivity_pcm": rod_reactivity_pcm or 0.0,
            "salt_flow_fraction": salt_flow_fraction or 1.0,
            "core_transit_time_s": tau_core or TAU_CORE_NOM,
            "loop_transit_time_s": tau_loop or 20.0,
            "salt_temp_coeff_pcm_per_k": alpha_salt or -5.0,
            "initial_power_mw": 500.0,
            "salt_temperature_k": T_SALT_NOM,
            "salt_velocity_m_s": 1.0,
        }
        trigger_id = ctx.triggered_id
        multiplier = int(time_multiplier or 1)
        paused = bool((run_state or {}).get("paused", False))

        if trigger_id == "msr-start-scenario" and scenario_clicks:
            updated = run_msr_scenario(selected_scenario or "pump_trip")
            paused = False
            full_rebuild = True
        elif trigger_id == "msr-reset-btn":
            updated = run_msr_control_step(None, raw, time_multiplier=1)
            paused = False
            full_rebuild = True
        elif trigger_id == "msr-pause-toggle":
            paused = not paused
            updated = session or run_msr_control_step(None, raw, time_multiplier=1)
            full_rebuild = session is None
        elif paused and session:
            updated = session
            full_rebuild = False
        elif session and session.get("scenario", {}).get("completed"):
            updated = session
            full_rebuild = False
        else:
            updated = run_msr_control_step(
                session, raw,
                step_seconds=MSR_STEP_SECONDS,
                time_multiplier=multiplier,
                max_history_points=MSR_MAX_HISTORY_POINTS,
            )
            full_rebuild = session is None or not session.get("history", {}).get("time_s")

        status = updated.get("status", {})
        ok = status.get("ok", True)
        color = "success" if ok else "danger"
        message = status.get("message", "MSR simulation status unavailable.")
        if paused:
            color = "warning" if ok else color
            t_str = _fmt_time(updated.get("time_s", 0.0))
            message = f"Simulation paused at {t_str}."

        scenario_done = bool(updated.get("scenario", {}).get("completed"))
        n_hist = history_length(updated)
        if scenario_done and n_hist > 1:
            if trigger_id == "msr-start-scenario":
                scrub_idx = n_hist - 1
            else:
                scrub_idx = clamp_scrub_index(updated, scrub_slider)
            display = slice_session_for_scrub(updated, scrub_idx, reactor="msr")
            scrub_at_end = scrub_idx >= n_hist - 1
        else:
            scrub_idx = None
            display = updated
            scrub_at_end = True

        power_fig = _power_figure(display, display_units)
        temp_fig = _temperature_figure(display, display_units)
        react_fig = _reactivity_figure(display, display_units)
        flow_fig = _flow_figure(display, display_units)
        beta_time_fig = _beta_time_figure(display, display_units)
        precursor_fig = _precursor_figure(display, display_units)
        control_fig = _control_schedule_figure(display, display_units)
        drift_fig = _drift_figure(display, display_units)
        if not scrub_at_end and scrub_idx is not None:
            raw_t = float((updated.get("history") or {}).get("time_s", [0])[scrub_idx])
            t_mark = time_values_seconds([raw_t], display_units)[0]
            for fig in (
                power_fig, temp_fig, react_fig, flow_fig,
                beta_time_fig, control_fig, drift_fig,
            ):
                _add_scrub_marker(fig, t_mark)

        run_state_out = {"paused": paused}
        interval_disabled = paused or scenario_done
        pause_label = "Resume" if paused else "Pause"
        pause_color = "success" if paused else "warning"

        beta_val = updated.get("beta_eff_flow", 0.0)
        beta_indicator = f"{beta_val:.5f}" if beta_val else "—"

        if scenario_done and n_hist > 1:
            scrub_style = scrubber_panel_style(updated)
            scrub_max = n_hist - 1
            scrub_val = scrub_idx if scrub_idx is not None else scrub_max
            scrub_label = format_scrub_label(updated, scrub_val)
        else:
            scrub_style = {"display": "none"}
            scrub_max = 0
            scrub_val = 0
            scrub_label = ""

        return (
            updated,
            message,
            color,
            _metric_row(display, display_units),
            power_fig,
            temp_fig,
            react_fig,
            flow_fig,
            beta_time_fig,
            precursor_fig,
            control_fig,
            drift_fig,
            _event_log_items(updated, display_units),
            run_state_out,
            interval_disabled,
            pause_label,
            pause_color,
            beta_indicator,
            scrub_style,
            scrub_max,
            scrub_val,
            scrub_label,
        )


def _register_sweep_callback(app: Dash) -> None:
    """Recompute the β_eff,flow curve whenever τ_core or τ_loop changes."""

    @app.callback(
        Output("msr-beta-graph", "figure"),
        Input("msr-tau-core", "value"),
        Input("msr-tau-loop", "value"),
        Input("msr-session-store", "data"),
    )
    def update_beta_figure(
        tau_core: Any,
        tau_loop: Any,
        session: dict[str, Any] | None,
    ) -> go.Figure:
        tc = float(tau_core or TAU_CORE_NOM)
        tl = float(tau_loop or 20.0)
        # Override with scenario config if one is running
        if session and session.get("config"):
            tc = float(session["config"].get("tau_core", tc))
            tl = float(session["config"].get("tau_loop", tl))
        sweep = run_msr_beta_flow_sweep(tau_core_nom=tc, tau_loop_nom=tl)
        return _beta_figure(sweep)


# ---------------------------------------------------------------------------
# Figure builders
# ---------------------------------------------------------------------------


def _base_layout(title: str = "", **kwargs) -> go.Layout:
    return go.Layout(
        template=_DARK_TEMPLATE,
        paper_bgcolor=_PAPER_BG,
        plot_bgcolor=_PLOT_BG,
        font=dict(color=_FONT_COLOR, family="monospace"),
        title=dict(text=title, font=dict(size=12)) if title else None,
        margin=dict(l=50, r=20, t=30, b=50),
        legend=dict(
            orientation="h",
            yanchor="bottom", y=1.02,
            xanchor="right", x=1,
        ),
        **kwargs,
    )


def _empty_power_figure() -> go.Figure:
    fig = go.Figure(layout=_base_layout())
    fig.add_trace(go.Scatter(x=[], y=[], name="Power (MWth)", line=dict(color=COLORS["power"])))
    fig.update_xaxes(title_text="Time (s)")
    fig.update_yaxes(title_text="Power (MWth)")
    return fig


def _msr_times(hist: dict[str, list[float]], display_units: dict[str, str] | None) -> list[float]:
    return time_values_seconds(hist.get("time_s", []), display_units)


def _power_figure(
    session: dict[str, Any],
    display_units: dict[str, str] | None = None,
) -> go.Figure:
    hist = session.get("history", {})
    t = _msr_times(hist, display_units)
    p = hist.get("power_mw", [])
    fig = go.Figure(layout=_base_layout())
    fig.add_trace(go.Scatter(
        x=t, y=p,
        name="Power (MWth)",
        line=dict(color=COLORS["power"], width=2),
    ))
    fig.update_xaxes(title_text=time_axis_label(display_units))
    fig.update_yaxes(title_text="Power (MWth)")
    return fig


def _empty_temperature_figure() -> go.Figure:
    fig = go.Figure(layout=_base_layout())
    fig.add_trace(go.Scatter(x=[], y=[], name="T_salt (K)", line=dict(color=COLORS["salt_temp"])))
    fig.update_xaxes(title_text="Time (s)")
    fig.update_yaxes(title_text="Temperature (K)")
    return fig


def _temperature_figure(
    session: dict[str, Any],
    display_units: dict[str, str] | None = None,
) -> go.Figure:
    hist = session.get("history", {})
    t = _msr_times(hist, display_units)
    temps = temperature_values_kelvin(hist.get("salt_temperature_k", []), display_units)
    fig = go.Figure(layout=_base_layout())
    fig.add_trace(go.Scatter(
        x=t, y=temps,
        name="T_salt",
        line=dict(color=COLORS["salt_temp"], width=2),
    ))
    fig.update_xaxes(title_text=time_axis_label(display_units))
    fig.update_yaxes(title_text=temperature_axis_label(display_units))
    return fig


def _empty_reactivity_figure() -> go.Figure:
    fig = go.Figure(layout=_base_layout())
    for name, color in [("External (pcm)", COLORS["rod"]), ("Temp feedback (pcm)", COLORS["temp_fb"]), ("Total (pcm)", COLORS["total"])]:
        fig.add_trace(go.Scatter(x=[], y=[], name=name, line=dict(color=color)))
    fig.update_xaxes(title_text="Time (s)")
    fig.update_yaxes(title_text="Reactivity (pcm)")
    return fig


def _reactivity_figure(
    session: dict[str, Any],
    display_units: dict[str, str] | None = None,
) -> go.Figure:
    hist = session.get("history", {})
    t = _msr_times(hist, display_units)
    ext  = hist.get("external_reactivity_pcm", [])
    temp = hist.get("temp_reactivity_pcm", [])
    tot  = hist.get("total_reactivity_pcm", [])
    fig = go.Figure(layout=_base_layout())
    fig.add_trace(go.Scatter(
        x=t, y=ext, name="External / rod (pcm)",
        line=dict(color=COLORS["rod"], width=2),
    ))
    fig.add_trace(go.Scatter(
        x=t, y=temp, name="Temp feedback (pcm)",
        line=dict(color=COLORS["temp_fb"], width=2),
    ))
    fig.add_trace(go.Scatter(
        x=t, y=tot, name="Total (pcm)",
        line=dict(color=COLORS["total"], width=1, dash="dot"),
    ))
    fig.add_hline(y=0, line=dict(color="#475569", width=1, dash="dash"))
    rho_base = float((session.get("config") or {}).get("base_reactivity_pcm", 0.0))
    if rho_base:
        fig.add_hline(
            y=rho_base,
            line=dict(color=COLORS["beta_static"], width=1, dash="dot"),
            annotation_text="ρ_base (criticality)",
            annotation_position="top left",
        )
    fig.update_xaxes(title_text=time_axis_label(display_units))
    fig.update_yaxes(title_text="Reactivity (pcm)")
    return fig


def _empty_flow_figure() -> go.Figure:
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(
        x=[], y=[], name="Flow fraction",
        line=dict(color=COLORS["flow"], width=2, shape="hv"),
    ), secondary_y=False)
    fig.add_trace(go.Scatter(
        x=[], y=[], name="ṁ (kg/s)",
        line=dict(color=COLORS["beta_point"], width=2, dash="dot", shape="hv"),
    ), secondary_y=True)
    fig.update_layout(_base_layout())
    fig.update_xaxes(title_text="Time (s)")
    fig.update_yaxes(title_text="Flow fraction", secondary_y=False)
    fig.update_yaxes(title_text="Salt mass flow (kg/s)", secondary_y=True, showgrid=False)
    return fig


def _flow_figure(
    session: dict[str, Any],
    display_units: dict[str, str] | None = None,
) -> go.Figure:
    hist = session.get("history", {})
    t = _msr_times(hist, display_units)
    flow_frac = hist.get("salt_flow_fraction", [])
    m_dot = [float(v) * M_DOT_SALT_NOM for v in flow_frac]
    cfg = session.get("config") or {}
    tau_c = float(cfg.get("tau_core", TAU_CORE_NOM))
    tau_l = float(cfg.get("tau_loop", 20.0))
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(
        x=t, y=list(flow_frac),
        name="Flow fraction",
        line=dict(color=COLORS["flow"], width=2, shape="hv"),
    ), secondary_y=False)
    fig.add_trace(go.Scatter(
        x=t, y=m_dot,
        name="ṁ (kg/s)",
        line=dict(color=COLORS["beta_point"], width=2, dash="dot", shape="hv"),
    ), secondary_y=True)
    fig.update_layout(
        _base_layout(),
        annotations=[
            dict(
                xref="paper", yref="paper", x=0.01, y=0.98,
                xanchor="left", yanchor="top", showarrow=False,
                text=f"τ_core = {tau_c:.1f} s, τ_loop = {tau_l:.1f} s",
                font=dict(size=10, color="#94a3b8"),
            ),
        ],
    )
    fig.update_xaxes(title_text=time_axis_label(display_units))
    fig.update_yaxes(title_text="Flow fraction", secondary_y=False)
    fig.update_yaxes(title_text="Salt mass flow (kg/s)", secondary_y=True, showgrid=False)
    return fig


def _empty_beta_time_figure() -> go.Figure:
    fig = go.Figure(layout=_base_layout())
    fig.add_trace(go.Scatter(
        x=[], y=[], name="β_eff,flow",
        line=dict(color=COLORS["beta_curve"], width=2),
    ))
    fig.add_hline(
        y=0.0065,
        line=dict(color=COLORS["beta_static"], dash="dash", width=1),
        annotation_text="β_eff static",
    )
    fig.update_xaxes(title_text="Time (s)")
    fig.update_yaxes(title_text="β_eff,flow")
    return fig


def _beta_time_figure(
    session: dict[str, Any],
    display_units: dict[str, str] | None = None,
) -> go.Figure:
    hist = session.get("history", {})
    t = _msr_times(hist, display_units)
    beta = hist.get("beta_eff_flow", [])
    fig = go.Figure(layout=_base_layout())
    fig.add_trace(go.Scatter(
        x=t, y=beta,
        name="β_eff,flow",
        line=dict(color=COLORS["beta_curve"], width=2),
    ))
    fig.add_hline(
        y=_PWR_BETA_STATIC,
        line=dict(color="#f87171", dash="dash", width=1),
        annotation_text="PWR β_eff (static)",
    )
    fig.add_hline(
        y=0.0065,
        line=dict(color=COLORS["beta_static"], dash="dot", width=1),
        annotation_text="MSR β at v→0",
    )
    fig.update_xaxes(title_text=time_axis_label(display_units))
    fig.update_yaxes(title_text="β_eff,flow")
    return fig


def _empty_beta_figure() -> go.Figure:
    fig = go.Figure(layout=_base_layout())
    fig.add_trace(go.Scatter(x=[], y=[], name="β_eff,flow", line=dict(color=COLORS["beta_curve"])))
    fig.add_hline(y=0.0065, line=dict(color=COLORS["beta_static"], dash="dash", width=1),
                  annotation_text="β_eff static", annotation_position="bottom right")
    fig.update_xaxes(title_text="Salt velocity (m/s)", type="log")
    fig.update_yaxes(title_text="β_eff,flow")
    return fig


def _beta_figure(sweep: dict[str, Any]) -> go.Figure:
    v = sweep.get("v_salt", [])
    b = sweep.get("beta_eff_flow", [])
    b_static = sweep.get("beta_eff_static", 0.0065)
    v_cur = sweep.get("v_salt_current", 1.0)
    b_cur = sweep.get("beta_current", b_static)

    fig = go.Figure(layout=_base_layout())

    # Main β_eff,flow curve
    fig.add_trace(go.Scatter(
        x=v, y=b,
        name="β_eff,flow (Eq. 24)",
        line=dict(color=COLORS["beta_curve"], width=2),
    ))

    fig.add_hline(
        y=_PWR_BETA_STATIC,
        line=dict(color="#f87171", dash="dash", width=1),
        annotation_text=f"PWR β_eff = {_PWR_BETA_STATIC:.5f}",
        annotation_position="top right",
    )
    fig.add_hline(
        y=b_static,
        line=dict(color=COLORS["beta_static"], dash="dot", width=1),
        annotation_text=f"MSR static = {b_static:.5f}",
        annotation_position="bottom right",
        annotation_font_color=COLORS["beta_static"],
    )

    # Current operating point
    fig.add_trace(go.Scatter(
        x=[v_cur], y=[b_cur],
        name=f"Operating point (v={v_cur:.2f} m/s)",
        mode="markers",
        marker=dict(color=COLORS["beta_point"], size=12, symbol="star"),
    ))

    fig.update_xaxes(title_text="Salt velocity (m/s)", type="log")
    fig.update_yaxes(title_text="β_eff,flow (dimensionless)")
    return fig


def _empty_control_schedule_figure() -> go.Figure:
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Scatter(
            x=[], y=[], mode="lines", name="Flow fraction",
            line={"color": COLORS["flow"], "width": 2, "shape": "hv"},
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=[], y=[], mode="lines", name="ρ_ext (pcm)",
            line={"color": COLORS["rod"], "width": 2, "shape": "hv"},
        ),
        secondary_y=True,
    )
    fig.update_layout(_base_layout("Operator Control Schedule"))
    fig.update_yaxes(title_text="ρ_ext (pcm)", secondary_y=True)
    return fig


def _control_schedule_figure(
    session: dict[str, Any],
    display_units: dict[str, str] | None = None,
) -> go.Figure:
    history = session.get("history", {})
    fig = _empty_control_schedule_figure()
    t = _msr_times(history, display_units)
    fig.data[0].x = t
    fig.data[0].y = history.get("salt_flow_fraction", [])
    fig.data[1].x = t
    fig.data[1].y = history.get("external_reactivity_pcm", [])
    fig.update_xaxes(title_text=time_axis_label(display_units))
    return fig


def _empty_drift_figure() -> go.Figure:
    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=tuple(_PRECURSOR_LABELS),
        vertical_spacing=0.12,
        horizontal_spacing=0.06,
    )
    for i in range(6):
        row, col = i // 3 + 1, i % 3 + 1
        fig.add_trace(
            go.Scatter(x=[], y=[], mode="lines", name="out", line={"color": COLORS["beta_curve"], "width": 1}),
            row=row, col=col,
        )
        fig.add_trace(
            go.Scatter(x=[], y=[], mode="lines", name="return", line={"color": COLORS["beta_point"], "width": 1, "dash": "dash"}),
            row=row, col=col,
        )
    fig.update_layout(_base_layout("Precursor Drift (all groups)"))
    return fig


def _drift_figure(
    session: dict[str, Any],
    display_units: dict[str, str] | None = None,
) -> go.Figure:
    history = session.get("history", {})
    fig = _empty_drift_figure()
    t = _msr_times(history, display_units)
    for g in range(6):
        row, col = g // 3 + 1, g % 3 + 1
        trace_out = (g * 2)
        trace_ret = trace_out + 1
        fig.data[trace_out].x = t
        fig.data[trace_out].y = history.get(f"precursor_flow_out_{g + 1}", [])
        fig.data[trace_ret].x = t
        fig.data[trace_ret].y = history.get(f"precursor_return_{g + 1}", [])
    return fig


def _empty_precursor_figure() -> go.Figure:
    """Return the base Plotly figure for all 6 delayed precursor groups."""
    fig = go.Figure(layout=_base_layout())
    for label, color in zip(_PRECURSOR_LABELS, _PRECURSOR_COLORS):
        fig.add_trace(go.Scatter(
            x=[], y=[], name=label,
            line=dict(color=color, width=2),
        ))
    fig.update_xaxes(title_text="Time (s)")
    fig.update_yaxes(title_text="Concentration (neutrons/cm³)")
    return fig


def _precursor_figure(
    session: dict[str, Any],
    display_units: dict[str, str] | None = None,
) -> go.Figure:
    """Build the 6-group in-core precursor concentration chart from session history."""
    hist = session.get("history", {})
    t = _msr_times(hist, display_units)
    fig = go.Figure(layout=_base_layout())
    for label, color, key in zip(_PRECURSOR_LABELS, _PRECURSOR_COLORS, _PRECURSOR_KEYS):
        fig.add_trace(go.Scatter(
            x=t, y=hist.get(key, []),
            name=label,
            line=dict(color=color, width=2),
        ))
    fig.update_xaxes(title_text=time_axis_label(display_units))
    fig.update_yaxes(title_text="Concentration (neutrons/cm³)")
    return fig


# ---------------------------------------------------------------------------
# Metrics row
# ---------------------------------------------------------------------------


def _scenario_banner(session: dict[str, Any]) -> dbc.Alert | html.Div:
    scenario = session.get("scenario") or {}
    label = scenario.get("label")
    if not label:
        return html.Div()
    description = scenario.get("description", "")
    color = "success" if scenario.get("completed") else "info"
    return dbc.Alert(
        [html.Strong(label), html.Span(f" — {description}" if description else "")],
        color=color,
        className="py-2 mb-2 small",
    )


def _metric_row(
    session: dict[str, Any],
    display_units: dict[str, str] | None = None,
) -> html.Div:
    metrics = session.get("metrics", {})
    t_s = float(session.get("time_s", 0.0))
    p_mw = float(metrics.get("power_mw", 0.0))
    T_k  = float(metrics.get("salt_temperature_k", T_SALT_NOM))
    T_disp = temperature_scalar_kelvin(T_k, display_units) or T_k
    temp_unit = temperature_unit_suffix(display_units)
    rho  = float(metrics.get("total_reactivity_pcm", 0.0))
    beta = float(metrics.get("beta_eff_flow", 0.0))
    P_pct = p_mw / (P_NOM_MSR / 1.0e6) * 100.0

    cfg = session.get("config") or {}
    tau_c = float(cfg.get("tau_core", TAU_CORE_NOM))
    tau_l = float(cfg.get("tau_loop", 20.0))
    row = dbc.Row([
        _metric_badge("Sim time", format_simulation_time(t_s, display_units)),
        _metric_badge("Power", f"{p_mw:.1f} MWth ({P_pct:.1f}%)"),
        _metric_badge("T_salt", f"{T_disp:.1f} {temp_unit}"),
        _metric_badge("ρ_total", f"{rho:+.1f} pcm"),
        _metric_badge("β_eff,flow", f"{beta:.5f}"),
        _metric_badge("τ_core / τ_loop", f"{tau_c:.0f} s / {tau_l:.0f} s"),
    ], className="g-2 metrics-row")
    banner = _scenario_banner(session)
    if isinstance(banner, dbc.Alert):
        return html.Div([banner, row])
    return row


def _register_msr_csv_callbacks(app: Dash) -> None:
    @app.callback(
        Output("msr-download-csv", "data"),
        Input("msr-save-csv", "n_clicks"),
        State("msr-session-store", "data"),
        prevent_initial_call=True,
    )
    def _download_msr_csv(n_clicks: int | None, session: dict[str, Any] | None) -> Any:
        if not n_clicks or not session:
            return no_update
        history = session.get("history") or {}
        if not history.get("time_s"):
            return no_update
        return {
            "content": history_to_csv_text(history),
            "filename": "msr_history.csv",
            "type": "text/csv",
        }


def _metric_badge(label: str, value: str) -> dbc.Col:
    return dbc.Col(
        dbc.Card(
            dbc.CardBody([
                html.Div(label, className="metric-label"),
                html.Div(value, className="metric-value"),
            ], className="p-2"),
            className="metric-badge",
        ),
        xs=6, sm=4, md="auto",
    )


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


def _fmt_time(t_s: float) -> str:
    """Format simulation time as Xh Ym Zs for display."""
    t_s = abs(t_s)
    h = int(t_s // 3600)
    m = int((t_s % 3600) // 60)
    s = t_s % 60
    if h:
        return f"{h}h {m:02d}m {s:04.1f}s"
    if m:
        return f"{m}m {s:04.1f}s"
    return f"{s:.1f}s"
