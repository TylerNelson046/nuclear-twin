"""PWR control panel and callbacks for the Dash reactor twin UI.

CURRENT charts (SPEC FR-10): power, temperatures, reactivity breakdown, Xe/I.
ADDITIONAL: normalized power overlay, extended metrics, HDF5 save/load.
See docs/UI_VISUALIZATION.md for the full inventory.
"""

from __future__ import annotations

import base64
from typing import Any

from dash import Dash, Input, Output, Patch, State, ctx, dcc, html, no_update
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from orchestrator.scenarios import pwr_scenario_options
from orchestrator.sim_runner import run_pwr_control_step, run_pwr_scenario
from physics.materials import PWR_DESIGN_LIMITS
from physics.pwr.thermal_hydraulics import M_DOT_NOM, T_IN_NOM
from ui.diagrams import (
    build_damage_panel,
    build_pwr_diagram,
    damage_history_figure,
    pwr_heatmap_figure,
    render_pwr_diagram,
    update_damage_panel_children,
)
from ui.diagrams.pwr_schematic import svg_to_img
from ui.scenario_scrubber import build_scenario_scrubber_card, format_scrub_label
from utils.data_handler import HDF5StorageError, load_pwr_session_from_bytes, session_to_hdf5_bytes
from utils.history_export import history_to_csv_text
from utils.display_units import (
    format_simulation_time,
    temperature_axis_label,
    temperature_scalar_kelvin,
    temperature_unit_suffix,
    temperature_values_kelvin,
    time_axis_label,
    time_values_seconds,
)
from utils.scenario_scrub import (
    clamp_scrub_index,
    history_length,
    scrubber_panel_style,
    slice_session_for_scrub,
)


DISCLAIMER = (
    "Models are simplified analytical representations for educational purposes "
    "and do not represent licensed reactor safety analysis"
)
LIVE_INTERVAL_MS = 200
LIVE_STEP_SECONDS = LIVE_INTERVAL_MS / 1000.0
LIVE_MIN_OUTPUT_POINTS = 9
LIVE_MAX_DISPLAY_POINTS = 2400
CHART_CONFIG = {
    "displayModeBar": True,
    "displaylogo": False,
    "doubleClick": "reset+autosize",
    "responsive": True,
    "scrollZoom": True,
    "toImageButtonOptions": {
        "format": "png",
        "filename": "pwr_timeseries",
        "height": 720,
        "width": 1280,
        "scale": 2,
    },
}
UIREVISION = "pwr-live-timeseries"
TIME_MULTIPLIER_OPTIONS = [
    {"label": "1x live", "value": 1},
    {"label": "60x accelerated", "value": 60},
    {"label": "3600x xenon timescale", "value": 3600},
]
COLORS = {
    "power": "#38bdf8",
    "fuel": "#fb7185",
    "coolant": "#2dd4bf",
    "net": "#f8fafc",
    "rod": "#a78bfa",
    "doppler": "#fb923c",
    "moderator": "#22d3ee",
    "boron": "#94a3b8",
    "xenon": "#c084fc",
    "iodine": "#facc15",
}
NOMINAL_MASS_FLOW_KG_S = M_DOT_NOM

# Delayed precursor group chart styling (6 Keepin groups, distinct colors)
_PRECURSOR_COLORS = ["#f87171", "#fb923c", "#facc15", "#4ade80", "#60a5fa", "#c084fc"]
_PRECURSOR_LABELS = ["C₁ (group 1)", "C₂ (group 2)", "C₃ (group 3)", "C₄ (group 4)", "C₅ (group 5)", "C₆ (group 6)"]
_PRECURSOR_KEYS = [f"precursor_C{i}" for i in range(1, 7)]


def build_pwr_panel() -> dbc.Card:
    """Build the Phase 1 PWR interactive control panel."""
    return dbc.Card(
        dbc.CardBody(
            [
                dcc.Store(id="pwr-session-store"),
                dcc.Store(id="pwr-run-state-store", data={"paused": False}),
                dcc.Interval(id="pwr-step-interval", interval=LIVE_INTERVAL_MS, n_intervals=0),
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                html.Div("Phase 1 Digital Twin", className="eyebrow"),
                                html.H3("PWR Control Panel", className="card-title mb-2"),
                                html.P(
                                    "Adjust reactor controls and watch the 11-state "
                                    "point-kinetics, thermal, and xenon model advance "
                                    "in short timesteps.",
                                    className="card-text text-secondary-emphasis mb-0",
                                ),
                            ],
                            lg=8,
                        ),
                        dbc.Col(
                            dbc.Alert(
                                DISCLAIMER,
                                color="warning",
                                className="disclaimer-alert mb-0",
                            ),
                            lg=4,
                        ),
                    ],
                    className="g-4 align-items-start mb-4",
                ),
                dbc.Alert(
                    "PWR session ready. Inputs are validated before each timestep.",
                    id="pwr-status-alert",
                    color="info",
                    className="status-alert mb-4",
                ),
                # ─── Operator view: reactor diagram + heatmap + key tiles ───
                dbc.Row(
                    [
                        dbc.Col(
                            dbc.Card(
                                dbc.CardBody([
                                    html.Div(
                                        id="pwr-diagram-container",
                                        children=build_pwr_diagram(),
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
                                            id="pwr-heatmap-graph",
                                            figure=pwr_heatmap_figure(900, 590, 565, 1.0, 0.2),
                                            config={"displaylogo": False},
                                        ),
                                    ]),
                                    className="operator-card mb-3",
                                ),
                                html.Div(
                                    id="pwr-operator-tiles",
                                    className="operator-tiles",
                                ),
                            ],
                            xl=5, lg=12, className="mb-3",
                        ),
                    ],
                    className="g-3 mb-3",
                ),
                # ─── Materials & remaining-life drawer ───
                dbc.Row(
                    [
                        dbc.Col(build_damage_panel("pwr"), lg=12, className="mb-3"),
                    ],
                    className="g-3 mb-4",
                ),
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                _controls_card(),
                                _scenario_card(),
                                build_scenario_scrubber_card("pwr"),
                                _event_log_card(),
                            ],
                            xl=3,
                            lg=4,
                            className="control-column",
                        ),
                        dbc.Col(
                            [
                                html.Div(id="pwr-metrics", className="mb-3"),
                                _graph_card(
                                    "Core Thermal Power",
                                    "Mouse wheel to zoom, drag to pan, range slider for full timeline.",
                                    "pwr-power-graph",
                                    _empty_power_figure(),
                                    class_name="mb-3",
                                ),
                                dbc.Row(
                                    [
                                        dbc.Col(
                                            _graph_card(
                                                "Fuel and Coolant Temperatures",
                                                "Independent y-axis zoom plus timeline scrubber.",
                                                "pwr-temperature-graph",
                                                _empty_temperature_figure(),
                                            ),
                                            xl=6,
                                        ),
                                        dbc.Col(
                                            _graph_card(
                                                "Reactivity Breakdown",
                                                "Pan and zoom reactivity components without losing history.",
                                                "pwr-reactivity-graph",
                                                _empty_reactivity_figure(),
                                            ),
                                            xl=6,
                                        ),
                                    ],
                                    className="g-3 mb-3",
                                ),
                                _graph_card(
                                    "Iodine and Xenon Concentrations",
                                    "Dual y-axis transient view with full-history range slider.",
                                    "pwr-xenon-iodine-graph",
                                    _empty_xenon_iodine_figure(),
                                    class_name="mb-3",
                                ),
                                dbc.Row(
                                    [
                                        dbc.Col(
                                            _graph_card(
                                                "Neutron Population & Thermal Flux",
                                                "Normalized population n and φ = n × φ_ref (n/cm²·s).",
                                                "pwr-neutron-graph",
                                                _empty_neutron_figure(),
                                            ),
                                            xl=6,
                                        ),
                                        dbc.Col(
                                            _graph_card(
                                                "Power vs Total Reactivity",
                                                "Phase-plane trajectory: thermal power vs net reactivity (pcm).",
                                                "pwr-power-rho-graph",
                                                _empty_power_rho_figure(),
                                            ),
                                            xl=6,
                                        ),
                                    ],
                                    className="g-3 mb-3",
                                ),
                                _graph_card(
                                    "State Derivatives (ODE RHS)",
                                    "dn/dt, dT/dt, dI/dt, dX/dt — stiffness and transient rate diagnostics.",
                                    "pwr-derivatives-graph",
                                    _empty_derivatives_figure(),
                                    class_name="mb-3",
                                ),
                                _graph_card(
                                    "Delayed Neutron Precursor Groups (C₁–C₆)",
                                    "Keepin 6-group concentrations — shows which groups dominate kinetics.",
                                    "pwr-precursor-graph",
                                    _empty_precursor_figure(),
                                    class_name="mb-3",
                                ),
                                dbc.Row(
                                    [
                                        dbc.Col(
                                            _graph_card(
                                                "Fuel–Coolant Temperature Difference",
                                                "ΔT = T_fuel − T_coolant. Rises with power; narrows at low flow.",
                                                "pwr-delta-t-graph",
                                                _empty_delta_t_figure(),
                                            ),
                                            xl=4,
                                        ),
                                        dbc.Col(
                                            _graph_card(
                                                "Control Inputs Schedule",
                                                "Boron (ppm) and coolant flow fraction vs time — captures scenario ramps.",
                                                "pwr-control-schedule-graph",
                                                _empty_control_schedule_figure(),
                                            ),
                                            xl=4,
                                        ),
                                        dbc.Col(
                                            _graph_card(
                                                "Xenon Worth vs Concentration",
                                                "Phase-plane: ρ_Xe (pcm) vs N_Xe — shows proportional worth vs flux history.",
                                                "pwr-xenon-worth-scatter",
                                                _empty_xenon_worth_figure(),
                                            ),
                                            xl=4,
                                        ),
                                    ],
                                    className="g-3",
                                ),
                            ],
                            xl=9,
                            lg=8,
                            className="graph-column",
                        ),
                    ],
                    className="g-4",
                ),
            ]
        ),
        className="pwr-panel shell-card",
    )


def register_pwr_callbacks(app: Dash) -> None:
    """Register PWR callbacks with the Dash app factory."""
    _register_pwr_hdf5_callbacks(app)
    _register_pwr_csv_callbacks(app)

    @app.callback(
        Output("pwr-session-store", "data"),
        Output("pwr-status-alert", "children"),
        Output("pwr-status-alert", "color"),
        Output("pwr-metrics", "children"),
        Output("pwr-power-graph", "figure"),
        Output("pwr-temperature-graph", "figure"),
        Output("pwr-reactivity-graph", "figure"),
        Output("pwr-xenon-iodine-graph", "figure"),
        Output("pwr-neutron-graph", "figure"),
        Output("pwr-power-rho-graph", "figure"),
        Output("pwr-precursor-graph", "figure"),
        Output("pwr-delta-t-graph", "figure"),
        Output("pwr-control-schedule-graph", "figure"),
        Output("pwr-xenon-worth-scatter", "figure"),
        Output("pwr-derivatives-graph", "figure"),
        Output("pwr-event-log", "children"),
        Output("pwr-run-state-store", "data"),
        Output("pwr-step-interval", "disabled"),
        Output("pwr-pause-toggle", "children"),
        Output("pwr-pause-toggle", "color"),
        Output("pwr-speed-summary", "children"),
        Output("pwr-scrubber-panel", "style"),
        Output("pwr-scrub-slider", "max"),
        Output("pwr-scrub-slider", "value"),
        Output("pwr-scrub-time-label", "children"),
        Input("pwr-step-interval", "n_intervals"),
        Input("pwr-scrub-slider", "value"),
        Input("display-units-store", "data"),
        Input("pwr-rod-reactivity", "value"),
        Input("pwr-boron-ppm", "value"),
        Input("pwr-flow-fraction", "value"),
        Input("pwr-inlet-temperature-c", "value"),
        Input("pwr-start-scenario", "n_clicks"),
        Input("pwr-reset-normal", "n_clicks"),
        Input("pwr-pause-toggle", "n_clicks"),
        Input("pwr-time-multiplier", "value"),
        State("pwr-session-store", "data"),
        State("pwr-run-state-store", "data"),
        State("pwr-scenario-select", "value"),
    )
    def update_pwr_panel(
        _n_intervals: int,
        scrub_slider: int | None,
        display_units: dict[str, str] | None,
        rod_reactivity_pcm: Any,
        boron_ppm: Any,
        coolant_flow_fraction: Any,
        inlet_temperature_c: Any,
        scenario_clicks: int | None,
        reset_clicks: int | None,
        pause_clicks: int | None,
        time_multiplier: Any,
        session: dict[str, Any] | None,
        run_state: dict[str, Any] | None,
        selected_scenario: str | None,
    ) -> tuple:
        del reset_clicks, pause_clicks

        controls = {
            "rod_reactivity_pcm": rod_reactivity_pcm,
            "boron_ppm": boron_ppm,
            "coolant_flow_fraction": coolant_flow_fraction,
            "inlet_temperature_k": _celsius_to_kelvin_input(inlet_temperature_c),
        }
        trigger_id = ctx.triggered_id
        multiplier = _coerce_time_multiplier(time_multiplier)
        paused = bool((run_state or {}).get("paused", False))

        if trigger_id == "pwr-start-scenario" and scenario_clicks:
            updated = run_pwr_scenario(selected_scenario or "rod_ejection")
            full_rebuild = True
            paused = False
        elif trigger_id == "pwr-reset-normal":
            updated = _run_live_step(None, controls, multiplier)
            if updated.get("status", {}).get("ok", True):
                updated["status"] = {"ok": True, "message": "Normal PWR simulation restarted."}
            full_rebuild = True
            paused = False
        elif trigger_id == "pwr-pause-toggle":
            paused = not paused
            updated = session or _run_live_step(None, controls, multiplier)
            full_rebuild = session is None
        elif paused and session:
            updated = session
            full_rebuild = False
        elif session and session.get("scenario", {}).get("completed"):
            updated = session
            full_rebuild = False
        else:
            updated = _run_live_step(session, controls, multiplier)
            # First-ever call (session was None) needs full figure initialization.
            full_rebuild = session is None or not session.get("history", {}).get("time_s")

        status = updated.get("status", {})
        color = "success" if status.get("ok") else "danger"
        message = status.get("message", "PWR simulation status unavailable.")
        if paused:
            color = "warning" if status.get("ok", True) else color
            message = f"Simulation paused at {format_simulation_time(updated.get('time_s'), display_units)}."

        if updated.get("history_rebuilt_last_step"):
            full_rebuild = True

        scenario_complete = bool(updated.get("scenario", {}).get("completed"))
        n_hist = history_length(updated)
        if scenario_complete and n_hist > 1:
            full_rebuild = True
            if trigger_id == "pwr-start-scenario":
                scrub_idx = n_hist - 1
            else:
                scrub_idx = clamp_scrub_index(updated, scrub_slider)
            display = slice_session_for_scrub(updated, scrub_idx, reactor="pwr")
            scrub_at_end = scrub_idx >= n_hist - 1
        else:
            scrub_idx = None
            display = updated
            scrub_at_end = True

        if full_rebuild:
            power_fig = _power_figure(display, display_units)
            temp_fig = _temperature_figure(display, display_units)
            react_fig = _reactivity_figure(display, display_units)
            xenon_fig = _xenon_iodine_figure(display, display_units)
            neutron_fig = _neutron_figure(display, display_units)
            power_rho_fig = _power_rho_figure(display)
            precursor_fig = _precursor_figure(display, display_units)
            delta_t_fig = _delta_t_figure(display, display_units)
            ctrl_sched_fig = _control_schedule_figure(display, display_units)
            xe_worth_fig = _xenon_worth_figure(display)
            deriv_fig = _derivatives_figure(display, display_units)
            if not scrub_at_end and scrub_idx is not None:
                raw_t = float((updated.get("history") or {}).get("time_s", [0])[scrub_idx])
                t_mark = time_values_seconds([raw_t], display_units)[0]
                for fig in (
                    power_fig, temp_fig, react_fig, xenon_fig, neutron_fig,
                    delta_t_fig, ctrl_sched_fig, deriv_fig,
                ):
                    _add_scrub_marker(fig, t_mark)
        elif trigger_id == "pwr-pause-toggle" or (paused and session):
            power_fig = no_update
            temp_fig = no_update
            react_fig = no_update
            xenon_fig = no_update
            neutron_fig = no_update
            power_rho_fig = no_update
            precursor_fig = no_update
            delta_t_fig = no_update
            ctrl_sched_fig = no_update
            xe_worth_fig = no_update
            deriv_fig = no_update
        else:
            n_new = updated.get("new_point_count_last_step", 0)
            n_trim = updated.get("front_trim_count_last_step", 0)
            power_fig = _patch_power_figure(display, n_new, n_trim, display_units)
            temp_fig = _patch_temperature_figure(display, n_new, n_trim, display_units)
            react_fig = _patch_reactivity_figure(display, n_new, n_trim, display_units)
            xenon_fig = _patch_xenon_iodine_figure(display, n_new, n_trim, display_units)
            neutron_fig = _patch_neutron_figure(display, n_new, n_trim, display_units)
            power_rho_fig = _patch_power_rho_figure(display, n_new, n_trim)
            precursor_fig = _patch_precursor_figure(display, n_new, n_trim, display_units)
            delta_t_fig = _patch_delta_t_figure(display, n_new, n_trim, display_units)
            ctrl_sched_fig = _patch_control_schedule_figure(display, n_new, n_trim, display_units)
            xe_worth_fig = _patch_xenon_worth_figure(display, n_new, n_trim)
            deriv_fig = _patch_derivatives_figure(display, n_new, n_trim, display_units)

        run_state_out = {"paused": paused}
        interval_disabled = paused or scenario_complete
        pause_label = "Resume" if paused else "Pause"
        pause_color = "success" if paused else "warning"

        if scenario_complete and n_hist > 1:
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
            xenon_fig,
            neutron_fig,
            power_rho_fig,
            precursor_fig,
            delta_t_fig,
            ctrl_sched_fig,
            xe_worth_fig,
            deriv_fig,
            _event_log_items(updated, display_units),
            run_state_out,
            interval_disabled,
            pause_label,
            pause_color,
            _speed_summary(multiplier),
            scrub_style,
            scrub_max,
            scrub_val,
            scrub_label,
        )

    @app.callback(
        Output("pwr-diagram-container", "children"),
        Output("pwr-heatmap-graph", "figure"),
        Output("pwr-damage-bars", "children"),
        Output("pwr-life-cards", "children"),
        Output("pwr-life-summary", "children"),
        Output("pwr-operator-tiles", "children"),
        Input("pwr-session-store", "data"),
    )
    def update_pwr_operator_view(session: dict[str, Any] | None):
        return _build_pwr_operator_view(session)


def _build_pwr_operator_view(session: dict[str, Any] | None):
    """Compute the operator-view outputs from the current session payload.

    Returns (diagram_html, heatmap_figure, damage_bars, life_cards, summary, tiles).
    """
    if not session:
        return (
            build_pwr_diagram(),
            pwr_heatmap_figure(900, 590, 565, 1.0, 0.2),
            [], [], "Awaiting first integration step…", [],
        )

    metrics = session.get("metrics") or {}
    materials = session.get("materials") or {}
    mat_state = materials.get("state") or {}
    mat_life = materials.get("remaining_life_years") or {}
    controls = session.get("controls") or {}

    power_mw = float(metrics.get("power_mw", 0.0))
    t_fuel = float(metrics.get("fuel_temperature_k", 900.0))
    t_cool = float(metrics.get("coolant_temperature_k", 590.0))
    t_in = float(controls.get("inlet_temperature_k", 565.0))
    rod_pcm = float(controls.get("rod_reactivity_pcm", 0.0))
    rod_inserted = max(0.0, min(1.0, -rod_pcm / 1000.0))
    p_norm = max(0.0, power_mw / 3000.0)

    rpv_frac = mat_state.get("rpv_drtndt_k", 0.0) / PWR_DESIGN_LIMITS["rpv_drtndt_k"]
    clad_frac = (
        mat_state.get("cladding_oxide_thickness_um", 0.0)
        / PWR_DESIGN_LIMITS["cladding_oxide_thickness_um"]
    )
    rod_frac = (
        mat_state.get("control_rod_b10_depletion_frac", 0.0)
        / PWR_DESIGN_LIMITS["control_rod_b10_depletion_frac"]
    )

    diagram_html = html.Div(
        svg_to_img(
            render_pwr_diagram(
                rod_position_fraction=rod_inserted,
                rpv_damage_fraction=rpv_frac,
                cladding_damage_fraction=clad_frac,
                rod_damage_fraction=rod_frac,
                fuel_power_normalized=p_norm,
            ),
            alt="PWR cutaway",
        ),
        className="reactor-diagram",
    )

    heatmap = pwr_heatmap_figure(t_fuel, t_cool, t_in, p_norm, rod_inserted)

    bars_input = [
        ("RPV ΔRT_NDT shift", mat_state.get("rpv_drtndt_k", 0.0),
         PWR_DESIGN_LIMITS["rpv_drtndt_k"], "K"),
        ("Cladding oxide thickness", mat_state.get("cladding_oxide_thickness_um", 0.0),
         PWR_DESIGN_LIMITS["cladding_oxide_thickness_um"], "µm"),
        ("Cladding H pickup", mat_state.get("cladding_hydrogen_ppm", 0.0),
         PWR_DESIGN_LIMITS["cladding_hydrogen_ppm"], "wppm"),
        ("Control rod B-10 depleted",
         mat_state.get("control_rod_b10_depletion_frac", 0.0) * 100.0,
         PWR_DESIGN_LIMITS["control_rod_b10_depletion_frac"] * 100.0, "%"),
        ("Fuel burnup", mat_state.get("fuel_burnup_gwd_per_mtu", 0.0),
         PWR_DESIGN_LIMITS["fuel_burnup_gwd_per_mtu"], "GWd/MTU"),
    ]
    life_input: dict[str, float] = {
        k: (float("inf") if v is None else float(v)) for k, v in mat_life.items()
    }
    life_label_map = {
        "rpv_embrittlement_years": "RPV embrittlement",
        "cladding_oxide_years": "Cladding oxide",
        "cladding_hydrogen_years": "Cladding hydrogen",
        "control_rod_years": "Control rod B-10",
        "fuel_burnup_years": "Fuel burnup",
    }
    bars, cards, summary = update_damage_panel_children(
        bars_input, life_input, life_label_map
    )

    op_hours = mat_state.get("operating_hours", 0.0)
    tiles = [
        _operator_tile("Thermal Power", f"{power_mw:.0f} MW",
                       f"{p_norm * 100:.1f}% of nominal"),
        _operator_tile("Fuel Temperature", f"{t_fuel - 273.15:.1f} °C",
                       f"ΔT to coolant {t_fuel - t_cool:.1f} K"),
        _operator_tile("Coolant Outlet", f"{t_cool - 273.15:.1f} °C",
                       f"inlet {t_in - 273.15:.1f} °C"),
        _operator_tile("Operating Hours", f"{op_hours:,.0f} h",
                       f"≈ {op_hours / 8760.0:.2f} yr at full power"),
    ]

    return diagram_html, heatmap, bars, cards, summary, tiles


def _operator_tile(label: str, value: str, sub: str) -> dbc.Card:
    return dbc.Card(
        dbc.CardBody([
            html.Div(label, className="op-tile-label"),
            html.Div(value, className="op-tile-value"),
            html.Div(sub, className="op-tile-sub"),
        ]),
        className="operator-tile",
    )


def _controls_card() -> dbc.Card:
    """Build the user input controls for the PWR panel."""
    return dbc.Card(
        dbc.CardBody(
            [
                html.Div("Live Boundary Conditions", className="eyebrow"),
                html.H5("Operator Inputs", className="card-title mb-3"),
                _control_rods_input(),
                _boron_input(),
                _flow_input(),
                _inlet_temperature_input(),
                _simulation_controls(),
            ]
        ),
        className="workstation-card control-card h-100",
    )


def _scenario_card() -> dbc.Card:
    """Build the Scenario Mode launcher card."""
    return dbc.Card(
        dbc.CardBody(
            [
                html.Div("Preset Transients", className="eyebrow"),
                html.H5("Scenario Mode", className="card-title mb-2"),
                html.P(
                    "Restart the PWR twin and immediately run a predefined transient "
                    "with time-dependent boundary conditions.",
                    className="small text-secondary-emphasis",
                ),
                dbc.Label("Select transient", html_for="pwr-scenario-select"),
                dcc.Dropdown(
                    id="pwr-scenario-select",
                    options=pwr_scenario_options(),
                    value="rod_ejection",
                    clearable=False,
                    className="dark-dropdown mb-3",
                ),
                dbc.Button(
                    "Restart With Scenario",
                    id="pwr-start-scenario",
                    color="primary",
                    className="w-100",
                ),
            ]
        ),
        className="workstation-card mt-3",
    )


def _graph_card(
    title: str,
    subtitle: str,
    graph_id: str,
    figure: go.Figure,
    *,
    class_name: str = "",
) -> dbc.Card:
    """Render a graph in a workstation card with built-in interaction guidance."""
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


def _simulation_controls() -> html.Div:
    """Build controls for pausing, resetting, and accelerating the live run."""
    return html.Div(
        [
            _input_label(
                "Simulation run control",
                "speed",
                "pwr-speed-help",
                "Accelerated speeds advance more simulated time per dashboard update.",
                html_for="pwr-time-multiplier",
            ),
            dcc.Dropdown(
                id="pwr-time-multiplier",
                options=TIME_MULTIPLIER_OPTIONS,
                value=1,
                clearable=False,
                searchable=False,
                className="dark-dropdown mb-2",
            ),
            html.Div(id="pwr-speed-summary", className="small text-secondary-emphasis mb-3"),
            dbc.ButtonGroup(
                [
                    dbc.Button(
                        "Pause",
                        id="pwr-pause-toggle",
                        color="warning",
                        outline=True,
                    ),
                    dbc.Button(
                        "Restart Normal",
                        id="pwr-reset-normal",
                        color="secondary",
                        outline=True,
                    ),
                ],
                className="w-100 simulation-button-group",
            ),
            dbc.FormText("Restart Normal clears scenario mode and starts from the current inputs."),
            html.Hr(className="my-3"),
            html.Div("Session persistence (FR-04)", className="eyebrow"),
            dbc.Button(
                "Download HDF5",
                id="pwr-save-hdf5",
                color="info",
                outline=True,
                className="w-100 mb-2",
            ),
            dcc.Download(id="pwr-download-hdf5"),
            dcc.Upload(
                id="pwr-upload-hdf5",
                children=dbc.Button(
                    "Upload HDF5",
                    color="info",
                    outline=True,
                    className="w-100",
                ),
                accept=".h5,.hdf5",
                multiple=False,
                className="w-100",
            ),
            dbc.FormText(
                "Save or restore the current time-series and final 11-state vector.",
                className="mt-2",
            ),
            dbc.Button(
                "Download CSV",
                id="pwr-save-csv",
                color="secondary",
                outline=True,
                className="w-100 mt-2",
            ),
            dcc.Download(id="pwr-download-csv"),
            dbc.FormText(
                "Export the plotted time-series columns for offline analysis.",
                className="mt-2",
            ),
        ],
        className="control-block mb-0",
    )


def _event_log_card() -> dbc.Card:
    """Build a scrollable console for plain-language PWR events."""
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
                id="pwr-event-log",
                children=[],
                flush=True,
                className="event-log",
            ),
        ],
        className="workstation-card event-log-card mt-3",
    )


def _control_rods_input() -> html.Div:
    """Slider for externally inserted control rod reactivity."""
    return html.Div(
        [
            _input_label(
                "Control rod reactivity insertion",
                "pcm",
                "pwr-rod-help",
                "Positive pcm inserts reactivity and raises neutron population; "
                "negative pcm represents additional shutdown worth.",
                html_for="pwr-rod-reactivity",
            ),
            dcc.Slider(
                id="pwr-rod-reactivity",
                min=-1000,
                max=1000,
                step=10,
                value=0,
                marks={-1000: "-1000 pcm", 0: "0", 1000: "+1000 pcm"},
                tooltip={"placement": "bottom", "always_visible": False},
                className="engineering-slider",
            ),
            dbc.FormText(
                "Bounded to ±1000 pcm to keep transients in the educational model envelope."
            ),
        ],
        className="control-block",
    )


def _boron_input() -> html.Div:
    """Numeric input for soluble boron concentration."""
    return html.Div(
        [
            _input_label(
                "Soluble boron concentration",
                "ppm",
                "pwr-boron-help",
                "Boric acid absorbs neutrons in the coolant. Higher ppm adds negative reactivity "
                "for slower, long-term power control.",
                html_for="pwr-boron-ppm",
            ),
            dbc.InputGroup(
                [
                    dbc.Input(
                        id="pwr-boron-ppm",
                        type="number",
                        min=0,
                        max=2500,
                        step=25,
                        value=0,
                        placeholder="0",
                    ),
                    dbc.InputGroupText("ppm"),
                ],
                className="engineering-input",
            ),
            dbc.FormText("Validated range: 0 to 2500 ppm."),
        ],
        className="control-block",
    )


def _flow_input() -> html.Div:
    """Slider for main coolant mass-flow fraction."""
    return html.Div(
        [
            _input_label(
                "Coolant mass flow rate",
                "kg/s",
                "pwr-flow-help",
                "Primary coolant flow removes heat from the lumped coolant node. Lower flow "
                "raises coolant temperature and strengthens moderator feedback.",
                html_for="pwr-flow-fraction",
            ),
            dcc.Slider(
                id="pwr-flow-fraction",
                min=0.2,
                max=1.2,
                step=0.05,
                value=1.0,
                marks={
                    0.2: "20%",
                    0.7: "70%",
                    1.0: "100%",
                    1.2: "120%",
                },
                tooltip={"placement": "bottom", "always_visible": False},
                className="engineering-slider",
            ),
            dbc.FormText(
                "1.00 nominal = "
                f"{NOMINAL_MASS_FLOW_KG_S:,.0f} kg/s; validated range: "
                f"{0.2 * NOMINAL_MASS_FLOW_KG_S:,.0f} to {1.2 * NOMINAL_MASS_FLOW_KG_S:,.0f} kg/s."
            ),
        ],
        className="control-block",
    )


def _inlet_temperature_input() -> html.Div:
    """Numeric input for core inlet temperature."""
    return html.Div(
        [
            _input_label(
                "Core inlet temperature",
                "°C",
                "pwr-inlet-temperature-help",
                "Coolant inlet temperature sets the heat sink for the core. Hotter inlet water "
                "reduces the margin for heat removal in this lumped model.",
                html_for="pwr-inlet-temperature-c",
            ),
            dbc.InputGroup(
                [
                    dbc.Input(
                        id="pwr-inlet-temperature-c",
                        type="number",
                        min=_kelvin_to_celsius_value(540.0),
                        max=_kelvin_to_celsius_value(610.0),
                        step=1,
                        value=round(_kelvin_to_celsius_value(T_IN_NOM) or 0.0, 1),
                        placeholder="291.9",
                    ),
                    dbc.InputGroupText("°C"),
                ],
                className="engineering-input",
            ),
            dbc.FormText("Validated range: 267 to 337 °C (540 to 610 K internally)."),
        ],
        className="control-block mb-0",
    )


def _input_label(
    label: str,
    units: str,
    tooltip_id: str,
    tooltip_text: str,
    *,
    html_for: str,
) -> html.Div:
    """Render a compact engineering input label with units and help text."""
    return html.Div(
        [
            dbc.Label(label, html_for=html_for, className="mb-0"),
            html.Div(
                [
                    html.Span(units, className="unit-badge"),
                    html.Span("?", id=tooltip_id, className="input-help", tabIndex=0),
                    dbc.Tooltip(tooltip_text, target=tooltip_id, placement="right"),
                ],
                className="d-flex align-items-center gap-2",
            ),
        ],
        className="control-label-row",
    )


def _scenario_banner(session: dict[str, Any]) -> dbc.Alert | html.Div:
    """Show active or completed scenario metadata when present."""
    scenario = session.get("scenario") or {}
    label = scenario.get("label")
    if not label:
        return html.Div()
    completed = scenario.get("completed", False)
    description = scenario.get("description", "")
    color = "success" if completed else "info"
    return dbc.Alert(
        [
            html.Strong(label),
            html.Span(f" — {description}" if description else "", className="ms-1"),
        ],
        color=color,
        className="py-2 mb-3 small",
    )


def _add_scrub_marker(fig: go.Figure, time_s: float) -> None:
    """Highlight the scrubbed simulation time on a time-series chart."""
    fig.add_vline(
        x=time_s,
        line_color="#facc15",
        line_width=1.5,
        line_dash="dash",
        opacity=0.85,
    )


def _hist_times(history: dict[str, list[float]], display_units: dict[str, str] | None) -> list[float]:
    return time_values_seconds(history.get("time_s", []), display_units)


def _metric_row(session: dict[str, Any], display_units: dict[str, str] | None = None) -> list:
    """Build compact metric cards from the latest PWR state."""
    metrics = session.get("metrics", {})
    p_norm = metrics.get("power_normalized")
    power_pct = f"{float(p_norm) * 100.0:.1f} %" if p_norm is not None else "--"
    temp_unit = temperature_unit_suffix(display_units)
    tiles = dbc.Row(
        [
            dbc.Col(
                _metric_card(
                    "Sim time",
                    format_simulation_time(session.get("time_s"), display_units),
                ),
                md=2,
            ),
            dbc.Col(
                _metric_card("Thermal Power", _format_value(metrics.get("power_mw"), "MWth")),
                md=2,
            ),
            dbc.Col(_metric_card("Power level", power_pct), md=2),
            dbc.Col(
                _metric_card(
                    "Fuel Temp",
                    _format_value(
                        temperature_scalar_kelvin(metrics.get("fuel_temperature_k"), display_units),
                        temp_unit,
                    ),
                ),
                md=2,
            ),
            dbc.Col(
                _metric_card(
                    "Coolant Temp",
                    _format_value(
                        temperature_scalar_kelvin(metrics.get("coolant_temperature_k"), display_units),
                        temp_unit,
                    ),
                ),
                md=2,
            ),
            dbc.Col(
                _metric_card("Reactivity", _format_value(metrics.get("total_reactivity_pcm"), "pcm")),
                md=2,
            ),
            dbc.Col(
                _metric_card(
                    "Xenon worth",
                    _format_value(metrics.get("xenon_reactivity_pcm"), "pcm"),
                ),
                md=2,
            ),
        ],
        className="g-3",
    )
    banner = _scenario_banner(session)
    if isinstance(banner, dbc.Alert):
        return [banner, tiles]
    return [tiles]


def _metric_card(label: str, value: str) -> dbc.Card:
    """Render one dashboard metric tile."""
    return dbc.Card(
        dbc.CardBody(
            [
                html.Div(label, className="text-muted small"),
                html.Div(value, className="fs-5 fw-semibold metric-value"),
            ]
        ),
        className="metric-card h-100",
    )


def _event_log_items(
    session: dict[str, Any],
    display_units: dict[str, str] | None = None,
) -> list[dbc.ListGroupItem]:
    """Render recent event records as bootstrap list items."""
    events = list(session.get("events", []))[-20:]
    if not events:
        return [
            dbc.ListGroupItem(
                "No events recorded yet.",
                className="event-log-item text-secondary-emphasis small",
            )
        ]

    items: list[dbc.ListGroupItem] = []
    for event in reversed(events):
        severity = event.get("severity", "info")
        items.append(
            dbc.ListGroupItem(
                [
                    html.Div(
                        [
                            html.Span(
                                format_simulation_time(event.get("time_s"), display_units),
                                className="fw-semibold me-2",
                            ),
                            html.Span(
                                str(event.get("message", "")),
                                className="small",
                            ),
                        ]
                    )
                ],
                color="warning" if severity == "warning" else None,
                className="event-log-item py-2",
            )
        )
    return items


def _format_event_time(value: Any) -> str:
    """Format event times for the console."""
    try:
        seconds = float(value)
    except (TypeError, ValueError):
        return "t=--"

    if seconds >= 3600.0:
        return f"t={seconds / 3600.0:.2f} h"
    if seconds >= 60.0:
        return f"t={seconds / 60.0:.1f} min"
    return f"t={seconds:.1f} s"


def _format_value(value: Any, units: str) -> str:
    """Format numeric dashboard values without exposing Python errors."""
    if value is None:
        return "--"
    try:
        return f"{float(value):,.2f} {units}"
    except (TypeError, ValueError):
        return "--"


def _coerce_time_multiplier(value: Any) -> int:
    """Return a supported accelerated-mode multiplier from Dash input."""
    try:
        multiplier = int(value)
    except (TypeError, ValueError):
        return 1
    valid = {option["value"] for option in TIME_MULTIPLIER_OPTIONS}
    return multiplier if multiplier in valid else 1


def _speed_summary(multiplier: int) -> str:
    """Describe the effective live speed in operator-facing terms."""
    if multiplier == 1:
        return "Smooth live mode: 1 simulated second per real second."
    if multiplier == 60:
        return "Accelerated mode: roughly 1 simulated minute per real second."
    return "Xenon timescale mode: roughly 1 simulated hour per real second."


def _run_live_step(
    session: dict[str, Any] | None,
    controls: dict[str, Any],
    multiplier: int,
) -> dict[str, Any]:
    """Advance a short lookahead window so the browser receives smaller, faster patches."""
    return run_pwr_control_step(
        session,
        controls,
        step_seconds=LIVE_STEP_SECONDS,
        time_multiplier=multiplier,
        min_output_points=LIVE_MIN_OUTPUT_POINTS,
        max_history_points=LIVE_MAX_DISPLAY_POINTS,
        history_retention="timeline",
    )


def _power_figure(
    session: dict[str, Any],
    display_units: dict[str, str] | None = None,
) -> go.Figure:
    """Build the transient power chart from session history (full rebuild)."""
    history = session.get("history", {})
    fig = _empty_power_figure()
    times = _hist_times(history, display_units)
    fig.data[0].x = times
    fig.data[0].y = history.get("power_mw", [])
    fig.data[1].x = times
    fig.data[1].y = [
        100.0 * float(v) for v in history.get("power_normalized", [])
    ]
    fig.update_xaxes(title_text=time_axis_label(display_units))
    return fig


def _patch_trim(patch: Patch, trace_idx: int, n: int) -> None:
    """Delete the first n elements from x and y of a trace via individual Patch deletes."""
    for _ in range(n):
        del patch["data"][trace_idx]["x"][0]
        del patch["data"][trace_idx]["y"][0]


def _patch_power_figure(
    session: dict[str, Any],
    n_new: int,
    n_trim: int,
    display_units: dict[str, str] | None = None,
) -> Patch:
    """Return a Patch extending the power trace with only the latest n_new points."""
    history = session.get("history", {})
    patch = Patch()
    new_times = _hist_times(
        {"time_s": history.get("time_s", [])[-n_new:] if n_new else []},
        display_units,
    )
    patch["data"][0]["x"].extend(new_times)
    patch["data"][0]["y"].extend(history.get("power_mw", [])[-n_new:] if n_new else [])
    patch["data"][1]["x"].extend(new_times)
    patch["data"][1]["y"].extend(
        [100.0 * float(v) for v in history.get("power_normalized", [])[-n_new:]]
        if n_new else []
    )
    if n_trim > 0:
        _patch_trim(patch, 0, n_trim)
        _patch_trim(patch, 1, n_trim)
    return patch


def _temperature_figure(
    session: dict[str, Any],
    display_units: dict[str, str] | None = None,
) -> go.Figure:
    """Build the fuel and coolant temperature chart from session history (full rebuild)."""
    history = session.get("history", {})
    fig = _empty_temperature_figure()
    times = _hist_times(history, display_units)
    fig.data[0].x = times
    fig.data[0].y = temperature_values_kelvin(history.get("fuel_temperature_k", []), display_units)
    fig.data[1].x = times
    fig.data[1].y = temperature_values_kelvin(history.get("coolant_temperature_k", []), display_units)
    fig.update_xaxes(title_text=time_axis_label(display_units))
    fig.update_yaxes(title_text=temperature_axis_label(display_units))
    return fig


def _patch_temperature_figure(
    session: dict[str, Any],
    n_new: int,
    n_trim: int,
    display_units: dict[str, str] | None = None,
) -> Patch:
    """Return a Patch extending fuel/coolant traces with the latest n_new points."""
    history = session.get("history", {})
    patch = Patch()
    new_times = _hist_times(
        {"time_s": history.get("time_s", [])[-n_new:] if n_new else []},
        display_units,
    )
    new_fuel = temperature_values_kelvin(
        history.get("fuel_temperature_k", [])[-n_new:] if n_new else [],
        display_units,
    )
    new_cool = temperature_values_kelvin(
        history.get("coolant_temperature_k", [])[-n_new:] if n_new else [],
        display_units,
    )
    for i, new_y in enumerate([new_fuel, new_cool]):
        patch["data"][i]["x"].extend(new_times)
        patch["data"][i]["y"].extend(new_y)
        if n_trim > 0:
            _patch_trim(patch, i, n_trim)
    return patch


_REACTIVITY_TRACE_KEYS = [
    "total_reactivity_pcm",
    "rod_reactivity_pcm",
    "doppler_reactivity_pcm",
    "moderator_reactivity_pcm",
    "boron_reactivity_pcm",
    "xenon_reactivity_pcm",
]


def _reactivity_figure(
    session: dict[str, Any],
    display_units: dict[str, str] | None = None,
) -> go.Figure:
    """Build the reactivity breakdown chart from session history (full rebuild)."""
    history = session.get("history", {})
    fig = _empty_reactivity_figure()
    times = _hist_times(history, display_units)
    for i, key in enumerate(_REACTIVITY_TRACE_KEYS):
        fig.data[i].x = times
        fig.data[i].y = history.get(key, [])
    fig.update_xaxes(title_text=time_axis_label(display_units))
    _apply_reactivity_reference_lines(fig, session)
    return fig


def _patch_reactivity_figure(
    session: dict[str, Any],
    n_new: int,
    n_trim: int,
    display_units: dict[str, str] | None = None,
) -> Patch:
    """Return a Patch extending all 6 reactivity traces with the latest n_new points."""
    history = session.get("history", {})
    patch = Patch()
    new_times = _hist_times(
        {"time_s": history.get("time_s", [])[-n_new:] if n_new else []},
        display_units,
    )
    for i, key in enumerate(_REACTIVITY_TRACE_KEYS):
        patch["data"][i]["x"].extend(new_times)
        patch["data"][i]["y"].extend(history.get(key, [])[-n_new:] if n_new else [])
        if n_trim > 0:
            _patch_trim(patch, i, n_trim)
    return patch


def _xenon_iodine_figure(
    session: dict[str, Any],
    display_units: dict[str, str] | None = None,
) -> go.Figure:
    """Build the iodine/xenon concentration chart with dual y-axes (full rebuild)."""
    history = session.get("history", {})
    fig = _empty_xenon_iodine_figure()
    times = _hist_times(history, display_units)
    fig.data[0].x = times
    fig.data[0].y = history.get("iodine_concentration", [])
    fig.data[1].x = times
    fig.data[1].y = history.get("xenon_concentration", [])
    fig.update_xaxes(title_text=time_axis_label(display_units))
    return fig


def _patch_xenon_iodine_figure(
    session: dict[str, Any],
    n_new: int,
    n_trim: int,
    display_units: dict[str, str] | None = None,
) -> Patch:
    """Return a Patch extending I-135 and Xe-135 traces with the latest n_new points."""
    history = session.get("history", {})
    patch = Patch()
    new_times = _hist_times(
        {"time_s": history.get("time_s", [])[-n_new:] if n_new else []},
        display_units,
    )
    for i, key in enumerate(["iodine_concentration", "xenon_concentration"]):
        patch["data"][i]["x"].extend(new_times)
        patch["data"][i]["y"].extend(history.get(key, [])[-n_new:] if n_new else [])
        if n_trim > 0:
            _patch_trim(patch, i, n_trim)
    return patch


_DERIVATIVE_KEYS = (
    ("dn_dt", "dn/dt"),
    ("dT_fuel_dt", "dT_fuel/dt"),
    ("dT_coolant_dt", "dT_cool/dt"),
    ("dI_dt", "dI/dt"),
    ("dX_dt", "dX/dt"),
)
_DERIVATIVE_COLORS = ["#f8fafc", "#fb7185", "#2dd4bf", "#facc15", "#c084fc"]


def _empty_derivatives_figure() -> go.Figure:
    fig = go.Figure()
    for (key, label), color in zip(_DERIVATIVE_KEYS, _DERIVATIVE_COLORS):
        fig.add_trace(
            go.Scatter(
                x=[], y=[], mode="lines", name=label,
                line=_smooth_line(color, 2),
            )
        )
    _apply_timeseries_layout(
        fig,
        title="State Derivatives",
        yaxis_title="Rate (various units/s)",
        height=360,
    )
    return fig


def _derivatives_figure(
    session: dict[str, Any],
    display_units: dict[str, str] | None = None,
) -> go.Figure:
    history = session.get("history", {})
    fig = _empty_derivatives_figure()
    times = _hist_times(history, display_units)
    for i, (key, _) in enumerate(_DERIVATIVE_KEYS):
        fig.data[i].x = times
        fig.data[i].y = history.get(key, [])
    fig.update_xaxes(title_text=time_axis_label(display_units))
    return fig


def _patch_derivatives_figure(
    session: dict[str, Any],
    n_new: int,
    n_trim: int,
    display_units: dict[str, str] | None = None,
) -> Patch:
    history = session.get("history", {})
    patch = Patch()
    new_times = _hist_times(
        {"time_s": history.get("time_s", [])[-n_new:] if n_new else []},
        display_units,
    )
    for i, (key, _) in enumerate(_DERIVATIVE_KEYS):
        patch["data"][i]["x"].extend(new_times)
        patch["data"][i]["y"].extend(history.get(key, [])[-n_new:] if n_new else [])
        if n_trim > 0:
            _patch_trim(patch, i, n_trim)
    return patch


def _apply_reactivity_reference_lines(fig: go.Figure, session: dict[str, Any]) -> None:
    """Add criticality base ρ and typical xenon equilibrium worth band (teaching reference)."""
    rho_base = float(session.get("base_reactivity_pcm", 0.0))
    shapes: list[dict[str, Any]] = [
        {
            "type": "rect",
            "xref": "paper",
            "yref": "y",
            "x0": 0,
            "x1": 1,
            "y0": -3000,
            "y1": -2500,
            "fillcolor": "rgba(192, 132, 252, 0.1)",
            "line": {"width": 0},
            "layer": "below",
        },
    ]
    annotations: list[dict[str, Any]] = [
        {
            "xref": "paper",
            "yref": "y",
            "x": 1.0,
            "y": -2750,
            "xanchor": "right",
            "text": "Typical Xe equilibrium band",
            "showarrow": False,
            "font": {"size": 10, "color": "#c084fc"},
        },
    ]
    if rho_base:
        shapes.append(
            {
                "type": "line",
                "xref": "paper",
                "yref": "y",
                "x0": 0,
                "x1": 1,
                "y0": rho_base,
                "y1": rho_base,
                "line": {"color": "#64748b", "width": 1, "dash": "longdash"},
            }
        )
        annotations.append(
            {
                "xref": "paper",
                "yref": "y",
                "x": 0,
                "y": rho_base,
                "xanchor": "left",
                "text": f"ρ_base = {rho_base:+.0f} pcm",
                "showarrow": False,
                "font": {"size": 10, "color": "#94a3b8"},
            }
        )
    fig.update_layout(shapes=shapes, annotations=annotations)


def _empty_neutron_figure() -> go.Figure:
    """Return dual-axis figure for neutron population and thermal flux."""
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Scatter(
            x=[], y=[], mode="lines", name="n (normalized)",
            line=_smooth_line(COLORS["net"], 2),
            hovertemplate="t=%{x:.1f} s<br>n=%{y:.4f}<extra></extra>",
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=[], y=[], mode="lines", name="φ (n/cm²·s)",
            line=_smooth_line(COLORS["moderator"], 2, "dot"),
            hovertemplate="t=%{x:.1f} s<br>φ=%{y:.3e}<extra></extra>",
        ),
        secondary_y=True,
    )
    _apply_timeseries_layout(
        fig,
        title="Neutron Population & Thermal Flux",
        yaxis_title="Population n",
        height=360,
    )
    fig.update_yaxes(title_text="Thermal flux φ (n/cm²·s)", secondary_y=True, type="log")
    return fig


def _neutron_figure(
    session: dict[str, Any],
    display_units: dict[str, str] | None = None,
) -> go.Figure:
    history = session.get("history", {})
    fig = _empty_neutron_figure()
    times = _hist_times(history, display_units)
    fig.data[0].x = times
    fig.data[0].y = history.get("neutron_population", [])
    fig.data[1].x = times
    fig.data[1].y = history.get("thermal_flux_n_cm2_s", [])
    fig.update_xaxes(title_text=time_axis_label(display_units))
    return fig


def _patch_neutron_figure(
    session: dict[str, Any],
    n_new: int,
    n_trim: int,
    display_units: dict[str, str] | None = None,
) -> Patch:
    history = session.get("history", {})
    patch = Patch()
    new_times = _hist_times(
        {"time_s": history.get("time_s", [])[-n_new:] if n_new else []},
        display_units,
    )
    for i, key in enumerate(["neutron_population", "thermal_flux_n_cm2_s"]):
        patch["data"][i]["x"].extend(new_times)
        patch["data"][i]["y"].extend(history.get(key, [])[-n_new:] if n_new else [])
        if n_trim > 0:
            _patch_trim(patch, i, n_trim)
    return patch


def _empty_power_rho_figure() -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=[], y=[], mode="lines+markers", name="Power–ρ trajectory",
            marker=dict(color=COLORS["power"], size=3, opacity=0.7),
            line=dict(color=COLORS["power"], width=2),
            hovertemplate="ρ=%{x:.1f} pcm<br>Power=%{y:.2f} MWth<extra></extra>",
        )
    )
    _apply_timeseries_layout(
        fig,
        title="Power vs Total Reactivity",
        yaxis_title="Thermal power (MWth)",
        height=360,
    )
    fig.update_xaxes(title_text="Total reactivity (pcm)", rangeslider={"visible": False})
    fig.update_layout(xaxis_title="Total reactivity (pcm)")
    return fig


def _power_rho_figure(session: dict[str, Any]) -> go.Figure:
    history = session.get("history", {})
    fig = _empty_power_rho_figure()
    fig.data[0].x = history.get("total_reactivity_pcm", [])
    fig.data[0].y = history.get("power_mw", [])
    return fig


def _patch_power_rho_figure(session: dict[str, Any], n_new: int, n_trim: int) -> Patch:
    history = session.get("history", {})
    patch = Patch()
    patch["data"][0]["x"].extend(history.get("total_reactivity_pcm", [])[-n_new:] if n_new else [])
    patch["data"][0]["y"].extend(history.get("power_mw", [])[-n_new:] if n_new else [])
    if n_trim > 0:
        _patch_trim(patch, 0, n_trim)
    return patch


def _empty_precursor_figure() -> go.Figure:
    """Return the base Plotly figure for all 6 delayed precursor groups with pre-seeded traces."""
    fig = go.Figure()
    for label, color in zip(_PRECURSOR_LABELS, _PRECURSOR_COLORS):
        fig.add_trace(go.Scatter(
            x=[], y=[], mode="lines", name=label,
            line=_smooth_line(color, 2),
            hovertemplate=f"t=%{{x:.1f}} s<br>{label}=%{{y:.3e}}<extra></extra>",
        ))
    _apply_timeseries_layout(
        fig,
        title="Delayed Neutron Precursor Groups",
        yaxis_title="Concentration (neutrons/cm³)",
        height=400,
    )
    return fig


def _precursor_figure(
    session: dict[str, Any],
    display_units: dict[str, str] | None = None,
) -> go.Figure:
    """Build the 6-group precursor chart from session history (full rebuild)."""
    history = session.get("history", {})
    fig = _empty_precursor_figure()
    times = _hist_times(history, display_units)
    for i, key in enumerate(_PRECURSOR_KEYS):
        fig.data[i].x = times
        fig.data[i].y = history.get(key, [])
    fig.update_xaxes(title_text=time_axis_label(display_units))
    return fig


def _patch_precursor_figure(
    session: dict[str, Any],
    n_new: int,
    n_trim: int,
    display_units: dict[str, str] | None = None,
) -> Patch:
    """Return a Patch extending all 6 precursor traces with the latest n_new points."""
    history = session.get("history", {})
    patch = Patch()
    new_times = _hist_times(
        {"time_s": history.get("time_s", [])[-n_new:] if n_new else []},
        display_units,
    )
    for i, key in enumerate(_PRECURSOR_KEYS):
        patch["data"][i]["x"].extend(new_times)
        patch["data"][i]["y"].extend(history.get(key, [])[-n_new:] if n_new else [])
        if n_trim > 0:
            _patch_trim(patch, i, n_trim)
    return patch


def _empty_delta_t_figure() -> go.Figure:
    """Return the base Plotly figure for fuel-coolant ΔT with pre-seeded trace."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[], y=[], mode="lines", name="ΔT (fuel − coolant)",
        line=_smooth_line("#f59e0b", 2),
        hovertemplate="t=%{x:.1f} s<br>ΔT=%{y:.1f} °C<extra></extra>",
    ))
    _apply_timeseries_layout(
        fig,
        title="Fuel–Coolant ΔT",
        yaxis_title="ΔT (°C)",
        height=300,
    )
    return fig


def _delta_t_figure(
    session: dict[str, Any],
    display_units: dict[str, str] | None = None,
) -> go.Figure:
    """Build the fuel-coolant ΔT chart from session history (full rebuild)."""
    history = session.get("history", {})
    fuel = temperature_values_kelvin(history.get("fuel_temperature_k", []), display_units)
    cool = temperature_values_kelvin(history.get("coolant_temperature_k", []), display_units)
    delta_t = [f - c for f, c in zip(fuel, cool)]
    fig = _empty_delta_t_figure()
    fig.data[0].x = _hist_times(history, display_units)
    fig.data[0].y = delta_t
    fig.update_xaxes(title_text=time_axis_label(display_units))
    return fig


def _patch_delta_t_figure(
    session: dict[str, Any],
    n_new: int,
    n_trim: int,
    display_units: dict[str, str] | None = None,
) -> Patch:
    """Return a Patch extending the ΔT trace with the latest n_new points."""
    history = session.get("history", {})
    patch = Patch()
    new_times = _hist_times(
        {"time_s": history.get("time_s", [])[-n_new:] if n_new else []},
        display_units,
    )
    fuel = temperature_values_kelvin(
        history.get("fuel_temperature_k", [])[-n_new:] if n_new else [],
        display_units,
    )
    cool = temperature_values_kelvin(
        history.get("coolant_temperature_k", [])[-n_new:] if n_new else [],
        display_units,
    )
    new_delta = [f - c for f, c in zip(fuel, cool)]
    patch["data"][0]["x"].extend(new_times)
    patch["data"][0]["y"].extend(new_delta)
    if n_trim > 0:
        _patch_trim(patch, 0, n_trim)
    return patch


_CONTROL_SCHEDULE_KEYS = (
    "boron_ppm",
    "coolant_flow_fraction",
    "rod_reactivity_pcm",
    "inlet_temperature_k",
)


def _empty_control_schedule_figure() -> go.Figure:
    """Return the base Plotly figure for operator control schedules (2×2)."""
    fig = make_subplots(
        rows=2,
        cols=2,
        specs=[[{}, {"secondary_y": True}], [{}, {}]],
        subplot_titles=("Boron", "Flow & ṁ", "Rod reactivity", "Inlet temperature"),
        vertical_spacing=0.14,
        horizontal_spacing=0.08,
    )
    fig.add_trace(
        go.Scatter(
            x=[], y=[], mode="lines", name="Boron (ppm)",
            line=_smooth_line(COLORS["boron"], 2, "dash"),
        ),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=[], y=[], mode="lines", name="Flow fraction",
            line={"color": COLORS["coolant"], "width": 2, "shape": "hv"},
        ),
        row=1, col=2, secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=[], y=[], mode="lines", name="ṁ (kg/s)",
            line={"color": COLORS["power"], "width": 2, "dash": "dot", "shape": "hv"},
        ),
        row=1, col=2, secondary_y=True,
    )
    fig.add_trace(
        go.Scatter(
            x=[], y=[], mode="lines", name="Rod (pcm)",
            line={"color": COLORS["rod"], "width": 2, "shape": "hv"},
        ),
        row=2, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=[], y=[], mode="lines", name="T_in (°C)",
            line=_smooth_line(COLORS["moderator"], 2),
        ),
        row=2, col=2,
    )
    _apply_timeseries_layout(
        fig,
        title="Control Inputs Schedule",
        yaxis_title="",
        height=420,
    )
    fig.update_yaxes(title_text="ppm", row=1, col=1)
    fig.update_yaxes(title_text="Fraction", row=1, col=2, secondary_y=False)
    fig.update_yaxes(title_text="ṁ (kg/s)", row=1, col=2, secondary_y=True, showgrid=False)
    fig.update_yaxes(title_text="pcm", row=2, col=1)
    fig.update_yaxes(title_text="°C", row=2, col=2)
    return fig


def _control_schedule_series(
    history: dict[str, list[float]],
    display_units: dict[str, str] | None = None,
) -> tuple[list[float], list[list[float]]]:
    """Extract aligned control schedule traces from history."""
    times = _hist_times(history, display_units)
    flow = history.get("coolant_flow_fraction", [])
    m_dot = [float(v) * NOMINAL_MASS_FLOW_KG_S for v in flow]
    inlet_display = temperature_values_kelvin(
        history.get("inlet_temperature_k", []),
        display_units,
    )
    return times, [
        history.get("boron_ppm", []),
        flow,
        m_dot,
        history.get("rod_reactivity_pcm", []),
        inlet_display,
    ]


def _control_schedule_figure(
    session: dict[str, Any],
    display_units: dict[str, str] | None = None,
) -> go.Figure:
    """Build the control inputs schedule chart from session history (full rebuild)."""
    history = session.get("history", {})
    fig = _empty_control_schedule_figure()
    times, series = _control_schedule_series(history, display_units)
    for idx, y_vals in enumerate(series):
        fig.data[idx].x = times
        fig.data[idx].y = y_vals
    return fig


def _patch_control_schedule_figure(
    session: dict[str, Any],
    n_new: int,
    n_trim: int,
    display_units: dict[str, str] | None = None,
) -> Patch:
    """Return a Patch extending all control schedule traces."""
    history = session.get("history", {})
    patch = Patch()
    trimmed = {k: (v[-n_new:] if n_new else []) for k, v in history.items() if isinstance(v, list)}
    new_times, series = _control_schedule_series(trimmed, display_units)
    for i, y_vals in enumerate(series):
        patch["data"][i]["x"].extend(new_times)
        patch["data"][i]["y"].extend(y_vals)
        if n_trim > 0:
            _patch_trim(patch, i, n_trim)
    return patch


def _empty_xenon_worth_figure() -> go.Figure:
    """Return the base Plotly figure for xenon reactivity worth vs Xe-135 concentration."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[], y=[], mode="lines+markers", name="Xe-135 worth trajectory",
        marker=dict(color=COLORS["xenon"], size=3, opacity=0.7),
        line=dict(color=COLORS["xenon"], width=2),
        hovertemplate="N_Xe=%{x:.3e} /cm³<br>ρ_Xe=%{y:.1f} pcm<extra></extra>",
    ))
    _apply_timeseries_layout(
        fig,
        title="Xenon Worth vs Concentration",
        yaxis_title="Xenon reactivity (pcm)",
        height=300,
    )
    fig.update_xaxes(
        title_text="Xe-135 concentration (atoms/cm³)",
        rangeslider={"visible": False},
    )
    return fig


def _xenon_worth_figure(session: dict[str, Any]) -> go.Figure:
    """Build the Xe worth vs concentration phase-plane from session history (full rebuild)."""
    history = session.get("history", {})
    fig = _empty_xenon_worth_figure()
    fig.data[0].x = history.get("xenon_concentration", [])
    fig.data[0].y = history.get("xenon_reactivity_pcm", [])
    return fig


def _patch_xenon_worth_figure(session: dict[str, Any], n_new: int, n_trim: int) -> Patch:
    """Return a Patch extending the Xe phase-plane trace with the latest n_new points."""
    history = session.get("history", {})
    patch = Patch()
    patch["data"][0]["x"].extend(history.get("xenon_concentration", [])[-n_new:] if n_new else [])
    patch["data"][0]["y"].extend(history.get("xenon_reactivity_pcm", [])[-n_new:] if n_new else [])
    if n_trim > 0:
        _patch_trim(patch, 0, n_trim)
    return patch


def _empty_power_figure() -> go.Figure:
    """Return the base Plotly figure for PWR power with MWth and % nominal traces."""
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Scatter(
            x=[], y=[], mode="lines", name="Thermal power",
            line=_smooth_line(COLORS["power"], 3),
            hovertemplate="t=%{x:.1f} s<br>Power=%{y:.2f} MWth<extra></extra>",
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=[], y=[], mode="lines", name="Power level",
            line=_smooth_line(COLORS["rod"], 2, "dot"),
            hovertemplate="t=%{x:.1f} s<br>Level=%{y:.1f} %<extra></extra>",
        ),
        secondary_y=True,
    )
    _apply_timeseries_layout(
        fig,
        title="Core Thermal Power",
        yaxis_title="Thermal power (MWth)",
        height=420,
    )
    fig.update_yaxes(title_text="% of nominal", secondary_y=True, showgrid=False)
    fig.add_hline(
        y=100.0,
        line_dash="dot",
        line_color="#64748b",
        line_width=1,
        secondary_y=True,
        annotation_text="100% nominal",
        annotation_position="right",
    )
    return fig


def _empty_temperature_figure() -> go.Figure:
    """Return the base Plotly figure for PWR temperatures with pre-seeded empty traces."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[], y=[], mode="lines", name="Fuel",
        line=_smooth_line(COLORS["fuel"], 3),
        hovertemplate="t=%{x:.1f} s<br>Fuel=%{y:.2f} °C<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=[], y=[], mode="lines", name="Coolant",
        line=_smooth_line(COLORS["coolant"], 3),
        hovertemplate="t=%{x:.1f} s<br>Coolant=%{y:.2f} °C<extra></extra>",
    ))
    _apply_timeseries_layout(
        fig,
        title="Fuel and Coolant Temperatures",
        yaxis_title="Temperature (°C)",
        height=400,
    )
    return fig


def _empty_reactivity_figure() -> go.Figure:
    """Return the base Plotly figure for reactivity components with pre-seeded empty traces."""
    _traces = [
        ("Net",      COLORS["net"],      3, None),
        ("Rod input",COLORS["rod"],      2, "dash"),
        ("Doppler",  COLORS["doppler"],  2, None),
        ("Moderator",COLORS["moderator"],2, None),
        ("Boron",    COLORS["boron"],    2, "dot"),
        ("Xenon",    COLORS["xenon"],    2, None),
    ]
    _keys = [
        "total_reactivity_pcm", "rod_reactivity_pcm", "doppler_reactivity_pcm",
        "moderator_reactivity_pcm", "boron_reactivity_pcm", "xenon_reactivity_pcm",
    ]
    fig = go.Figure()
    for (name, color, width, dash), key in zip(_traces, _keys):
        line = _smooth_line(color, width, dash)
        fig.add_trace(go.Scatter(
            x=[], y=[], mode="lines", name=name, line=line,
            hovertemplate=f"t=%{{x:.1f}} s<br>{name}=%{{y:.2f}} pcm<extra></extra>",
        ))
    _apply_timeseries_layout(
        fig,
        title="Reactivity Breakdown",
        yaxis_title="Reactivity (pcm)",
        height=400,
    )
    return fig


def _empty_xenon_iodine_figure() -> go.Figure:
    """Return the base Plotly figure for I/Xe concentrations with pre-seeded empty traces."""
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(
        x=[], y=[], mode="lines", name="I-135",
        line=_smooth_line(COLORS["iodine"], 3),
        hovertemplate="t=%{x:.1f} s<br>I-135=%{y:.3e} atoms/cm^3<extra></extra>",
    ), secondary_y=False)
    fig.add_trace(go.Scatter(
        x=[], y=[], mode="lines", name="Xe-135",
        line=_smooth_line(COLORS["xenon"], 3),
        hovertemplate="t=%{x:.1f} s<br>Xe-135=%{y:.3e} atoms/cm^3<extra></extra>",
    ), secondary_y=True)
    _apply_timeseries_layout(
        fig,
        title="Iodine and Xenon Concentrations",
        yaxis_title="I-135 (atoms/cm^3)",
        height=420,
    )
    fig.update_yaxes(title_text="Xe-135 (atoms/cm^3)", secondary_y=True)
    return fig


def _smooth_line(color: str, width: int, dash: str | None = None) -> dict[str, Any]:
    """Return a Plotly line style that visually smooths dense live traces."""
    line: dict[str, Any] = {
        "color": color,
        "width": width,
        "shape": "spline",
        "smoothing": 1.15,
    }
    if dash:
        line["dash"] = dash
    return line


def _apply_timeseries_layout(
    fig: go.Figure,
    *,
    title: str,
    yaxis_title: str,
    height: int,
) -> None:
    """Apply a consistent engineering-workstation style to time-series figures."""
    fig.update_layout(
        title={"text": "", "x": 0.02, "xanchor": "left"},
        template="plotly_dark",
        paper_bgcolor="#111827",
        plot_bgcolor="#0f172a",
        font={"family": "Inter, Arial, sans-serif", "color": "#dbeafe"},
        height=height,
        autosize=True,
        margin={"l": 64, "r": 64, "t": 24, "b": 64},
        dragmode="pan",
        xaxis_title="Simulation time (s)",
        yaxis_title=yaxis_title,
        legend={
            "orientation": "h",
            "y": 1.12,
            "x": 0.0,
            "bgcolor": "rgba(15, 23, 42, 0.75)",
        },
        hovermode="x unified",
        hoverlabel={
            "bgcolor": "#020617",
            "bordercolor": "#334155",
            "font_color": "#e2e8f0",
        },
        uirevision=UIREVISION,
    )
    fig.update_xaxes(
        fixedrange=False,
        minallowed=0,
        rangemode="nonnegative",
        rangeslider={
            "visible": True,
            "bgcolor": "#020617",
            "bordercolor": "#334155",
            "borderwidth": 1,
            "thickness": 0.12,
        },
        showgrid=True,
        gridcolor="#243449",
        zeroline=False,
        linecolor="#475569",
    )
    fig.update_yaxes(
        fixedrange=False,
        showgrid=True,
        gridcolor="#243449",
        zeroline=True,
        zerolinecolor="#64748b",
        linecolor="#475569",
    )


def _kelvin_to_celsius(values: list[float]) -> list[float]:
    """Convert stored Kelvin values into Celsius for operator-facing plots."""
    return [float(value) - 273.15 for value in values]


def _kelvin_to_celsius_value(value: Any) -> float | None:
    """Convert one Kelvin value into Celsius while preserving empty UI state."""
    if value is None or value == "":
        return None
    try:
        return float(value) - 273.15
    except (TypeError, ValueError):
        return None


def _celsius_to_kelvin_input(value: Any) -> float | None:
    """Convert the operator-facing Celsius input to the runner's Kelvin schema."""
    if value is None or value == "":
        return None
    try:
        return float(value) + 273.15
    except (TypeError, ValueError):
        return None


def _register_pwr_csv_callbacks(app: Dash) -> None:
    """CSV export of the current PWR session history."""

    @app.callback(
        Output("pwr-download-csv", "data"),
        Input("pwr-save-csv", "n_clicks"),
        State("pwr-session-store", "data"),
        prevent_initial_call=True,
    )
    def _download_pwr_csv(n_clicks: int | None, session: dict[str, Any] | None) -> Any:
        if not n_clicks or not session:
            return no_update
        history = session.get("history") or {}
        if not history.get("time_s"):
            return no_update
        return {
            "content": history_to_csv_text(history),
            "filename": "pwr_history.csv",
            "type": "text/csv",
        }


def _register_pwr_hdf5_callbacks(app: Dash) -> None:
    """HDF5 save (download) and load (upload) for PWR sessions (FR-04)."""

    @app.callback(
        Output("pwr-download-hdf5", "data"),
        Input("pwr-save-hdf5", "n_clicks"),
        State("pwr-session-store", "data"),
        prevent_initial_call=True,
    )
    def _download_pwr_session(n_clicks: int | None, session: dict[str, Any] | None) -> Any:
        if not n_clicks or not session or not session.get("history", {}).get("time_s"):
            return no_update
        try:
            payload = session_to_hdf5_bytes(session)
        except (ValueError, HDF5StorageError) as exc:
            return no_update
        return {
            "content": base64.b64encode(payload).decode("ascii"),
            "filename": "pwr_session.h5",
            "type": "application/x-hdf5",
            "base64": True,
        }

    @app.callback(
        Output("pwr-session-store", "data", allow_duplicate=True),
        Output("pwr-status-alert", "children", allow_duplicate=True),
        Output("pwr-status-alert", "color", allow_duplicate=True),
        Output("pwr-metrics", "children", allow_duplicate=True),
        Output("pwr-power-graph", "figure", allow_duplicate=True),
        Output("pwr-temperature-graph", "figure", allow_duplicate=True),
        Output("pwr-reactivity-graph", "figure", allow_duplicate=True),
        Output("pwr-xenon-iodine-graph", "figure", allow_duplicate=True),
        Output("pwr-neutron-graph", "figure", allow_duplicate=True),
        Output("pwr-power-rho-graph", "figure", allow_duplicate=True),
        Output("pwr-precursor-graph", "figure", allow_duplicate=True),
        Output("pwr-delta-t-graph", "figure", allow_duplicate=True),
        Output("pwr-control-schedule-graph", "figure", allow_duplicate=True),
        Output("pwr-xenon-worth-scatter", "figure", allow_duplicate=True),
        Output("pwr-derivatives-graph", "figure", allow_duplicate=True),
        Output("pwr-event-log", "children", allow_duplicate=True),
        Input("pwr-upload-hdf5", "contents"),
        State("pwr-upload-hdf5", "filename"),
        prevent_initial_call=True,
    )
    def _load_pwr_session(contents: str | None, filename: str | None) -> tuple:
        if not contents:
            return (no_update,) * 16
        try:
            _, encoded = contents.split(",", 1)
            loaded = load_pwr_session_from_bytes(base64.b64decode(encoded))
        except (ValueError, HDF5StorageError, OSError) as exc:
            return (
                no_update,
                f"HDF5 load failed: {exc}",
                "danger",
                no_update,
                *((no_update,) * 11),
                no_update,
            )

        loaded["status"] = {
            "ok": True,
            "message": f"Loaded session from {filename or 'upload'}.",
        }
        loaded["history_rebuilt_last_step"] = True
        return (
            loaded,
            loaded["status"]["message"],
            "success",
            _metric_row(loaded),
            _power_figure(loaded),
            _temperature_figure(loaded),
            _reactivity_figure(loaded),
            _xenon_iodine_figure(loaded),
            _neutron_figure(loaded),
            _power_rho_figure(loaded),
            _precursor_figure(loaded),
            _delta_t_figure(loaded),
            _control_schedule_figure(loaded),
            _xenon_worth_figure(loaded),
            _derivatives_figure(loaded),
            _event_log_items(loaded),
        )