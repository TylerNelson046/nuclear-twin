"""Dash boilerplate application with a placeholder reactor selector."""

from collections.abc import Callable

from dash import Dash, Input, Output, dcc, html
import dash_bootstrap_components as dbc

from ui.display_units_bar import (
    DEFAULT_DISPLAY_UNITS,
    build_display_units_controls,
    register_display_units_callback,
)
from ui.comparison_panel import build_comparison_panel, register_comparison_callbacks
from ui.helion_panel import build_helion_panel, register_helion_callbacks
from ui.msr_panel import build_msr_panel, register_msr_callbacks
from ui.pwr_panel import build_pwr_panel, register_pwr_callbacks


REACTOR_PANELS: dict[str, Callable[[], dbc.Card]] = {
    "pwr": build_pwr_panel,
    "helion": build_helion_panel,
    "msr": build_msr_panel,
    "comparison": build_comparison_panel,
}


def create_app() -> Dash:
    """Create the Dash app shell for local pre-phase validation."""
    app = Dash(
        __name__,
        external_stylesheets=[dbc.themes.BOOTSTRAP],
        suppress_callback_exceptions=True,
        title="Nuclear Reactor Digital Twin",
    )

    app.layout = html.Div(
        dbc.Container(
            [
                dcc.Store(id="display-units-store", data=DEFAULT_DISPLAY_UNITS),
                html.Header(
                    [
                        html.Div("Nuclear Reactor Digital Twin Platform", className="eyebrow"),
                        html.H1("Engineering Workstation"),
                        html.P(
                            "Interactive reduced-order reactor twins for point kinetics, "
                            "thermal response, and feedback visualization.",
                            className="lead",
                        ),
                    ],
                    className="app-header",
                ),
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                dbc.Label("Select reactor twin", html_for="reactor-selector"),
                                dcc.Dropdown(
                                    id="reactor-selector",
                                    options=[
                                        {"label": "Pressurized Water Reactor (PWR)", "value": "pwr"},
                                        {"label": "Helion FRC Fusion Reactor", "value": "helion"},
                                        {"label": "Molten Salt Reactor (MSR)", "value": "msr"},
                                        {
                                            "label": "PWR vs MSR Comparison",
                                            "value": "comparison",
                                        },
                                    ],
                                    value="pwr",
                                    clearable=False,
                                    className="dark-dropdown",
                                ),
                            ],
                            xl=4,
                            lg=5,
                        )
                    ],
                    className="mb-4",
                ),
                build_display_units_controls(),
                html.Main(id="reactor-panel"),
            ],
            fluid="xl",
            className="app-container",
        ),
        className="app-shell",
    )

    @app.callback(
        Output("reactor-panel", "children"),
        Input("reactor-selector", "value"),
    )
    def render_reactor_panel(reactor_type: str) -> dbc.Card:
        panel_factory = REACTOR_PANELS.get(reactor_type, build_pwr_panel)
        return panel_factory()

    register_display_units_callback(app)
    register_pwr_callbacks(app)
    register_helion_callbacks(app)
    register_msr_callbacks(app)
    register_comparison_callbacks(app)

    return app
