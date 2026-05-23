"""Global display-units controls shared across reactor panels."""

from __future__ import annotations

from dash import Dash, Input, Output, dcc, html
import dash_bootstrap_components as dbc

DEFAULT_DISPLAY_UNITS = {"temperature": "celsius", "time": "seconds"}


def build_display_units_controls() -> dbc.Row:
    """Temperature and time unit selectors (SPEC §4.6.4)."""
    return dbc.Row(
        [
            dbc.Col(
                [
                    dbc.Label("Temperature display", html_for="display-temperature-units"),
                    dcc.Dropdown(
                        id="display-temperature-units",
                        options=[
                            {"label": "°C (engineering)", "value": "celsius"},
                            {"label": "K (SI)", "value": "kelvin"},
                        ],
                        value="celsius",
                        clearable=False,
                        searchable=False,
                        className="dark-dropdown",
                    ),
                ],
                md=3,
            ),
            dbc.Col(
                [
                    dbc.Label("Time display", html_for="display-time-units"),
                    dcc.Dropdown(
                        id="display-time-units",
                        options=[
                            {"label": "Seconds", "value": "seconds"},
                            {"label": "Minutes", "value": "minutes"},
                            {"label": "Hours", "value": "hours"},
                        ],
                        value="seconds",
                        clearable=False,
                        searchable=False,
                        className="dark-dropdown",
                    ),
                ],
                md=3,
            ),
        ],
        className="mb-4 g-3 align-items-end",
    )


def register_display_units_callback(app: Dash) -> None:
    """Sync dropdowns into the global display-units store."""

    @app.callback(
        Output("display-units-store", "data"),
        Input("display-temperature-units", "value"),
        Input("display-time-units", "value"),
    )
    def _update_display_units(temperature: str, time_u: str) -> dict[str, str]:
        return {
            "temperature": temperature or "celsius",
            "time": time_u or "seconds",
        }
