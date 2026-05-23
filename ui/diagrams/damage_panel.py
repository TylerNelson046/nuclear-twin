"""Shared diagnostics drawer: damage bars + remaining-life cards.

This widget is reactor-agnostic. The caller passes:
- a list of ``(label, current, limit, units)`` tuples to render damage bars
- a dict of ``label → years remaining`` to render the projection cards

It exposes its rendered children via :func:`update_damage_panel_children`
so a Dash callback can patch it without rebuilding the wrapping ``Card``.
"""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np
import plotly.graph_objects as go
from dash import html
import dash_bootstrap_components as dbc


def build_damage_panel(panel_id: str) -> dbc.Card:
    """Return the empty diagnostics drawer card (panel_id e.g. 'pwr-damage')."""
    return dbc.Card(
        [
            dbc.CardHeader(
                [
                    html.Span("Materials & Remaining Life", className="card-title-text"),
                    html.Span("illustrative — not licensed analysis",
                              className="diagram-disclaimer"),
                ],
                className="d-flex justify-content-between align-items-center",
            ),
            dbc.CardBody(
                [
                    html.Div(id=f"{panel_id}-damage-bars", className="damage-bars"),
                    html.Hr(),
                    html.Div(id=f"{panel_id}-life-cards", className="life-cards"),
                    html.Div(id=f"{panel_id}-life-summary", className="life-summary"),
                ]
            ),
        ],
        className="damage-panel",
    )


def update_damage_panel_children(
    damage_bars: Iterable[tuple[str, float, float, str]],
    remaining_life_years: dict[str, float],
    life_label_map: dict[str, str] | None = None,
) -> tuple[list, list, str]:
    """Return (bars_children, cards_children, summary_text).

    ``damage_bars``: iterable of (label, current, limit, units).
    ``remaining_life_years``: dict of key → years (may be math.inf).
    ``life_label_map``: optional pretty name override for each key.
    """
    bars = []
    for label, current, limit, units in damage_bars:
        if limit <= 0:
            frac = 0.0
        else:
            frac = max(0.0, min(1.0, current / limit))
        color = _bar_color(frac)
        bars.append(
            html.Div(
                [
                    html.Div(
                        [
                            html.Span(label, className="damage-bar-label"),
                            html.Span(
                                f"{current:.3g} / {limit:.3g} {units}",
                                className="damage-bar-value",
                            ),
                        ],
                        className="damage-bar-header",
                    ),
                    html.Div(
                        html.Div(
                            style={
                                "width": f"{frac * 100:.1f}%",
                                "background": color,
                            },
                            className="damage-bar-fill",
                        ),
                        className="damage-bar-track",
                    ),
                ],
                className="damage-bar",
            )
        )

    cards = []
    finite_years = [v for v in remaining_life_years.values() if math.isfinite(v)]
    binding_min = min(finite_years) if finite_years else math.inf
    for key, years in remaining_life_years.items():
        pretty = (life_label_map or {}).get(key, key.replace("_", " ").title())
        if math.isinf(years):
            display = "∞"
            sub = "no measurable damage at current load"
        elif years <= 0:
            display = "0"
            sub = "design limit reached"
        elif years < 1:
            display = f"{years * 12:.1f} mo"
            sub = "limit reached at this load"
        else:
            display = f"{years:.1f} yr"
            sub = "at current operating point"

        is_binding = math.isfinite(years) and abs(years - binding_min) < 1e-9
        cards.append(
            html.Div(
                [
                    html.Div(pretty, className="life-card-label"),
                    html.Div(display, className="life-card-value"),
                    html.Div(sub, className="life-card-sub"),
                ],
                className="life-card" + (" life-card-binding" if is_binding else ""),
            )
        )

    if math.isinf(binding_min):
        summary = "No measurable damage accumulation at the current operating point."
    elif binding_min <= 0:
        summary = "⚠ A material limit has been reached. Reduce power or replace affected component."
    elif binding_min < 1:
        summary = f"Binding limit projected in {binding_min * 12:.1f} months at the current load."
    else:
        summary = f"Binding limit projected in {binding_min:.1f} years at the current load."

    return bars, cards, summary


def _bar_color(frac: float) -> str:
    if frac < 0.5:
        return "linear-gradient(90deg, #2ecc71, #27ae60)"
    if frac < 0.8:
        return "linear-gradient(90deg, #f4d35e, #e2a93b)"
    return "linear-gradient(90deg, #e74c3c, #c0392b)"


def damage_history_figure(
    time_array: list[float] | np.ndarray,
    series: dict[str, list[float]],
    y_label: str = "Damage indicator",
) -> go.Figure:
    """Time-series of accumulated damage indicators (one trace per component)."""
    fig = go.Figure()
    palette = ["#ff7b3a", "#88b3ff", "#9b59b6", "#2ecc71", "#f4d35e", "#e74c3c"]
    for idx, (name, values) in enumerate(series.items()):
        fig.add_trace(
            go.Scatter(
                x=time_array, y=values, mode="lines",
                name=name, line=dict(color=palette[idx % len(palette)], width=2),
            )
        )
    fig.update_layout(
        title=dict(text="Damage accumulation history", font=dict(size=12)),
        xaxis=dict(title="time (h)", showgrid=False),
        yaxis=dict(title=y_label, showgrid=True, gridcolor="#1f2a35"),
        margin=dict(l=50, r=20, t=40, b=40),
        height=260,
        paper_bgcolor="#0f1620",
        plot_bgcolor="#0f1620",
        font=dict(color="#dceaf3", size=11),
        legend=dict(orientation="h", x=0.5, xanchor="center", y=1.15),
    )
    return fig
