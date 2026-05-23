"""Smoke tests for the Dash application shell and reactor panels."""

from dash import Dash
import dash_bootstrap_components as dbc

from ui.app import REACTOR_PANELS, create_app
from ui.helion_panel import build_helion_panel


def test_dash_app_factory_builds_placeholder_shell() -> None:
    app = create_app()

    assert isinstance(app, Dash)
    assert app.title == "Nuclear Reactor Digital Twin"
    assert set(REACTOR_PANELS) == {"pwr", "helion", "msr", "comparison"}
    assert "reactor-panel.children" in app.callback_map


def test_helion_callbacks_registered() -> None:
    app = create_app()
    callback_ids = " ".join(app.callback_map.keys())
    assert "helion-pulse-store" in callback_ids
    assert "helion-sweep-store" in callback_ids
    assert "helion-chart-temp" in callback_ids
    assert "helion-chart-reactivity" in callback_ids
    assert "helion-chart-heatmap" in callback_ids


def test_build_helion_panel_returns_card() -> None:
    panel = build_helion_panel()
    assert isinstance(panel, dbc.Card)


def test_build_helion_panel_contains_disclaimer() -> None:
    from ui.pwr_panel import DISCLAIMER
    panel = build_helion_panel()
    panel_str = str(panel)
    assert "educational purposes" in panel_str or "simplified" in panel_str


def test_build_helion_panel_contains_run_button() -> None:
    panel = build_helion_panel()
    panel_str = str(panel)
    assert "helion-run-btn" in panel_str


def test_build_helion_panel_contains_stores() -> None:
    panel = build_helion_panel()
    panel_str = str(panel)
    assert "helion-pulse-store" in panel_str
    assert "helion-sweep-store" in panel_str
