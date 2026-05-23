"""MSR loop schematic (inline SVG) + salt-temperature loop heatmap.

Layout: core → riser → heat exchanger → downcomer → pump → core. The salt
loop is the *only* fluid carrying the fuel; precursors drift around the
loop and re-enter the core after one transit time. We expose the salt
temperature as a 1-D function of loop-angle (theta) so the visualisation
can show where the salt is hottest and where the heat exchanger removes
energy.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import plotly.graph_objects as go
from dash import html

from ui.diagrams.pwr_schematic import svg_to_img


def build_msr_diagram() -> html.Div:
    """Return a Dash Div with the MSR loop SVG + legend."""
    return html.Div(
        [
            html.Div(
                id="msr-diagram",
                children=svg_to_img(_msr_svg_static(), alt="MSR primary loop"),
                className="reactor-diagram",
            ),
            html.Div(
                id="msr-diagram-legend",
                children=[
                    html.Span([html.I(className="legend-swatch swatch-fuel-salt"),
                               "Fuel salt"]),
                    html.Span([html.I(className="legend-swatch swatch-coolant-salt"),
                               "Secondary salt"]),
                    html.Span([html.I(className="legend-swatch swatch-hot"),
                               "Hot salt"]),
                    html.Span([html.I(className="legend-swatch swatch-cold"),
                               "Cold salt"]),
                    html.Span([html.I(className="legend-swatch swatch-damage"),
                               "Cumulative corrosion"]),
                ],
                className="diagram-legend",
            ),
        ],
        className="reactor-diagram-card",
    )


def _msr_svg_static() -> str:
    """MSR loop SVG with named segments for live colouring."""
    return """
    <svg viewBox="0 0 500 480" xmlns="http://www.w3.org/2000/svg"
         width="100%" preserveAspectRatio="xMidYMid meet">
      <defs>
        <linearGradient id="msr-grad" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stop-color="#23303a"/>
          <stop offset="100%" stop-color="#0e1518"/>
        </linearGradient>
        <marker id="msr-arrow" viewBox="0 0 10 10" refX="5" refY="5"
                markerWidth="5" markerHeight="5" orient="auto-start-reverse">
          <path d="M 0 0 L 10 5 L 0 10 z" fill="#88b3ff"/>
        </marker>
      </defs>

      <!-- Core (the salt fission region) -->
      <rect id="msr-core" x="60" y="170" width="120" height="160" rx="14"
            fill="#7a3c1c" stroke="#cfa672" stroke-width="3"/>
      <text x="120" y="200" fill="#fffaf2" font-size="13" text-anchor="middle"
            font-weight="600">CORE</text>
      <text x="120" y="220" fill="#fffaf2" font-size="10" text-anchor="middle">
        graphite + fuel salt
      </text>

      <!-- Riser (core outlet to HX) -->
      <path id="msr-riser" d="M 120 170 L 120 70 L 300 70"
            stroke="#d96a4a" stroke-width="14" fill="none" marker-end="url(#msr-arrow)"/>

      <!-- Heat exchanger -->
      <rect id="msr-hx" x="300" y="60" width="120" height="120" rx="10"
            fill="#1c4f7a" stroke="#88b3ff" stroke-width="3"/>
      <text x="360" y="105" fill="#dceaf3" font-size="12" text-anchor="middle"
            font-weight="600">PRIMARY</text>
      <text x="360" y="125" fill="#dceaf3" font-size="12" text-anchor="middle"
            font-weight="600">HEAT EXCH</text>

      <!-- Secondary loop (cold side) -->
      <path id="msr-secondary" d="M 460 60 L 460 240 L 460 280 L 460 400"
            stroke="#88b3ff" stroke-width="6" fill="none"
            stroke-dasharray="6 3"/>
      <text x="468" y="280" fill="#88b3ff" font-size="11">to power conversion</text>

      <!-- Downcomer (HX outlet back toward pump) -->
      <path id="msr-downcomer" d="M 300 180 L 120 180 L 120 360"
            stroke="#3a85d9" stroke-width="14" fill="none"
            marker-end="url(#msr-arrow)"/>

      <!-- Pump -->
      <circle id="msr-pump" cx="120" cy="380" r="22"
              fill="#4a5b6a" stroke="#aabbcc" stroke-width="2"/>
      <text x="120" y="385" fill="#fff" font-size="11" text-anchor="middle">PUMP</text>

      <!-- Return to core bottom -->
      <path id="msr-return" d="M 120 402 L 120 430 L 200 430 L 200 350 L 180 350"
            stroke="#3a85d9" stroke-width="10" fill="none"
            marker-end="url(#msr-arrow)"/>

      <!-- Damage overlays -->
      <rect id="msr-damage-pipe" x="40" y="50" width="430" height="400"
            fill="#e63946" fill-opacity="0" pointer-events="none"/>
      <rect id="msr-damage-hx" x="300" y="60" width="120" height="120"
            fill="#9b59b6" fill-opacity="0" pointer-events="none"/>

      <!-- Labels -->
      <text x="250" y="22" fill="#dceaf3" font-size="14" text-anchor="middle"
            font-weight="600">MSR Primary Loop (fuel salt)</text>
      <text x="250" y="470" fill="#9bb0c0" font-size="10" text-anchor="middle"
            font-style="italic">
        Precursors drift with salt → fewer delayed neutrons in core ⇒ smaller β_eff,flow
      </text>
    </svg>
    """


def msr_loop_heatmap_figure(
    t_salt_hot_k: float,
    t_salt_cold_k: float,
    flow_fraction: float,
    n_points: int = 60,
) -> go.Figure:
    """1-D loop temperature plot (theta around the loop).

    The salt heats up in the core, cools in the HX, and returns to the
    core. Flow fraction adjusts the temperature swing (low flow ⇒ wider
    swing). This is a *visualization* of the lumped scalar T_salt(t), not
    an additional integration.
    """
    theta = np.linspace(0.0, 1.0, n_points)
    mean_t = 0.5 * (t_salt_hot_k + t_salt_cold_k)
    swing = (t_salt_hot_k - t_salt_cold_k)
    # Heating ramp in core (theta 0.00–0.25), constant in riser (0.25–0.45),
    # cooling in HX (0.45–0.70), constant in downcomer (0.70–1.00).
    profile = np.full_like(theta, t_salt_cold_k)
    core_mask = theta <= 0.25
    profile[core_mask] = t_salt_cold_k + swing * (theta[core_mask] / 0.25)
    riser_mask = (theta > 0.25) & (theta <= 0.45)
    profile[riser_mask] = t_salt_hot_k
    hx_mask = (theta > 0.45) & (theta <= 0.70)
    profile[hx_mask] = t_salt_hot_k - swing * ((theta[hx_mask] - 0.45) / 0.25)
    # Else stays at t_salt_cold_k

    # Flow scaling: at low flow the swing is exaggerated; at high flow it
    # shrinks because residence time falls.
    flow = max(0.05, flow_fraction)
    profile = mean_t + (profile - mean_t) * (1.0 / flow)

    fig = go.Figure(
        data=[
            go.Scatter(
                x=theta,
                y=profile,
                mode="lines",
                line=dict(color="#ff7b3a", width=4),
                fill="tozeroy",
                fillcolor="rgba(255, 123, 58, 0.15)",
                name="Salt temperature",
            )
        ]
    )
    # Annotate segments.
    for label, x_center in [
        ("Core", 0.125),
        ("Riser", 0.35),
        ("HX", 0.575),
        ("Downcomer", 0.85),
    ]:
        fig.add_annotation(
            x=x_center,
            y=t_salt_cold_k - 30,
            text=label,
            showarrow=False,
            font=dict(size=10, color="#aabbcc"),
        )

    fig.update_layout(
        title=dict(text="Fuel-salt temperature around loop", font=dict(size=12)),
        xaxis=dict(title="Loop position θ (0=core inlet)", range=[0, 1], showgrid=False),
        yaxis=dict(title="T_salt (K)", showgrid=True, gridcolor="#1f2a35"),
        margin=dict(l=50, r=20, t=40, b=40),
        height=320,
        paper_bgcolor="#0f1620",
        plot_bgcolor="#0f1620",
        font=dict(color="#dceaf3", size=11),
    )
    return fig


def render_msr_diagram(
    t_salt_hot_k: float,
    t_salt_cold_k: float,
    corrosion_fraction: float,
    hx_thinning_fraction: float,
) -> str:
    """Return the full SVG string with state-dependent attributes baked in."""
    from ui.diagrams.pwr_schematic import _apply_svg_patches
    base = _msr_svg_static()
    patches = msr_diagram_patches(
        t_salt_hot_k, t_salt_cold_k, corrosion_fraction, hx_thinning_fraction
    )
    return _apply_svg_patches(base, patches)


def msr_diagram_patches(
    t_salt_hot_k: float,
    t_salt_cold_k: float,
    corrosion_fraction: float,
    hx_thinning_fraction: float,
) -> dict[str, dict[str, Any]]:
    """SVG element IDs → attribute updates."""
    # Tint the core / riser warmer and downcomer cooler based on the hot
    # leg / cold leg split.
    def _temp_color(t_k: float) -> str:
        # Map 800–1100 K → cool to hot palette.
        x = max(0.0, min(1.0, (t_k - 800.0) / 300.0))
        r = int(120 + 130 * x)
        g = int(80 + 60 * (1.0 - abs(x - 0.5) * 2))
        b = int(40 + 120 * (1.0 - x))
        return f"rgb({r},{g},{b})"

    return {
        "msr-core": {"fill": _temp_color(t_salt_hot_k)},
        "msr-riser": {"stroke": _temp_color(t_salt_hot_k)},
        "msr-downcomer": {"stroke": _temp_color(t_salt_cold_k)},
        "msr-return": {"stroke": _temp_color(t_salt_cold_k)},
        "msr-damage-pipe": {
            "fill-opacity": f"{min(0.45, 0.45 * corrosion_fraction):.3f}",
        },
        "msr-damage-hx": {
            "fill-opacity": f"{min(0.50, 0.50 * hx_thinning_fraction):.3f}",
        },
    }
