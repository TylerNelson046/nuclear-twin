"""PWR cutaway schematic (inline SVG) + axial temperature heatmap.

Renders a stylized vertical cross-section of a commercial PWR vessel with:
- vessel pressure boundary (outline)
- 17x17 fuel-assembly grid (axial mid-plane)
- control rod column positions (insertable, animated)
- coolant inlet/outlet nozzles
- damage shading overlay driven by material state

The companion `pwr_heatmap_figure` returns a Plotly heatmap of axial-radial
temperature derived from the lumped scalar fuel/coolant temperatures via a
chopped-cosine power shape.
"""

from __future__ import annotations

import base64
import math
from typing import Any

import numpy as np
import plotly.graph_objects as go
from dash import html


def svg_to_img(svg: str, alt: str = "") -> html.Img:
    """Embed an SVG string as a data-URI ``html.Img`` so Dash will render it.

    Dash does not expose dangerouslySetInnerHTML; base64 + data URI is the
    most portable way to surface arbitrary SVG markup from a callback.
    """
    encoded = base64.b64encode(svg.encode("utf-8")).decode("ascii")
    return html.Img(
        src=f"data:image/svg+xml;base64,{encoded}",
        alt=alt,
        style={"width": "100%", "height": "auto", "display": "block"},
    )


# Geometry constants (SVG user units; not physical).
_VESSEL_X = 60.0
_VESSEL_Y = 30.0
_VESSEL_W = 380.0
_VESSEL_H = 540.0
_CORE_X = 110.0
_CORE_Y = 130.0
_CORE_W = 280.0
_CORE_H = 340.0
_ASSEMBLY_GRID = 9  # 9x9 visualisation (not 17x17 — readability)
_ROD_COLUMNS = [1, 3, 5, 7]  # which columns carry visible rods


def build_pwr_diagram() -> html.Div:
    """Return a Dash Div containing the inline PWR schematic SVG.

    The SVG uses descriptive element IDs so a callback can patch fills,
    rod positions, and damage overlays without re-rendering the whole tree.
    """
    return html.Div(
        [
            html.Div(
                id="pwr-diagram",
                children=svg_to_img(_pwr_svg_static(), alt="PWR cutaway"),
                className="reactor-diagram",
            ),
            html.Div(
                id="pwr-diagram-legend",
                children=_diagram_legend(),
                className="diagram-legend",
            ),
        ],
        className="reactor-diagram-card",
    )


def _diagram_legend() -> list[Any]:
    return [
        html.Span([html.I(className="legend-swatch swatch-fuel"), "Fuel assemblies"]),
        html.Span([html.I(className="legend-swatch swatch-rod"), "Control rods"]),
        html.Span([html.I(className="legend-swatch swatch-coolant"), "Coolant flow"]),
        html.Span([html.I(className="legend-swatch swatch-vessel"), "RPV wall"]),
        html.Span([html.I(className="legend-swatch swatch-damage"), "Cumulative damage"]),
    ]


def _pwr_svg_static() -> str:
    """Static SVG with named elements; live updates patch fill colours only."""
    cell_w = _CORE_W / _ASSEMBLY_GRID
    cell_h = _CORE_H / _ASSEMBLY_GRID

    assemblies = []
    for row in range(_ASSEMBLY_GRID):
        for col in range(_ASSEMBLY_GRID):
            x = _CORE_X + col * cell_w
            y = _CORE_Y + row * cell_h
            assemblies.append(
                f'<rect id="pwr-assembly-{row}-{col}" '
                f'x="{x:.1f}" y="{y:.1f}" '
                f'width="{cell_w - 1:.1f}" height="{cell_h - 1:.1f}" '
                f'fill="#3b6ea5" stroke="#1c3958" stroke-width="0.5"/>'
            )

    rods = []
    for col in _ROD_COLUMNS:
        x = _CORE_X + col * cell_w + cell_w * 0.3
        rods.append(
            f'<rect id="pwr-rod-{col}" x="{x:.1f}" y="{_CORE_Y - 90:.1f}" '
            f'width="{cell_w * 0.4:.1f}" height="90" '
            f'fill="#888" stroke="#333" stroke-width="0.5" rx="2"/>'
        )

    # Coolant arrows and labels
    coolant_arrows = """
      <path id="pwr-arrow-inlet" d="M 30 200 L 110 200 M 95 192 L 110 200 L 95 208"
            stroke="#3aa3ff" stroke-width="3" fill="none"/>
      <path id="pwr-arrow-outlet" d="M 390 380 L 470 380 M 455 372 L 470 380 L 455 388"
            stroke="#ff6b3a" stroke-width="3" fill="none"/>
      <text x="20" y="195" fill="#aabbcc" font-size="11">cold leg</text>
      <text x="395" y="375" fill="#ffaa88" font-size="11">hot leg</text>
    """

    # Damage overlay rectangles — one above the vessel for RPV, one
    # over the core for cladding, one over rods for B-10. Live callback
    # adjusts the fill-opacity to indicate damage progress.
    damage_layer = """
      <rect id="pwr-damage-rpv" x="60" y="30" width="380" height="540"
            fill="#e63946" fill-opacity="0" pointer-events="none"/>
      <rect id="pwr-damage-cladding" x="110" y="130" width="280" height="340"
            fill="#f4a261" fill-opacity="0" pointer-events="none"/>
      <rect id="pwr-damage-rods" x="110" y="40" width="280" height="90"
            fill="#9b59b6" fill-opacity="0" pointer-events="none"/>
    """

    svg = f"""
    <svg viewBox="0 0 500 600" xmlns="http://www.w3.org/2000/svg"
         width="100%" preserveAspectRatio="xMidYMid meet">
      <defs>
        <linearGradient id="pwr-vessel-grad" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stop-color="#23303a"/>
          <stop offset="100%" stop-color="#0e1518"/>
        </linearGradient>
      </defs>

      <!-- Outer vessel wall -->
      <rect id="pwr-vessel" x="{_VESSEL_X}" y="{_VESSEL_Y}"
            width="{_VESSEL_W}" height="{_VESSEL_H}" rx="60" ry="40"
            fill="url(#pwr-vessel-grad)" stroke="#7a8a96" stroke-width="3"/>

      <!-- Vessel head bolts (decoration) -->
      <ellipse cx="250" cy="40" rx="160" ry="20" fill="#1a262e" stroke="#7a8a96" stroke-width="2"/>

      <!-- Coolant region inside vessel -->
      <rect id="pwr-coolant-region" x="{_VESSEL_X + 30}" y="{_VESSEL_Y + 90}"
            width="{_VESSEL_W - 60}" height="{_VESSEL_H - 180}"
            fill="#0d3a5a" opacity="0.7"/>

      <!-- Core baffle -->
      <rect x="{_CORE_X - 4}" y="{_CORE_Y - 4}" width="{_CORE_W + 8}" height="{_CORE_H + 8}"
            fill="none" stroke="#aabbcc" stroke-width="2"/>

      <!-- Fuel assemblies -->
      {''.join(assemblies)}

      <!-- Control rods -->
      {''.join(rods)}

      <!-- Coolant nozzles -->
      {coolant_arrows}

      <!-- Damage overlays (opacity = damage fraction) -->
      {damage_layer}

      <!-- Labels -->
      <text x="250" y="22" fill="#dceaf3" font-size="14" text-anchor="middle"
            font-family="Inter,Segoe UI,sans-serif" font-weight="600">
        PWR Pressure Vessel
      </text>
      <text x="250" y="500" fill="#dceaf3" font-size="12" text-anchor="middle"
            font-family="Inter,Segoe UI,sans-serif">
        Core Mid-plane (17×17 assemblies shown 9×9 for clarity)
      </text>
    </svg>
    """
    return svg


def pwr_heatmap_figure(
    t_fuel_k: float,
    t_coolant_k: float,
    t_inlet_k: float,
    power_fraction: float,
    rod_inserted_fraction: float,
    n_axial: int = 30,
    n_radial: int = 9,
) -> go.Figure:
    """Generate an axial-radial heatmap of estimated fuel temperature.

    The PWR engine carries scalar lumped fuel + coolant temperatures. To
    visualize spatial structure we reconstruct an analytical chopped-cosine
    axial flux shape with a Bessel-like radial form factor, then map the
    scalar to a peak-to-average distribution. This is an *illustrative*
    overlay — the underlying physics is still 0-D lumped.

    Control-rod insertion suppresses the top of the axial shape proportionally
    so the user can see the depressed flux region track the rod position.
    """
    axial = np.linspace(0.0, 1.0, n_axial)
    radial = np.linspace(-1.0, 1.0, n_radial)

    # Chopped cosine axial shape, peaking near mid-plane.
    base_axial = np.cos(np.pi * (axial - 0.5))
    base_axial = np.clip(base_axial, 0.1, None)

    # Rod suppression: the top fraction of the core sees a multiplicative
    # depression scaling with rod insertion.
    rod_mask = np.ones_like(axial)
    insertion_top = 1.0 - rod_inserted_fraction
    rod_mask = np.where(axial > insertion_top, 1.0 - 0.6 * rod_inserted_fraction, rod_mask)
    axial_shape = base_axial * rod_mask

    # Radial cosine — peak in centre, suppressed at periphery (baffle).
    radial_shape = np.cos(np.pi * radial / 2.0)
    radial_shape = np.clip(radial_shape, 0.3, None)

    grid = np.outer(axial_shape, radial_shape) * power_fraction
    if grid.max() > 0.0:
        grid_norm = grid / grid.max()
    else:
        grid_norm = grid

    # Map normalized power to temperature via lumped fuel-coolant delta.
    delta = max(0.0, t_fuel_k - t_coolant_k)
    t_grid = t_coolant_k + delta * grid_norm

    # Inject a coolant heat-up gradient — bottom = inlet, top = outlet.
    coolant_axial = t_inlet_k + (t_coolant_k - t_inlet_k) * axial
    t_grid = t_grid + (coolant_axial - t_coolant_k)[:, None]

    fig = go.Figure(
        data=go.Heatmap(
            z=t_grid,
            x=radial,
            y=axial,
            colorscale="Inferno",
            colorbar=dict(title="K", thickness=12),
            zmin=t_inlet_k - 5,
            zmax=t_inlet_k + delta + 30,
        )
    )
    fig.update_layout(
        title=dict(text="Core temperature (axial × radial)", font=dict(size=12)),
        xaxis=dict(title="Radial position (−1=west, +1=east)", showgrid=False),
        yaxis=dict(title="Axial fraction (0=bot, 1=top)", showgrid=False),
        margin=dict(l=50, r=20, t=40, b=40),
        height=320,
        paper_bgcolor="#0f1620",
        plot_bgcolor="#0f1620",
        font=dict(color="#dceaf3", size=11),
    )
    return fig


def render_pwr_diagram(
    rod_position_fraction: float,
    rpv_damage_fraction: float,
    cladding_damage_fraction: float,
    rod_damage_fraction: float,
    fuel_power_normalized: float,
) -> str:
    """Return the full SVG string with state-dependent attributes baked in.

    Called from the live callback once per tick. Cheaper than a clientside
    SVG-attribute patch dance because the SVG is small (~3 KB).
    """
    base = _pwr_svg_static()
    patches = pwr_diagram_patches(
        rod_position_fraction,
        rpv_damage_fraction,
        cladding_damage_fraction,
        rod_damage_fraction,
        fuel_power_normalized,
    )
    return _apply_svg_patches(base, patches)


def _apply_svg_patches(svg: str, patches: dict[str, dict[str, Any]]) -> str:
    """Apply per-element attribute patches to an inline SVG string.

    Uses a string-replace approach keyed on each element's ``id="..."``
    anchor. It modifies whitespace-separated attribute strings without
    parsing the SVG, which is enough for the controlled markup we ship.
    """
    import re

    for element_id, attrs in patches.items():
        pattern = re.compile(
            r'(<[a-zA-Z][a-zA-Z0-9]*[^>]*?id="' + re.escape(element_id) + r'"[^>]*?)(/?>)'
        )

        def _sub(match: re.Match[str], _attrs: dict[str, Any] = attrs) -> str:
            head = match.group(1)
            tail = match.group(2)
            for k, v in _attrs.items():
                attr_pattern = re.compile(r'\b' + re.escape(k) + r'="[^"]*"')
                if attr_pattern.search(head):
                    head = attr_pattern.sub(f'{k}="{v}"', head)
                else:
                    head = head + f' {k}="{v}"'
            return head + tail

        svg = pattern.sub(_sub, svg)
    return svg


def pwr_diagram_patches(
    rod_position_fraction: float,
    rpv_damage_fraction: float,
    cladding_damage_fraction: float,
    rod_damage_fraction: float,
    fuel_power_normalized: float,
) -> dict[str, dict[str, Any]]:
    """Return a dict of SVG element IDs → attribute updates for one frame.

    Consumed by a Dash clientside callback that patches the inline SVG
    without re-rendering the whole component.
    """
    # Rods slide vertically — y position interpolates between -90 (fully out)
    # and the core top (fully in).
    cell_h = _CORE_H / _ASSEMBLY_GRID
    rod_y_out = _CORE_Y - 90.0
    rod_y_in = _CORE_Y + cell_h * 0.5
    rod_y = rod_y_out + (rod_y_in - rod_y_out) * max(0.0, min(1.0, rod_position_fraction))

    patches: dict[str, dict[str, Any]] = {}
    for col in _ROD_COLUMNS:
        patches[f"pwr-rod-{col}"] = {"y": f"{rod_y:.1f}"}

    # Damage overlays — fill-opacity proportional to damage fraction.
    patches["pwr-damage-rpv"] = {
        "fill-opacity": f"{min(0.45, 0.45 * rpv_damage_fraction):.3f}",
    }
    patches["pwr-damage-cladding"] = {
        "fill-opacity": f"{min(0.40, 0.40 * cladding_damage_fraction):.3f}",
    }
    patches["pwr-damage-rods"] = {
        "fill-opacity": f"{min(0.50, 0.50 * rod_damage_fraction):.3f}",
    }

    # Assembly fills tint warmer with higher local power. We use the same
    # chopped-cosine shape as the heatmap so the two views agree.
    power_clamped = max(0.0, min(1.5, fuel_power_normalized))
    for row in range(_ASSEMBLY_GRID):
        local_axial = math.cos(math.pi * (row / (_ASSEMBLY_GRID - 1) - 0.5))
        local_axial = max(0.1, local_axial)
        for col in range(_ASSEMBLY_GRID):
            local_radial = math.cos(math.pi * (col / (_ASSEMBLY_GRID - 1) - 0.5))
            local_radial = max(0.3, local_radial)
            heat = power_clamped * local_axial * local_radial
            r = int(min(255, 60 + 180 * heat))
            g = int(min(255, 110 + 60 * heat))
            b = int(max(40, 165 - 100 * heat))
            patches[f"pwr-assembly-{row}-{col}"] = {"fill": f"rgb({r},{g},{b})"}
    return patches
