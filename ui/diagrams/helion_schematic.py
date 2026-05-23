"""Helion FRC schematic + sub-microsecond pulse phase visualization.

The Helion machine fires two field-reversed-configuration (FRC) plasmoids
from opposite ends of a linear chamber, accelerates them together, and
compresses the merged plasmoid in a central compression chamber. We render
five conceptual stages plus a Plotly figure that slices the pulse history
into the same phases so the user sees what's happening when.

Stages (left → right on the schematic):
  formation_west → acceleration_west → compression → acceleration_east →
  formation_east

The "super slo-mo" pulse view re-plots the µs-scale pulse trace with stage
annotations and a moving phase indicator driven by `t_now`.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import plotly.graph_objects as go
from dash import html

from ui.diagrams.pwr_schematic import svg_to_img


def build_helion_diagram() -> html.Div:
    """Return a Dash Div with the Helion machine SVG + legend."""
    return html.Div(
        [
            html.Div(
                id="helion-diagram",
                children=svg_to_img(_helion_svg_static(), alt="Helion FRC chamber"),
                className="reactor-diagram",
            ),
            html.Div(
                id="helion-diagram-legend",
                children=[
                    html.Span([html.I(className="legend-swatch swatch-formation"),
                               "Formation chamber"]),
                    html.Span([html.I(className="legend-swatch swatch-accel"),
                               "Acceleration coils"]),
                    html.Span([html.I(className="legend-swatch swatch-compression"),
                               "Compression chamber"]),
                    html.Span([html.I(className="legend-swatch swatch-plasma"),
                               "Plasma (FRC)"]),
                    html.Span([html.I(className="legend-swatch swatch-damage"),
                               "Coil/first-wall damage"]),
                ],
                className="diagram-legend",
            ),
        ],
        className="reactor-diagram-card",
    )


def _helion_svg_static() -> str:
    """Helion machine SVG."""
    # Coil rings (vertical lines representing coil cross sections).
    accel_coils_west = "".join(
        f'<rect id="helion-coil-w-{i}" x="{120 + i * 18}" y="180" '
        f'width="6" height="80" fill="#4a5b6a" stroke="#7a8a96" stroke-width="0.5"/>'
        for i in range(7)
    )
    accel_coils_east = "".join(
        f'<rect id="helion-coil-e-{i}" x="{310 + i * 18}" y="180" '
        f'width="6" height="80" fill="#4a5b6a" stroke="#7a8a96" stroke-width="0.5"/>'
        for i in range(7)
    )
    return f"""
    <svg viewBox="0 0 600 460" xmlns="http://www.w3.org/2000/svg"
         width="100%" preserveAspectRatio="xMidYMid meet">
      <defs>
        <radialGradient id="helion-plasma-grad" cx="50%" cy="50%" r="50%">
          <stop offset="0%" stop-color="#ffec88" stop-opacity="1"/>
          <stop offset="40%" stop-color="#ff8b3a" stop-opacity="0.9"/>
          <stop offset="100%" stop-color="#ff3a3a" stop-opacity="0"/>
        </radialGradient>
        <linearGradient id="helion-chamber-grad" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stop-color="#1a232c"/>
          <stop offset="100%" stop-color="#0a1014"/>
        </linearGradient>
      </defs>

      <!-- Chamber backplate -->
      <rect x="40" y="170" width="520" height="100" rx="40" ry="40"
            fill="url(#helion-chamber-grad)" stroke="#7a8a96" stroke-width="2"/>

      <!-- Formation chamber west (left end-cap) -->
      <rect id="helion-formation-w" x="40" y="170" width="80" height="100"
            rx="40" ry="40" fill="#284054" opacity="0.85"/>

      <!-- Acceleration coils west -->
      {accel_coils_west}

      <!-- Compression chamber centre -->
      <rect id="helion-compression" x="260" y="170" width="80" height="100"
            rx="20" ry="20" fill="#2e4a7a" stroke="#88b3ff" stroke-width="2"/>

      <!-- Acceleration coils east -->
      {accel_coils_east}

      <!-- Formation chamber east -->
      <rect id="helion-formation-e" x="480" y="170" width="80" height="100"
            rx="40" ry="40" fill="#284054" opacity="0.85"/>

      <!-- Plasmoids (initially at formation chambers; live callback moves x) -->
      <circle id="helion-plasma-w" cx="80" cy="220" r="18"
              fill="url(#helion-plasma-grad)" opacity="0.6"/>
      <circle id="helion-plasma-e" cx="520" cy="220" r="18"
              fill="url(#helion-plasma-grad)" opacity="0.6"/>
      <circle id="helion-plasma-merged" cx="300" cy="220" r="0"
              fill="url(#helion-plasma-grad)" opacity="0.0"/>

      <!-- Damage overlay (first wall + coils) -->
      <rect id="helion-damage-coils" x="120" y="180" width="220" height="80"
            fill="#9b59b6" fill-opacity="0" pointer-events="none"/>
      <rect id="helion-damage-firstwall" x="260" y="160" width="80" height="120"
            fill="#e63946" fill-opacity="0" pointer-events="none"/>

      <!-- Capacitor bank icons -->
      <rect x="60" y="320" width="80" height="60" rx="6"
            fill="#1c1c2c" stroke="#88b3ff" stroke-width="2"/>
      <text x="100" y="345" fill="#88b3ff" font-size="10" text-anchor="middle">CAP BANK W</text>
      <text x="100" y="365" fill="#dceaf3" font-size="9" text-anchor="middle">~5 MJ</text>

      <rect x="460" y="320" width="80" height="60" rx="6"
            fill="#1c1c2c" stroke="#88b3ff" stroke-width="2"/>
      <text x="500" y="345" fill="#88b3ff" font-size="10" text-anchor="middle">CAP BANK E</text>
      <text x="500" y="365" fill="#dceaf3" font-size="9" text-anchor="middle">~5 MJ</text>

      <rect x="250" y="320" width="100" height="60" rx="6"
            fill="#2c1c1c" stroke="#ff8b3a" stroke-width="2"/>
      <text x="300" y="345" fill="#ff8b3a" font-size="10" text-anchor="middle">COMPRESSION</text>
      <text x="300" y="365" fill="#dceaf3" font-size="9" text-anchor="middle">~20 MJ</text>

      <!-- Stage labels above chamber -->
      <text x="80" y="160" fill="#aabbcc" font-size="10" text-anchor="middle">FRC W</text>
      <text x="220" y="160" fill="#aabbcc" font-size="10" text-anchor="middle">accel W</text>
      <text x="300" y="160" fill="#88b3ff" font-size="11" text-anchor="middle"
            font-weight="600">COMPRESS</text>
      <text x="380" y="160" fill="#aabbcc" font-size="10" text-anchor="middle">accel E</text>
      <text x="520" y="160" fill="#aabbcc" font-size="10" text-anchor="middle">FRC E</text>

      <!-- Title + caption -->
      <text x="300" y="22" fill="#dceaf3" font-size="14" text-anchor="middle"
            font-weight="600">Helion FRC — Pulse Sequence View</text>
      <text x="300" y="425" fill="#9bb0c0" font-size="10" text-anchor="middle"
            font-style="italic">
        D-He³ fusion is aneutronic in the dominant branch; D-D side-reactions drive first-wall damage.
      </text>
    </svg>
    """


# Default pulse phase boundaries (fractions of the pulse duration).
# Calibrated against Helion's published Trenta / Polaris pulse traces.
HELION_PHASES: list[tuple[str, float, float]] = [
    ("Formation", 0.00, 0.12),
    ("Acceleration", 0.12, 0.35),
    ("Merge", 0.35, 0.45),
    ("Compression", 0.45, 0.65),
    ("Burn", 0.65, 0.78),
    ("Expansion", 0.78, 1.00),
]


def helion_pulse_phase_figure(
    t_array_s: list[float] | np.ndarray,
    t_kev_array: list[float] | np.ndarray,
    p_fusion_w: list[float] | np.ndarray,
    cursor_s: float | None = None,
) -> go.Figure:
    """Slow-motion pulse trace: T_keV + P_fusion on a log time axis.

    A vertical cursor (driven by the scrubber) shows the current frame
    relative to the pulse trace. Phase boundaries are annotated as shaded
    bands so the user can see the formation → compression → burn evolution.
    """
    t = np.asarray(t_array_s, dtype=float)
    if t.size == 0:
        fig = go.Figure()
        fig.update_layout(
            paper_bgcolor="#0f1620", plot_bgcolor="#0f1620",
            font=dict(color="#dceaf3", size=11),
            title="Run a pulse to see the slow-motion phase view",
        )
        return fig

    duration = float(t[-1]) if t[-1] > 0 else 1.0
    t_us = t * 1.0e6  # microseconds for axis

    fig = go.Figure()
    # Phase shading.
    for label, t0, t1 in HELION_PHASES:
        fig.add_vrect(
            x0=t0 * duration * 1.0e6,
            x1=t1 * duration * 1.0e6,
            fillcolor="#88b3ff" if "Compression" in label or "Burn" in label else "#3a4a5a",
            opacity=0.10,
            line_width=0,
            annotation_text=label,
            annotation_position="top left",
            annotation_font=dict(size=9, color="#aabbcc"),
        )

    fig.add_trace(go.Scatter(
        x=t_us, y=t_kev_array, name="T_ion (keV)",
        line=dict(color="#ff8b3a", width=3),
        yaxis="y1",
    ))
    fig.add_trace(go.Scatter(
        x=t_us, y=p_fusion_w, name="P_fusion (W)",
        line=dict(color="#88b3ff", width=2),
        yaxis="y2",
    ))

    if cursor_s is not None:
        fig.add_vline(
            x=cursor_s * 1.0e6,
            line=dict(color="#ffec88", width=2, dash="dot"),
            annotation_text="now",
            annotation_position="top",
        )

    fig.update_layout(
        title=dict(text="Pulse phase view (slow-motion)", font=dict(size=12)),
        xaxis=dict(title="time (µs)", showgrid=False),
        yaxis=dict(
            title=dict(text="T_ion (keV)", font=dict(color="#ff8b3a")),
            tickfont=dict(color="#ff8b3a"),
            showgrid=False,
        ),
        yaxis2=dict(
            title=dict(text="P_fusion (W)", font=dict(color="#88b3ff")),
            tickfont=dict(color="#88b3ff"),
            anchor="x", overlaying="y", side="right",
            type="log", showgrid=False,
        ),
        margin=dict(l=50, r=60, t=40, b=40),
        height=320,
        paper_bgcolor="#0f1620",
        plot_bgcolor="#0f1620",
        font=dict(color="#dceaf3", size=11),
        legend=dict(orientation="h", x=0.5, xanchor="center", y=1.15),
    )
    return fig


def helion_phase_for_time(t_now_s: float, t_total_s: float) -> str:
    """Return the phase label for the given pulse-local time."""
    if t_total_s <= 0.0:
        return "—"
    frac = max(0.0, min(1.0, t_now_s / t_total_s))
    for label, t0, t1 in HELION_PHASES:
        if t0 <= frac < t1:
            return label
    return HELION_PHASES[-1][0]


def render_helion_diagram(
    t_now_s: float,
    t_total_s: float,
    fw_damage_fraction: float,
    coil_damage_fraction: float,
    t_keV_now: float,
) -> str:
    """Return the full SVG string with state-dependent attributes baked in."""
    from ui.diagrams.pwr_schematic import _apply_svg_patches
    base = _helion_svg_static()
    patches = helion_diagram_patches(
        t_now_s, t_total_s, fw_damage_fraction, coil_damage_fraction, t_keV_now
    )
    return _apply_svg_patches(base, patches)


def helion_diagram_patches(
    t_now_s: float,
    t_total_s: float,
    fw_damage_fraction: float,
    coil_damage_fraction: float,
    t_keV_now: float,
) -> dict[str, dict[str, Any]]:
    """Plasmoid x-positions, merged plasmoid radius, damage overlays.

    During Formation phase the two plasmoids sit in the formation chambers.
    During Acceleration they travel toward the centre. At Merge they collide,
    fade out, and the merged plasmoid fades in. During Compression the merged
    radius shrinks. During Burn the merged plasmoid is brightest. During
    Expansion it grows and fades.
    """
    if t_total_s <= 0.0:
        frac = 0.0
    else:
        frac = max(0.0, min(1.0, t_now_s / t_total_s))

    formation_end = 0.12
    accel_end = 0.45
    burn_end = 0.78

    # Default plasmoid positions.
    cx_w, cx_e, cx_merged = 80.0, 520.0, 300.0
    op_w, op_e, op_m = 0.6, 0.6, 0.0
    r_merged = 0.0

    if frac < formation_end:
        op_w = 0.6
        op_e = 0.6
    elif frac < accel_end:
        travel = (frac - formation_end) / (accel_end - formation_end)
        cx_w = 80.0 + (300.0 - 80.0) * travel
        cx_e = 520.0 - (520.0 - 300.0) * travel
        op_w = 0.6 * (1.0 - travel)
        op_e = 0.6 * (1.0 - travel)
        op_m = travel
        r_merged = 8.0 + 12.0 * travel
    elif frac < burn_end:
        # Compression + burn — merged shrinks then brightens.
        compress = (frac - accel_end) / (burn_end - accel_end)
        op_w = 0.0
        op_e = 0.0
        op_m = 1.0
        r_merged = max(4.0, 20.0 - 14.0 * compress)
    else:
        # Expansion.
        exp = (frac - burn_end) / (1.0 - burn_end)
        op_w = 0.0
        op_e = 0.0
        op_m = max(0.0, 1.0 - exp)
        r_merged = 6.0 + 30.0 * exp

    # Plasma colour tints with temperature: blue at low, white-hot at >50 keV.
    t_clip = max(0.0, min(1.0, t_keV_now / 100.0))
    r = int(120 + 130 * t_clip)
    g = int(120 + 130 * t_clip)
    b = int(255 - 100 * t_clip)
    plasma_color = f"rgb({r},{g},{b})"

    return {
        "helion-plasma-w": {
            "cx": f"{cx_w:.1f}",
            "opacity": f"{op_w:.3f}",
            "fill": "url(#helion-plasma-grad)",
        },
        "helion-plasma-e": {
            "cx": f"{cx_e:.1f}",
            "opacity": f"{op_e:.3f}",
            "fill": "url(#helion-plasma-grad)",
        },
        "helion-plasma-merged": {
            "cx": f"{cx_merged:.1f}",
            "r": f"{r_merged:.1f}",
            "opacity": f"{op_m:.3f}",
            "fill": plasma_color,
        },
        "helion-damage-firstwall": {
            "fill-opacity": f"{min(0.50, 0.50 * fw_damage_fraction):.3f}",
        },
        "helion-damage-coils": {
            "fill-opacity": f"{min(0.40, 0.40 * coil_damage_fraction):.3f}",
        },
    }
