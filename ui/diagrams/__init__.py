"""Reactor schematic diagrams + Plotly heatmap helpers for the UI layer."""

from ui.diagrams.pwr_schematic import (
    build_pwr_diagram,
    pwr_heatmap_figure,
    render_pwr_diagram,
)
from ui.diagrams.msr_schematic import (
    build_msr_diagram,
    msr_loop_heatmap_figure,
    render_msr_diagram,
)
from ui.diagrams.helion_schematic import (
    build_helion_diagram,
    helion_pulse_phase_figure,
    render_helion_diagram,
    helion_phase_for_time,
)
from ui.diagrams.damage_panel import (
    build_damage_panel,
    update_damage_panel_children,
    damage_history_figure,
)

__all__ = [
    "build_pwr_diagram",
    "pwr_heatmap_figure",
    "render_pwr_diagram",
    "build_msr_diagram",
    "msr_loop_heatmap_figure",
    "render_msr_diagram",
    "build_helion_diagram",
    "helion_pulse_phase_figure",
    "render_helion_diagram",
    "helion_phase_for_time",
    "build_damage_panel",
    "update_damage_panel_children",
    "damage_history_figure",
]
