"""Material aging and remaining-life models for the three reactor twins.

This module is an illustrative add-on to the core physics layer. It tracks
slow damage indicators (fluence accumulation, oxide growth, B-10 depletion,
salt-side corrosion, pulse fatigue) and projects remaining design life by
linear extrapolation of the current rate.

These models are first-order, calibrated to published design-basis values for
commercial reactor materials. They are NOT a substitute for licensed
materials-surveillance analyses. The companion UI labels every projection as
"illustrative" so users do not mistake them for engineering predictions.

See `aging.py` for the per-reactor step functions and remaining-life
projections; `state.py` for the serializable state dataclasses and
default-initialization helpers.
"""

from physics.materials.state import (
    PWRMaterialState,
    MSRMaterialState,
    HelionMaterialState,
    fresh_pwr_material_state,
    fresh_msr_material_state,
    fresh_helion_material_state,
)
from physics.materials.aging import (
    step_pwr_aging,
    step_msr_aging,
    step_helion_pulse_aging,
    pwr_remaining_life,
    msr_remaining_life,
    helion_remaining_life,
    PWR_DESIGN_LIMITS,
    MSR_DESIGN_LIMITS,
    HELION_DESIGN_LIMITS,
)

__all__ = [
    "PWRMaterialState",
    "MSRMaterialState",
    "HelionMaterialState",
    "fresh_pwr_material_state",
    "fresh_msr_material_state",
    "fresh_helion_material_state",
    "step_pwr_aging",
    "step_msr_aging",
    "step_helion_pulse_aging",
    "pwr_remaining_life",
    "msr_remaining_life",
    "helion_remaining_life",
    "PWR_DESIGN_LIMITS",
    "MSR_DESIGN_LIMITS",
    "HELION_DESIGN_LIMITS",
]
