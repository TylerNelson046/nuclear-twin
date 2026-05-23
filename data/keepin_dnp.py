"""Keepin-style U-235 6-group delayed neutron parameters.

The relative group abundances are loaded from the parsed IAEA U-235 thermal
6-group table and scaled to the nominal U-235 effective delayed neutron fraction
used by the reactor kinetics specification.
"""

from __future__ import annotations

from dataclasses import dataclass

from data.iaea_delayed_neutron import load_u235_thermal_6group


BETA_EFF_U235_THERMAL = 0.0065


@dataclass(frozen=True)
class KeepinGroup:
    """One delayed-neutron precursor group for point kinetics."""

    group: int
    beta_i: float
    lambda_i: float
    relative_abundance: float
    half_life_s: float


def load_keepin_u235_groups(
    beta_eff: float = BETA_EFF_U235_THERMAL,
) -> tuple[KeepinGroup, ...]:
    """Return U-235 six-group beta fractions and decay constants."""
    iaea_groups = load_u235_thermal_6group()
    relative_sum = sum(group.relative_abundance for group in iaea_groups)

    return tuple(
        KeepinGroup(
            group=group.group,
            beta_i=beta_eff * group.relative_abundance / relative_sum,
            lambda_i=group.decay_constant_s_inv,
            relative_abundance=group.relative_abundance,
            half_life_s=group.half_life_s,
        )
        for group in iaea_groups
    )


def beta_fractions(beta_eff: float = BETA_EFF_U235_THERMAL) -> tuple[float, ...]:
    """Return beta_i values for all six U-235 precursor groups."""
    return tuple(group.beta_i for group in load_keepin_u235_groups(beta_eff))


def decay_constants() -> tuple[float, ...]:
    """Return lambda_i decay constants for all six U-235 precursor groups."""
    return tuple(group.lambda_i for group in load_keepin_u235_groups())
