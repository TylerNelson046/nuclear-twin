"""Parsed IAEA delayed-neutron group data.

This module contains the U-235 thermal-spectrum 6-group delayed neutron table
from the IAEA Reference Database for Beta-Delayed Neutron Emission.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from math import log
from pathlib import Path


SOURCE_URL = "https://nds.iaea.org/beta-delayed-neutron/databases/delayedn_fy_ado.html"
SOURCE_DESCRIPTION = (
    "IAEA Reference Database for Beta-Delayed Neutron Emission, U-235 thermal "
    "spectrum, 6-groups original model."
)
U235_THERMAL_6GROUP_CSV = Path(__file__).with_name("iaea_u235_thermal_6group.csv")


@dataclass(frozen=True)
class DelayedNeutronGroup:
    """One delayed-neutron group from the parsed IAEA table."""

    group: int
    half_life_s: float
    half_life_uncertainty_s: float
    relative_abundance: float
    relative_abundance_uncertainty: float

    @property
    def decay_constant_s_inv(self) -> float:
        """Return lambda = ln(2) / half-life for this precursor group."""
        return log(2.0) / self.half_life_s


def load_u235_thermal_6group(
    path: Path = U235_THERMAL_6GROUP_CSV,
) -> tuple[DelayedNeutronGroup, ...]:
    """Load the U-235 thermal-spectrum 6-group delayed-neutron table."""
    with path.open(newline="", encoding="utf-8") as csv_file:
        rows = csv.DictReader(csv_file)
        groups = tuple(
            DelayedNeutronGroup(
                group=int(row["group"]),
                half_life_s=float(row["half_life_s"]),
                half_life_uncertainty_s=float(row["half_life_uncertainty_s"]),
                relative_abundance=float(row["relative_abundance"]),
                relative_abundance_uncertainty=float(
                    row["relative_abundance_uncertainty"]
                ),
            )
            for row in rows
        )

    if len(groups) != 6:
        raise ValueError(f"Expected 6 delayed-neutron groups, found {len(groups)}")

    return groups


def decay_constants_s_inv() -> tuple[float, ...]:
    """Return decay constants for the U-235 thermal-spectrum 6-group table."""
    return tuple(group.decay_constant_s_inv for group in load_u235_thermal_6group())
