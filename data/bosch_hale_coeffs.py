"""Bosch-Hale D-He3 fusion reactivity coefficients.

Constants are from Bosch and Hale, "Improved formulas for fusion cross-sections
and thermal reactivities", Nuclear Fusion 32(4), 1992, Table IV. The fitted
temperature range for D-He3 is 0.5 to 190 keV.
"""

from dataclasses import dataclass


SOURCE_DOI = "https://doi.org/10.1088/0029-5515/32/4/i07"
DHE3_TEMPERATURE_RANGE_KEV = (0.5, 190.0)


@dataclass(frozen=True)
class BoschHaleCoefficients:
    """Coefficient set for one Bosch-Hale reaction fit."""

    reaction: str
    bg: float
    mrc2: float
    c1: float
    c2: float
    c3: float
    c4: float
    c5: float
    c6: float
    c7: float
    temperature_min_kev: float
    temperature_max_kev: float

    @property
    def polynomial_coefficients(self) -> tuple[float, ...]:
        """Return C1 through C7 in Bosch-Hale table order."""
        return (self.c1, self.c2, self.c3, self.c4, self.c5, self.c6, self.c7)


DHE3_BOSCH_HALE = BoschHaleCoefficients(
    reaction="D(He3,p)He4",
    bg=68.7508,
    mrc2=1_124_572.0,
    c1=5.51036e-10,
    c2=0.00641918,
    c3=-0.00202896,
    c4=-1.91080e-5,
    c5=1.35776e-4,
    c6=0.0,
    c7=0.0,
    temperature_min_kev=DHE3_TEMPERATURE_RANGE_KEV[0],
    temperature_max_kev=DHE3_TEMPERATURE_RANGE_KEV[1],
)