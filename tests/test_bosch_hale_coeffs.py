"""Tests for Bosch-Hale D-He3 coefficient data."""

import pytest

from data.bosch_hale_coeffs import DHE3_BOSCH_HALE, SOURCE_DOI


def test_dhe3_bosch_hale_coefficients_match_table_iv() -> None:
    assert DHE3_BOSCH_HALE.reaction == "D(He3,p)He4"
    assert DHE3_BOSCH_HALE.bg == pytest.approx(68.7508)
    assert DHE3_BOSCH_HALE.mrc2 == pytest.approx(1_124_572.0)
    assert DHE3_BOSCH_HALE.polynomial_coefficients == pytest.approx(
        (
            5.51036e-10,
            0.00641918,
            -0.00202896,
            -1.91080e-5,
            1.35776e-4,
            0.0,
            0.0,
        )
    )


def test_dhe3_temperature_bounds_are_recorded() -> None:
    assert DHE3_BOSCH_HALE.temperature_min_kev == 0.5
    assert DHE3_BOSCH_HALE.temperature_max_kev == 190.0
    assert SOURCE_DOI.endswith("32/4/i07")
