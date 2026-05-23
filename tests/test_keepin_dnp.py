"""Tests for U-235 delayed-neutron point kinetics parameters."""

from math import log

import pytest

from data.keepin_dnp import (
    BETA_EFF_U235_THERMAL,
    beta_fractions,
    decay_constants,
    load_keepin_u235_groups,
)


def test_keepin_groups_scale_relative_abundances_to_beta_eff() -> None:
    groups = load_keepin_u235_groups()

    assert [group.group for group in groups] == [1, 2, 3, 4, 5, 6]
    assert sum(group.beta_i for group in groups) == pytest.approx(
        BETA_EFF_U235_THERMAL
    )
    assert beta_fractions() == pytest.approx(tuple(group.beta_i for group in groups))


def test_keepin_decay_constants_match_half_lives() -> None:
    groups = load_keepin_u235_groups()

    assert decay_constants() == pytest.approx(
        tuple(log(2.0) / group.half_life_s for group in groups)
    )
