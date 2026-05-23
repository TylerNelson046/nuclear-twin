"""Tests for parsed IAEA delayed-neutron reference data."""

from math import log

import pytest

from data import SOURCE_URL, decay_constants_s_inv, load_u235_thermal_6group


def test_u235_thermal_6group_table_loads_all_groups() -> None:
    groups = load_u235_thermal_6group()

    assert [group.group for group in groups] == [1, 2, 3, 4, 5, 6]
    assert [group.half_life_s for group in groups] == [
        53.9,
        22.3,
        6.40,
        2.26,
        0.494,
        0.179,
    ]
    assert sum(group.relative_abundance for group in groups) == pytest.approx(1.0)


def test_decay_constants_are_derived_from_half_lives() -> None:
    groups = load_u235_thermal_6group()
    decay_constants = decay_constants_s_inv()

    assert len(decay_constants) == 6
    assert decay_constants == pytest.approx(
        tuple(log(2.0) / group.half_life_s for group in groups)
    )


def test_iaea_source_metadata_is_recorded() -> None:
    assert SOURCE_URL.startswith("https://nds.iaea.org/beta-delayed-neutron/")


def test_load_raises_when_group_count_is_wrong(tmp_path) -> None:
    """load_u235_thermal_6group raises ValueError when CSV contains != 6 groups.

    Covers the validation guard at iaea_delayed_neutron.py line 58-59.
    """
    bad_csv = tmp_path / "bad_groups.csv"
    # Write a valid-format CSV with only 5 rows (one group short)
    bad_csv.write_text(
        "group,half_life_s,half_life_uncertainty_s,"
        "relative_abundance,relative_abundance_uncertainty\n"
        "1,53.9,0.3,0.033,0.003\n"
        "2,22.3,0.2,0.219,0.010\n"
        "3,6.40,0.1,0.196,0.008\n"
        "4,2.26,0.05,0.395,0.015\n"
        "5,0.494,0.010,0.115,0.005\n"
        # Group 6 intentionally omitted — only 5 groups
    )

    with pytest.raises(ValueError, match="Expected 6 delayed-neutron groups, found 5"):
        load_u235_thermal_6group(path=bad_csv)
