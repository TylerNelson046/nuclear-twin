"""Unit tests for the HDF5 save/load layer (utils/data_handler.py).

Validates:
  1. save_pwr_session creates a file at the specified path.
  2. Full roundtrip: loaded history arrays are bit-for-bit identical to saved arrays.
  3. Final 11-state ODE vector is preserved exactly.
  4. Every dataset carries a non-empty 'units' attribute.
  5. The /metadata group has reactor_type, timestamp, and spec_version.
  6. Loading a missing file raises FileNotFoundError.
  7. Loading a corrupt (non-HDF5) file raises HDF5StorageError.
  8. Loading a structurally incomplete HDF5 file raises HDF5StorageError.
  9. save_pwr_session raises ValueError for missing session keys.
  10. save_pwr_session raises ValueError when state is not 11 elements.

State vector layout (ARCHITECTURE.md §4.1):
  y[0]    n           neutron population
  y[1:7]  C1–C6       delayed-neutron precursor concentrations  (neutrons/cm³)
  y[7]    T_fuel_K    lumped fuel temperature                   (K)
  y[8]    T_cool_K    lumped coolant temperature                (K)
  y[9]    I_135       I-135 number density                      (atoms/cm³)
  y[10]   Xe_135      Xe-135 number density                     (atoms/cm³)
"""

from __future__ import annotations

import json
import os

import h5py
import numpy as np
import numpy.testing as npt
import pytest

from utils.data_handler import (
    HDF5StorageError,
    SPEC_VERSION,
    _HISTORY_MAP,
    _REACTIVITY_MAP,
    _REQUIRED_PATHS,
    _STATE_LAYOUT,
    load_pwr_session,
    save_pwr_session,
)


# ---------------------------------------------------------------------------
# Shared mock session fixture
# ---------------------------------------------------------------------------

N = 60  # history length used across all tests


@pytest.fixture
def mock_session() -> dict:
    """Realistic PWR session dict matching sim_runner output structure."""
    rng = np.random.default_rng(seed=42)
    t = np.linspace(0.0, 590.0, N)

    history = {
        "time_s": t.tolist(),
        "power_mw": (3000.0 + 10 * rng.standard_normal(N)).tolist(),
        "power_normalized": (1.0 + 1e-3 * rng.standard_normal(N)).tolist(),
        "fuel_temperature_k": (900.0 + 2 * rng.standard_normal(N)).tolist(),
        "coolant_temperature_k": (590.0 + rng.standard_normal(N)).tolist(),
        "rod_reactivity_pcm": np.zeros(N).tolist(),
        "doppler_reactivity_pcm": (-250.0 + rng.standard_normal(N)).tolist(),
        "moderator_reactivity_pcm": (-800.0 + rng.standard_normal(N)).tolist(),
        "boron_reactivity_pcm": (-500.0 * np.ones(N)).tolist(),
        "xenon_reactivity_pcm": (-2750.0 * np.ones(N)).tolist(),
        "total_reactivity_pcm": (0.01 * rng.standard_normal(N)).tolist(),
        "iodine_concentration": (1.5e14 * np.ones(N)).tolist(),
        "xenon_concentration": (8.0e13 * np.ones(N)).tolist(),
    }

    state = [
        1.0e8,   # n
        2.5e6,   # C1
        1.4e7,   # C2
        1.3e7,   # C3
        2.8e7,   # C4
        8.5e6,   # C5
        2.9e6,   # C6
        900.0,   # T_fuel_K
        590.0,   # T_cool_K
        1.5e14,  # I-135
        8.0e13,  # Xe-135
    ]

    return {
        "time_s": 590.0,
        "state": state,
        "history": history,
        "base_reactivity_pcm": -100.0,
        "reference_state": {
            "fuel_temperature_ref_k": 900.0,
            "coolant_temperature_ref_k": 590.0,
        },
        "controls": {
            "rod_reactivity_pcm": 0.0,
            "boron_ppm": 1000.0,
            "coolant_flow_fraction": 1.0,
            "inlet_temperature_k": 565.0,
        },
        "time_multiplier": 1,
    }


# ---------------------------------------------------------------------------
# 1. File creation
# ---------------------------------------------------------------------------

def test_save_creates_file(tmp_path, mock_session):
    path = tmp_path / "run.h5"
    assert not path.exists()
    save_pwr_session(mock_session, path)
    assert path.exists()
    assert path.stat().st_size > 0


# ---------------------------------------------------------------------------
# 2 & 3. Full roundtrip — history arrays and state vector
# ---------------------------------------------------------------------------

@pytest.fixture
def roundtrip(tmp_path, mock_session):
    """Save then load a session, returning (original_session, loaded_result)."""
    path = tmp_path / "roundtrip.h5"
    save_pwr_session(mock_session, path)
    loaded = load_pwr_session(path)
    return mock_session, loaded


def test_roundtrip_time_series(roundtrip):
    original, loaded = roundtrip
    for key in _HISTORY_MAP:
        orig_arr = np.array(original["history"][key], dtype=np.float64)
        load_arr = np.array(loaded["history"][key], dtype=np.float64)
        npt.assert_array_equal(
            load_arr, orig_arr,
            err_msg=f"History array '{key}' changed across save/load roundtrip",
        )


def test_roundtrip_reactivity_series(roundtrip):
    original, loaded = roundtrip
    for key in _REACTIVITY_MAP:
        orig_arr = np.array(original["history"][key], dtype=np.float64)
        load_arr = np.array(loaded["history"][key], dtype=np.float64)
        npt.assert_array_equal(
            load_arr, orig_arr,
            err_msg=f"Reactivity array '{key}' changed across save/load roundtrip",
        )


def test_roundtrip_state_vector(roundtrip):
    original, loaded = roundtrip
    orig_state = np.array(original["state"], dtype=np.float64)
    load_state = np.array(loaded["state"], dtype=np.float64)
    npt.assert_array_equal(load_state, orig_state)


def test_roundtrip_simulation_time(roundtrip):
    original, loaded = roundtrip
    assert loaded["time_s"] == pytest.approx(original["time_s"])


def test_roundtrip_base_reactivity(roundtrip):
    original, loaded = roundtrip
    assert loaded["base_reactivity_pcm"] == pytest.approx(
        original["base_reactivity_pcm"]
    )


def test_roundtrip_reference_state(roundtrip):
    original, loaded = roundtrip
    assert loaded["reference_state"] == original["reference_state"]


def test_roundtrip_controls(roundtrip):
    original, loaded = roundtrip
    assert loaded["controls"] == original["controls"]


def test_roundtrip_state_length(roundtrip):
    _, loaded = roundtrip
    assert len(loaded["state"]) == 11


# ---------------------------------------------------------------------------
# 4. Units attributes on every dataset
# ---------------------------------------------------------------------------

def test_all_history_datasets_have_units(tmp_path, mock_session):
    path = tmp_path / "units.h5"
    save_pwr_session(mock_session, path)
    with h5py.File(path, "r") as f:
        for _, (ds_name, expected_units) in _HISTORY_MAP.items():
            ds = f[f"/pwr/{ds_name}"]
            assert "units" in ds.attrs, f"/pwr/{ds_name} missing 'units' attribute"
            assert ds.attrs["units"] == expected_units


def test_all_reactivity_datasets_have_units(tmp_path, mock_session):
    path = tmp_path / "units.h5"
    save_pwr_session(mock_session, path)
    with h5py.File(path, "r") as f:
        for _, (ds_name, expected_units) in _REACTIVITY_MAP.items():
            ds = f[f"/pwr/reactivity/{ds_name}"]
            assert "units" in ds.attrs, f"/pwr/reactivity/{ds_name} missing 'units'"
            assert ds.attrs["units"] == expected_units


def test_all_state_datasets_have_units(tmp_path, mock_session):
    path = tmp_path / "units.h5"
    save_pwr_session(mock_session, path)
    with h5py.File(path, "r") as f:
        for var_name, _, expected_units in _STATE_LAYOUT:
            ds = f[f"/pwr/state/{var_name}"]
            assert "units" in ds.attrs, f"/pwr/state/{var_name} missing 'units'"
            assert ds.attrs["units"] == expected_units


def test_simulation_time_dataset_has_units(tmp_path, mock_session):
    path = tmp_path / "units.h5"
    save_pwr_session(mock_session, path)
    with h5py.File(path, "r") as f:
        assert f["/metadata/simulation_time_s"].attrs["units"] == "s"


# ---------------------------------------------------------------------------
# 5. Metadata group contents
# ---------------------------------------------------------------------------

def test_metadata_reactor_type(tmp_path, mock_session):
    path = tmp_path / "meta.h5"
    save_pwr_session(mock_session, path)
    with h5py.File(path, "r") as f:
        assert f["/metadata"].attrs["reactor_type"] == "PWR"


def test_metadata_spec_version(tmp_path, mock_session):
    path = tmp_path / "meta.h5"
    save_pwr_session(mock_session, path)
    with h5py.File(path, "r") as f:
        assert f["/metadata"].attrs["spec_version"] == SPEC_VERSION


def test_metadata_timestamp_present(tmp_path, mock_session):
    path = tmp_path / "meta.h5"
    save_pwr_session(mock_session, path)
    with h5py.File(path, "r") as f:
        ts = f["/metadata"].attrs["timestamp"]
        assert isinstance(ts, str) and len(ts) > 0


def test_metadata_parameters_is_valid_json(tmp_path, mock_session):
    path = tmp_path / "meta.h5"
    save_pwr_session(mock_session, path)
    with h5py.File(path, "r") as f:
        raw = f["/metadata/parameters"][()]
        text = raw.decode("utf-8") if isinstance(raw, bytes) else str(raw)
        obj = json.loads(text)
    assert "controls" in obj
    assert "reference_state" in obj
    assert "base_reactivity_pcm" in obj


# ---------------------------------------------------------------------------
# 6. FileNotFoundError for missing file
# ---------------------------------------------------------------------------

def test_load_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="not found"):
        load_pwr_session(tmp_path / "does_not_exist.h5")


# ---------------------------------------------------------------------------
# 7. HDF5StorageError for corrupt file
# ---------------------------------------------------------------------------

def test_load_corrupt_file_raises(tmp_path):
    corrupt = tmp_path / "corrupt.h5"
    corrupt.write_bytes(b"THIS IS NOT AN HDF5 FILE\x00\xff\xfe")
    with pytest.raises(HDF5StorageError, match="Cannot read HDF5 file"):
        load_pwr_session(corrupt)


# ---------------------------------------------------------------------------
# 8. HDF5StorageError for incomplete HDF5 structure
# ---------------------------------------------------------------------------

def test_load_missing_dataset_raises(tmp_path, mock_session):
    """An HDF5 file that omits /pwr/time_s must raise HDF5StorageError."""
    path = tmp_path / "incomplete.h5"
    save_pwr_session(mock_session, path)

    # Remove a required dataset by copying everything except /pwr/time_s
    incomplete = tmp_path / "incomplete2.h5"
    with h5py.File(path, "r") as src, h5py.File(incomplete, "w") as dst:
        src.copy("metadata", dst)
        # Copy /pwr but skip time_s
        pwr_src = src["pwr"]
        pwr_dst = dst.require_group("pwr")
        for key in pwr_src:
            if key != "time_s":
                pwr_src.copy(key, pwr_dst)

    with pytest.raises(HDF5StorageError, match="missing required datasets"):
        load_pwr_session(incomplete)


# ---------------------------------------------------------------------------
# 9. ValueError for missing session keys
# ---------------------------------------------------------------------------

def test_save_raises_on_missing_history():
    bad = {"state": [0.0] * 11, "time_s": 0.0}  # no "history"
    with pytest.raises(ValueError, match="missing required keys"):
        save_pwr_session(bad, "/tmp/should_not_be_created.h5")


def test_save_raises_on_missing_state():
    bad = {"history": {}, "time_s": 0.0}  # no "state"
    with pytest.raises(ValueError, match="missing required keys"):
        save_pwr_session(bad, "/tmp/should_not_be_created.h5")


def test_save_raises_on_missing_time_s():
    bad = {"history": {}, "state": [0.0] * 11}  # no "time_s"
    with pytest.raises(ValueError, match="missing required keys"):
        save_pwr_session(bad, "/tmp/should_not_be_created.h5")


# ---------------------------------------------------------------------------
# 10. ValueError when state vector is not 11 elements
# ---------------------------------------------------------------------------

def test_save_raises_on_wrong_state_length():
    bad = {"history": {}, "state": [0.0] * 10, "time_s": 0.0}
    with pytest.raises(ValueError, match="11 elements"):
        save_pwr_session(bad, "/tmp/should_not_be_created.h5")


def test_save_raises_on_oversized_state():
    bad = {"history": {}, "state": [0.0] * 12, "time_s": 0.0}
    with pytest.raises(ValueError, match="11 elements"):
        save_pwr_session(bad, "/tmp/should_not_be_created.h5")


# ---------------------------------------------------------------------------
# Edge case: empty history arrays
# ---------------------------------------------------------------------------

def test_roundtrip_empty_history(tmp_path):
    """Empty history lists survive save/load without error."""
    from utils.data_handler import _HISTORY_MAP, _REACTIVITY_MAP

    empty_history = {k: [] for k in list(_HISTORY_MAP) + list(_REACTIVITY_MAP)}
    session = {
        "time_s": 0.0,
        "state": [1.0e8, 2.5e6, 1.4e7, 1.3e7, 2.8e7, 8.5e6, 2.9e6,
                  900.0, 590.0, 1.5e14, 8.0e13],
        "history": empty_history,
    }
    path = tmp_path / "empty.h5"
    save_pwr_session(session, path)
    loaded = load_pwr_session(path)
    for key in _HISTORY_MAP:
        assert loaded["history"][key] == []
    for key in _REACTIVITY_MAP:
        assert loaded["history"][key] == []


# ---------------------------------------------------------------------------
# State vector index correctness
# ---------------------------------------------------------------------------

def test_state_vector_index_mapping(tmp_path, mock_session):
    """Each named state variable maps to the correct ODE vector index."""
    path = tmp_path / "state_idx.h5"
    save_pwr_session(mock_session, path)
    orig = np.array(mock_session["state"], dtype=np.float64)

    with h5py.File(path, "r") as f:
        for var_name, idx, _ in _STATE_LAYOUT:
            stored = float(f[f"/pwr/state/{var_name}"][()])
            assert stored == pytest.approx(orig[idx]), (
                f"/pwr/state/{var_name} (index {idx}): "
                f"expected {orig[idx]}, got {stored}"
            )


# ---------------------------------------------------------------------------
# All required paths present in a freshly written file
# ---------------------------------------------------------------------------

def test_all_required_paths_written(tmp_path, mock_session):
    path = tmp_path / "full.h5"
    save_pwr_session(mock_session, path)
    with h5py.File(path, "r") as f:
        for req_path in _REQUIRED_PATHS:
            assert req_path in f, f"Required path '{req_path}' not found in HDF5 file"
