"""HDF5 persistence layer for PWR simulation states and time-series history.

Implements FR-04 (save/load simulation states) and NFR-05 (human-inspectable
HDF5 with descriptive dataset names and units metadata).

HDF5 file layout
----------------
/metadata/                  group
    attrs: reactor_type, timestamp, spec_version
    simulation_time_s       scalar float64, units=s
    parameters              variable-length JSON string
                            (controls, reference_state, base_reactivity_pcm,
                             time_multiplier)

/pwr/                       group — time-series simulation history
    time_s                  float64[N]   units=s
    power_mw                float64[N]   units=MW
    power_normalized        float64[N]   units=dimensionless
    T_fuel_K                float64[N]   units=K
    T_cool_K                float64[N]   units=K
    I_135_atoms_per_cm3     float64[N]   units=atoms/cm3
    Xe_135_atoms_per_cm3    float64[N]   units=atoms/cm3
    reactivity/             subgroup
        rod_pcm             float64[N]   units=pcm
        doppler_pcm         float64[N]   units=pcm
        moderator_pcm       float64[N]   units=pcm
        boron_pcm           float64[N]   units=pcm
        xenon_pcm           float64[N]   units=pcm
        total_pcm           float64[N]   units=pcm
    state/                  subgroup — final ODE vector (warm-restart snapshot)
        n                   float64      units=dimensionless   (index 0)
        C1 … C6             float64      units=neutrons/cm3    (indices 1–6)
        T_fuel_K            float64      units=K               (index 7)
        T_cool_K            float64      units=K               (index 8)
        I_135               float64      units=atoms/cm3       (index 9)
        Xe_135              float64      units=atoms/cm3       (index 10)

State vector indices follow ARCHITECTURE.md §4.1 and SPEC §4.6.2.
"""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from typing import Any

import h5py
import numpy as np

SPEC_VERSION = "1.0"

# ---------------------------------------------------------------------------
# Dataset mapping tables
# ---------------------------------------------------------------------------

# session history key → (HDF5 dataset name relative to /pwr/, units string)
_HISTORY_MAP: dict[str, tuple[str, str]] = {
    "time_s":                ("time_s",               "s"),
    "power_mw":              ("power_mw",             "MW"),
    "power_normalized":      ("power_normalized",      "dimensionless"),
    "fuel_temperature_k":    ("T_fuel_K",             "K"),
    "coolant_temperature_k": ("T_cool_K",             "K"),
    "iodine_concentration":  ("I_135_atoms_per_cm3",  "atoms/cm3"),
    "xenon_concentration":   ("Xe_135_atoms_per_cm3", "atoms/cm3"),
}

# session history key → (HDF5 dataset name relative to /pwr/reactivity/, units)
_REACTIVITY_MAP: dict[str, tuple[str, str]] = {
    "rod_reactivity_pcm":       ("rod_pcm",       "pcm"),
    "doppler_reactivity_pcm":   ("doppler_pcm",   "pcm"),
    "moderator_reactivity_pcm": ("moderator_pcm", "pcm"),
    "boron_reactivity_pcm":     ("boron_pcm",     "pcm"),
    "xenon_reactivity_pcm":     ("xenon_pcm",     "pcm"),
    "total_reactivity_pcm":     ("total_pcm",     "pcm"),
}

# 11-state ODE vector layout: (dataset name, state-vector index, units)
_STATE_LAYOUT: list[tuple[str, int, str]] = [
    ("n",        0,  "dimensionless"),
    ("C1",       1,  "neutrons/cm3"),
    ("C2",       2,  "neutrons/cm3"),
    ("C3",       3,  "neutrons/cm3"),
    ("C4",       4,  "neutrons/cm3"),
    ("C5",       5,  "neutrons/cm3"),
    ("C6",       6,  "neutrons/cm3"),
    ("T_fuel_K", 7,  "K"),
    ("T_cool_K", 8,  "K"),
    ("I_135",    9,  "atoms/cm3"),
    ("Xe_135",   10, "atoms/cm3"),
]

# Paths that must exist in any valid file written by this module.
_REQUIRED_PATHS: tuple[str, ...] = (
    "/metadata",
    "/metadata/parameters",
    "/metadata/simulation_time_s",
    "/pwr/time_s",
    "/pwr/power_mw",
    "/pwr/T_fuel_K",
    "/pwr/T_cool_K",
    "/pwr/reactivity/total_pcm",
    "/pwr/state/n",
    "/pwr/state/T_fuel_K",
    "/pwr/state/Xe_135",
)


# ---------------------------------------------------------------------------
# Public exception
# ---------------------------------------------------------------------------

class HDF5StorageError(RuntimeError):
    """Raised when an HDF5 save or load operation fails in a defined way."""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def save_pwr_session(
    session: dict[str, Any],
    filepath: str | os.PathLike,
) -> None:
    """Serialize a PWR simulation session to an HDF5 file.

    Writes the full time-series history, final 11-state ODE vector, and
    all metadata needed to resume or re-plot the run.

    Args:
        session:  Session dict produced by ``sim_runner.run_pwr_control_step``
                  or ``run_pwr_accelerated_batch``.  Must contain ``history``,
                  ``state`` (11-element list), and ``time_s``.
        filepath: Destination ``.h5`` file path.  The parent directory must
                  already exist.

    Raises:
        ValueError:       If ``session`` is missing required keys or ``state``
                          is not 11 elements.
        HDF5StorageError: On any I/O or HDF5 serialization failure.
    """
    _validate_session_keys(session)

    history = session["history"]
    state_vec = np.asarray(session["state"], dtype=np.float64)

    params_payload: dict[str, Any] = {
        "controls": session.get("controls", {}),
        "reference_state": session.get("reference_state", {}),
        "base_reactivity_pcm": float(session.get("base_reactivity_pcm", 0.0)),
        "time_multiplier": session.get("time_multiplier", 1),
    }
    params_json: str = json.dumps(params_payload)

    try:
        with h5py.File(filepath, "w") as f:
            _write_metadata(f, session, params_json)
            _write_history(f, history)
            _write_state_vector(f, state_vec)
    except OSError as exc:
        raise HDF5StorageError(
            f"Failed to write HDF5 file '{filepath}': {exc}"
        ) from exc


def session_to_hdf5_bytes(session: dict[str, Any]) -> bytes:
    """Serialize a PWR session to HDF5 bytes for browser download (FR-04)."""
    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
        path = tmp.name
    try:
        save_pwr_session(session, path)
        with open(path, "rb") as handle:
            return handle.read()
    finally:
        if os.path.exists(path):
            os.unlink(path)


def load_pwr_session_from_bytes(data: bytes) -> dict[str, Any]:
    """Load a PWR session from HDF5 bytes uploaded through the Dash UI."""
    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
        tmp.write(data)
        path = tmp.name
    try:
        return load_pwr_session(path)
    finally:
        if os.path.exists(path):
            os.unlink(path)


def load_pwr_session(filepath: str | os.PathLike) -> dict[str, Any]:
    """Load a PWR simulation session from an HDF5 file.

    Validates the file structure, then returns a dict compatible with the
    ``sim_runner`` session layout so the engine can plot historical runs or
    resume calculations from the stored snapshot.

    Args:
        filepath: Path to a ``.h5`` file written by ``save_pwr_session``.

    Returns:
        Dict with keys:
            ``history``             — matches ``_empty_history()`` structure
            ``state``               — 11-element list (final ODE state)
            ``time_s``              — simulation clock at save time (s)
            ``base_reactivity_pcm`` — critical base reactivity
            ``reference_state``     — ``{fuel_temperature_ref_k, coolant_temperature_ref_k}``
            ``controls``            — last applied control dict
            ``metadata``            — ``{reactor_type, timestamp, spec_version, parameters}``

    Raises:
        FileNotFoundError:  If ``filepath`` does not exist.
        HDF5StorageError:   If the file is corrupt, not a valid HDF5 file, or
                            missing required datasets.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Simulation file not found: '{filepath}'")

    try:
        with h5py.File(filepath, "r") as f:
            _validate_file_structure(f, filepath)

            meta_grp = f["metadata"]
            reactor_type   = str(meta_grp.attrs.get("reactor_type", ""))
            timestamp      = str(meta_grp.attrs.get("timestamp", ""))
            spec_version   = str(meta_grp.attrs.get("spec_version", ""))
            simulation_time_s = float(meta_grp["simulation_time_s"][()])

            raw = meta_grp["parameters"][()]
            params_json = raw.decode("utf-8") if isinstance(raw, bytes) else str(raw)
            params_dict: dict[str, Any] = json.loads(params_json)

            history = _read_history(f)
            state_vec = _read_state_vector(f)

    except OSError as exc:
        raise HDF5StorageError(
            f"Cannot read HDF5 file '{filepath}': {exc}"
        ) from exc
    except (KeyError, ValueError) as exc:
        raise HDF5StorageError(
            f"Invalid or incomplete HDF5 file '{filepath}': {exc}"
        ) from exc

    return {
        "time_s":               simulation_time_s,
        "state":                state_vec.tolist(),
        "history":              history,
        "base_reactivity_pcm":  params_dict.get("base_reactivity_pcm", 0.0),
        "reference_state":      params_dict.get("reference_state", {}),
        "controls":             params_dict.get("controls", {}),
        "metadata": {
            "reactor_type": reactor_type,
            "timestamp":    timestamp,
            "spec_version": spec_version,
            "parameters":   params_dict,
        },
    }


# ---------------------------------------------------------------------------
# Internal helpers — write
# ---------------------------------------------------------------------------

def _write_metadata(
    f: h5py.File,
    session: dict[str, Any],
    params_json: str,
) -> None:
    meta = f.require_group("metadata")
    meta.attrs["reactor_type"] = "PWR"
    meta.attrs["timestamp"]    = datetime.now(timezone.utc).isoformat()
    meta.attrs["spec_version"] = SPEC_VERSION

    ds = meta.create_dataset(
        "simulation_time_s",
        data=np.float64(session.get("time_s", 0.0)),
    )
    ds.attrs["units"] = "s"

    meta.create_dataset(
        "parameters",
        data=params_json,
        dtype=h5py.string_dtype(),
    )


def _write_history(f: h5py.File, history: dict[str, list]) -> None:
    pwr = f.require_group("pwr")

    for session_key, (ds_name, units) in _HISTORY_MAP.items():
        arr = np.asarray(history.get(session_key, []), dtype=np.float64)
        _create_ds(pwr, ds_name, arr, units)

    react = pwr.require_group("reactivity")
    for session_key, (ds_name, units) in _REACTIVITY_MAP.items():
        arr = np.asarray(history.get(session_key, []), dtype=np.float64)
        _create_ds(react, ds_name, arr, units)


def _write_state_vector(f: h5py.File, state_vec: np.ndarray) -> None:
    state_grp = f.require_group("pwr/state")
    for var_name, idx, units in _STATE_LAYOUT:
        ds = state_grp.create_dataset(var_name, data=np.float64(state_vec[idx]))
        ds.attrs["units"] = units


def _create_ds(
    group: h5py.Group,
    name: str,
    arr: np.ndarray,
    units: str,
) -> h5py.Dataset:
    """Create a dataset with a units attribute; uses gzip only for non-empty arrays."""
    if arr.size > 0:
        ds = group.create_dataset(
            name, data=arr, compression="gzip", compression_opts=4,
        )
    else:
        ds = group.create_dataset(name, data=arr)
    ds.attrs["units"] = units
    return ds


# ---------------------------------------------------------------------------
# Internal helpers — read
# ---------------------------------------------------------------------------

def _read_history(f: h5py.File) -> dict[str, list[float]]:
    pwr = f["pwr"]
    history: dict[str, list[float]] = {}

    for session_key, (ds_name, _) in _HISTORY_MAP.items():
        history[session_key] = pwr[ds_name][()].tolist()

    react = pwr["reactivity"]
    for session_key, (ds_name, _) in _REACTIVITY_MAP.items():
        history[session_key] = react[ds_name][()].tolist()

    return history


def _read_state_vector(f: h5py.File) -> np.ndarray:
    state_grp = f["pwr/state"]
    vec = np.zeros(11, dtype=np.float64)
    for var_name, idx, _ in _STATE_LAYOUT:
        vec[idx] = float(state_grp[var_name][()])
    return vec


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def _validate_session_keys(session: dict[str, Any]) -> None:
    required = {"history", "state", "time_s"}
    missing = required - session.keys()
    if missing:
        raise ValueError(
            f"Session dict is missing required keys: {sorted(missing)}"
        )
    state = session["state"]
    if len(state) != 11:
        raise ValueError(
            f"session['state'] must have 11 elements (got {len(state)})"
        )


def _validate_file_structure(f: h5py.File, filepath: str | os.PathLike) -> None:
    missing = [p for p in _REQUIRED_PATHS if p not in f]
    if missing:
        raise HDF5StorageError(
            f"HDF5 file '{filepath}' is missing required datasets: {missing}"
        )
