"""benchmarks/benchmark_helion_pulse.py — Helion FRC performance benchmark.

Measures the Helion plasma energy balance ODE integration (SPEC Eq. 16–21)
and the 2D ignition boundary sweep execution time, validating NFR-02 targets:

    Without Numba:  ODE timestep < 100 ms per UI callback
    With Numba:     ODE timestep < 20 ms per UI callback (Bosch-Hale kernel)

Run directly (Numba JIT fully enabled):
    cd nuclear-twin
    poetry run python benchmarks/benchmark_helion_pulse.py
"""

from __future__ import annotations

import os
import sys
import time

os.environ["NUMBA_DISABLE_JIT"] = "0"

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_THIS_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np

from physics.helion.bosch_hale import prewarm_jit as prewarm_bosch_hale_jit
from physics.helion.bosch_hale import sigma_v_kernel
from physics.helion.engine import integrate_helion_pulse
from physics.helion.engine import prewarm_jit as prewarm_helion_jit
from physics.helion.parameters import HelionParameters
from physics.helion.sweep import ignition_boundary_sweep

NFR_02_WITH_NUMBA_MS    = 20.0
NFR_02_WITHOUT_NUMBA_MS = 100.0

N_SIGMA_REPEATS = 2000
N_PULSE_REPS    = 5
N_SWEEP_REPS    = 3


def _baseline_params() -> HelionParameters:
    return HelionParameters(
        ion_temperature_kev=20.0,
        plasma_density_m3=1.0e21,
        compression_ratio=10.0,
        magnetic_field_t=5.0,
        plasma_volume_m3=1.0,
        pulse_duration_s=1.0e-5,
    )


def _time_sigma_v() -> tuple[float, float]:
    """Return (median_us, min_us) for sigma_v_kernel post-warmup."""
    for _ in range(10):
        sigma_v_kernel(20.0)
    times = np.empty(N_SIGMA_REPEATS)
    for i in range(N_SIGMA_REPEATS):
        t0 = time.perf_counter()
        sigma_v_kernel(20.0)
        times[i] = time.perf_counter() - t0
    return float(np.median(times)) * 1e6, float(np.min(times)) * 1e6


def _time_pulse(params: HelionParameters, n_reps: int) -> tuple[float, int]:
    """Return (median_ms, n_func_evals) for repeated pulse integrations."""
    result = integrate_helion_pulse(params)
    nfev = result.nfev

    times = np.empty(n_reps)
    for i in range(n_reps):
        t0 = time.perf_counter()
        integrate_helion_pulse(params)
        times[i] = time.perf_counter() - t0
    return float(np.median(times)) * 1e3, nfev


def _time_sweep(params: HelionParameters, n_rc: int, n_density: int) -> float:
    """Return median_ms for ignition_boundary_sweep on an n_rc × n_density grid."""
    rc_grid = np.logspace(0, 3, n_rc)
    n_grid  = np.logspace(19, 23, n_density)
    ignition_boundary_sweep(rc_grid, n_grid, params)  # warmup

    times = np.empty(N_SWEEP_REPS)
    for i in range(N_SWEEP_REPS):
        t0 = time.perf_counter()
        ignition_boundary_sweep(rc_grid, n_grid, params)
        times[i] = time.perf_counter() - t0
    return float(np.median(times)) * 1e3


def _pass_fail(val: float, target: float) -> str:
    return "PASS" if val < target else "FAIL"


def main() -> None:
    print("=" * 66)
    print("Helion FRC Performance Benchmark (SPEC NFR-02)")
    print(f"  NFR-02 with Numba:    < {NFR_02_WITH_NUMBA_MS:.0f} ms per pulse integration")
    print(f"  NFR-02 without Numba: < {NFR_02_WITHOUT_NUMBA_MS:.0f} ms per pulse integration")
    print("=" * 66)

    params = _baseline_params()

    print("\n[1/5] Pre-warming Numba JIT (Bosch-Hale + plasma energy kernels)...")
    t_warm = time.perf_counter()
    prewarm_bosch_hale_jit()
    prewarm_helion_jit()
    warmup_ms = (time.perf_counter() - t_warm) * 1e3
    print(f"      JIT compile time: {warmup_ms:.0f} ms  (one-time at app startup)")

    print(f"\n[2/5] sigma_v_kernel single-call timing ({N_SIGMA_REPEATS} reps)...")
    sv_med, sv_min = _time_sigma_v()
    print(f"      Median: {sv_med:.2f} µs     Min: {sv_min:.2f} µs")

    print(f"\n[3/5] Baseline pulse 1×10⁻⁵ s integration (WITH Numba JIT, {N_PULSE_REPS} reps)...")
    base_ms, base_nfev = _time_pulse(params, N_PULSE_REPS)
    print(f"      Median: {base_ms:.2f} ms   [{_pass_fail(base_ms, NFR_02_WITH_NUMBA_MS)} vs {NFR_02_WITH_NUMBA_MS:.0f}ms]")
    print(f"      Solver RHS evals: {base_nfev}")

    high_rc_params = HelionParameters(
        ion_temperature_kev=20.0,
        plasma_density_m3=1.0e21,
        compression_ratio=100.0,
        magnetic_field_t=10.0,
        plasma_volume_m3=1.0,
        pulse_duration_s=1.0e-5,
    )
    print(f"\n[4/5] High-compression pulse (Rc=100, WITH Numba JIT, {N_PULSE_REPS} reps)...")
    hrc_ms, hrc_nfev = _time_pulse(high_rc_params, N_PULSE_REPS)
    print(f"      Median: {hrc_ms:.2f} ms   [{_pass_fail(hrc_ms, NFR_02_WITH_NUMBA_MS)} vs {NFR_02_WITH_NUMBA_MS:.0f}ms]")
    print(f"      Solver RHS evals: {hrc_nfev}")

    sweep_n = 20
    print(f"\n[5/5] Ignition boundary sweep {sweep_n}×{sweep_n} ({N_SWEEP_REPS} reps)...")
    sweep_ms = _time_sweep(params, sweep_n, sweep_n)
    sweep_target = NFR_02_WITH_NUMBA_MS * sweep_n * sweep_n
    print(f"      Median: {sweep_ms:.1f} ms   (grid: {sweep_n}×{sweep_n} = {sweep_n**2} pulses)")
    print(f"      Per-pulse: {sweep_ms / sweep_n**2:.2f} ms   "
          f"[{_pass_fail(sweep_ms / sweep_n**2, NFR_02_WITH_NUMBA_MS)} vs {NFR_02_WITH_NUMBA_MS:.0f}ms]")

    print("\n" + "=" * 66)
    print("SUMMARY")
    print(f"  sigma_v_kernel (JIT):                    {sv_med:.2f} µs")
    print(f"  Baseline pulse 1µs (JIT):                {base_ms:.2f} ms  "
          f"[{_pass_fail(base_ms, NFR_02_WITH_NUMBA_MS)}]")
    print(f"  High-Rc pulse (JIT):                     {hrc_ms:.2f} ms  "
          f"[{_pass_fail(hrc_ms, NFR_02_WITH_NUMBA_MS)}]")
    print(f"  Sweep per-pulse avg (JIT):               {sweep_ms / sweep_n**2:.2f} ms  "
          f"[{_pass_fail(sweep_ms / sweep_n**2, NFR_02_WITH_NUMBA_MS)}]")
    print("=" * 66)

    nfr_pass = (
        base_ms < NFR_02_WITH_NUMBA_MS
        and hrc_ms < NFR_02_WITH_NUMBA_MS
        and (sweep_ms / sweep_n**2) < NFR_02_WITH_NUMBA_MS
    )
    if nfr_pass:
        print("RESULT: NFR-02 SATISFIED — Helion pulse meets the 20ms target.")
    else:
        print("RESULT: NFR-02 NOT MET — one or more Helion operations exceeded target.")
    print("=" * 66)

    sys.exit(0 if nfr_pass else 1)


if __name__ == "__main__":
    main()
