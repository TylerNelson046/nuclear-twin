"""
benchmarks/benchmark_pwr_rhs.py — PWR hot-loop performance benchmark.

Measures pwr_11_state_system (the full 11-state PWR ODE RHS) execution time
with Numba JIT fully enabled, validating SPEC NFR-02 targets (SPEC §5.2):

    Without Numba:  ODE timestep < 100 ms per UI callback
    With Numba:     ODE timestep < 20 ms per UI callback

SPEC §7.4: "Target: full ODE timestep integration in under 20ms per UI
callback cycle on local hardware."  The UI callback window modeled here is
0.5 s of simulated physics time (2 Hz update rate).

Run directly (bypasses conftest.py NUMBA_DISABLE_JIT=1 override):
    cd nuclear-twin
    poetry run python benchmarks/benchmark_pwr_rhs.py
"""

from __future__ import annotations

import os
import subprocess
import sys
import time

# Must be set before any numba import so the decorator compiles (not passthrough).
# The root conftest.py uses setdefault("1") which would disable JIT for pytest;
# this script runs outside pytest so we set it explicitly here.
os.environ["NUMBA_DISABLE_JIT"] = "0"

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_THIS_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np

from physics.pwr.engine import (
    PWRControls,
    PWRModelConfig,
    build_initial_state,
    critical_base_reactivity_pcm,
    integrate_pwr_11state,
    make_params_array,
    prewarm_jit_11state,
    pwr_11_state_system,
)
from physics.pwr.reactivity import ReferenceState
from physics.pwr.thermal_hydraulics import T_COOL_NOM, T_FUEL_NOM, T_IN_NOM

# NFR-02 targets (SPEC §5.2, ARCHITECTURE.md §7)
NFR_02_WITH_NUMBA_MS    = 20.0
NFR_02_WITHOUT_NUMBA_MS = 100.0

N_RHS_REPEATS = 500     # single-call loops (post-warmup)
UI_WINDOW_S   = 0.5     # simulated seconds per UI callback cycle
UI_STEPS      = 51      # t_eval resolution


def _build_equilibrium():
    reference_state = ReferenceState(
        fuel_temperature_ref_k=T_FUEL_NOM,
        coolant_temperature_ref_k=T_COOL_NOM,
    )
    controls = PWRControls(
        rod_reactivity_pcm=0.0,
        boron_ppm=0.0,
        coolant_flow_fraction=1.0,
        inlet_temperature_k=T_IN_NOM,
    )
    y0 = build_initial_state()
    base_pcm = critical_base_reactivity_pcm(y0, controls, reference_state)
    config = PWRModelConfig(base_reactivity_pcm=base_pcm)
    params = make_params_array(controls, reference_state, config)
    return y0, params, reference_state, config


def _build_transient(reference_state, config, rod_pcm=50.0):
    """Return params for a +50 pcm rod withdrawal transient."""
    controls = PWRControls(
        rod_reactivity_pcm=rod_pcm,
        boron_ppm=0.0,
        coolant_flow_fraction=1.0,
        inlet_temperature_k=T_IN_NOM,
    )
    return make_params_array(controls, reference_state, config)


def _time_rhs(y0, params):
    """Return (median_us, min_us) for pwr_11_state_system post-warmup."""
    # Discard first few calls (cache load / branch prediction warmup)
    for _ in range(5):
        pwr_11_state_system(0.0, y0, params)

    times = np.empty(N_RHS_REPEATS)
    for i in range(N_RHS_REPEATS):
        t0 = time.perf_counter()
        pwr_11_state_system(0.0, y0, params)
        times[i] = time.perf_counter() - t0
    return float(np.median(times)) * 1e6, float(np.min(times)) * 1e6


def _time_integration(y0, params, t_end=UI_WINDOW_S, n_reps=5):
    """Return (median_ms, nfev) for repeated ODE integrations."""
    t_eval = np.linspace(0.0, t_end, UI_STEPS)

    # Warmup: first integration triggers any residual JIT specialisation
    result = integrate_pwr_11state(y0, (0.0, t_end), params, t_eval=t_eval)
    nfev = result.nfev

    times = np.empty(n_reps)
    for i in range(n_reps):
        t0 = time.perf_counter()
        integrate_pwr_11state(y0, (0.0, t_end), params, t_eval=t_eval)
        times[i] = time.perf_counter() - t0

    return float(np.median(times)) * 1e3, nfev


def _time_integration_no_jit(t_end=UI_WINDOW_S):
    """Measure integration time in a subprocess with JIT fully disabled."""
    helper = os.path.join(_THIS_DIR, "_nojit_helper.py")
    _write_helper(helper, t_end)
    proc = subprocess.run(
        [sys.executable, helper],
        capture_output=True, text=True, timeout=120,
    )
    os.remove(helper)
    try:
        return float(proc.stdout.strip().split()[-1])
    except Exception:
        return float("nan")


def _write_helper(path: str, t_end: float) -> None:
    code = f"""\
import os, sys, time, numpy as np
os.environ["NUMBA_DISABLE_JIT"] = "1"
sys.path.insert(0, {repr(_PROJECT_ROOT)})
from physics.pwr.engine import (
    PWRControls, PWRModelConfig, build_initial_state,
    critical_base_reactivity_pcm, integrate_pwr_11state, make_params_array,
)
from physics.pwr.reactivity import ReferenceState
from physics.pwr.thermal_hydraulics import T_COOL_NOM, T_FUEL_NOM, T_IN_NOM

ref = ReferenceState(fuel_temperature_ref_k=T_FUEL_NOM,
                     coolant_temperature_ref_k=T_COOL_NOM)
ctrl = PWRControls(rod_reactivity_pcm=0.0, boron_ppm=0.0,
                   coolant_flow_fraction=1.0, inlet_temperature_k=T_IN_NOM)
y0 = build_initial_state()
base_pcm = critical_base_reactivity_pcm(y0, ctrl, ref)
cfg = PWRModelConfig(base_reactivity_pcm=base_pcm)
params = make_params_array(ctrl, ref, cfg)
t_eval = np.linspace(0.0, {t_end}, {UI_STEPS})

# Warmup
integrate_pwr_11state(y0, (0.0, {t_end}), params, t_eval=t_eval)

times = []
for _ in range(5):
    t0 = time.perf_counter()
    integrate_pwr_11state(y0, (0.0, {t_end}), params, t_eval=t_eval)
    times.append(time.perf_counter() - t0)
print(f"{{float(np.median(times)) * 1e3:.2f}}")
"""
    with open(path, "w") as f:
        f.write(code)


def _pass_fail(val, target):
    return "PASS" if val < target else "FAIL"


def main() -> None:
    print("=" * 66)
    print("PWR Hot-Loop Performance Benchmark (SPEC NFR-02)")
    print(f"  NFR-02 with Numba:    < {NFR_02_WITH_NUMBA_MS:.0f} ms per {UI_WINDOW_S:.1f}s UI callback")
    print(f"  NFR-02 without Numba: < {NFR_02_WITHOUT_NUMBA_MS:.0f} ms per {UI_WINDOW_S:.1f}s UI callback")
    print("=" * 66)

    # 1. Build equilibrium -----------------------------------------------
    print("\n[1/6] Building critical 11-state equilibrium...")
    y0, eq_params, ref_state, config = _build_equilibrium()

    # 2. Pre-warm JIT -----------------------------------------------------
    print("[2/6] Pre-warming Numba JIT (first-call compilation)...")
    t_warm = time.perf_counter()
    prewarm_jit_11state()
    warmup_ms = (time.perf_counter() - t_warm) * 1e3
    print(f"      JIT compile time: {warmup_ms:.0f} ms  (one-time at app startup)")

    # 3. RHS single-call timing ------------------------------------------
    print(f"\n[3/6] pwr_11_state_system single-call timing ({N_RHS_REPEATS} reps)...")
    rhs_med, rhs_min = _time_rhs(y0, eq_params)
    print(f"      Median: {rhs_med:.2f} µs     Min: {rhs_min:.2f} µs")

    # 4. Steady-state integration — JIT ----------------------------------
    print(f"\n[4/6] Steady-state {UI_WINDOW_S:.1f}s integration (WITH Numba JIT, 5 reps)...")
    ss_ms_jit, ss_nfev = _time_integration(y0, eq_params)
    rhs_cost_ms = ss_nfev * rhs_med / 1e3
    print(f"      Median: {ss_ms_jit:.2f} ms   [{_pass_fail(ss_ms_jit, NFR_02_WITH_NUMBA_MS)} vs {NFR_02_WITH_NUMBA_MS:.0f}ms]")
    print(f"      Solver RHS evals: {ss_nfev}  (RHS cost: {rhs_cost_ms:.3f} ms, "
          f"solver overhead: {ss_ms_jit - rhs_cost_ms:.2f} ms)")

    # 5. Transient integration — JIT -------------------------------------
    print(f"\n[5/6] +50 pcm rod transient {UI_WINDOW_S:.1f}s integration (WITH Numba JIT, 5 reps)...")
    tr_params = _build_transient(ref_state, config)
    tr_ms_jit, tr_nfev = _time_integration(y0, tr_params)
    tr_rhs_cost = tr_nfev * rhs_med / 1e3
    print(f"      Median: {tr_ms_jit:.2f} ms   [{_pass_fail(tr_ms_jit, NFR_02_WITH_NUMBA_MS)} vs {NFR_02_WITH_NUMBA_MS:.0f}ms]")
    print(f"      Solver RHS evals: {tr_nfev}  (RHS cost: {tr_rhs_cost:.3f} ms)")

    # 6. No-JIT baseline — subprocess ------------------------------------
    print(f"\n[6/6] Steady-state {UI_WINDOW_S:.1f}s integration (WITHOUT Numba JIT, subprocess)...")
    ss_ms_nojit = _time_integration_no_jit(UI_WINDOW_S)
    speedup = ss_ms_nojit / max(ss_ms_jit, 0.001)
    print(f"      Median: {ss_ms_nojit:.2f} ms   [{_pass_fail(ss_ms_nojit, NFR_02_WITHOUT_NUMBA_MS)} vs {NFR_02_WITHOUT_NUMBA_MS:.0f}ms]")
    print(f"      Speedup (JIT vs no-JIT): {speedup:.1f}×")
    print()
    print("      Note: for steady-state (few solver steps), scipy Radau Python")
    print("      overhead dominates. JIT benefit scales with transient severity.")

    # Summary ------------------------------------------------------------
    print("\n" + "=" * 66)
    print("SUMMARY")
    print(f"  Single RHS call (JIT):                   {rhs_med:.2f} µs")
    print(f"  Steady-state integration {UI_WINDOW_S:.1f}s (JIT):   {ss_ms_jit:.2f} ms  "
          f"[{_pass_fail(ss_ms_jit, NFR_02_WITH_NUMBA_MS)}]")
    print(f"  Transient +50pcm integration {UI_WINDOW_S:.1f}s (JIT): {tr_ms_jit:.2f} ms  "
          f"[{_pass_fail(tr_ms_jit, NFR_02_WITH_NUMBA_MS)}]")
    print(f"  Steady-state integration {UI_WINDOW_S:.1f}s (no JIT): {ss_ms_nojit:.2f} ms  "
          f"[{_pass_fail(ss_ms_nojit, NFR_02_WITHOUT_NUMBA_MS)}]")
    print("=" * 66)

    nfr_pass = ss_ms_jit < NFR_02_WITH_NUMBA_MS and tr_ms_jit < NFR_02_WITH_NUMBA_MS
    if nfr_pass:
        print("RESULT: NFR-02 SATISFIED — Numba acceleration meets the 20ms target.")
    else:
        print("RESULT: NFR-02 NOT MET — one or more integration windows exceeded target.")
    print("=" * 66)

    sys.exit(0 if nfr_pass else 1)


if __name__ == "__main__":
    main()
