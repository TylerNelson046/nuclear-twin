"""
scripts/benchmark_pwr_system.py — Full-system PWR performance benchmark.

Evaluates the complete 11-state PWR ODE solver against SPEC NFR-02 targets
(SPEC §5.2, ARCHITECTURE.md §7):

    Without Numba:  ODE solver shall complete each UI-callback timestep < 100 ms
    With Numba:     ODE solver shall complete each UI-callback timestep < 20 ms

One UI-callback timestep = 0.5 s of simulated physics (2 Hz update rate).

Benchmark sections
──────────────────
  A. Full 100-second transient   — total wall time, real-time factor, peak memory
  B. Chunked 0.5-s UI windows    — per-callback distribution, NFR-02 pass/fail
  C. No-Numba baseline           — subprocess measurement, speedup factor
  D. CPU bottleneck profile      — cProfile top-10 functions (Python + Numba overhead)

Run (from nuclear-twin/):
    poetry run python scripts/benchmark_pwr_system.py
"""

from __future__ import annotations

import cProfile
import io
import os
import pstats
import subprocess
import sys
import tempfile
import time
import tracemalloc

# Force Numba JIT on for this process (root conftest.py sets NUMBA_DISABLE_JIT=1
# for pytest; this script runs outside pytest).
os.environ["NUMBA_DISABLE_JIT"] = "0"

_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPTS_DIR)
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

# ─── NFR-02 constants (SPEC §5.2, ARCHITECTURE.md §7) ─────────────────────────
NFR_NUMBA_MS: float = 20.0      # ms per 0.5-s UI callback (with Numba)
NFR_NOJIT_MS: float = 100.0     # ms per 0.5-s UI callback (without Numba)

# ─── Benchmark configuration ───────────────────────────────────────────────────
TRANSIENT_DURATION_S: float = 100.0    # full transient window for Section A
ROD_INSERTION_PCM:    float = 50.0     # +50 pcm step insertion at t=0

UI_WINDOW_S:   float = 0.5             # simulated seconds per UI callback cycle
N_UI_WINDOWS:  int   = 20             # number of 0.5-s windows for Section B
SECTION_B_REPS: int  = 5              # timing repetitions per window (median)
T_EVAL_DENSITY: int  = 51             # output points per 0.5-s window

# ─── Layout widths ─────────────────────────────────────────────────────────────
_W = 68


def _banner(title: str) -> None:
    print(f"\n{'─' * _W}")
    print(f" {title}")
    print(f"{'─' * _W}")


def _pf(value: float, target: float, *, unit: str = "ms", lower_is_better: bool = True) -> str:
    ok = value < target if lower_is_better else value > target
    tag = "PASS" if ok else "FAIL"
    margin = target - value if lower_is_better else value - target
    return f"{value:.2f} {unit}  [{tag} vs {target:.0f}{unit}, margin {abs(margin):.2f}{unit}]"


# ─── Setup helpers ─────────────────────────────────────────────────────────────

def _build_equilibrium() -> tuple[np.ndarray, np.ndarray, ReferenceState, PWRModelConfig]:
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
    params_eq = make_params_array(controls, reference_state, config)
    return y0, params_eq, reference_state, config


def _transient_params(ref: ReferenceState, cfg: PWRModelConfig) -> np.ndarray:
    controls = PWRControls(
        rod_reactivity_pcm=ROD_INSERTION_PCM,
        boron_ppm=0.0,
        coolant_flow_fraction=1.0,
        inlet_temperature_k=T_IN_NOM,
    )
    return make_params_array(controls, ref, cfg)


# ─── Section A: Full 100-second transient ─────────────────────────────────────

def section_a_full_transient(
    y0: np.ndarray,
    params: np.ndarray,
) -> dict:
    """Time and memory-profile the full TRANSIENT_DURATION_S integration."""
    t_eval = np.linspace(0.0, TRANSIENT_DURATION_S, int(TRANSIENT_DURATION_S * 10 + 1))

    # -- Memory baseline before integration
    tracemalloc.start()
    tracemalloc.clear_traces()

    t_start = time.perf_counter()
    result = integrate_pwr_11state(y0, (0.0, TRANSIENT_DURATION_S), params, t_eval=t_eval)
    wall_ms = (time.perf_counter() - t_start) * 1e3

    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    real_time_factor = TRANSIENT_DURATION_S / (wall_ms * 1e-3)

    return {
        "wall_ms": wall_ms,
        "nfev": result.nfev,
        "njac": result.njev,
        "success": result.success,
        "peak_kib": peak_bytes / 1024.0,
        "real_time_factor": real_time_factor,
        "mean_rhs_us": (wall_ms / result.nfev) * 1e3 if result.nfev > 0 else float("nan"),
        "n_output_points": len(result.t),
        "final_state": result.y[:, -1],
    }


# ─── Section B: Chunked UI-callback windows ────────────────────────────────────

def section_b_chunked_windows(
    y0: np.ndarray,
    params: np.ndarray,
) -> dict:
    """Time N_UI_WINDOWS consecutive 0.5-s integrations; report per-callback distribution."""
    t_eval_chunk = np.linspace(0.0, UI_WINDOW_S, T_EVAL_DENSITY)

    # First, advance through each window sequentially to get realistic solver state
    # history (precursor warmup, solver step-size adaptation).
    window_medians_ms = []
    current_state = y0.copy()

    for _ in range(N_UI_WINDOWS):
        # Collect SECTION_B_REPS timings for this window, re-running from same state
        times_ms = []
        for _ in range(SECTION_B_REPS):
            t0 = time.perf_counter()
            res = integrate_pwr_11state(
                current_state,
                (0.0, UI_WINDOW_S),
                params,
                t_eval=t_eval_chunk,
            )
            times_ms.append((time.perf_counter() - t0) * 1e3)

        window_medians_ms.append(float(np.median(times_ms)))
        # Advance state to end of this window for the next iteration
        current_state = res.y[:, -1].copy()  # noqa: F821  (last rep's result)

    arr = np.array(window_medians_ms)
    return {
        "median_ms": float(np.median(arr)),
        "mean_ms":   float(np.mean(arr)),
        "max_ms":    float(np.max(arr)),
        "min_ms":    float(np.min(arr)),
        "p95_ms":    float(np.percentile(arr, 95)),
        "all_ms":    arr,
        "n_windows": N_UI_WINDOWS,
    }


# ─── Section C: No-Numba baseline (subprocess) ────────────────────────────────

def section_c_nojit_baseline() -> float:
    """Return median ms for one 0.5-s integration with Numba JIT disabled."""
    nojit_code = f"""\
import os, sys, time, numpy as np
os.environ["NUMBA_DISABLE_JIT"] = "1"
sys.path.insert(0, {repr(_PROJECT_ROOT)})
from physics.pwr.engine import (
    PWRControls, PWRModelConfig, build_initial_state,
    critical_base_reactivity_pcm, integrate_pwr_11state, make_params_array,
)
from physics.pwr.reactivity import ReferenceState
from physics.pwr.thermal_hydraulics import T_COOL_NOM, T_FUEL_NOM, T_IN_NOM

ref  = ReferenceState(fuel_temperature_ref_k=T_FUEL_NOM,
                      coolant_temperature_ref_k=T_COOL_NOM)
ctrl = PWRControls(rod_reactivity_pcm={ROD_INSERTION_PCM},
                   boron_ppm=0.0, coolant_flow_fraction=1.0,
                   inlet_temperature_k=T_IN_NOM)
y0  = build_initial_state()
base = critical_base_reactivity_pcm(y0, ctrl, ref)
cfg  = PWRModelConfig(base_reactivity_pcm=base)
params = make_params_array(ctrl, ref, cfg)
t_eval = np.linspace(0.0, {UI_WINDOW_S}, {T_EVAL_DENSITY})

# Warmup pass
integrate_pwr_11state(y0, (0.0, {UI_WINDOW_S}), params, t_eval=t_eval)

times = []
for _ in range(5):
    t0 = time.perf_counter()
    integrate_pwr_11state(y0, (0.0, {UI_WINDOW_S}), params, t_eval=t_eval)
    times.append(time.perf_counter() - t0)
print(f"{{float(np.median(times)) * 1e3:.4f}}")
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as fh:
        fh.write(nojit_code)
        tmp_path = fh.name

    try:
        proc = subprocess.run(
            [sys.executable, tmp_path],
            capture_output=True, text=True, timeout=120,
        )
        return float(proc.stdout.strip().split()[-1])
    except Exception:
        return float("nan")
    finally:
        os.unlink(tmp_path)


# ─── Section D: cProfile bottleneck analysis ──────────────────────────────────

def section_d_profile(y0: np.ndarray, params: np.ndarray) -> str:
    """Run cProfile over a 10-s transient and return formatted top-10 report."""
    t_eval = np.linspace(0.0, 10.0, 201)

    profiler = cProfile.Profile()
    profiler.enable()
    integrate_pwr_11state(y0, (0.0, 10.0), params, t_eval=t_eval)
    profiler.disable()

    buf = io.StringIO()
    stats = pstats.Stats(profiler, stream=buf)
    stats.strip_dirs()
    stats.sort_stats("cumulative")
    stats.print_stats(15)
    return buf.getvalue()


def _summarise_profile(raw: str) -> list[str]:
    """Extract the table rows from cProfile output, skip header lines."""
    lines = raw.splitlines()
    in_table = False
    rows = []
    for line in lines:
        if "cumtime" in line:
            in_table = True
            rows.append(line)
            continue
        if in_table and line.strip():
            rows.append(line)
    return rows[:16]  # header + 15 rows


# ─── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * _W)
    print(" Nuclear PWR 11-State System — Full Performance Benchmark")
    print(f" SPEC NFR-02 (SPEC §5.2 · ARCHITECTURE.md §7)")
    print("=" * _W)
    print(f"  NFR-02 with Numba    : < {NFR_NUMBA_MS:.0f} ms per {UI_WINDOW_S:.1f} s simulated UI callback")
    print(f"  NFR-02 without Numba : < {NFR_NOJIT_MS:.0f} ms per {UI_WINDOW_S:.1f} s simulated UI callback")
    print(f"  Transient scenario   : +{ROD_INSERTION_PCM:.0f} pcm step rod insertion (Doppler self-regulation)")
    print("=" * _W)

    # ── Setup ──────────────────────────────────────────────────────────────────
    print("\n[SETUP 1/2] Building critical 11-state equilibrium at nominal operating point...")
    y0, eq_params, ref_state, config = _build_equilibrium()
    tr_params = _transient_params(ref_state, config)
    print(f"            State vector: {y0.shape[0]} states  |  "
          f"n₀ = {y0[0]:.4f}  T_fuel = {y0[7]:.1f} K  T_cool = {y0[8]:.1f} K")

    print("[SETUP 2/2] Pre-warming Numba JIT (first-call compilation)...")
    t_warm = time.perf_counter()
    prewarm_jit_11state()
    warmup_ms = (time.perf_counter() - t_warm) * 1e3
    print(f"            JIT compile time: {warmup_ms:.0f} ms  (one-time; excluded from NFR-02 targets)")

    # ── Section A ──────────────────────────────────────────────────────────────
    _banner(f"SECTION A: Full {TRANSIENT_DURATION_S:.0f}-Second Transient  (WITH Numba)")
    print(f"  Integrating {TRANSIENT_DURATION_S:.0f} s of simulated physics as a single solve_ivp call...")
    a = section_a_full_transient(y0, tr_params)

    if not a["success"]:
        print("  WARNING: ODE solver reported failure for full transient!")

    print(f"\n  Total wall time          : {a['wall_ms']:.2f} ms  ({a['wall_ms']/1e3:.4f} s)")
    print(f"  Simulated physics time   : {TRANSIENT_DURATION_S:.0f} s")
    print(f"  Real-time factor         : {a['real_time_factor']:,.0f}×  "
          f"(simulator is {a['real_time_factor']:,.0f}× faster than wall clock)")
    print(f"  Solver RHS evaluations   : {a['nfev']}")
    print(f"  Jacobian evaluations     : {a['njac']}")
    print(f"  Mean wall time / RHS call: {a['mean_rhs_us']:.3f} µs")
    print(f"  Peak heap Δ (tracemalloc): {a['peak_kib']:.1f} KiB")
    print(f"  Output time-series points: {a['n_output_points']}")

    implied_callback_ms = a["wall_ms"] / (TRANSIENT_DURATION_S / UI_WINDOW_S)
    print(f"\n  Implied cost/UI callback : {implied_callback_ms:.3f} ms  "
          f"(wall time ÷ {int(TRANSIENT_DURATION_S/UI_WINDOW_S)} callbacks)")
    print(f"  vs NFR-02 target         : {_pf(implied_callback_ms, NFR_NUMBA_MS)}")

    # ── Section B ──────────────────────────────────────────────────────────────
    _banner(f"SECTION B: Chunked {UI_WINDOW_S:.1f}-s UI-Callback Windows  (WITH Numba, {N_UI_WINDOWS} windows)")
    print(f"  Timing {N_UI_WINDOWS} consecutive {UI_WINDOW_S:.1f}-s windows (each repeated {SECTION_B_REPS}×, median taken)...")
    print(f"  This is the most direct measurement of NFR-02 compliance.\n")
    b = section_b_chunked_windows(y0, tr_params)

    print(f"  Median per callback : {_pf(b['median_ms'], NFR_NUMBA_MS)}")
    print(f"  Mean per callback   : {_pf(b['mean_ms'],   NFR_NUMBA_MS)}")
    print(f"  95th percentile     : {_pf(b['p95_ms'],    NFR_NUMBA_MS)}")
    print(f"  Worst callback      : {_pf(b['max_ms'],    NFR_NUMBA_MS)}")
    print(f"  Best callback       : {b['min_ms']:.2f} ms")

    n_pass = int(np.sum(b["all_ms"] < NFR_NUMBA_MS))
    print(f"\n  Windows passing NFR-02 : {n_pass}/{N_UI_WINDOWS}  "
          f"({'ALL PASS' if n_pass == N_UI_WINDOWS else f'{N_UI_WINDOWS - n_pass} FAIL'})")

    # ── Section C ──────────────────────────────────────────────────────────────
    _banner("SECTION C: No-Numba Baseline  (subprocess, JIT disabled)")
    print(f"  Running 5-rep measurement in subprocess with NUMBA_DISABLE_JIT=1...")
    print(f"  (This isolates pure Python + SciPy overhead without any compiled kernel.)\n")
    nojit_ms = section_c_nojit_baseline()
    speedup = nojit_ms / max(b["median_ms"], 0.001)

    print(f"  Median per callback (no JIT): {_pf(nojit_ms, NFR_NOJIT_MS)}")
    print(f"  Numba speedup               : {speedup:.1f}×  "
          f"(JIT median {b['median_ms']:.2f} ms vs no-JIT {nojit_ms:.2f} ms)")

    # ── Section D ──────────────────────────────────────────────────────────────
    _banner("SECTION D: CPU Bottleneck Profile  (cProfile, 10-s transient, WITH Numba)")
    print("  Top functions by cumulative time — identifies where CPU cycles are spent.\n")
    profile_raw = section_d_profile(y0, tr_params)
    profile_rows = _summarise_profile(profile_raw)
    for row in profile_rows:
        print(f"  {row}")

    # ── NFR-02 Verdict ─────────────────────────────────────────────────────────
    print("\n" + "=" * _W)
    print(" FINAL VERDICT — NFR-02 COMPLIANCE SUMMARY")
    print("=" * _W)

    jit_pass   = b["median_ms"] < NFR_NUMBA_MS
    nojit_pass = nojit_ms < NFR_NOJIT_MS
    all_pass   = n_pass == N_UI_WINDOWS
    overall    = jit_pass and nojit_pass

    def tick(ok: bool) -> str:
        return "PASS" if ok else "FAIL"

    print(f"  [{tick(jit_pass)}]  With Numba  — median callback : "
          f"{b['median_ms']:.2f} ms  (target < {NFR_NUMBA_MS:.0f} ms)")
    print(f"  [{tick(all_pass)}]  With Numba  — all {N_UI_WINDOWS} windows    : "
          f"{n_pass}/{N_UI_WINDOWS} passed")
    print(f"  [{tick(nojit_pass)}]  Without Numba — median callback: "
          f"{nojit_ms:.2f} ms  (target < {NFR_NOJIT_MS:.0f} ms)")
    print(f"  [INFO]  Real-time factor (100 s transient) : {a['real_time_factor']:,.0f}×")
    print(f"  [INFO]  Numba speedup                      : {speedup:.1f}×")
    print(f"  [INFO]  JIT compile time (one-time)        : {warmup_ms:.0f} ms")

    print()
    if overall:
        print("  RESULT: NFR-02 SATISFIED — Numba-accelerated solver meets all targets.")
    else:
        print("  RESULT: NFR-02 NOT MET — see FAIL rows above for bottleneck targets.")
        if not jit_pass:
            print(f"          Bottleneck: Numba-path median {b['median_ms']:.2f} ms exceeds {NFR_NUMBA_MS:.0f} ms.")
            print("          ACTION: Profile Section D output and target highest cumtime functions.")
        if not nojit_pass:
            print(f"          Bottleneck: No-JIT path {nojit_ms:.2f} ms exceeds {NFR_NOJIT_MS:.0f} ms.")
            print("          ACTION: Review SciPy Radau Python overhead; consider pre-factored Jacobian.")

    print("=" * _W)
    sys.exit(0 if overall else 1)


if __name__ == "__main__":
    main()
