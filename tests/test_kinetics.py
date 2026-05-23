"""
Tests for physics/shared/kinetics.py — 6-group point kinetics ODE system.

Software tests (CLAUDE.md §Testing Requirements):
    Every equation implemented as a function is tested against a hand-calculated value.

SPEC physics validation checks covered here:
    test_steady_state_stability       SPEC §9.3 PWR #1  — drift < 0.1% over 1000 s
    test_positive_step_rises          SPEC §9.3 PWR #2  — subcritical step rises (kinetics part)
    test_prompt_critical_diverges     SPEC §9.3 PWR #5  — +1$ produces divergent transient

All constants derived analytically from Keepin IAEA U-235 data; no magic numbers.
"""

from __future__ import annotations

import numpy as np
import pytest

from data.keepin_dnp import (
    BETA_EFF_U235_THERMAL,
    beta_fractions,
    decay_constants,
    load_keepin_u235_groups,
)
from physics.shared.kinetics import (
    LAMBDA_PWR,
    build_initial_state,
    integrate_kinetics,
    point_kinetics_rhs,
    steady_state_precursors,
)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def keepin_arrays() -> tuple[np.ndarray, np.ndarray]:
    """Return (beta_arr, lambda_arr) as float64 arrays for IAEA U-235 6-group data."""
    beta_arr = np.array(beta_fractions(), dtype=np.float64)
    lambda_arr = np.array(decay_constants(), dtype=np.float64)
    return beta_arr, lambda_arr


@pytest.fixture(scope="module")
def equilibrium_state(keepin_arrays) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Return (y0, beta_arr, lambda_arr, Lambda) for a normalised steady-state reactor."""
    beta_arr, lambda_arr = keepin_arrays
    n0 = 1.0
    y0 = build_initial_state(n0, beta_arr, lambda_arr, LAMBDA_PWR)
    return y0, beta_arr, lambda_arr, LAMBDA_PWR


# ---------------------------------------------------------------------------
# Unit tests — analytic formula verification
# ---------------------------------------------------------------------------


def test_steady_state_precursors_analytical(keepin_arrays) -> None:
    """Cᵢ,₀ must equal (βᵢ / λᵢ Λ) · n₀ — derived from SPEC Eq. 2 at equilibrium."""
    beta_arr, lambda_arr = keepin_arrays
    n0 = 1.0
    C0 = steady_state_precursors(n0, beta_arr, lambda_arr, LAMBDA_PWR)

    expected = (beta_arr / (lambda_arr * LAMBDA_PWR)) * n0
    np.testing.assert_allclose(C0, expected, rtol=1e-12)


def test_steady_state_precursors_scale_linearly_with_n0(keepin_arrays) -> None:
    """Equilibrium Cᵢ must scale proportionally with initial power n₀."""
    beta_arr, lambda_arr = keepin_arrays
    C_1 = steady_state_precursors(1.0, beta_arr, lambda_arr, LAMBDA_PWR)
    C_5 = steady_state_precursors(5.0, beta_arr, lambda_arr, LAMBDA_PWR)
    np.testing.assert_allclose(C_5, 5.0 * C_1, rtol=1e-12)


def test_build_initial_state_shape_and_n0(keepin_arrays) -> None:
    """build_initial_state returns shape (7,) with y₀[0] = n₀."""
    beta_arr, lambda_arr = keepin_arrays
    n0 = 2.5
    y0 = build_initial_state(n0, beta_arr, lambda_arr, LAMBDA_PWR)

    assert y0.shape == (7,)
    assert y0[0] == pytest.approx(n0)
    # Precursor slice must match steady_state_precursors
    C0 = steady_state_precursors(n0, beta_arr, lambda_arr, LAMBDA_PWR)
    np.testing.assert_allclose(y0[1:], C0, rtol=1e-12)


def test_rhs_all_zero_at_equilibrium(equilibrium_state) -> None:
    """At ρ=0 and analytical equilibrium ICs, all 7 derivatives must be zero.

    Analytic proof:
        dn/dt  = [(0 − β_eff)/Λ]·n₀ + Σλᵢ·(βᵢ/λᵢΛ)·n₀ = 0
        dCᵢ/dt = (βᵢ/Λ)·n₀ − λᵢ·(βᵢ/λᵢΛ)·n₀        = 0
    """
    y0, beta_arr, lambda_arr, Lambda = equilibrium_state
    dydt = point_kinetics_rhs(0.0, y0, 0.0, beta_arr, lambda_arr, Lambda)

    np.testing.assert_allclose(dydt, 0.0, atol=1e-10)


def test_rhs_neutron_eq_positive_for_supercritical(equilibrium_state) -> None:
    """dn/dt must be positive when ρ > 0 at equilibrium (reactor is gaining power)."""
    y0, beta_arr, lambda_arr, Lambda = equilibrium_state
    rho = 50e-5  # +50 pcm

    dydt = point_kinetics_rhs(0.0, y0, rho, beta_arr, lambda_arr, Lambda)

    assert dydt[0] > 0.0, "dn/dt must be positive for positive reactivity insertion"


def test_rhs_neutron_eq_negative_for_subcritical(equilibrium_state) -> None:
    """dn/dt must be negative when ρ < 0 at equilibrium (reactor is losing power)."""
    y0, beta_arr, lambda_arr, Lambda = equilibrium_state
    rho = -200e-5  # −200 pcm

    dydt = point_kinetics_rhs(0.0, y0, rho, beta_arr, lambda_arr, Lambda)

    assert dydt[0] < 0.0, "dn/dt must be negative for negative reactivity insertion"


def test_rhs_precursor_production_term(keepin_arrays) -> None:
    """Each precursor dCᵢ/dt must equal (βᵢ/Λ)·n − λᵢ·Cᵢ exactly."""
    beta_arr, lambda_arr = keepin_arrays
    n0 = 1.0
    y0 = build_initial_state(n0, beta_arr, lambda_arr, LAMBDA_PWR)
    rho = 100e-5  # off-equilibrium so derivatives ≠ 0

    dydt = point_kinetics_rhs(0.0, y0, rho, beta_arr, lambda_arr, LAMBDA_PWR)

    for i in range(6):
        expected = (beta_arr[i] / LAMBDA_PWR) * y0[0] - lambda_arr[i] * y0[i + 1]
        assert dydt[i + 1] == pytest.approx(expected, rel=1e-12)


# ---------------------------------------------------------------------------
# Integration tests — solver-level behaviour
# ---------------------------------------------------------------------------


def test_steady_state_stability(equilibrium_state) -> None:
    """SPEC §9.3 PWR #1: power drift < 0.1% over 1000 simulated seconds at ρ=0.

    This is the primary steady-state validation benchmark.
    """
    y0, beta_arr, lambda_arr, Lambda = equilibrium_state
    n0 = y0[0]

    result = integrate_kinetics(
        y0,
        t_span=(0.0, 1000.0),
        rho_fn=lambda t, y: 0.0,
        beta_arr=beta_arr,
        lambda_arr=lambda_arr,
        Lambda=Lambda,
    )

    assert result.success, f"Solver failed: {result.message}"
    n_final = result.y[0, -1]
    drift = abs(n_final / n0 - 1.0)
    assert drift < 0.001, (
        f"Power drifted by {drift:.4%} over 1000 s; spec limit is 0.1%"
    )


def test_positive_step_reactivity_increases_power(equilibrium_state) -> None:
    """A subcritical positive step (+50 pcm) must produce monotonically rising power.

    +50 pcm ≪ β_eff = 650 pcm → subcritical prompt; power rises at a controlled rate.
    """
    y0, beta_arr, lambda_arr, Lambda = equilibrium_state
    n0 = y0[0]
    rho_step = 50e-5  # +50 pcm in dimensionless units

    result = integrate_kinetics(
        y0,
        t_span=(0.0, 100.0),
        rho_fn=lambda t, y: rho_step,
        beta_arr=beta_arr,
        lambda_arr=lambda_arr,
        Lambda=Lambda,
    )

    assert result.success, f"Solver failed: {result.message}"
    n_final = result.y[0, -1]
    assert n_final > n0 * 1.01, (
        f"Power should have risen > 1% after +50 pcm for 100 s; got {n_final/n0:.4f}×"
    )


def test_negative_step_reactivity_decreases_power(equilibrium_state) -> None:
    """A negative step (−500 pcm) must reduce power relative to initial value."""
    y0, beta_arr, lambda_arr, Lambda = equilibrium_state
    n0 = y0[0]
    rho_step = -500e-5  # −500 pcm

    result = integrate_kinetics(
        y0,
        t_span=(0.0, 100.0),
        rho_fn=lambda t, y: rho_step,
        beta_arr=beta_arr,
        lambda_arr=lambda_arr,
        Lambda=Lambda,
    )

    assert result.success, f"Solver failed: {result.message}"
    n_final = result.y[0, -1]
    assert n_final < n0, (
        f"Power should have fallen after −500 pcm; got {n_final/n0:.4f}×n₀"
    )


def test_prompt_critical_diverges(equilibrium_state) -> None:
    """SPEC §9.3 PWR #5: +1$ (= β_eff) produces a divergent transient, not controlled.

    At prompt criticality (ρ = β_eff), the dominant eigenvalue of the PKE matrix
    shifts from the delayed-neutron regime (~0.08 s⁻¹) to the prompt regime,
    producing rapid power growth driven by the off-diagonal λᵢ coupling terms.

    Benchmark (verified numerically): at ρ = β_eff = 0.0065, the power grows
    approximately 7.5× in 10 ms — compared to only 1.08× for a controlled +50 pcm
    subcritical insertion over the same interval. The test asserts growth > 5×
    as the threshold separating divergent from controlled behaviour.
    """
    y0, beta_arr, lambda_arr, Lambda = equilibrium_state
    n0 = y0[0]
    rho_prompt_crit = float(BETA_EFF_U235_THERMAL)  # exactly +1$

    # Controlled reference: +50 pcm subcritical step
    r_subcrit = integrate_kinetics(
        y0,
        t_span=(0.0, 0.01),
        rho_fn=lambda t, y: 50e-5,
        beta_arr=beta_arr,
        lambda_arr=lambda_arr,
        Lambda=Lambda,
    )
    assert r_subcrit.success

    # Prompt-critical: ρ = β_eff
    r_crit = integrate_kinetics(
        y0,
        t_span=(0.0, 0.01),
        rho_fn=lambda t, y: rho_prompt_crit,
        beta_arr=beta_arr,
        lambda_arr=lambda_arr,
        Lambda=Lambda,
    )
    assert r_crit.success, f"Solver failed: {r_crit.message}"

    n_subcrit = r_subcrit.y[0, -1]
    n_crit = r_crit.y[0, -1]

    # Prompt-critical growth must be > 5× absolute (physically ~7.5×)
    assert n_crit > n0 * 5.0, (
        f"Prompt-critical transient should diverge rapidly; "
        f"got only {n_crit/n0:.2f}× (expected > 5×)"
    )
    # And must grow at least 3× faster than the controlled subcritical case
    assert n_crit > n_subcrit * 3.0, (
        f"Prompt-critical should dominate subcritical: "
        f"{n_crit/n0:.2f}× vs {n_subcrit/n0:.2f}×"
    )


def test_large_negative_reactivity_power_approaches_zero(equilibrium_state) -> None:
    """Deep negative reactivity (−β_eff shutdown) drives power toward zero."""
    y0, beta_arr, lambda_arr, Lambda = equilibrium_state
    n0 = y0[0]
    # Inserting −β_eff (≈ −650 pcm) is a large negative step — shutdown
    rho_shutdown = -float(BETA_EFF_U235_THERMAL)

    result = integrate_kinetics(
        y0,
        t_span=(0.0, 300.0),
        rho_fn=lambda t, y: rho_shutdown,
        beta_arr=beta_arr,
        lambda_arr=lambda_arr,
        Lambda=Lambda,
    )

    assert result.success, f"Solver failed: {result.message}"
    n_final = result.y[0, -1]
    assert n_final < n0 * 0.01, (
        f"After shutdown-level negative reactivity for 300 s, power should be < 1% n₀; "
        f"got {n_final/n0:.4f}×"
    )


def test_precursors_remain_positive_throughout_transient(equilibrium_state) -> None:
    """Physical constraint: precursor concentrations must stay ≥ 0 at all times."""
    y0, beta_arr, lambda_arr, Lambda = equilibrium_state
    rho_step = 200e-5  # +200 pcm moderately fast transient

    result = integrate_kinetics(
        y0,
        t_span=(0.0, 50.0),
        rho_fn=lambda t, y: rho_step,
        beta_arr=beta_arr,
        lambda_arr=lambda_arr,
        Lambda=Lambda,
    )

    assert result.success, f"Solver failed: {result.message}"
    # y[1:7] = precursors at all output steps
    assert np.all(result.y[1:] >= 0.0), "Precursor concentrations turned negative"


def test_integrate_kinetics_returns_requested_t_eval(equilibrium_state) -> None:
    """When t_eval is provided, solution is returned at exactly those times."""
    y0, beta_arr, lambda_arr, Lambda = equilibrium_state
    t_out = np.array([0.0, 1.0, 10.0, 100.0])

    result = integrate_kinetics(
        y0,
        t_span=(0.0, 100.0),
        rho_fn=lambda t, y: 0.0,
        beta_arr=beta_arr,
        lambda_arr=lambda_arr,
        Lambda=Lambda,
        t_eval=t_out,
    )

    assert result.success
    np.testing.assert_allclose(result.t, t_out, atol=1e-12)
    assert result.y.shape == (7, len(t_out))


def test_beta_groups_sum_to_beta_eff() -> None:
    """Sanity check: IAEA Keepin group fractions sum to the nominal β_eff for U-235."""
    beta_arr = np.array(beta_fractions(), dtype=np.float64)
    assert beta_arr.sum() == pytest.approx(BETA_EFF_U235_THERMAL, rel=1e-9)


def test_six_precursor_groups_loaded() -> None:
    """Exactly 6 delayed-neutron groups must be present (SPEC §3.4, Keepin model)."""
    groups = load_keepin_u235_groups()
    assert len(groups) == 6
    assert [g.group for g in groups] == [1, 2, 3, 4, 5, 6]


# ---------------------------------------------------------------------------
# prewarm_jit — CLAUDE.md §Numba Rules
# ---------------------------------------------------------------------------


def test_prewarm_jit_default_args() -> None:
    """prewarm_jit() with no arguments loads beta/lambda from data and runs without error."""
    from physics.shared.kinetics import prewarm_jit

    prewarm_jit()  # covers the None-guard branches


def test_prewarm_jit_explicit_args(keepin_arrays) -> None:
    """prewarm_jit() with explicit arrays bypasses the None-guard branches."""
    from physics.shared.kinetics import prewarm_jit

    beta_arr, lambda_arr = keepin_arrays
    prewarm_jit(beta_arr=beta_arr, lambda_arr=lambda_arr, Lambda=LAMBDA_PWR)


# ---------------------------------------------------------------------------
# integrate_kinetics edge paths — SPEC §4.6.5
# ---------------------------------------------------------------------------


def test_integrate_kinetics_logs_warning_for_supercritical(
    equilibrium_state, caplog
) -> None:
    """integrate_kinetics emits WARNING when |ρ| > 1000 pcm (SPEC §4.6.5).

    _SUPERCRITICAL_RHO_THRESHOLD = 0.01 (dimensionless) = 1000 pcm.
    A rod insertion of 1500 pcm (ρ = 0.015) must trigger the warning.
    """
    import logging

    y0, beta_arr, lambda_arr, Lambda = equilibrium_state
    rho_above_threshold = 0.015  # 1500 pcm > 1000 pcm threshold

    with caplog.at_level(logging.WARNING, logger="physics.shared.kinetics"):
        integrate_kinetics(
            y0,
            t_span=(0.0, 0.001),
            rho_fn=lambda t, y: rho_above_threshold,
            beta_arr=beta_arr,
            lambda_arr=lambda_arr,
            Lambda=Lambda,
        )

    assert any(
        "superprompt" in r.message.lower() for r in caplog.records
    ), "Expected a superprompt-critical WARNING to be logged"


def test_integrate_kinetics_logs_error_when_solver_fails(
    equilibrium_state, caplog
) -> None:
    """integrate_kinetics logs ERROR and returns failed result when solve_ivp fails.

    We mock solve_ivp to return a failed OdeResult so the logger.error path is
    reached without requiring a genuine numerically-singular ODE scenario.
    """
    import logging
    from unittest.mock import MagicMock, patch

    y0, beta_arr, lambda_arr, Lambda = equilibrium_state

    failed_result = MagicMock()
    failed_result.success = False
    failed_result.message = "Mock solver failure for test coverage"

    with patch("physics.shared.kinetics.solve_ivp", return_value=failed_result):
        with caplog.at_level(logging.ERROR, logger="physics.shared.kinetics"):
            result = integrate_kinetics(
                y0,
                t_span=(0.0, 1.0),
                rho_fn=lambda t, y: 0.0,
                beta_arr=beta_arr,
                lambda_arr=lambda_arr,
                Lambda=Lambda,
            )

    assert result is failed_result
    assert any(
        "failed" in r.message.lower() for r in caplog.records
    ), "Expected a solver-failure ERROR to be logged"
