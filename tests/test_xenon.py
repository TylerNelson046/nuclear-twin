"""
Tests for physics/shared/xenon.py — I-135/Xe-135 ODE system.

Software tests (CLAUDE.md §Testing Requirements):
    Each equation is tested against a hand-calculated value.

SPEC physics validation checks covered here:
    test_xenon_equilibrium_worth    SPEC §9.3 PWR #3  — xenon worth -2500 to -3000 pcm at full power
    test_post_shutdown_xenon_peak   SPEC §9.3 PWR #4  — peak 6-10 hr post-shutdown

All analytic reference values are derived from SPEC §4.5.4 constants; no magic numbers.

Reactor parameters used throughout (typical generic LWR, SPEC §3.2):
    PHI_FULL   = 3.1e13 neutrons/cm²/s   — thermal flux at full power
    SIGMA_F    = 0.3 cm⁻¹                — macroscopic fission cross section
    SIGMA_A    = 0.55 cm⁻¹               — total macroscopic absorption XS
                                            (≈ ν·Σ_f for a one-group critical reactor;
                                             used only for reactivity worth calculation)
"""

from __future__ import annotations

import numpy as np
import pytest

from physics.shared.xenon import (
    GAMMA_I,
    GAMMA_X,
    LAMBDA_I,
    LAMBDA_X,
    SIGMA_AX,
    integrate_xenon,
    steady_state_xenon,
    xenon_iodine_rhs,
)

# ---------------------------------------------------------------------------
# Shared test parameters
# ---------------------------------------------------------------------------

PHI_FULL: float = 3.1e13   # neutrons/cm²/s — full-power thermal flux
SIGMA_F: float = 0.3       # cm⁻¹ — macroscopic fission cross section
SIGMA_A: float = 0.55      # cm⁻¹ — total absorption XS for worth calculation


@pytest.fixture(scope="module")
def equilibrium_state() -> tuple[np.ndarray, float, float]:
    """Return (y0, phi, Sigma_f) for equilibrium at full power."""
    y0 = steady_state_xenon(PHI_FULL, SIGMA_F)
    return y0, PHI_FULL, SIGMA_F


# ---------------------------------------------------------------------------
# Unit tests — analytic formula verification
# ---------------------------------------------------------------------------


def test_steady_state_xenon_analytical() -> None:
    """Equilibrium I₀ and X₀ must match the closed-form SPEC §4.6.2 expressions.

    I₀ = γ_I · Σ_f · φ / λ_I
    X₀ = (γ_X + γ_I) · Σ_f · φ / (λ_X + σ_aX · φ)
    """
    fission_rate = SIGMA_F * PHI_FULL

    I0_expected = GAMMA_I * fission_rate / LAMBDA_I
    X0_expected = (GAMMA_X + GAMMA_I) * fission_rate / (LAMBDA_X + SIGMA_AX * PHI_FULL)

    y0 = steady_state_xenon(PHI_FULL, SIGMA_F)

    np.testing.assert_allclose(y0[0], I0_expected, rtol=1e-12)
    np.testing.assert_allclose(y0[1], X0_expected, rtol=1e-12)


def test_steady_state_zero_flux_returns_zeros() -> None:
    """At φ = 0 there is no fission source; equilibrium concentrations are zero."""
    y0 = steady_state_xenon(0.0, SIGMA_F)
    np.testing.assert_array_equal(y0, [0.0, 0.0])


def test_rhs_zero_at_equilibrium(equilibrium_state) -> None:
    """At equilibrium ICs and equilibrium flux, all derivatives must be zero.

    Analytic proof:
        dI/dt = γ_I·Σ_f·φ − λ_I·I₀ = γ_I·Σ_f·φ − γ_I·Σ_f·φ = 0
        dX/dt = γ_X·Σ_f·φ + λ_I·I₀ − (λ_X + σ_aX·φ)·X₀
              = (γ_X + γ_I)·Σ_f·φ − (λ_X + σ_aX·φ)·X₀ = 0
    """
    y0, phi, Sigma_f = equilibrium_state
    dydt = xenon_iodine_rhs(0.0, y0, phi, Sigma_f)

    np.testing.assert_allclose(dydt, 0.0, atol=1e-3)  # atoms/cm³/s tolerance


def test_rhs_iodine_decreases_at_zero_flux() -> None:
    """With φ = 0 and nonzero I, dI/dt must be negative (pure decay)."""
    y = np.array([1.0e16, 1.0e15])
    dydt = xenon_iodine_rhs(0.0, y, 0.0, SIGMA_F)

    assert dydt[0] < 0.0, "dI/dt must be negative when φ = 0 (only radioactive decay)"


def test_rhs_xenon_rises_after_shutdown() -> None:
    """Immediately after shutdown, dX/dt > 0 when I is large relative to X.

    Post-shutdown: no burnout or direct production, but iodine feeds xenon
    faster than xenon decays (I_0 >> X_0 / (λ_X / λ_I) at full-power equilibrium).
    """
    y0 = steady_state_xenon(PHI_FULL, SIGMA_F)
    # Set phi=0 to simulate shutdown
    dydt = xenon_iodine_rhs(0.0, y0, 0.0, SIGMA_F)

    assert dydt[1] > 0.0, "dX/dt must be positive right after shutdown (iodine decay feeds xenon)"


def test_rhs_iodine_production_term(equilibrium_state) -> None:
    """Each dI/dt term must equal γ_I·Σ_f·φ − λ_I·I exactly (SPEC Eq. 14)."""
    y0, phi, Sigma_f = equilibrium_state
    # Perturb to get nonzero derivatives
    y_perturbed = y0 * 0.5
    dydt = xenon_iodine_rhs(0.0, y_perturbed, phi, Sigma_f)

    expected_dI = GAMMA_I * Sigma_f * phi - LAMBDA_I * y_perturbed[0]
    assert dydt[0] == pytest.approx(expected_dI, rel=1e-12)


def test_rhs_xenon_production_term(equilibrium_state) -> None:
    """dX/dt must equal γ_X·Σ_f·φ + λ_I·I − (λ_X + σ_aX·φ)·X exactly (SPEC Eq. 15)."""
    y0, phi, Sigma_f = equilibrium_state
    y_perturbed = y0 * 0.5
    dydt = xenon_iodine_rhs(0.0, y_perturbed, phi, Sigma_f)

    expected_dX = (
        GAMMA_X * Sigma_f * phi
        + LAMBDA_I * y_perturbed[0]
        - (LAMBDA_X + SIGMA_AX * phi) * y_perturbed[1]
    )
    assert dydt[1] == pytest.approx(expected_dX, rel=1e-12)


def test_equilibrium_scales_linearly_with_flux() -> None:
    """Doubling flux doubles equilibrium I (linear source term in Eq. 14).

    Xenon equilibrium does not scale exactly linearly because σ_aX·φ in the
    denominator creates a saturating burnout term — verified to increase.
    """
    y_1 = steady_state_xenon(PHI_FULL, SIGMA_F)
    y_2 = steady_state_xenon(2 * PHI_FULL, SIGMA_F)

    np.testing.assert_allclose(y_2[0], 2.0 * y_1[0], rtol=1e-12)
    # Xenon: should increase but saturate — just verify it rose
    assert y_2[1] > y_1[1], "X₀ must increase with flux"


# ---------------------------------------------------------------------------
# Integration tests — physics validation (SPEC §9.3)
# ---------------------------------------------------------------------------


def test_steady_state_stability(equilibrium_state) -> None:
    """At equilibrium ICs and constant full-power flux, concentrations stay stable.

    Tolerance: < 0.01% drift over 36,000 s (~10 hr), which spans several xenon
    half-lives and is a stringent check of the numerical equilibrium.
    """
    y0, phi, Sigma_f = equilibrium_state

    result = integrate_xenon(
        y0,
        t_span=(0.0, 36_000.0),
        phi_fn=lambda t, y: phi,
        Sigma_f=Sigma_f,
    )

    assert result.success, f"Solver failed: {result.message}"

    I_drift = abs(result.y[0, -1] / y0[0] - 1.0)
    X_drift = abs(result.y[1, -1] / y0[1] - 1.0)

    assert I_drift < 1e-4, f"I-135 drifted {I_drift:.2e} from equilibrium"
    assert X_drift < 1e-4, f"Xe-135 drifted {X_drift:.2e} from equilibrium"


def test_xenon_equilibrium_worth(equilibrium_state) -> None:
    """SPEC §9.3 PWR #3: xenon worth within −2500 to −3000 pcm at full-power equilibrium.

    Xenon worth formula (one-group perturbation theory):
        ρ_Xe = −σ_aX · X / Σ_a

    With Σ_a = 0.55 cm⁻¹ (typical PWR one-group absorption XS ≈ ν·Σ_f at criticality),
    PHI_FULL = 3.1×10¹³, Σ_f = 0.3 cm⁻¹ → X_eq ≈ 6.07×10¹⁵ atoms/cm³
    → ρ_Xe ≈ −0.02869 ≡ −2869 pcm (within range).
    """
    y0, phi, Sigma_f = equilibrium_state

    # Integrate 50 hr from zero to verify buildup reaches equilibrium
    t_50hr = 50.0 * 3600.0
    t_eval = np.array([t_50hr])
    result = integrate_xenon(
        np.zeros(2),
        t_span=(0.0, t_50hr),
        phi_fn=lambda t, y: phi,
        Sigma_f=Sigma_f,
        t_eval=t_eval,
    )

    assert result.success, f"Solver failed: {result.message}"
    X_50hr = result.y[1, -1]

    rho_xe_pcm = -SIGMA_AX * X_50hr / SIGMA_A * 1e5

    assert -3000 <= rho_xe_pcm <= -2500, (
        f"Xenon worth at 50 hr = {rho_xe_pcm:.0f} pcm; "
        f"SPEC requires −2500 to −3000 pcm"
    )


def test_post_shutdown_xenon_peak(equilibrium_state) -> None:
    """SPEC §9.3 PWR #4: xenon peaks between 6 and 10 hours post-shutdown.

    Following full-power shutdown (φ → 0), iodine decay continues feeding
    xenon while neutron burnout vanishes. The peak occurs when
    λ_I · I(t) = λ_X · X(t); analytically ~8.5 hr for these parameters.
    """
    y0, phi, Sigma_f = equilibrium_state

    # Simulate 40 hr post-shutdown at fine time resolution to locate the peak
    t_shutdown = 0.0
    t_end = 40.0 * 3600.0
    t_eval = np.linspace(0.0, t_end, 2000)

    result = integrate_xenon(
        y0,
        t_span=(t_shutdown, t_end),
        phi_fn=lambda t, y: 0.0,  # immediate full-power shutdown
        Sigma_f=Sigma_f,
        t_eval=t_eval,
    )

    assert result.success, f"Solver failed: {result.message}"

    xenon = result.y[1]
    peak_idx = int(np.argmax(xenon))
    t_peak_hr = result.t[peak_idx] / 3600.0

    assert 6.0 <= t_peak_hr <= 10.0, (
        f"Xenon peak at {t_peak_hr:.2f} hr post-shutdown; SPEC requires 6–10 hr"
    )

    # Peak value must exceed equilibrium xenon (that's what makes it a peak)
    assert xenon[peak_idx] > y0[1], (
        f"Peak xenon ({xenon[peak_idx]:.3e}) must exceed pre-shutdown equilibrium "
        f"({y0[1]:.3e})"
    )


def test_post_shutdown_xenon_returns_to_zero(equilibrium_state) -> None:
    """SPEC §9.3 PWR #4 (continued): xenon returns to near-zero well after the peak.

    With λ_X t½ = 9.2 hr, xenon is still ~24% of equilibrium at 50 hr.
    By 80 hr post-shutdown (~8.7 xenon half-lives from the peak) it drops to
    ~3% of equilibrium, satisfying the "near-zero" criterion. Iodine reaches
    < 1% by 50 hr (t½ = 6.7 hr).
    """
    y0, phi, Sigma_f = equilibrium_state
    t_end = 80.0 * 3600.0

    result = integrate_xenon(
        y0,
        t_span=(0.0, t_end),
        phi_fn=lambda t, y: 0.0,
        Sigma_f=Sigma_f,
    )

    assert result.success, f"Solver failed: {result.message}"

    I_final_frac = result.y[0, -1] / y0[0]
    X_final_frac = result.y[1, -1] / y0[1]

    assert I_final_frac < 0.01, (
        f"I-135 at 80 hr = {I_final_frac:.3%} of equilibrium; expected < 1%"
    )
    assert X_final_frac < 0.10, (
        f"Xe-135 at 80 hr = {X_final_frac:.3%} of equilibrium; expected < 10%"
    )


def test_startup_from_zero_approaches_equilibrium() -> None:
    """Starting from zero, concentrations must approach equilibrium at full power.

    After 50 hr at full power from [I=0, X=0], both concentrations must reach
    within 1% of their analytic equilibrium values (SPEC §4.6.2).
    """
    y_eq = steady_state_xenon(PHI_FULL, SIGMA_F)
    t_end = 50.0 * 3600.0

    result = integrate_xenon(
        np.zeros(2),
        t_span=(0.0, t_end),
        phi_fn=lambda t, y: PHI_FULL,
        Sigma_f=SIGMA_F,
    )

    assert result.success, f"Solver failed: {result.message}"

    I_frac = abs(result.y[0, -1] / y_eq[0] - 1.0)
    X_frac = abs(result.y[1, -1] / y_eq[1] - 1.0)

    assert I_frac < 0.01, (
        f"I-135 after 50 hr at full power is {I_frac:.2%} away from equilibrium"
    )
    assert X_frac < 0.01, (
        f"Xe-135 after 50 hr at full power is {X_frac:.2%} away from equilibrium"
    )


def test_no_negative_concentrations_during_transient(equilibrium_state) -> None:
    """Physical constraint: I and X must remain ≥ 0 at all times.

    Tests through a power ramp down followed by zero-power coast.
    """
    y0, phi, Sigma_f = equilibrium_state
    t_ramp_end = 3600.0    # 1 hr ramp from full to zero
    t_total = 72.0 * 3600.0  # 72 hr total

    def phi_fn(t, y):
        if t < t_ramp_end:
            return phi * (1.0 - t / t_ramp_end)  # linear ramp down
        return 0.0

    result = integrate_xenon(
        y0,
        t_span=(0.0, t_total),
        phi_fn=phi_fn,
        Sigma_f=Sigma_f,
    )

    assert result.success, f"Solver failed: {result.message}"
    assert np.all(result.y[0] >= 0.0), "I-135 concentration went negative"
    assert np.all(result.y[1] >= 0.0), "Xe-135 concentration went negative"


def test_integrate_xenon_returns_requested_t_eval() -> None:
    """When t_eval is provided, solution is returned at exactly those times."""
    y0 = steady_state_xenon(PHI_FULL, SIGMA_F)
    t_out = np.array([0.0, 3600.0, 18000.0, 36000.0])

    result = integrate_xenon(
        y0,
        t_span=(0.0, 36_000.0),
        phi_fn=lambda t, y: PHI_FULL,
        Sigma_f=SIGMA_F,
        t_eval=t_out,
    )

    assert result.success
    np.testing.assert_allclose(result.t, t_out, atol=1e-12)
    assert result.y.shape == (2, len(t_out))


# ---------------------------------------------------------------------------
# integrate_xenon edge path — SPEC §4.6.5
# ---------------------------------------------------------------------------


def test_integrate_xenon_logs_error_when_solver_fails(caplog) -> None:
    """integrate_xenon logs ERROR and returns a failed result when solve_ivp fails.

    We mock solve_ivp to return a failed OdeResult so the logger.error path is
    reached without requiring a genuine numerically-singular ODE scenario.
    """
    import logging
    from unittest.mock import MagicMock, patch

    y0 = steady_state_xenon(PHI_FULL, SIGMA_F)

    failed_result = MagicMock()
    failed_result.success = False
    failed_result.message = "Mock xenon solver failure for test coverage"

    with patch("physics.shared.xenon.solve_ivp", return_value=failed_result):
        with caplog.at_level(logging.ERROR, logger="physics.shared.xenon"):
            result = integrate_xenon(
                y0,
                t_span=(0.0, 3600.0),
                phi_fn=lambda t, y: PHI_FULL,
                Sigma_f=SIGMA_F,
            )

    assert result is failed_result
    assert any(
        "failed" in r.message.lower() for r in caplog.records
    ), "Expected a solver-failure ERROR to be logged"
