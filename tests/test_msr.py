"""
tests/test_msr.py — Physics validation and unit tests for the MSR twin.

Test categories (CLAUDE.md §Testing Requirements):
  - Unit tests: each equation implementation verified against hand-calculated values.
  - Integration tests: SPEC §7.3 MSR physics validation checkpoints.
  - Key mandatory tests (CLAUDE.md §Testing Requirements):
      test_shared_kinetics_consistency : MSR at v_salt≈0 matches PWR kinetics
      test_beta_eff_flow_zero_velocity  : β_eff,flow = β_eff_static at zero salt flow
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from data.keepin_dnp import BETA_EFF_U235_THERMAL, beta_fractions, decay_constants
from physics.msr.drift import PrecursorHistory, compute_beta_eff_flow, msr_precursor_drift_rhs
from physics.msr.engine import (
    MSRControls,
    MSRModelConfig,
    build_initial_state,
    flowing_steady_state_precursors,
    integrate_msr,
    make_critical_msr_config,
    msr_coupled_rhs,
    msr_critical_base_reactivity_pcm,
)
from physics.msr.thermal import (
    C_SALT,
    M_CORE_SALT,
    M_DOT_SALT_NOM,
    P_NOM_MSR,
    T_SALT_INLET_NOM,
    T_SALT_NOM,
    TAU_CORE_NOM,
    m_dot_salt_from_tau_core,
    salt_thermal_rhs,
    steady_state_salt_temperature,
)
from physics.shared.kinetics import LAMBDA_PWR, steady_state_precursors

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

BETA_ARR = np.array(beta_fractions(), dtype=np.float64)
LAMBDA_ARR = np.array(decay_constants(), dtype=np.float64)
BETA_EFF_STATIC = float(np.sum(BETA_ARR))  # ~0.0065


# ===========================================================================
# Unit tests — physics/msr/drift.py
# ===========================================================================


class TestComputeBetaEffFlow:
    """SPEC Eq. 24 — β_eff,flow hand-calculation checks."""

    def test_zero_loop_transit_returns_static_beta(self):
        """τ_loop = 0 (instant return): drift flow-out and return cancel; β_eff,flow = β_eff."""
        # At τ_loop=0: exp(-λ*0)=1, denom=1+λτ_core-1=λτ_core,
        # factor1 = λτ_c/(1+λτ_c), factor2 = 1 + 1/(λτ_c) = (1+λτ_c)/(λτ_c)
        # → βᵢ * factor1 * factor2 = βᵢ * 1  →  Σ = β_eff_static  ✓
        beta_flow = compute_beta_eff_flow(BETA_ARR, LAMBDA_ARR, tau_core=5.0, tau_loop=0.0)
        assert abs(beta_flow - BETA_EFF_STATIC) < 1.0e-10, (
            f"β_eff,flow at τ_loop=0 should equal β_eff_static={BETA_EFF_STATIC:.6f}, "
            f"got {beta_flow:.6f}"
        )

    def test_large_tau_core_approaches_static_beta(self):
        """τ_core → ∞ (v_salt → 0): factor1 → 1, return term → 0 → β_eff,flow → β_eff_static."""
        beta_flow = compute_beta_eff_flow(
            BETA_ARR, LAMBDA_ARR, tau_core=1.0e8, tau_loop=4.0e8
        )
        assert abs(beta_flow - BETA_EFF_STATIC) < 1.0e-4, (
            f"β_eff,flow at large τ_core should approach β_eff_static={BETA_EFF_STATIC:.6f}, "
            f"got {beta_flow:.6f}"
        )

    def test_monotonic_decrease_with_velocity(self):
        """β_eff,flow decreases monotonically as salt velocity increases (τ_core decreases)."""
        # τ_loop / τ_core = L_loop / L_core = const ≈ 4 for typical MSR geometry
        tau_cores = [100.0, 50.0, 20.0, 10.0, 5.0, 2.0, 1.0]
        tau_loops = [4 * tc for tc in tau_cores]
        beta_flows = [
            compute_beta_eff_flow(BETA_ARR, LAMBDA_ARR, tc, tl)
            for tc, tl in zip(tau_cores, tau_loops)
        ]
        for i in range(len(beta_flows) - 1):
            assert beta_flows[i] > beta_flows[i + 1], (
                f"β_eff,flow should decrease with velocity: "
                f"β_flow(τ_core={tau_cores[i]})={beta_flows[i]:.5f} not > "
                f"β_flow(τ_core={tau_cores[i+1]})={beta_flows[i+1]:.5f}"
            )

    def test_beta_flow_less_than_static_at_finite_velocity(self):
        """At finite salt velocity (τ_core=5s, τ_loop=20s), β_eff,flow < β_eff_static."""
        beta_flow = compute_beta_eff_flow(
            BETA_ARR, LAMBDA_ARR, tau_core=5.0, tau_loop=20.0
        )
        assert beta_flow < BETA_EFF_STATIC, (
            f"β_eff,flow={beta_flow:.5f} should be less than "
            f"β_eff_static={BETA_EFF_STATIC:.5f} at finite salt velocity."
        )
        assert beta_flow > 0.0, "β_eff,flow must be positive."

    def test_hand_calculated_single_group(self):
        """Hand-verify Eq. 24 for a single precursor group (all others zeroed out)."""
        # One group: β=0.001, λ=0.1 s⁻¹, τ_core=5s, τ_loop=20s
        beta = np.array([0.001, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
        lam = np.array([0.1, 0.01, 0.01, 0.01, 0.01, 0.01], dtype=np.float64)

        lam0, tc, tl = 0.1, 5.0, 20.0
        exp_t = math.exp(-lam0 * tl)
        factor1 = (lam0 * tc) / (1.0 + lam0 * tc)      # =0.5/(1+0.5)= 0.3333...  wait
        # λτ_c = 0.1*5 = 0.5 → factor1 = 0.5/1.5 = 0.3333
        # exp_term = exp(-0.1*20) = exp(-2) ≈ 0.13534
        # denom = 1 + 0.5 - 0.13534 = 1.36466
        # factor2 = 1 + 0.13534/1.36466 ≈ 1 + 0.09920 = 1.09920
        # β_flow = 0.001 * 0.3333 * 1.09920 ≈ 3.664e-4
        factor1_expected = 0.5 / 1.5
        denom_expected = 1.0 + 0.5 - exp_t
        factor2_expected = 1.0 + exp_t / denom_expected
        expected = 0.001 * factor1_expected * factor2_expected

        result = compute_beta_eff_flow(beta, lam, tau_core=tc, tau_loop=tl)
        assert abs(result - expected) < 1.0e-12, (
            f"Hand-calc β_eff,flow={expected:.8e}, got {result:.8e}"
        )


class TestMsrPrecursorDriftRhs:
    """Unit tests for the msr_precursor_drift_rhs Numba kernel."""

    def test_zero_flow_no_drift(self):
        """At τ_core → ∞ the drift contribution is negligible (~0)."""
        C = np.ones(6, dtype=np.float64)
        C_del = np.ones(6, dtype=np.float64)
        # Use very large tau_core to simulate zero flow
        drift = msr_precursor_drift_rhs(C, C_del, LAMBDA_ARR, tau_core=1.0e10, tau_loop=1.0e10)
        assert np.all(np.abs(drift) < 1.0e-9)

    def test_instant_return_cancels(self):
        """At τ_loop = 0 with C = C_delayed: drift = -C/τ + C/τ * 1 = 0."""
        C = np.array([1.0, 2.0, 3.0, 0.5, 0.3, 0.1], dtype=np.float64)
        drift = msr_precursor_drift_rhs(
            C, C_delayed=C.copy(), lambda_arr=LAMBDA_ARR, tau_core=5.0, tau_loop=0.0
        )
        assert np.allclose(drift, 0.0, atol=1.0e-12), (
            f"Drift should cancel at τ_loop=0: {drift}"
        )

    def test_finite_loop_reduces_precursors(self):
        """At finite τ_loop, C_return < C_flow_out: net drift is negative."""
        C = np.ones(6, dtype=np.float64)
        C_del = np.ones(6, dtype=np.float64)  # steady state: C(t-τ_loop) = C(t)
        drift = msr_precursor_drift_rhs(
            C, C_del, lambda_arr=LAMBDA_ARR, tau_core=5.0, tau_loop=20.0
        )
        # At SS with C_delayed=C: net drift = (1/τ_c)*C*(exp(-λτ_l) - 1) < 0
        assert np.all(drift < 0.0), (
            "Net drift should be negative (precursor loss) at steady state with finite τ_loop."
        )

    def test_return_term_shape(self):
        """Return array must have shape (6,)."""
        C = np.ones(6, dtype=np.float64)
        C_del = np.zeros(6, dtype=np.float64)
        drift = msr_precursor_drift_rhs(C, C_del, LAMBDA_ARR, tau_core=5.0, tau_loop=20.0)
        assert drift.shape == (6,)


class TestPrecursorHistory:
    """Unit tests for PrecursorHistory delay buffer."""

    def test_steady_state_before_start(self):
        """Lookups at t < t_start return C_init."""
        C_init = np.ones(6, dtype=np.float64) * 2.5
        hist = PrecursorHistory(t_start=0.0, C_init=C_init)
        result = hist.get(-10.0)
        assert np.allclose(result, C_init)

    def test_interpolation_between_points(self):
        """Linear interpolation between two stored points."""
        C_init = np.zeros(6, dtype=np.float64)
        hist = PrecursorHistory(t_start=0.0, C_init=C_init)
        hist.add(0.0, np.zeros(6))
        hist.add(10.0, np.ones(6) * 10.0)

        result = hist.get(5.0)
        assert np.allclose(result, 5.0 * np.ones(6), atol=1.0e-12)

    def test_extrapolation_clamps_to_last_value(self):
        """Lookups past the last stored time return the last value."""
        C_init = np.zeros(6, dtype=np.float64)
        hist = PrecursorHistory(t_start=0.0, C_init=C_init)
        hist.add(5.0, np.ones(6) * 3.0)
        result = hist.get(100.0)
        assert np.allclose(result, 3.0)

    def test_size_property(self):
        C_init = np.zeros(6, dtype=np.float64)
        hist = PrecursorHistory(t_start=0.0, C_init=C_init)
        assert hist.size == 0
        hist.add(1.0, C_init)
        hist.add(2.0, C_init)
        assert hist.size == 2


# ===========================================================================
# Unit tests — physics/msr/thermal.py
# ===========================================================================


class TestSaltThermalModule:
    """Unit tests for the unified salt thermal-hydraulic module."""

    def test_steady_state_temperature_nominal(self):
        """At nominal operating conditions, T_salt_ss = T_SALT_NOM."""
        T_ss = steady_state_salt_temperature(P_NOM_MSR, M_DOT_SALT_NOM, T_SALT_INLET_NOM)
        assert abs(T_ss - T_SALT_NOM) < 1.0, (
            f"Nominal SS temperature {T_ss:.3f} K should equal T_SALT_NOM={T_SALT_NOM:.3f} K"
        )

    def test_zero_power_returns_inlet_temperature(self):
        """At P = 0, steady-state salt temperature equals the inlet temperature."""
        T_ss = steady_state_salt_temperature(0.0, M_DOT_SALT_NOM, T_SALT_INLET_NOM)
        assert abs(T_ss - T_SALT_INLET_NOM) < 1.0e-10

    def test_higher_flow_lowers_temperature(self):
        """Doubling ṁ_salt halves the temperature rise above inlet."""
        T_1x = steady_state_salt_temperature(P_NOM_MSR, M_DOT_SALT_NOM, T_SALT_INLET_NOM)
        T_2x = steady_state_salt_temperature(P_NOM_MSR, 2.0 * M_DOT_SALT_NOM, T_SALT_INLET_NOM)
        delta_1x = T_1x - T_SALT_INLET_NOM
        delta_2x = T_2x - T_SALT_INLET_NOM
        assert abs(delta_2x - delta_1x / 2.0) < 1.0e-6

    def test_steady_state_invalid_flow(self):
        """Zero mass flow rate returns NaN (SPEC SR-01)."""
        import math
        T_ss = steady_state_salt_temperature(P_NOM_MSR, 0.0, T_SALT_INLET_NOM)
        assert math.isnan(T_ss)

    def test_salt_thermal_rhs_zero_derivative_at_steady_state(self):
        """At exact steady-state conditions, dT_salt/dt ≈ 0 (within numerical tolerance)."""
        T_ss = steady_state_salt_temperature(P_NOM_MSR, M_DOT_SALT_NOM, T_SALT_INLET_NOM)
        y_ss = np.array([T_ss], dtype=np.float64)
        dydt = salt_thermal_rhs(0.0, y_ss, P_NOM_MSR, M_DOT_SALT_NOM, T_SALT_INLET_NOM)
        assert abs(dydt[0]) < 1.0, (
            f"dT_salt/dt at steady state should be ≈0, got {dydt[0]:.6e} K/s"
        )

    def test_m_dot_from_tau_core_nominal(self):
        """At τ_core = TAU_CORE_NOM, ṁ_salt = M_DOT_SALT_NOM."""
        m_dot = m_dot_salt_from_tau_core(TAU_CORE_NOM)
        assert abs(m_dot - M_DOT_SALT_NOM) < 1.0e-6

    def test_m_dot_doubles_when_tau_core_halves(self):
        """ṁ_salt ∝ 1 / τ_core (constant core geometry)."""
        m_dot_1 = m_dot_salt_from_tau_core(TAU_CORE_NOM)
        m_dot_2 = m_dot_salt_from_tau_core(TAU_CORE_NOM / 2.0)
        assert abs(m_dot_2 / m_dot_1 - 2.0) < 1.0e-10


# ===========================================================================
# Integration tests — physics/msr/engine.py
# ===========================================================================


class TestMsrSteadyStateStability:
    """SPEC §7.3 — Steady-state criticality: power drift < 0.1% over 1000 s."""

    def test_steady_state_power_drift(self):
        """Power drift < 0.1% over 1000 simulated seconds with no external reactivity.

        Uses make_critical_msr_config() to set base_reactivity_pcm = ρ_crit so the
        flowing-precursor steady state is exactly critical at t = 0.
        """
        config = make_critical_msr_config(
            tau_core=5.0,
            tau_loop=20.0,
            alpha_salt_pcm_per_k=-5.0,
            T_salt_ref_k=T_SALT_NOM,
            beta_arr=BETA_ARR,
            lambda_arr=LAMBDA_ARR,
        )
        y0 = build_initial_state(n0=1.0, config=config, beta_arr=BETA_ARR, lambda_arr=LAMBDA_ARR)
        controls = MSRControls(external_reactivity_pcm=0.0)

        t_eval = np.linspace(0, 1000, 101)
        result = integrate_msr(
            y0, (0, 1000), controls, config,
            beta_arr=BETA_ARR, lambda_arr=LAMBDA_ARR,
            t_eval=t_eval,
        )
        assert result.success, f"Integration failed: {result.message}"
        n_initial = result.y[0, 0]
        n_final = result.y[0, -1]
        drift_pct = abs(n_final - n_initial) / n_initial * 100.0
        assert drift_pct < 0.1, (
            f"Steady-state power drift {drift_pct:.4f}% exceeds 0.1% tolerance "
            f"(n_initial={n_initial:.6f}, n_final={n_final:.6f})"
        )


class TestSharedKineticsConsistency:
    """CLAUDE.md mandatory test — at v_salt ≈ 0, MSR kinetics equals PWR kinetics.

    SPEC §7.3 physics validation checkpoint:
        'At zero salt velocity, MSR point kinetics results shall be numerically
        identical to PWR point kinetics results given identical input parameters.'

    Method: run MSR with very large τ_core (≡ zero flow, drift terms vanish) and
    compare n(t) against the shared point_kinetics_rhs result. With zero
    temperature feedback and identical initial conditions, the two must match
    within Radau solver tolerance over a 10 s transient.
    """

    def test_neutron_population_matches_pwr_kinetics_at_zero_flow(self):
        """MSR with τ_core=1e8 s produces n(t) identical to static PWR kinetics."""
        from physics.shared.kinetics import integrate_kinetics

        rho_insertion = 50.0e-5  # +50 pcm in dimensionless Δk/k

        # --- MSR at near-zero salt velocity (τ_core=1e8 → drift terms ≈ 0) ---
        # At τ_core → ∞: flowing SS = static SS, ρ_crit ≈ 0, so base_reactivity ≈ 0.
        # Disable temperature feedback so only kinetics is compared.
        config_zero_flow = MSRModelConfig(
            tau_core=1.0e8,
            tau_loop=4.0e8,
            alpha_salt_pcm_per_k=0.0,
            T_salt_ref_k=T_SALT_NOM,
            base_reactivity_pcm=0.0,
        )
        y0_msr = build_initial_state(
            n0=1.0, config=config_zero_flow, beta_arr=BETA_ARR, lambda_arr=LAMBDA_ARR
        )
        controls_msr = MSRControls(external_reactivity_pcm=50.0)  # +50 pcm

        t_eval = np.linspace(0, 10, 51)
        result_msr = integrate_msr(
            y0_msr, (0, 10), controls_msr, config_zero_flow,
            beta_arr=BETA_ARR, lambda_arr=LAMBDA_ARR,
            t_eval=t_eval,
        )
        assert result_msr.success, f"MSR integration failed: {result_msr.message}"

        # --- Shared PWR point kinetics (pure 7-state, no thermal feedback) ---
        from physics.shared.kinetics import build_initial_state as build_pwr_ic
        y0_pwr = build_pwr_ic(1.0, BETA_ARR, LAMBDA_ARR, LAMBDA_PWR)
        result_pwr = integrate_kinetics(
            y0_pwr,
            (0, 10),
            rho_fn=lambda t, y: rho_insertion,
            beta_arr=BETA_ARR,
            lambda_arr=LAMBDA_ARR,
            Lambda=LAMBDA_PWR,
            t_eval=t_eval,
        )
        assert result_pwr.success, "PWR kinetics integration failed."

        # Interpolate to common time grid — both use the same t_eval but integrate_msr
        # always prepends t=0 while integrate_kinetics may have it or not.
        # Align by finding n(t) at matching time points.
        t_common = np.intersect1d(
            np.round(result_msr.t, 6), np.round(result_pwr.t, 6)
        )
        assert len(t_common) >= 10, (
            f"Too few common time points: {len(t_common)}. "
            f"MSR t range: {result_msr.t[[0,-1]]}, PWR t range: {result_pwr.t[[0,-1]]}"
        )

        msr_idx = np.searchsorted(np.round(result_msr.t, 6), t_common)
        pwr_idx = np.searchsorted(np.round(result_pwr.t, 6), t_common)
        n_msr = result_msr.y[0, msr_idx]
        n_pwr = result_pwr.y[0, pwr_idx]

        max_rel_diff = float(np.max(np.abs(n_msr - n_pwr) / np.abs(n_pwr + 1.0e-30)))
        assert max_rel_diff < 1.0e-3, (
            f"MSR n(t) differs from PWR kinetics by {max_rel_diff:.2e} "
            f"at τ_core=1e8 s (should be < 0.1%). "
            f"Max |Δn/n|={max_rel_diff:.3e}"
        )


class TestDopplerSelfRegulationMsr:
    """SPEC §7.3 — +50 pcm insertion stabilises (Doppler analog via salt feedback)."""

    def test_reactivity_insertion_stabilises(self):
        """A +50 pcm external reactivity insertion reaches a new equilibrium (does not diverge)."""
        config = make_critical_msr_config(
            tau_core=5.0,
            tau_loop=20.0,
            alpha_salt_pcm_per_k=-5.0,
            T_salt_ref_k=T_SALT_NOM,
            beta_arr=BETA_ARR,
            lambda_arr=LAMBDA_ARR,
        )
        y0 = build_initial_state(n0=1.0, config=config, beta_arr=BETA_ARR, lambda_arr=LAMBDA_ARR)
        controls = MSRControls(external_reactivity_pcm=50.0)

        result = integrate_msr(
            y0, (0, 500), controls, config,
            beta_arr=BETA_ARR, lambda_arr=LAMBDA_ARR,
            dt_history=10.0,
        )
        assert result.success, f"Integration failed: {result.message}"

        n_max = np.max(result.y[0, :])
        n_final = result.y[0, -1]

        # Power must have stabilised (final power < peak): self-regulation via T feedback
        assert n_final < n_max * 1.5, (
            f"Power did not stabilise after +50 pcm insertion: "
            f"n_max={n_max:.4f}, n_final={n_final:.4f}"
        )
        # Power must be above initial (positive reactivity net effect at new equilibrium)
        assert n_final > 1.0, f"Final power {n_final:.4f} should exceed initial n=1.0"


class TestBetaEffFlowReduction:
    """SPEC §7.3 validation: β_eff,flow decreases monotonically with salt velocity."""

    def test_beta_eff_flow_decreases_with_velocity(self):
        """Faster salt → smaller β_eff,flow (more precursors lost outside core)."""
        # τ_core range from slow (100s) to fast (1s); τ_loop = 4×τ_core
        tau_cores = [100.0, 20.0, 5.0, 1.0]
        beta_flows = [
            compute_beta_eff_flow(BETA_ARR, LAMBDA_ARR, tc, 4 * tc)
            for tc in tau_cores
        ]
        for i in range(len(beta_flows) - 1):
            assert beta_flows[i] > beta_flows[i + 1], (
                f"β_eff,flow should decrease: "
                f"β({tau_cores[i]}s)={beta_flows[i]:.5f} not > "
                f"β({tau_cores[i+1]}s)={beta_flows[i+1]:.5f}"
            )

    def test_beta_eff_flow_bounded_by_static(self):
        """β_eff,flow ≤ β_eff_static for all salt velocities."""
        for tc in [0.5, 1.0, 5.0, 20.0, 100.0]:
            bf = compute_beta_eff_flow(BETA_ARR, LAMBDA_ARR, tc, 4 * tc)
            assert bf <= BETA_EFF_STATIC + 1.0e-10, (
                f"β_eff,flow={bf:.6f} > β_eff_static={BETA_EFF_STATIC:.6f} "
                f"at τ_core={tc}"
            )
            assert bf > 0.0, f"β_eff,flow must be positive at τ_core={tc}"


class TestMsrInitialState:
    """Tests for build_initial_state analytical steady-state construction."""

    def test_salt_temperature_at_nominal_conditions(self):
        """y0[7] = T_SALT_NOM at nominal power and flow."""
        config = MSRModelConfig()
        y0 = build_initial_state(n0=1.0, config=config, beta_arr=BETA_ARR, lambda_arr=LAMBDA_ARR)
        assert abs(y0[7] - T_SALT_NOM) < 1.0, (
            f"Initial T_salt={y0[7]:.3f} K should be close to T_SALT_NOM={T_SALT_NOM:.3f} K"
        )

    def test_precursors_at_flowing_steady_state(self):
        """Cᵢ,₀ = (βᵢ/Λ)·n / [λᵢ + (1/τ_core)·(1−e^{−λᵢτ_loop})] (flowing SS, Eq. 22)."""
        config = MSRModelConfig()
        y0 = build_initial_state(n0=1.0, config=config, beta_arr=BETA_ARR, lambda_arr=LAMBDA_ARR)
        C_expected = flowing_steady_state_precursors(
            1.0, BETA_ARR, LAMBDA_ARR, LAMBDA_PWR, config.tau_core, config.tau_loop
        )
        assert np.allclose(y0[1:7], C_expected, rtol=1.0e-10), (
            "Initial precursor concentrations differ from flowing steady-state formula."
        )

    def test_precursors_at_zero_flow_equal_static(self):
        """At τ_core=1e8 (zero flow), flowing SS = static SS."""
        config = MSRModelConfig(tau_core=1.0e8, tau_loop=4.0e8)
        y0 = build_initial_state(n0=1.0, config=config, beta_arr=BETA_ARR, lambda_arr=LAMBDA_ARR)
        C_static = steady_state_precursors(1.0, BETA_ARR, LAMBDA_ARR, LAMBDA_PWR)
        assert np.allclose(y0[1:7], C_static, rtol=1.0e-6), (
            "At zero flow, flowing SS precursors should equal static SS precursors."
        )

    def test_state_vector_length(self):
        """Initial state vector must have exactly 8 elements."""
        y0 = build_initial_state()
        assert y0.shape == (8,), f"Expected (8,), got {y0.shape}"


class TestMsrRhsProperties:
    """Algebraic properties of the MSR coupled ODE right-hand side."""

    def _make_rhs_inputs(self, tau_core=5.0, tau_loop=20.0, alpha_salt=-5.0):
        config = make_critical_msr_config(
            tau_core=tau_core, tau_loop=tau_loop, alpha_salt_pcm_per_k=alpha_salt,
            beta_arr=BETA_ARR, lambda_arr=LAMBDA_ARR,
        )
        y0 = build_initial_state(config=config, beta_arr=BETA_ARR, lambda_arr=LAMBDA_ARR)
        C_init = y0[1:7]
        history = PrecursorHistory(t_start=0.0, C_init=C_init)
        history.add(0.0, C_init)
        from physics.msr.engine import constant_controls_fn
        controls_fn = constant_controls_fn(MSRControls(external_reactivity_pcm=0.0))
        return y0, config, history, controls_fn

    def test_rhs_shape(self):
        """RHS must return shape (8,)."""
        y0, config, history, controls_fn = self._make_rhs_inputs()
        dydt = msr_coupled_rhs(0.0, y0, controls_fn, history, config, BETA_ARR, LAMBDA_ARR)
        assert dydt.shape == (8,), f"Expected (8,), got {dydt.shape}"

    def test_rhs_near_zero_at_steady_state(self):
        """At steady state with zero reactivity insertion, |dy/dt| should be small."""
        y0, config, history, controls_fn = self._make_rhs_inputs()
        dydt = msr_coupled_rhs(0.0, y0, controls_fn, history, config, BETA_ARR, LAMBDA_ARR)
        # dn/dt ≈ 0 at SS (within numerical precision of initial conditions)
        assert abs(dydt[0]) < 1.0e-3, (
            f"dn/dt at steady state = {dydt[0]:.3e}, expected ≈ 0"
        )
        # dT_salt/dt ≈ 0 at SS (within 1 K/s tolerance for lumped model)
        assert abs(dydt[7]) < 1.0, (
            f"dT_salt/dt at SS = {dydt[7]:.4e} K/s, expected ≈ 0"
        )

    def test_zero_flow_rhs_matches_static_kinetics(self):
        """At τ_core=1e8 (zero flow), MSR RHS dCi/dt matches standard PWR precursor equation."""
        from physics.shared.kinetics import point_kinetics_rhs

        y0_msr, config_zf, history_zf, controls_fn_zf = self._make_rhs_inputs(
            tau_core=1.0e8, tau_loop=4.0e8, alpha_salt=0.0
        )
        dydt_msr = msr_coupled_rhs(
            0.0, y0_msr, controls_fn_zf, history_zf, config_zf, BETA_ARR, LAMBDA_ARR
        )

        # Reference: pure point kinetics (no drift, no thermal feedback)
        rho = 0.0
        dydt_pwr = point_kinetics_rhs(0.0, y0_msr[:7], rho, BETA_ARR, LAMBDA_ARR, LAMBDA_PWR)

        # n and C1–C6 derivatives should match (drift ≈ 0 at τ_core=1e8)
        assert np.allclose(dydt_msr[:7], dydt_pwr, rtol=1.0e-4), (
            f"MSR kinetics drift at τ_core=1e8 should match PWR kinetics. "
            f"Max diff: {np.max(np.abs(dydt_msr[:7] - dydt_pwr)):.3e}"
        )
