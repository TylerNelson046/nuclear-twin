"""Tests for the full 11-state PWR ODE system — pwr_11_state_system.

Covers:
    1. Params array structure and output shape.
    2. Steady-state correctness: all 11 derivatives ≈ 0 at the validated
       critical equilibrium (verified block by block).
    3. Transient robustness: 100-second window after a +50 pcm rod withdrawal.
       The stiff Radau solver must converge; all physical states must remain
       non-negative; Doppler + moderator feedback must limit the excursion.

State vector layout (ARCHITECTURE.md §4.1, SPEC §4.6.2):
    y[0]    n         neutron population (normalised)
    y[1:7]  C₁–C₆    delayed-neutron precursor concentrations
    y[7]    T_fuel    lumped fuel temperature (K)
    y[8]    T_cool    lumped coolant temperature (K)
    y[9]    I         I-135 number density (atoms/cm³)
    y[10]   X         Xe-135 number density (atoms/cm³)

ODE solver: scipy Radau, rtol=1e-6, atol=1e-9 (CLAUDE.md §Architecture Rules).
"""

from __future__ import annotations

import numpy as np
import pytest

from physics.pwr.engine import (
    PARAMS_LEN,
    PWRControls,
    PWRModelConfig,
    build_initial_state,
    critical_base_reactivity_pcm,
    integrate_pwr_11state,
    make_params_array,
    pwr_11_state_system,
)
from physics.pwr.reactivity import ReferenceState
from physics.pwr.thermal_hydraulics import T_COOL_NOM, T_FUEL_NOM, T_IN_NOM


# ---------------------------------------------------------------------------
# Module-scoped fixture: validated critical 11-state equilibrium
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def critical_setup():
    """Build the full-power critical 11-state equilibrium.

    Uses the same strategy as test_pwr_coupled_engine.py: start from the
    analytical steady-state ICs, compute the base reactivity that exactly
    cancels equilibrium xenon worth, and return both the state vector and
    the flat params array ready for pwr_11_state_system.

    Returns:
        (y0, params, reference_state, controls, config)
    """
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
    return y0, params, reference_state, controls, config


# ---------------------------------------------------------------------------
# 1. Params array structure and output shape
# ---------------------------------------------------------------------------

class TestParamsArray:
    def test_length_is_params_len(self, critical_setup):
        """make_params_array returns exactly PARAMS_LEN float64 elements."""
        _, params, _, _, _ = critical_setup
        assert params.shape == (PARAMS_LEN,)

    def test_dtype_is_float64(self, critical_setup):
        """All elements of the params vector are float64."""
        _, params, _, _, _ = critical_setup
        assert params.dtype == np.float64

    def test_output_shape_is_11(self, critical_setup):
        """pwr_11_state_system returns a (11,) array."""
        y0, params, _, _, _ = critical_setup
        dydt = pwr_11_state_system(0.0, y0, params)
        assert dydt.shape == (11,)

    def test_output_dtype_is_float64(self, critical_setup):
        """Derivative array is float64 to match the ODE state vector."""
        y0, params, _, _, _ = critical_setup
        dydt = pwr_11_state_system(0.0, y0, params)
        assert dydt.dtype == np.float64


# ---------------------------------------------------------------------------
# 2. Steady-state: all 11 derivatives ≈ 0 at the critical equilibrium
#
# Physical basis: the analytical steady-state ICs (SPEC §4.6.2) exactly
# satisfy each sub-equation; numerical error comes only from floating-point
# cancellation of large nearly-equal terms.
#
# Tolerances justified:
#   Kinetics [0:7]  atol=1e-10 — ρ=0 at equilibrium, subtracted terms exact
#   Thermal  [7:9]  atol=1e-8  — R_FC and M_DOT_NOM derived from exact SS
#   Xenon   [9:11]  atol=1e-3  — terms ~5e11 atoms/cm³/s; FP error ~5e-4
# ---------------------------------------------------------------------------

class TestSteadyStateDerivatives:
    def test_kinetics_block_zero(self, critical_setup):
        """At the critical equilibrium dy/dt[:7] ≈ 0 (kinetics block, atol 1e-10)."""
        y0, params, _, _, _ = critical_setup
        dydt = pwr_11_state_system(0.0, y0, params)
        np.testing.assert_allclose(
            dydt[:7], 0.0, atol=1e-10,
            err_msg="Kinetics RHS non-zero at analytically constructed steady state",
        )

    def test_neutron_population_derivative_zero(self, critical_setup):
        """dn/dt must be zero at the critical equilibrium (Eq. 1 check)."""
        y0, params, _, _, _ = critical_setup
        dydt = pwr_11_state_system(0.0, y0, params)
        assert abs(dydt[0]) < 1e-10, f"dn/dt = {dydt[0]:.3e} at steady state"

    def test_all_precursor_derivatives_zero(self, critical_setup):
        """dCᵢ/dt = 0 for all 6 groups at steady state (Eq. 2 check)."""
        y0, params, _, _, _ = critical_setup
        dydt = pwr_11_state_system(0.0, y0, params)
        for i in range(6):
            assert abs(dydt[i + 1]) < 1e-10, (
                f"dC{i+1}/dt = {dydt[i+1]:.3e} at steady state"
            )

    def test_thermal_block_zero(self, critical_setup):
        """At the critical equilibrium dy/dt[7:9] ≈ 0 K/s (thermal block, atol 1e-8)."""
        y0, params, _, _, _ = critical_setup
        dydt = pwr_11_state_system(0.0, y0, params)
        np.testing.assert_allclose(
            dydt[7:9], 0.0, atol=1e-8,
            err_msg="Thermal block (fuel/coolant temperatures) non-zero at steady state",
        )

    def test_fuel_temperature_derivative_zero(self, critical_setup):
        """dT_fuel/dt = 0 at steady state — heat in equals heat out (Eq. 9)."""
        y0, params, _, _, _ = critical_setup
        dydt = pwr_11_state_system(0.0, y0, params)
        assert abs(dydt[7]) < 1e-8, f"dT_fuel/dt = {dydt[7]:.3e} K/s at steady state"

    def test_coolant_temperature_derivative_zero(self, critical_setup):
        """dT_cool/dt = 0 at steady state — advection balances conduction (Eq. 10)."""
        y0, params, _, _, _ = critical_setup
        dydt = pwr_11_state_system(0.0, y0, params)
        assert abs(dydt[8]) < 1e-8, f"dT_cool/dt = {dydt[8]:.3e} K/s at steady state"

    def test_xenon_block_zero(self, critical_setup):
        """At the critical equilibrium dy/dt[9:11] ≈ 0 atoms/cm³/s (xenon block)."""
        y0, params, _, _, _ = critical_setup
        dydt = pwr_11_state_system(0.0, y0, params)
        np.testing.assert_allclose(
            dydt[9:11], 0.0, atol=1e-3,
            err_msg="Xenon/Iodine RHS non-zero at equilibrium flux",
        )

    def test_all_11_derivatives_near_zero(self, critical_setup):
        """Composite steady-state test: all 11 dy/dt ≈ 0 with block-appropriate tolerances."""
        y0, params, _, _, _ = critical_setup
        dydt = pwr_11_state_system(0.0, y0, params)

        assert dydt.shape == (11,), "RHS must return 11 derivatives"
        np.testing.assert_allclose(dydt[:7],   0.0, atol=1e-10,
                                   err_msg="Kinetics block failure")
        np.testing.assert_allclose(dydt[7:9],  0.0, atol=1e-8,
                                   err_msg="Thermal block failure")
        np.testing.assert_allclose(dydt[9:11], 0.0, atol=1e-3,
                                   err_msg="Xenon block failure")


# ---------------------------------------------------------------------------
# 3. Transient: 100-second window after +50 pcm rod withdrawal
#
# The stiff Radau solver must integrate successfully over the coupled timescales
# (Λ ≈ 1e-5 s kinetics through ~1e4 s xenon dynamics).  All physical states
# must remain non-negative and bounded.  Doppler + moderator feedback must
# visibly suppress the power excursion by t = 100 s.
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def transient_result(critical_setup):
    """Run a 100-second transient after +50 pcm rod withdrawal.

    Uses the same baseline y0 as the steady-state tests but applies a +50 pcm
    rod reactivity insertion.  Returns the full scipy OdeResult.
    """
    y0, _, reference_state, _, config = critical_setup
    perturbed_controls = PWRControls(
        rod_reactivity_pcm=50.0,
        boron_ppm=0.0,
        coolant_flow_fraction=1.0,
        inlet_temperature_k=T_IN_NOM,
    )
    params = make_params_array(perturbed_controls, reference_state, config)
    t_eval = np.linspace(0.0, 100.0, 201)
    return integrate_pwr_11state(y0, (0.0, 100.0), params, t_eval=t_eval)


class TestTransient100s:
    def test_solver_succeeds(self, transient_result):
        """Radau must converge across the 100-second stiff transient."""
        assert transient_result.success, (
            f"ODE solver failed: {transient_result.message}"
        )

    def test_output_shape(self, transient_result):
        """ODE output must have 11 state rows and the requested 201 time steps."""
        assert transient_result.y.shape == (11, 201)

    def test_integration_reaches_t_end(self, transient_result):
        """Solver must integrate to t = 100 s without early termination."""
        assert transient_result.t[-1] == pytest.approx(100.0, abs=0.01)

    def test_no_negative_states(self, transient_result):
        """All physical states n, Cᵢ, T_fuel, T_cool, I, X must remain ≥ 0 (SPEC §4.6.5)."""
        assert np.all(transient_result.y >= 0.0), (
            "One or more state variables went negative during the transient. "
            f"Negative at (state, step): {list(zip(*np.where(transient_result.y < 0.0)))}"
        )

    def test_power_excursion_occurs(self, transient_result):
        """Power must rise above its initial value — positive reactivity insertion."""
        power = transient_result.y[0]
        assert power.max() > power[0] * 1.01, (
            f"No power excursion detected (max/initial = {power.max()/power[0]:.4f}). "
            "The +50 pcm insertion should raise power."
        )

    def test_power_excursion_self_limited(self, transient_result):
        """Power at t = 100 s must be below the peak — feedback suppresses the excursion."""
        power = transient_result.y[0]
        assert power[-1] < power.max() * 0.99, (
            f"Power at t=100s ({power[-1]:.4f}) is at or above the peak ({power.max():.4f}). "
            "Doppler/moderator feedback is not suppressing the excursion."
        )

    def test_final_power_bounded_by_feedback(self, transient_result):
        """Final power must be bounded well below prompt-critical levels.

        +50 pcm << beta_eff ≈ 650 pcm, so the excursion is deeply sub-prompt.
        The settled power must be within a few percent of the initial value
        once Doppler and moderator feedbacks have cancelled the insertion.
        """
        power = transient_result.y[0]
        # From test_pwr_coupled_engine.py the settled power after +50 pcm is < 1.05;
        # allow slightly more headroom here since we stop at 100 s rather than 200 s.
        assert power[-1] < power[0] * 1.08, (
            f"Final power {power[-1]:.4f} exceeds 8% above initial {power[0]:.4f}. "
            "Feedback coefficients may not be engaging correctly."
        )

    def test_fuel_temperature_rises(self, transient_result):
        """Fuel temperature must rise above the reference — Doppler feedback is active."""
        T_fuel = transient_result.y[7]
        assert T_fuel[-1] > T_FUEL_NOM, (
            f"Fuel temperature at t=100s ({T_fuel[-1]:.1f} K) did not exceed "
            f"reference ({T_FUEL_NOM} K). Doppler feedback may be disconnected."
        )

    def test_coolant_temperature_rises(self, transient_result):
        """Coolant temperature must rise above the reference — moderator feedback is active."""
        T_cool = transient_result.y[8]
        assert T_cool[-1] > T_COOL_NOM, (
            f"Coolant temperature at t=100s ({T_cool[-1]:.1f} K) did not exceed "
            f"reference ({T_COOL_NOM} K). Moderator feedback may be disconnected."
        )

    def test_iodine_non_negative_and_bounded(self, transient_result):
        """I-135 must remain non-negative and must not diverge in 100 seconds.

        I-135 half-life is 6.7 hours; over 100 seconds the concentration changes
        by at most ~0.04%, so relative change must be < 5%.
        """
        I = transient_result.y[9]
        assert np.all(I >= 0.0), "I-135 went negative"
        I0 = I[0]
        if I0 > 0.0:
            max_rel_change = float(np.max(np.abs(I - I0))) / I0
            assert max_rel_change < 0.05, (
                f"I-135 changed by {max_rel_change:.1%} in 100 s "
                f"(t½ ≈ 6.7 hr; < 5% expected)"
            )

    def test_xenon_non_negative_and_bounded(self, transient_result):
        """Xe-135 must remain non-negative and must not diverge in 100 seconds.

        Xe-135 half-life is 9.2 hours; over 100 seconds the concentration changes
        by at most ~0.03%, so relative change must be < 5%.
        """
        X = transient_result.y[10]
        assert np.all(X >= 0.0), "Xe-135 went negative"
        X0 = X[0]
        if X0 > 0.0:
            max_rel_change = float(np.max(np.abs(X - X0))) / X0
            assert max_rel_change < 0.05, (
                f"Xe-135 changed by {max_rel_change:.1%} in 100 s "
                f"(t½ ≈ 9.2 hr; < 5% expected)"
            )

    def test_precursors_non_negative(self, transient_result):
        """All 6 delayed-neutron precursor groups must remain non-negative."""
        C_all = transient_result.y[1:7]
        assert np.all(C_all >= 0.0), (
            "One or more precursor groups went negative: "
            f"{list(zip(*np.where(C_all < 0.0)))}"
        )
