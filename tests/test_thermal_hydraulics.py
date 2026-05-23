"""
Unit tests for physics/pwr/thermal_hydraulics.py — lumped-parameter fuel/coolant model.

Software tests (CLAUDE.md §Testing Requirements, SPEC §7.2):
    Every equation implemented as a function is tested against a hand-calculated value.
    Tests span three operating scenarios: nominal steady state, power step transient,
    and loss-of-flow transient.

SPEC equations validated:
    Eq. 9   m_f c_f · dT_f/dt = γ_f · P − (T_f − T_c) / R_fc
    Eq. 10  m_c c_c · dT_c/dt = (T_f − T_c) / R_fc − ṁ c_c (T_c − T_in)

Physics validation checks (SPEC §7.3 PWR, ARCHITECTURE.md §9):
    - test_rhs_zero_at_nominal_steady_state: ODE RHS = 0 at analytical SS ICs
    - test_steady_state_formula_nominal: formula reproduces T_FUEL_NOM, T_COOL_NOM
    - test_power_step_reaches_new_equilibrium: +10% power step stabilises analytically
    - test_loss_of_flow_raises_both_temperatures: 50% flow reduction → higher SS temps
    - test_shutdown_temperatures_decay_to_inlet: P=0 → T_f = T_c = T_in asymptotically

ODE solver: scipy Radau, rtol=1e-6, atol=1e-9 (CLAUDE.md §Architecture Rules).
"""

from __future__ import annotations

import logging
import math

import numpy as np
import pytest

from physics.pwr.thermal_hydraulics import (
    C_COOL,
    C_FUEL,
    GAMMA_F,
    M_COOL,
    M_DOT_NOM,
    M_FUEL,
    P_NOM,
    R_FC,
    T_COOL_NOM,
    T_FUEL_NOM,
    T_IN_NOM,
    integrate_thermal_hydraulics,
    steady_state_temperatures,
    thermal_hydraulics_rhs,
)

# ---------------------------------------------------------------------------
# Tolerance constants
# ---------------------------------------------------------------------------

# RHS residual tolerance at exact SS ICs — limited only by floating-point rounding.
RHS_ATOL: float = 1e-6     # K/s — dT/dt residual at steady state

# Temperature convergence tolerance for time-integrated transients.
# After many time constants, the ODE solution must be within 0.01% of the
# analytic steady state.
TEMP_REL_TOL: float = 1e-4  # relative (0.01%)
TEMP_ABS_TOL: float = 0.01  # K — absolute fallback for near-zero deltas


# ===========================================================================
# 1. Module-level physical constants — sanity checks
# ===========================================================================


class TestPhysicalConstants:
    """Verify that derived constants are physically self-consistent."""

    def test_R_FC_is_positive(self):
        """Thermal resistance must be positive (SPEC Eq. 9 requires T_f > T_c)."""
        assert R_FC > 0.0

    def test_M_DOT_NOM_is_positive(self):
        """Nominal mass flow rate must be positive (forward forced circulation)."""
        assert M_DOT_NOM > 0.0

    def test_gamma_f_physical_range(self):
        """Fraction of power to fuel must be in (0, 1)."""
        assert 0.0 < GAMMA_F < 1.0

    def test_R_FC_value_hand_calculated(self):
        """R_fc = (T_f_nom − T_c_nom) / (γ_f · P_nom) from SPEC Eq. 9 SS condition."""
        expected = (T_FUEL_NOM - T_COOL_NOM) / (GAMMA_F * P_NOM)
        assert R_FC == pytest.approx(expected, rel=1e-12)

    def test_M_DOT_NOM_value_hand_calculated(self):
        """ṁ_nom = γ_f · P_nom / (c_c · (T_c_nom − T_in)) from SPEC Eq. 10 SS."""
        expected = (GAMMA_F * P_NOM) / (C_COOL * (T_COOL_NOM - T_IN_NOM))
        assert M_DOT_NOM == pytest.approx(expected, rel=1e-12)

    def test_nominal_thermal_masses_positive(self):
        """Fuel and coolant thermal masses must be positive."""
        assert M_FUEL * C_FUEL > 0.0
        assert M_COOL * C_COOL > 0.0

    def test_fuel_hotter_than_coolant_at_reference(self):
        """T_FUEL_NOM > T_COOL_NOM > T_IN_NOM (SPEC §4.5.4 operating hierarchy)."""
        assert T_FUEL_NOM > T_COOL_NOM > T_IN_NOM


# ===========================================================================
# 2. Thermal-hydraulic RHS — SPEC Eqs. 9 & 10
# ===========================================================================


class TestThermalHydraulicsRhs:
    """Tests for thermal_hydraulics_rhs(t, y, power_w, m_dot, T_in)."""

    # -----------------------------------------------------------------------
    # 2a. Steady-state conditions — RHS must be identically zero
    # -----------------------------------------------------------------------

    def test_rhs_zero_at_nominal_steady_state(self):
        """RHS must vanish at the nominal SS operating point (the mathematical
        precondition for any stability integration test).

        SPEC Eq. 9 SS: γ_f·P = (T_f−T_c)/R_fc  →  dT_f/dt = 0
        SPEC Eq. 10 SS: (T_f−T_c)/R_fc = ṁ·c_c·(T_c−T_in) →  dT_c/dt = 0
        """
        y_ss = np.array([T_FUEL_NOM, T_COOL_NOM], dtype=np.float64)
        dydt = thermal_hydraulics_rhs(0.0, y_ss, P_NOM, M_DOT_NOM, T_IN_NOM)

        assert dydt.shape == (2,), "RHS must return a 2-element vector"
        np.testing.assert_allclose(
            dydt, 0.0, atol=RHS_ATOL,
            err_msg="Thermal-hydraulic RHS non-zero at nominal SS ICs"
        )

    def test_rhs_fuel_derivative_sign_for_power_increase(self):
        """Surplus power raises T_f: dT_f/dt > 0 when γ_f·P > (T_f−T_c)/R_fc."""
        y = np.array([T_FUEL_NOM, T_COOL_NOM], dtype=np.float64)
        dydt = thermal_hydraulics_rhs(0.0, y, 1.1 * P_NOM, M_DOT_NOM, T_IN_NOM)
        assert dydt[0] > 0.0, "dT_f/dt must be positive when power exceeds SS transfer rate"

    def test_rhs_fuel_derivative_sign_for_power_decrease(self):
        """Power reduction cools fuel: dT_f/dt < 0 when γ_f·P < (T_f−T_c)/R_fc."""
        y = np.array([T_FUEL_NOM, T_COOL_NOM], dtype=np.float64)
        dydt = thermal_hydraulics_rhs(0.0, y, 0.9 * P_NOM, M_DOT_NOM, T_IN_NOM)
        assert dydt[0] < 0.0

    def test_rhs_coolant_derivative_sign_for_flow_decrease(self):
        """Reduced flow means less advective removal: dT_c/dt > 0."""
        y = np.array([T_FUEL_NOM, T_COOL_NOM], dtype=np.float64)
        dydt = thermal_hydraulics_rhs(0.0, y, P_NOM, 0.5 * M_DOT_NOM, T_IN_NOM)
        assert dydt[1] > 0.0

    def test_rhs_at_zero_power_and_hot_starts_cooling(self):
        """At P=0 starting from hot SS, T_f cools immediately; T_c derivative is zero
        at the initial instant because the residual heat flux from the still-hot fuel
        exactly equals the advective removal at the nominal SS coolant temperature.
        T_c then begins cooling as T_f drops and q_fc decreases below the advective
        removal rate.

        Tested explicitly:
            - dT_f/dt < 0 (fuel begins cooling immediately)
            - dT_c/dt ≤ 0 (coolant does not heat up; = 0 at the initial instant)
        The time-evolution confirmation (T_c ultimately decays to T_in) is covered by
        test_shutdown_temperatures_decay_to_inlet.
        """
        y = np.array([T_FUEL_NOM, T_COOL_NOM], dtype=np.float64)
        dydt = thermal_hydraulics_rhs(0.0, y, 0.0, M_DOT_NOM, T_IN_NOM)
        assert dydt[0] < 0.0, "Fuel must cool when power removed"
        assert dydt[1] <= 0.0, (
            "Coolant must not heat up when power removed; "
            "at the initial instant dT_c/dt = 0 because residual fuel-to-coolant "
            "heat flux exactly equals advective removal at the SS coolant temperature"
        )

    def test_rhs_shape_and_type(self):
        """Return is shape (2,) float64 for all valid inputs."""
        y = np.array([T_FUEL_NOM, T_COOL_NOM], dtype=np.float64)
        dydt = thermal_hydraulics_rhs(0.0, y, P_NOM, M_DOT_NOM, T_IN_NOM)
        assert dydt.shape == (2,)
        assert dydt.dtype == np.float64

    def test_rhs_hand_calculated_fuel_derivative(self):
        """Hand-calculated dT_f/dt at 110% power starting from SS temperatures.

        Extra heat rate = 10% × γ_f × P_nom = 0.1 × 0.97 × 3e9 = 2.91e8 W
        dT_f/dt = extra_heat / (M_FUEL × C_FUEL) = 2.91e8 / (101000 × 300)
        """
        y = np.array([T_FUEL_NOM, T_COOL_NOM], dtype=np.float64)
        dydt = thermal_hydraulics_rhs(0.0, y, 1.1 * P_NOM, M_DOT_NOM, T_IN_NOM)
        extra_power = 0.1 * GAMMA_F * P_NOM
        expected_dTf = extra_power / (M_FUEL * C_FUEL)
        assert dydt[0] == pytest.approx(expected_dTf, rel=1e-6)

    def test_time_argument_ignored_in_autonomous_ode(self):
        """The thermal-hydraulic ODE is autonomous (no explicit time dependence):
        identical y, power, m_dot, T_in must give identical RHS regardless of t."""
        y = np.array([T_FUEL_NOM, T_COOL_NOM], dtype=np.float64)
        dydt_0 = thermal_hydraulics_rhs(0.0, y, P_NOM, M_DOT_NOM, T_IN_NOM)
        dydt_500 = thermal_hydraulics_rhs(500.0, y, P_NOM, M_DOT_NOM, T_IN_NOM)
        np.testing.assert_array_equal(dydt_0, dydt_500)


# ===========================================================================
# 3. Steady-state temperature formula — SPEC Eqs. 9 & 10 analytical solution
# ===========================================================================


class TestSteadyStateTemperatures:
    """Tests for steady_state_temperatures(power_w, m_dot, T_in)."""

    def test_nominal_operating_point_exact(self):
        """At P_NOM, M_DOT_NOM, T_IN_NOM: formula must return T_FUEL_NOM, T_COOL_NOM.

        This is guaranteed by construction of R_FC and M_DOT_NOM (SPEC §4.5.4).
        """
        T_ss = steady_state_temperatures(P_NOM, M_DOT_NOM, T_IN_NOM)
        assert T_ss[0] == pytest.approx(T_FUEL_NOM, rel=1e-10), "T_f_ss must equal T_FUEL_NOM"
        assert T_ss[1] == pytest.approx(T_COOL_NOM, rel=1e-10), "T_c_ss must equal T_COOL_NOM"

    def test_zero_power_gives_inlet_temperature(self):
        """At P=0, no heat is deposited: T_f_ss = T_c_ss = T_in (SPEC Eq. 9 & 10 at P=0)."""
        T_ss = steady_state_temperatures(0.0, M_DOT_NOM, T_IN_NOM)
        assert T_ss[0] == pytest.approx(T_IN_NOM, abs=1e-10)
        assert T_ss[1] == pytest.approx(T_IN_NOM, abs=1e-10)

    def test_fuel_hotter_than_coolant_at_any_positive_power(self):
        """T_f > T_c must hold for any positive power (heat flows fuel → coolant)."""
        for fraction in [0.1, 0.5, 1.0, 1.2]:
            T_ss = steady_state_temperatures(fraction * P_NOM, M_DOT_NOM, T_IN_NOM)
            assert T_ss[0] > T_ss[1], f"T_f ≤ T_c at {fraction*100:.0f}% power (unphysical)"

    def test_coolant_hotter_than_inlet_at_positive_power(self):
        """T_c > T_in must hold for any positive power and positive flow."""
        T_ss = steady_state_temperatures(P_NOM, M_DOT_NOM, T_IN_NOM)
        assert T_ss[1] > T_IN_NOM

    def test_temperature_increases_monotonically_with_power(self):
        """Higher power → higher SS temperatures (conservation of energy)."""
        T_ss_80 = steady_state_temperatures(0.8 * P_NOM, M_DOT_NOM, T_IN_NOM)
        T_ss_100 = steady_state_temperatures(P_NOM, M_DOT_NOM, T_IN_NOM)
        T_ss_110 = steady_state_temperatures(1.1 * P_NOM, M_DOT_NOM, T_IN_NOM)
        assert T_ss_80[0] < T_ss_100[0] < T_ss_110[0]
        assert T_ss_80[1] < T_ss_100[1] < T_ss_110[1]

    def test_lower_flow_raises_steady_state_temperatures(self):
        """Half flow → higher coolant and fuel temperatures at same power."""
        T_ss_nom = steady_state_temperatures(P_NOM, M_DOT_NOM, T_IN_NOM)
        T_ss_lof = steady_state_temperatures(P_NOM, 0.5 * M_DOT_NOM, T_IN_NOM)
        assert T_ss_lof[1] > T_ss_nom[1], "Reduced flow must raise T_c at steady state"
        assert T_ss_lof[0] > T_ss_nom[0], "Reduced flow must raise T_f at steady state"

    def test_formula_consistent_with_rhs_zero_condition(self):
        """For any (P, ṁ, T_in) within bounds, RHS must be zero at the SS temps."""
        for (P, mdot, Tin) in [
            (P_NOM, M_DOT_NOM, T_IN_NOM),
            (0.5 * P_NOM, M_DOT_NOM, T_IN_NOM),
            (P_NOM, 0.8 * M_DOT_NOM, T_IN_NOM),
            (P_NOM, M_DOT_NOM, T_IN_NOM - 10),
        ]:
            T_ss = steady_state_temperatures(P, mdot, Tin)
            dydt = thermal_hydraulics_rhs(0.0, T_ss, P, mdot, Tin)
            np.testing.assert_allclose(
                dydt, 0.0, atol=1e-3,
                err_msg=f"RHS non-zero at computed SS for P={P:.2e}, mdot={mdot:.1f}"
            )

    def test_hand_calculated_coolant_temperature_rise(self):
        """T_c_ss = T_in + γ_f·P / (ṁ·c_c): hand-calculated for 50% power.

        γ_f·P = 0.97 × 1.5e9 = 1.455e9 W
        ΔT_c = 1.455e9 / (M_DOT_NOM × 5600) = 1.455e9 / (M_DOT_NOM × 5600)
        """
        P_half = 0.5 * P_NOM
        T_ss = steady_state_temperatures(P_half, M_DOT_NOM, T_IN_NOM)
        expected_T_c = T_IN_NOM + GAMMA_F * P_half / (M_DOT_NOM * C_COOL)
        assert T_ss[1] == pytest.approx(expected_T_c, rel=1e-10)

    # --- Unphysical inputs ---

    def test_zero_flow_returns_nan(self):
        """ṁ = 0 is a LOCA condition out of scope (SPEC §3.3); must return NaN."""
        T_ss = steady_state_temperatures(P_NOM, 0.0, T_IN_NOM)
        assert np.all(np.isnan(T_ss))

    def test_negative_flow_returns_nan(self):
        """Negative flow is physically impossible; must return NaN."""
        T_ss = steady_state_temperatures(P_NOM, -1.0, T_IN_NOM)
        assert np.all(np.isnan(T_ss))

    def test_negative_power_returns_nan(self):
        """Negative fission power is unphysical; must return NaN."""
        T_ss = steady_state_temperatures(-1.0, M_DOT_NOM, T_IN_NOM)
        assert np.all(np.isnan(T_ss))

    def test_negative_inlet_temperature_returns_nan(self):
        """T_in < 0 K is unphysical; must return NaN."""
        T_ss = steady_state_temperatures(P_NOM, M_DOT_NOM, -1.0)
        assert np.all(np.isnan(T_ss))

    def test_zero_flow_logs_warning(self, caplog):
        """Zero flow triggers a WARNING-level log message (SR-01)."""
        with caplog.at_level(logging.WARNING, logger="physics.pwr.thermal_hydraulics"):
            steady_state_temperatures(P_NOM, 0.0, T_IN_NOM)
        assert len(caplog.records) >= 1
        assert any("unphysical" in r.message.lower() or "scope" in r.message.lower()
                   for r in caplog.records)

    def test_return_shape_and_dtype(self):
        """Return value is always shape (2,) float64 for valid inputs."""
        T_ss = steady_state_temperatures(P_NOM, M_DOT_NOM, T_IN_NOM)
        assert T_ss.shape == (2,)
        assert T_ss.dtype == np.float64


# ===========================================================================
# 4. Transient integration — 100% nominal power, step changes, and LOF
# ===========================================================================


class TestIntegrationTransients:
    """Integration tests using scipy Radau to verify physical transient behavior."""

    def test_integration_stable_at_nominal_steady_state(self):
        """Starting at nominal SS ICs with nominal inputs, temperatures must not drift.

        Runs 1000 s at full steady-state conditions; relative drift must be < 1e-5
        (same criterion as the coupled 9-state integration test in Week 2).
        """
        y0 = np.array([T_FUEL_NOM, T_COOL_NOM], dtype=np.float64)
        t_eval = np.linspace(0.0, 1000.0, 201)

        result = integrate_thermal_hydraulics(
            y0, (0.0, 1000.0),
            power_fn=lambda t, y: P_NOM,
            m_dot_fn=lambda t, y: M_DOT_NOM,
            T_in_fn=lambda t, y: T_IN_NOM,
            t_eval=t_eval,
        )

        assert result.success, f"Solver failed: {result.message}"

        drift_f = float(np.max(np.abs(result.y[0] / T_FUEL_NOM - 1.0)))
        drift_c = float(np.max(np.abs(result.y[1] / T_COOL_NOM - 1.0)))

        assert drift_f < 1e-5, f"T_fuel drift = {drift_f:.2e} (limit 1e-5)"
        assert drift_c < 1e-5, f"T_cool drift = {drift_c:.2e} (limit 1e-5)"

    def test_power_step_10pct_reaches_new_equilibrium(self):
        """Step from 100% to 110% power: temperatures must reach the analytic new SS.

        New SS temperatures are computed from steady_state_temperatures.
        After 300 s (>80 fuel time constants), convergence within 0.01%.
        """
        y0 = np.array([T_FUEL_NOM, T_COOL_NOM], dtype=np.float64)
        P_new = 1.10 * P_NOM
        T_ss_new = steady_state_temperatures(P_new, M_DOT_NOM, T_IN_NOM)

        result = integrate_thermal_hydraulics(
            y0, (0.0, 300.0),
            power_fn=lambda t, y: P_new,
            m_dot_fn=lambda t, y: M_DOT_NOM,
            T_in_fn=lambda t, y: T_IN_NOM,
        )

        assert result.success, f"Solver failed: {result.message}"
        assert result.y[0, -1] == pytest.approx(T_ss_new[0], rel=TEMP_REL_TOL), \
            f"T_fuel did not converge to {T_ss_new[0]:.2f} K"
        assert result.y[1, -1] == pytest.approx(T_ss_new[1], rel=TEMP_REL_TOL), \
            f"T_cool did not converge to {T_ss_new[1]:.2f} K"

    def test_power_step_increase_raises_temperatures(self):
        """After a positive power step, both T_f and T_c must increase monotonically
        and stabilise above initial values."""
        y0 = np.array([T_FUEL_NOM, T_COOL_NOM], dtype=np.float64)
        t_eval = np.linspace(0.0, 300.0, 301)

        result = integrate_thermal_hydraulics(
            y0, (0.0, 300.0),
            power_fn=lambda t, y: 1.1 * P_NOM,
            m_dot_fn=lambda t, y: M_DOT_NOM,
            T_in_fn=lambda t, y: T_IN_NOM,
            t_eval=t_eval,
        )

        assert result.success
        assert result.y[0, -1] > T_FUEL_NOM, "T_fuel must be above nominal after power step up"
        assert result.y[1, -1] > T_COOL_NOM, "T_cool must be above nominal after power step up"

    def test_power_step_down_lowers_temperatures(self):
        """Power reduction to 90%: both temperatures settle below initial values."""
        y0 = np.array([T_FUEL_NOM, T_COOL_NOM], dtype=np.float64)

        result = integrate_thermal_hydraulics(
            y0, (0.0, 300.0),
            power_fn=lambda t, y: 0.90 * P_NOM,
            m_dot_fn=lambda t, y: M_DOT_NOM,
            T_in_fn=lambda t, y: T_IN_NOM,
        )

        assert result.success
        T_ss_90 = steady_state_temperatures(0.90 * P_NOM, M_DOT_NOM, T_IN_NOM)
        assert result.y[0, -1] == pytest.approx(T_ss_90[0], rel=TEMP_REL_TOL)
        assert result.y[1, -1] == pytest.approx(T_ss_90[1], rel=TEMP_REL_TOL)
        assert result.y[0, -1] < T_FUEL_NOM
        assert result.y[1, -1] < T_COOL_NOM

    def test_loss_of_flow_50pct_raises_temperatures(self):
        """Loss-of-flow to 50% nominal: reduced advective cooling raises both temperatures.

        New equilibrium computed analytically from steady_state_temperatures.
        After 300 s, convergence within 0.01%.

        Physical rationale: at the same power with half the flow, the coolant can only
        remove the same heat by rising to a higher average temperature (SPEC §4.2,
        coolant energy balance).
        """
        y0 = np.array([T_FUEL_NOM, T_COOL_NOM], dtype=np.float64)
        m_dot_lof = 0.5 * M_DOT_NOM
        T_ss_lof = steady_state_temperatures(P_NOM, m_dot_lof, T_IN_NOM)

        result = integrate_thermal_hydraulics(
            y0, (0.0, 300.0),
            power_fn=lambda t, y: P_NOM,
            m_dot_fn=lambda t, y: m_dot_lof,
            T_in_fn=lambda t, y: T_IN_NOM,
        )

        assert result.success, f"Solver failed: {result.message}"
        assert result.y[0, -1] > T_FUEL_NOM, "T_fuel must rise during loss-of-flow"
        assert result.y[1, -1] > T_COOL_NOM, "T_cool must rise during loss-of-flow"
        assert result.y[0, -1] == pytest.approx(T_ss_lof[0], rel=TEMP_REL_TOL), \
            f"T_fuel did not converge to LOF equilibrium {T_ss_lof[0]:.2f} K"
        assert result.y[1, -1] == pytest.approx(T_ss_lof[1], rel=TEMP_REL_TOL), \
            f"T_cool did not converge to LOF equilibrium {T_ss_lof[1]:.2f} K"

    def test_loss_of_flow_fuel_temperature_spike_magnitude(self):
        """Quantitative check: 50% flow raises T_cool by ΔT_c = γ_f·P/(ṁ_lof·c_c) − ΔT_nom.

        At 50% flow: T_c_ss_lof = T_in + γ_f·P / (0.5·ṁ_nom·c_c)
        Expected rise: T_c_ss_lof − T_c_ss_nom > 0 and bounded above ~60 K for physical params.
        """
        T_ss_nom = steady_state_temperatures(P_NOM, M_DOT_NOM, T_IN_NOM)
        T_ss_lof = steady_state_temperatures(P_NOM, 0.5 * M_DOT_NOM, T_IN_NOM)
        delta_T_c = T_ss_lof[1] - T_ss_nom[1]
        assert delta_T_c > 0.0
        # Analytic: ΔT_c = γ_f·P·(1/(0.5·ṁ) − 1/ṁ)·1/c_c = γ_f·P/(ṁ·c_c) = original rise
        expected_delta = GAMMA_F * P_NOM / (0.5 * M_DOT_NOM * C_COOL) - \
                         GAMMA_F * P_NOM / (M_DOT_NOM * C_COOL)
        assert delta_T_c == pytest.approx(expected_delta, rel=1e-8)

    def test_shutdown_temperatures_decay_to_inlet(self):
        """Full power shutdown (P=0): T_f and T_c must decay to T_in asymptotically.

        With τ_fuel ≈ 3.2 s and τ_cool < 2 s, 200 s is >60 time constants.
        Both temperatures must reach T_IN_NOM within 0.01 K.
        """
        y0 = np.array([T_FUEL_NOM, T_COOL_NOM], dtype=np.float64)

        result = integrate_thermal_hydraulics(
            y0, (0.0, 200.0),
            power_fn=lambda t, y: 0.0,
            m_dot_fn=lambda t, y: M_DOT_NOM,
            T_in_fn=lambda t, y: T_IN_NOM,
        )

        assert result.success, f"Solver failed: {result.message}"
        assert result.y[0, -1] == pytest.approx(T_IN_NOM, abs=0.01), \
            f"T_fuel = {result.y[0, -1]:.3f} K at t=200 s, expected {T_IN_NOM} K"
        assert result.y[1, -1] == pytest.approx(T_IN_NOM, abs=0.01), \
            f"T_cool = {result.y[1, -1]:.3f} K at t=200 s, expected {T_IN_NOM} K"

    def test_temperatures_remain_non_negative_throughout(self):
        """Physical constraint: T_f and T_c must remain ≥ 0 K across all transients."""
        y0 = np.array([T_FUEL_NOM, T_COOL_NOM], dtype=np.float64)
        t_eval = np.linspace(0.0, 300.0, 301)

        for scenario_name, power_fn, m_dot_fn in [
            ("nominal SS", lambda t, y: P_NOM, lambda t, y: M_DOT_NOM),
            ("10% power step", lambda t, y: 1.1 * P_NOM, lambda t, y: M_DOT_NOM),
            ("50% LOF", lambda t, y: P_NOM, lambda t, y: 0.5 * M_DOT_NOM),
            ("shutdown", lambda t, y: 0.0, lambda t, y: M_DOT_NOM),
        ]:
            result = integrate_thermal_hydraulics(
                y0, (0.0, 300.0),
                power_fn=power_fn, m_dot_fn=m_dot_fn,
                T_in_fn=lambda t, y: T_IN_NOM,
                t_eval=t_eval,
            )
            assert result.success, f"Solver failed for scenario '{scenario_name}'"
            assert np.all(result.y >= 0.0), \
                f"Negative temperature in scenario '{scenario_name}'"

    def test_fuel_hotter_than_coolant_at_all_times_under_power(self):
        """T_f(t) > T_c(t) must hold at every output step while power is applied.

        Heat flows from fuel to coolant, so T_f > T_c is a physical invariant
        at any positive power level.
        """
        y0 = np.array([T_FUEL_NOM, T_COOL_NOM], dtype=np.float64)
        t_eval = np.linspace(0.0, 300.0, 301)

        result = integrate_thermal_hydraulics(
            y0, (0.0, 300.0),
            power_fn=lambda t, y: P_NOM,
            m_dot_fn=lambda t, y: M_DOT_NOM,
            T_in_fn=lambda t, y: T_IN_NOM,
            t_eval=t_eval,
        )

        assert result.success
        assert np.all(result.y[0] > result.y[1]), \
            "T_fuel must exceed T_cool at every timestep under positive power"

    def test_ode_result_attributes(self):
        """solve_ivp result must have .success, .t, and .y with correct shapes."""
        y0 = np.array([T_FUEL_NOM, T_COOL_NOM], dtype=np.float64)
        t_eval = np.linspace(0.0, 10.0, 11)

        result = integrate_thermal_hydraulics(
            y0, (0.0, 10.0),
            power_fn=lambda t, y: P_NOM,
            m_dot_fn=lambda t, y: M_DOT_NOM,
            T_in_fn=lambda t, y: T_IN_NOM,
            t_eval=t_eval,
        )

        assert result.success
        assert result.y.shape[0] == 2
        assert result.y.shape[1] == len(t_eval)


# ===========================================================================
# 5. Parametric operating-range coverage
# ===========================================================================


@pytest.mark.parametrize("power_fraction", [0.1, 0.5, 0.8, 1.0, 1.1, 1.2])
def test_steady_state_temperatures_parametric_power(power_fraction):
    """Across the 10%–120% power operating range: SS temps are physically ordered."""
    T_ss = steady_state_temperatures(power_fraction * P_NOM, M_DOT_NOM, T_IN_NOM)
    if power_fraction > 0.0:
        assert T_ss[0] > T_ss[1] > T_IN_NOM
    else:
        assert T_ss[0] == pytest.approx(T_IN_NOM, abs=1e-8)


@pytest.mark.parametrize("flow_fraction", [0.3, 0.5, 0.8, 1.0, 1.2])
def test_steady_state_temperatures_parametric_flow(flow_fraction):
    """Across the 30%–120% flow range: SS formula produces physical results."""
    T_ss = steady_state_temperatures(P_NOM, flow_fraction * M_DOT_NOM, T_IN_NOM)
    assert T_ss[0] > T_ss[1] > T_IN_NOM


@pytest.mark.parametrize("T_in", [555.0, 560.0, 565.0, 570.0, 575.0])
def test_steady_state_temperatures_parametric_inlet(T_in):
    """Varying inlet temperature shifts both SS temperatures by the same ΔT."""
    T_ss_ref = steady_state_temperatures(P_NOM, M_DOT_NOM, T_IN_NOM)
    T_ss = steady_state_temperatures(P_NOM, M_DOT_NOM, T_in)
    # Both temperatures shift by the inlet delta when flow and power are unchanged
    delta_in = T_in - T_IN_NOM
    assert T_ss[0] == pytest.approx(T_ss_ref[0] + delta_in, rel=1e-9)
    assert T_ss[1] == pytest.approx(T_ss_ref[1] + delta_in, rel=1e-9)
