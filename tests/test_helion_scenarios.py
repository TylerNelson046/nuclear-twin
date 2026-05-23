"""Week 7 tests — Helion FRC scenarios and 2D ignition boundary sweep.

Covers:
    - Predefined Helion scenario registry and pulse integration
    - 2D ignition boundary sweep: shape, Q values, monotonicity
    - Q factor validation across sub-ignition and ignition regimes
    - Energy conservation at multiple operating points (SPEC §7.3)
    - Temperature scaling validation (SPEC Eq. 19, T ∝ Rᶜ^{2/3})
"""

from __future__ import annotations

import numpy as np
import pytest

from orchestrator.helion_scenarios import (
    HelionScenario,
    available_helion_scenarios,
    get_helion_scenario,
    helion_scenario_options,
)
from physics.helion.compression import apply_adiabatic_compression
from physics.helion.engine import integrate_helion_pulse
from physics.helion.parameters import HelionParameters
from physics.helion.sweep import IgnitionBoundaryResult, ignition_boundary_sweep


# ---------------------------------------------------------------------------
# Pulse simulation: predefined Helion scenarios
# ---------------------------------------------------------------------------


class TestHelionScenarios:
    """Verify predefined scenario registry, integration success, and physics."""

    def test_three_scenarios_registered(self) -> None:
        assert len(available_helion_scenarios()) == 3

    def test_all_expected_scenario_ids_present(self) -> None:
        scenarios = available_helion_scenarios()
        assert "baseline_pulse" in scenarios
        assert "high_compression" in scenarios
        assert "sub_ignition_diagnostic" in scenarios

    def test_get_scenario_returns_helion_scenario(self) -> None:
        assert isinstance(get_helion_scenario("baseline_pulse"), HelionScenario)

    def test_get_unknown_scenario_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="Unknown Helion scenario"):
            get_helion_scenario("does_not_exist")

    def test_scenario_options_returns_dash_compatible_dicts(self) -> None:
        options = helion_scenario_options()
        assert len(options) == 3
        for opt in options:
            assert "label" in opt
            assert "value" in opt

    def test_all_scenarios_integrate_successfully(self) -> None:
        for sid, scenario in available_helion_scenarios().items():
            result = integrate_helion_pulse(scenario.parameters, P_heat_w=scenario.P_heat_w)
            assert result.success, f"Scenario '{sid}' integration failed: {result.message}"

    def test_baseline_pulse_is_sub_ignition(self) -> None:
        """Baseline scenario (nominal operating point) must have Q < 1."""
        scenario = get_helion_scenario("baseline_pulse")
        result = integrate_helion_pulse(scenario.parameters)
        assert result.success
        assert result.Q < 1.0, f"Baseline expected Q < 1; got Q = {result.Q:.4f}"

    def test_sub_ignition_diagnostic_q_far_below_unity(self) -> None:
        """Low-density diagnostic scenario must be deeply sub-ignition (Q < 0.01)."""
        scenario = get_helion_scenario("sub_ignition_diagnostic")
        result = integrate_helion_pulse(scenario.parameters)
        assert result.success
        assert result.Q < 0.01, f"Sub-ignition diagnostic expected Q << 1; got Q = {result.Q:.4e}"

    def test_all_scenario_pulses_produce_nonnegative_W(self) -> None:
        """Plasma thermal energy must remain non-negative throughout every scenario."""
        for sid, scenario in available_helion_scenarios().items():
            result = integrate_helion_pulse(scenario.parameters, n_t_points=100)
            assert result.success
            assert np.all(result.W >= 0.0), f"W went negative in scenario '{sid}'"

    def test_all_scenario_pulses_produce_nonnegative_powers(self) -> None:
        """P_fusion, P_brem, and P_cond must be non-negative in all scenarios."""
        for sid, scenario in available_helion_scenarios().items():
            result = integrate_helion_pulse(scenario.parameters, n_t_points=100)
            assert result.success
            assert np.all(result.P_fusion >= 0.0), f"P_fusion negative in '{sid}'"
            assert np.all(result.P_brem   >= 0.0), f"P_brem negative in '{sid}'"
            assert np.all(result.P_cond   >= 0.0), f"P_cond negative in '{sid}'"

    def test_all_scenarios_have_consistent_array_lengths(self) -> None:
        for sid, scenario in available_helion_scenarios().items():
            result = integrate_helion_pulse(scenario.parameters, n_t_points=50)
            n = len(result.t)
            assert len(result.W)        == n, f"W length mismatch in '{sid}'"
            assert len(result.T_kev)    == n, f"T_kev length mismatch in '{sid}'"
            assert len(result.P_fusion) == n, f"P_fusion length mismatch in '{sid}'"
            assert len(result.P_brem)   == n, f"P_brem length mismatch in '{sid}'"
            assert len(result.P_cond)   == n, f"P_cond length mismatch in '{sid}'"


# ---------------------------------------------------------------------------
# 2D ignition boundary sweep
# ---------------------------------------------------------------------------


class TestIgnitionBoundarySweep:
    """Verify sweep structure, Q values, monotonicity, and ignition region."""

    def _base_params(self) -> HelionParameters:
        return HelionParameters(
            ion_temperature_kev=10.0,
            plasma_density_m3=1.0e21,
            compression_ratio=10.0,
            magnetic_field_t=5.0,
            plasma_volume_m3=1.0,
            pulse_duration_s=1.0e-5,
        )

    def test_result_is_ignition_boundary_result(self) -> None:
        rc = np.array([5.0, 10.0])
        nd = np.array([1e20, 1e21])
        result = ignition_boundary_sweep(rc, nd, self._base_params(), n_t_points=50)
        assert isinstance(result, IgnitionBoundaryResult)

    def test_q_map_shape_matches_input_grids(self) -> None:
        rc = np.array([5.0, 10.0, 50.0])
        nd = np.array([1e20, 1e21, 1e22])
        result = ignition_boundary_sweep(rc, nd, self._base_params(), n_t_points=50)
        assert result.Q_map.shape == (3, 3)

    def test_ignition_mask_is_q_map_greater_than_one(self) -> None:
        rc = np.array([5.0, 20.0, 100.0])
        nd = np.array([1e20, 1e21])
        result = ignition_boundary_sweep(rc, nd, self._base_params(), n_t_points=50)
        np.testing.assert_array_equal(result.ignition_mask, result.Q_map > 1.0)

    def test_q_map_all_nonnegative(self) -> None:
        rc = np.array([5.0, 20.0, 100.0])
        nd = np.array([1e20, 1e21])
        result = ignition_boundary_sweep(rc, nd, self._base_params(), n_t_points=50)
        assert np.all(result.Q_map >= 0.0)

    def test_result_echoes_input_axes(self) -> None:
        rc = np.array([10.0, 50.0])
        nd = np.array([1e21, 1e22])
        result = ignition_boundary_sweep(rc, nd, self._base_params(), n_t_points=50)
        np.testing.assert_array_equal(result.compression_ratios, rc)
        np.testing.assert_array_equal(result.densities_m3, nd)

    def test_n_failed_zero_for_valid_grid(self) -> None:
        """A grid entirely within Pydantic bounds must have zero failures."""
        rc = np.array([5.0, 10.0])
        nd = np.array([1e20, 1e21])
        result = ignition_boundary_sweep(rc, nd, self._base_params(), n_t_points=50)
        assert result.n_failed == 0

    def test_low_density_low_compression_is_sub_ignition(self) -> None:
        """Low n and low Rᶜ must produce Q < 1 at every grid point."""
        rc = np.array([2.0, 5.0])
        nd = np.array([1e19, 5e19])
        result = ignition_boundary_sweep(rc, nd, self._base_params(), n_t_points=50)
        assert np.all(result.Q_map < 1.0), (
            f"Expected all sub-ignition; Q_map = {result.Q_map}"
        )

    def test_extreme_compression_and_density_has_ignition_region(self) -> None:
        """At Rᶜ=1000 and n=1e23 m⁻³ at least one Q > 1 (confirmed by Week 6 test)."""
        rc = np.array([500.0, 1000.0])
        nd = np.array([1e22, 1e23])
        base = HelionParameters(
            ion_temperature_kev=1.0,
            plasma_density_m3=1.0e21,
            compression_ratio=10.0,
            magnetic_field_t=5.0,
            plasma_volume_m3=1.0,
            pulse_duration_s=1.0e-5,
        )
        result = ignition_boundary_sweep(rc, nd, base, n_t_points=50)
        assert np.any(result.ignition_mask), (
            f"Expected at least one ignition point; Q_map = {result.Q_map}"
        )

    def test_q_increases_with_compression_at_fixed_density(self) -> None:
        """Q must be monotonically non-decreasing as Rᶜ increases at fixed n.

        Physical basis: Q ∝ Rᶜ^{1/3} × ⟨σv⟩(T_initial × Rᶜ^{2/3}).
        Both factors increase (or hold constant when T clamps) with Rᶜ.
        """
        rc = np.array([5.0, 20.0, 100.0, 500.0])
        nd = np.array([1.0e21])
        result = ignition_boundary_sweep(rc, nd, self._base_params(), n_t_points=50)
        q_vals = result.Q_map[:, 0]
        for k in range(len(q_vals) - 1):
            assert q_vals[k] <= q_vals[k + 1], (
                f"Q decreased from Rᶜ={rc[k]} to Rᶜ={rc[k+1]}: "
                f"{q_vals[k]:.4e} → {q_vals[k+1]:.4e}"
            )

    def test_q_increases_with_density_at_fixed_compression(self) -> None:
        """Q must be monotonically non-decreasing as n increases at fixed Rᶜ.

        Physical basis: P_fusion ∝ n² while W_initial ∝ n and τ_E ∝ n,
        so Q ∝ n. Conduction and Bremsstrahlung losses do not break monotonicity.
        """
        rc = np.array([50.0])
        nd = np.array([1.0e20, 1.0e21, 1.0e22])
        result = ignition_boundary_sweep(rc, nd, self._base_params(), n_t_points=50)
        q_vals = result.Q_map[0, :]
        for k in range(len(q_vals) - 1):
            assert q_vals[k] <= q_vals[k + 1], (
                f"Q decreased from n={nd[k]:.1e} to n={nd[k+1]:.1e}: "
                f"{q_vals[k]:.4e} → {q_vals[k+1]:.4e}"
            )

    def test_out_of_bounds_grid_increments_n_failed(self) -> None:
        """Grid points outside Pydantic bounds are counted in n_failed, not silently lost."""
        rc = np.array([0.5, 10.0])   # Rc=0.5 is below ge=1 bound
        nd = np.array([1.0e21])
        result = ignition_boundary_sweep(rc, nd, self._base_params(), n_t_points=50)
        assert result.n_failed >= 1, "Expected at least one failure for Rc=0.5"
        # Valid point must still be computed
        assert result.Q_map[1, 0] > 0.0


# ---------------------------------------------------------------------------
# Q factor validation
# ---------------------------------------------------------------------------


class TestQFactorValidation:
    """Validate Q factor calculation properties and boundary conditions."""

    def test_q_nonnegative_at_default_params(self) -> None:
        result = integrate_helion_pulse(HelionParameters())
        assert result.Q >= 0.0

    def test_q_near_zero_at_deep_sub_ignition(self) -> None:
        """Extremely unfavourable conditions (low T, Rc=1, low n) give Q ≈ 0."""
        params = HelionParameters(
            ion_temperature_kev=0.5,
            plasma_density_m3=1.0e19,
            compression_ratio=1.0,
            magnetic_field_t=1.0,
            plasma_volume_m3=0.01,
            pulse_duration_s=1.0e-7,
        )
        result = integrate_helion_pulse(params)
        assert result.success
        assert result.Q < 1.0e-6, f"Expected near-zero Q; got {result.Q:.4e}"

    def test_q_exceeds_unity_at_ignition_conditions(self) -> None:
        """High density and extreme compression must yield Q > 1 (SPEC §7.3)."""
        params = HelionParameters(
            ion_temperature_kev=1.0,
            plasma_density_m3=1.0e23,
            compression_ratio=1000.0,
            magnetic_field_t=5.0,
            plasma_volume_m3=1.0,
            pulse_duration_s=1.0e-5,
        )
        result = integrate_helion_pulse(params)
        assert result.success
        assert result.Q > 1.0, f"Expected Q > 1 at ignition conditions; got Q = {result.Q:.4f}"

    def test_higher_density_gives_higher_q(self) -> None:
        """Doubling pre-compression density roughly doubles Q (P_fusion ∝ n²,
        W_initial ∝ n, so Q ∝ n at fixed Rᶜ and B)."""
        base = HelionParameters(
            ion_temperature_kev=10.0,
            plasma_density_m3=1.0e21,
            compression_ratio=50.0,
            magnetic_field_t=5.0,
            plasma_volume_m3=1.0,
            pulse_duration_s=1.0e-5,
        )
        high_n = HelionParameters(
            ion_temperature_kev=base.ion_temperature_kev,
            plasma_density_m3=2.0e21,
            compression_ratio=base.compression_ratio,
            magnetic_field_t=base.magnetic_field_t,
            plasma_volume_m3=base.plasma_volume_m3,
            pulse_duration_s=base.pulse_duration_s,
        )
        r1 = integrate_helion_pulse(base)
        r2 = integrate_helion_pulse(high_n)
        assert r1.success and r2.success
        assert r2.Q > r1.Q, (
            f"Expected Q to increase with density; "
            f"Q(1e21)={r1.Q:.4e}, Q(2e21)={r2.Q:.4e}"
        )

    def test_higher_compression_gives_higher_q(self) -> None:
        """Increasing Rᶜ must increase Q at fixed pre-compression conditions."""
        base = HelionParameters(
            ion_temperature_kev=5.0,
            plasma_density_m3=1.0e21,
            compression_ratio=10.0,
            magnetic_field_t=5.0,
            plasma_volume_m3=1.0,
            pulse_duration_s=1.0e-5,
        )
        high_rc = HelionParameters(
            ion_temperature_kev=base.ion_temperature_kev,
            plasma_density_m3=base.plasma_density_m3,
            compression_ratio=100.0,
            magnetic_field_t=base.magnetic_field_t,
            plasma_volume_m3=base.plasma_volume_m3,
            pulse_duration_s=base.pulse_duration_s,
        )
        r1 = integrate_helion_pulse(base)
        r2 = integrate_helion_pulse(high_rc)
        assert r1.success and r2.success
        assert r2.Q > r1.Q, (
            f"Expected Q to increase with Rᶜ; "
            f"Q(Rc=10)={r1.Q:.4e}, Q(Rc=100)={r2.Q:.4e}"
        )


# ---------------------------------------------------------------------------
# Energy conservation validation (SPEC §7.3)
# ---------------------------------------------------------------------------


class TestEnergyConservation:
    """SPEC §7.3: W_initial + E_fusion = W_final + E_brem + E_cond within 1%."""

    def _assert_energy_balance(
        self, params: HelionParameters, tol: float = 0.01
    ) -> None:
        result = integrate_helion_pulse(params, n_t_points=500)
        assert result.success

        E_brem   = float(np.trapezoid(result.P_brem,   result.t))
        E_cond   = float(np.trapezoid(result.P_cond,   result.t))
        E_fusion = float(np.trapezoid(result.P_fusion, result.t))
        W_final  = float(result.W[-1])

        lhs = result.W_initial + E_fusion       # energy in
        rhs = W_final + E_brem + E_cond         # energy out
        rel_err = abs(lhs - rhs) / max(abs(lhs), 1.0)
        assert rel_err < tol, (
            f"Energy conservation error {rel_err:.4f} > {tol:.2f} "
            f"(LHS={lhs:.4e} J, RHS={rhs:.4e} J)"
        )

    def test_energy_conservation_baseline_scenario(self) -> None:
        self._assert_energy_balance(
            get_helion_scenario("baseline_pulse").parameters
        )

    def test_energy_conservation_sub_ignition_scenario(self) -> None:
        self._assert_energy_balance(
            get_helion_scenario("sub_ignition_diagnostic").parameters
        )

    def test_energy_conservation_high_compression_scenario(self) -> None:
        self._assert_energy_balance(
            get_helion_scenario("high_compression").parameters
        )

    def test_energy_conservation_at_high_temperature(self) -> None:
        params = HelionParameters(
            ion_temperature_kev=50.0,
            plasma_density_m3=1.0e21,
            compression_ratio=2.0,
            magnetic_field_t=5.0,
            plasma_volume_m3=1.0,
            pulse_duration_s=1.0e-5,
        )
        self._assert_energy_balance(params)


# ---------------------------------------------------------------------------
# Temperature scaling validation (SPEC Eq. 19, T ∝ Rᶜ^{2/3})
# ---------------------------------------------------------------------------


class TestTemperatureScaling:
    """Validate T_final = T_initial × Rᶜ^{2/3} across the physics stack."""

    def test_post_compression_temperature_in_pulse_result(self) -> None:
        """T_kev[0] in pulse result matches adiabatic formula T_initial × Rᶜ^{2/3}."""
        T0, Rc = 5.0, 27.0  # 27^(2/3) = 9, so T_final = 45 keV exactly
        params = HelionParameters(
            ion_temperature_kev=T0,
            plasma_density_m3=1.0e21,
            compression_ratio=Rc,
            magnetic_field_t=5.0,
            plasma_volume_m3=1.0,
            pulse_duration_s=1.0e-5,
        )
        result = integrate_helion_pulse(params, n_t_points=50)
        assert result.success
        T_expected = T0 * Rc ** (2.0 / 3.0)  # = 45.0 keV
        assert result.T_kev[0] == pytest.approx(T_expected, rel=0.01)

    def test_temperature_scaling_holds_across_compression_ratios(self) -> None:
        """T_final / T_initial = Rᶜ^{2/3} for a range of Rᶜ values."""
        T0, n0, V0 = 10.0, 1.0e21, 1.0
        for Rc in [2.0, 4.0, 8.0, 16.0, 64.0]:
            T_final, _, _, _ = apply_adiabatic_compression(T0, n0, Rc, V0)
            expected = T0 * Rc ** (2.0 / 3.0)
            assert T_final == pytest.approx(expected, rel=1e-9), (
                f"T scaling mismatch at Rᶜ={Rc}: got {T_final:.6f}, expected {expected:.6f}"
            )

    def test_density_scaling_holds_across_compression_ratios(self) -> None:
        """n_final = n_initial × Rᶜ for a range of Rᶜ values."""
        T0, n0, V0 = 10.0, 1.0e21, 1.0
        for Rc in [2.0, 5.0, 10.0, 100.0]:
            _, n_final, _, _ = apply_adiabatic_compression(T0, n0, Rc, V0)
            assert n_final == pytest.approx(n0 * Rc, rel=1e-12)

    def test_higher_compression_produces_higher_initial_temperature(self) -> None:
        """Post-compression T_kev[0] is strictly increasing with Rᶜ in pulse results."""
        T0, n0 = 5.0, 1.0e21
        rc_vals = [5.0, 20.0, 100.0]
        T_initials = []
        for rc in rc_vals:
            params = HelionParameters(
                ion_temperature_kev=T0,
                plasma_density_m3=n0,
                compression_ratio=rc,
                magnetic_field_t=5.0,
                plasma_volume_m3=1.0,
                pulse_duration_s=1.0e-5,
            )
            result = integrate_helion_pulse(params, n_t_points=10)
            assert result.success
            T_initials.append(result.T_kev[0])

        assert T_initials[0] < T_initials[1] < T_initials[2], (
            f"T_kev[0] not monotonically increasing with Rᶜ: {T_initials}"
        )

    def test_energy_scales_as_rc_two_thirds(self) -> None:
        """W_initial ∝ Rᶜ^{2/3}: compressing same initial plasma to higher Rᶜ
        stores more energy (PdV work done on the plasma)."""
        T0, n0, V0 = 10.0, 1.0e21, 1.0
        _, _, _, W_ref  = apply_adiabatic_compression(T0, n0,  1.0, V0)
        _, _, _, W_10   = apply_adiabatic_compression(T0, n0, 10.0, V0)
        _, _, _, W_100  = apply_adiabatic_compression(T0, n0, 100.0, V0)
        assert W_10  == pytest.approx(W_ref  * 10.0  ** (2.0 / 3.0), rel=1e-9)
        assert W_100 == pytest.approx(W_ref  * 100.0 ** (2.0 / 3.0), rel=1e-9)
