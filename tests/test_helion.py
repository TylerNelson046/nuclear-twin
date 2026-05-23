"""Tests for the Helion FRC Phase 2 physics engine.

Covers:
    - Bosch-Hale ⟨σv⟩ computation (SPEC Eq. 18)
    - Adiabatic compression (SPEC Eq. 19)
    - Bremsstrahlung radiation loss (SPEC Eq. 20)
    - Confinement time scaling and conduction loss (SPEC Eq. 21)
    - Plasma temperature ↔ energy conversion
    - Full plasma energy balance ODE (SPEC Eq. 16–17)
    - Physics validation: Bremsstrahlung dominance below ~30 keV
    - Physics validation: Q > 1 at ignition conditions
    - Physics validation: energy conservation within 1% (SPEC §7.3)

NUMBA_DISABLE_JIT=1 is set by conftest.py so @njit kernels run as plain Python.
"""

from __future__ import annotations

import numpy as np
import pytest

from data.bosch_hale_coeffs import DHE3_BOSCH_HALE
from physics.helion.bosch_hale import T_MAX_KEV, T_MIN_KEV, sigma_v_dhe3, sigma_v_kernel
from physics.helion.compression import (
    apply_adiabatic_compression,
    plasma_temperature_kev,
)
from physics.helion.engine import (
    E_DHE3,
    HELION_PARAMS_LEN,
    _J_PER_KEV,
    integrate_helion_pulse,
    make_params_array,
    plasma_energy_rhs_kernel,
)
from physics.helion.losses import (
    C_TAU,
    TAU_E_UNCERTAINTY_LOWER,
    TAU_E_UNCERTAINTY_UPPER,
    bremsstrahlung_power,
    conduction_loss,
    confinement_time,
)
from physics.helion.parameters import HelionParameters


# ---------------------------------------------------------------------------
# Bosch-Hale ⟨σv⟩ (SPEC Eq. 18)
# ---------------------------------------------------------------------------


class TestBoschHaleReactivity:
    """Verify ⟨σv⟩(T) numerical values and boundary behaviour."""

    def test_sigma_v_is_positive_over_valid_range(self) -> None:
        for T in [1.0, 10.0, 50.0, 100.0, 190.0]:
            assert sigma_v_dhe3(T) > 0.0

    def test_sigma_v_known_value_at_50_kev(self) -> None:
        """At T=50 keV, ⟨σv⟩ should be ≈5.6×10⁻²³ m³/s per B&H Table I."""
        sv = sigma_v_dhe3(50.0)
        assert sv == pytest.approx(5.562e-23, rel=0.05), (
            f"⟨σv⟩(50 keV) = {sv:.4e} m³/s, expected ≈5.56×10⁻²³ m³/s"
        )

    def test_sigma_v_known_value_at_100_kev(self) -> None:
        """At T=100 keV, ⟨σv⟩ should be ≈1.71×10⁻²² m³/s per B&H Table I."""
        sv = sigma_v_dhe3(100.0)
        assert sv == pytest.approx(1.713e-22, rel=0.05), (
            f"⟨σv⟩(100 keV) = {sv:.4e} m³/s, expected ≈1.71×10⁻²² m³/s"
        )

    def test_sigma_v_known_value_at_10_kev(self) -> None:
        """At T=10 keV, ⟨σv⟩ should be ≈2.1×10⁻²⁵ m³/s per B&H Table I."""
        sv = sigma_v_dhe3(10.0)
        assert sv == pytest.approx(2.1e-25, rel=0.10), (
            f"⟨σv⟩(10 keV) = {sv:.4e} m³/s, expected ≈2.1×10⁻²⁵ m³/s"
        )

    def test_sigma_v_monotonically_increasing_10_to_150_kev(self) -> None:
        """SPEC §7.3: fusion power increases monotonically with T in 10–100 keV range."""
        temps = np.arange(10.0, 151.0, 10.0)
        sv_vals = [sigma_v_dhe3(T) for T in temps]
        for i in range(len(sv_vals) - 1):
            assert sv_vals[i] < sv_vals[i + 1], (
                f"⟨σv⟩ decreased from T={temps[i]} to T={temps[i+1]} keV"
            )

    def test_sigma_v_clamps_below_minimum(self) -> None:
        """T below 0.5 keV returns clamped value equal to T=0.5 keV result."""
        sv_min = sigma_v_dhe3(T_MIN_KEV)
        sv_sub = sigma_v_dhe3(0.1)
        assert sv_sub == pytest.approx(sv_min, rel=1e-10)

    def test_sigma_v_clamps_above_maximum(self) -> None:
        """T above 190 keV returns clamped value equal to T=190 keV result."""
        sv_max = sigma_v_dhe3(T_MAX_KEV)
        sv_sup = sigma_v_dhe3(300.0)
        assert sv_sup == pytest.approx(sv_max, rel=1e-10)

    def test_kernel_and_wrapper_agree(self) -> None:
        """sigma_v_kernel and sigma_v_dhe3 return identical results at 50 keV."""
        bh = DHE3_BOSCH_HALE
        sv_kernel = sigma_v_kernel(
            50.0,
            bh.bg, bh.mrc2,
            bh.c1, bh.c2, bh.c3, bh.c4, bh.c5, bh.c6, bh.c7,
        )
        sv_wrapper = sigma_v_dhe3(50.0)
        assert sv_kernel == pytest.approx(sv_wrapper, rel=1e-12)

    def test_sigma_v_units_consistent_m3_per_s(self) -> None:
        """⟨σv⟩ must be in m³/s range, not cm³/s (1e6× smaller than cm³/s value)."""
        sv = sigma_v_dhe3(50.0)
        # D-He3 ⟨σv⟩ at 50 keV is ~5e-17 cm³/s = ~5e-23 m³/s
        assert 1e-26 < sv < 1e-19, f"Unexpected unit range: {sv:.3e} m³/s"


# ---------------------------------------------------------------------------
# Adiabatic compression (SPEC Eq. 19)
# ---------------------------------------------------------------------------


class TestAdiabaticCompression:
    """Verify post-compression temperature, density, volume, and energy."""

    def test_temperature_scales_as_rc_to_two_thirds(self) -> None:
        """T_final = T_initial × Rᶜ^{2/3} exactly."""
        T0, n0, Rc, V0 = 5.0, 1.0e21, 8.0, 1.0
        T_final, _, _, _ = apply_adiabatic_compression(T0, n0, Rc, V0)
        expected = T0 * Rc ** (2.0 / 3.0)
        assert T_final == pytest.approx(expected, rel=1e-12)

    def test_density_scales_as_rc(self) -> None:
        """n_final = n_initial × Rᶜ exactly."""
        T0, n0, Rc, V0 = 5.0, 1.0e21, 10.0, 1.0
        _, n_final, _, _ = apply_adiabatic_compression(T0, n0, Rc, V0)
        assert n_final == pytest.approx(n0 * Rc, rel=1e-12)

    def test_volume_decreases_by_rc(self) -> None:
        """V_final = V_initial / Rᶜ exactly."""
        T0, n0, Rc, V0 = 5.0, 1.0e21, 100.0, 2.0
        _, _, V_final, _ = apply_adiabatic_compression(T0, n0, Rc, V0)
        assert V_final == pytest.approx(V0 / Rc, rel=1e-12)

    def test_energy_scales_as_rc_to_two_thirds(self) -> None:
        """W_final = W_initial × Rᶜ^{2/3} (compression does PdV work on the plasma)."""
        T0, n0, Rc, V0 = 20.0, 1.0e21, 27.0, 1.0
        _, _, _, W_initial = apply_adiabatic_compression(T0, n0, 1.0, V0)
        _, _, _, W_final   = apply_adiabatic_compression(T0, n0, Rc,  V0)
        assert W_final == pytest.approx(W_initial * Rc ** (2.0 / 3.0), rel=1e-9)

    def test_no_compression_is_identity(self) -> None:
        """Rᶜ = 1.0 leaves all state unchanged."""
        T0, n0, V0 = 30.0, 1.0e21, 0.5
        T_f, n_f, V_f, _ = apply_adiabatic_compression(T0, n0, 1.0, V0)
        assert T_f == pytest.approx(T0, rel=1e-12)
        assert n_f == pytest.approx(n0, rel=1e-12)
        assert V_f == pytest.approx(V0, rel=1e-12)


# ---------------------------------------------------------------------------
# Plasma temperature ↔ energy conversion
# ---------------------------------------------------------------------------


class TestPlasmaTemperatureConversion:
    """Round-trip and edge-case tests for W ↔ T conversion."""

    def test_round_trip_consistency(self) -> None:
        """plasma_temperature_kev inverts apply_adiabatic_compression energy output."""
        T0, n0, Rc, V0 = 20.0, 1.0e21, 10.0, 1.0
        T_final, n_final, V_final, W_final = apply_adiabatic_compression(T0, n0, Rc, V0)
        T_recovered = plasma_temperature_kev(W_final, n_final, V_final)
        assert T_recovered == pytest.approx(T_final, rel=1e-9)

    def test_temperature_zero_for_nonpositive_energy(self) -> None:
        """W ≤ 0 returns T = 0.0 (floored state, no division by zero)."""
        assert plasma_temperature_kev(0.0, 1e21, 1.0) == 0.0
        assert plasma_temperature_kev(-1.0, 1e21, 1.0) == 0.0

    def test_temperature_proportional_to_energy(self) -> None:
        """T scales linearly with W at fixed n and V."""
        W1, W2 = 1.0e8, 2.0e8
        T1 = plasma_temperature_kev(W1, 1e21, 1.0)
        T2 = plasma_temperature_kev(W2, 1e21, 1.0)
        assert T2 == pytest.approx(2.0 * T1, rel=1e-9)


# ---------------------------------------------------------------------------
# Bremsstrahlung power (SPEC Eq. 20)
# ---------------------------------------------------------------------------


class TestBremsstrahlungPower:
    """Verify Bremsstrahlung formula scaling and sign."""

    def test_bremsstrahlung_is_positive(self) -> None:
        assert bremsstrahlung_power(50.0, 1e21, 1.0) > 0.0

    def test_bremsstrahlung_scales_quadratically_with_density(self) -> None:
        """P_Brem ∝ n² (SPEC Eq. 20): doubling n quadruples P_Brem."""
        P1 = bremsstrahlung_power(50.0, 1e21, 1.0)
        P2 = bremsstrahlung_power(50.0, 2e21, 1.0)
        assert P2 == pytest.approx(4.0 * P1, rel=1e-9)

    def test_bremsstrahlung_scales_with_sqrt_T(self) -> None:
        """P_Brem ∝ T^{1/2}: quadrupling T doubles P_Brem."""
        P1 = bremsstrahlung_power(25.0, 1e21, 1.0)
        P2 = bremsstrahlung_power(100.0, 1e21, 1.0)
        assert P2 == pytest.approx(2.0 * P1, rel=1e-9)

    def test_bremsstrahlung_scales_linearly_with_volume(self) -> None:
        """P_Brem ∝ V: doubling V doubles P_Brem."""
        P1 = bremsstrahlung_power(50.0, 1e21, 1.0)
        P2 = bremsstrahlung_power(50.0, 1e21, 2.0)
        assert P2 == pytest.approx(2.0 * P1, rel=1e-9)

    def test_bremsstrahlung_hand_calculation(self) -> None:
        """Hand-verify a single numerical point.

        At T=100 keV, n=1e21 m⁻³, V=1 m³:
            n_e = 1.5×10²¹ m⁻³,  Z_eff = 5/3
            P = 5.35e-37 × (5/3) × (1.5e21)² × 10.0 × 1
              = 5.35e-37 × 1.667 × 2.25e42 × 10
              = 2.006e7 W
        """
        P = bremsstrahlung_power(100.0, 1e21, 1.0)
        assert P == pytest.approx(2.006e7, rel=0.01)


# ---------------------------------------------------------------------------
# Confinement time and conduction loss (SPEC Eq. 21)
# ---------------------------------------------------------------------------


class TestConfinementAndConduction:
    """Verify τ_E parametric scaling and conduction loss formula."""

    def test_tau_e_positive(self) -> None:
        assert confinement_time(1e21, 5.0) > 0.0

    def test_tau_e_increases_with_density(self) -> None:
        """Higher density → better confinement (FRC empirical trend)."""
        assert confinement_time(2e21, 5.0) > confinement_time(1e21, 5.0)

    def test_tau_e_increases_with_field(self) -> None:
        """Higher B → better confinement."""
        assert confinement_time(1e21, 10.0) > confinement_time(1e21, 5.0)

    def test_tau_e_scales_quadratically_with_B(self) -> None:
        """τ_E ∝ B²: doubling B quadruples τ_E."""
        tau1 = confinement_time(1e21, 5.0)
        tau2 = confinement_time(1e21, 10.0)
        assert tau2 == pytest.approx(4.0 * tau1, rel=1e-9)

    def test_conduction_loss_equals_w_over_tau(self) -> None:
        """P_cond = W / τ_E by definition."""
        n, B, W = 1e21, 5.0, 1e7
        tau_e = confinement_time(n, B)
        P_cond = conduction_loss(W, n, B)
        assert P_cond == pytest.approx(W / tau_e, rel=1e-12)

    def test_uncertainty_band_constants_bracketing_nominal(self) -> None:
        """Uncertainty multipliers straddle 1.0 (nominal), per Risk R-02."""
        assert TAU_E_UNCERTAINTY_LOWER < 1.0
        assert TAU_E_UNCERTAINTY_UPPER > 1.0


# ---------------------------------------------------------------------------
# Plasma energy balance RHS (SPEC Eq. 16)
# ---------------------------------------------------------------------------


class TestPlasmaEnergyRhs:
    """Unit tests for the plasma_energy_rhs_kernel ODE right-hand side."""

    def _make_default_params(
        self,
        n: float = 1e21,
        V: float = 1.0,
        P_heat: float = 0.0,
        B: float = 5.0,
    ) -> np.ndarray:
        return make_params_array(n, V, P_heat, B)

    def test_params_array_has_correct_length(self) -> None:
        params = self._make_default_params()
        assert params.shape == (HELION_PARAMS_LEN,)
        assert params.dtype == np.float64

    def test_rhs_returns_shape_1(self) -> None:
        params = self._make_default_params()
        y0 = np.array([1e8], dtype=np.float64)
        dydt = plasma_energy_rhs_kernel(0.0, y0, params)
        assert dydt.shape == (1,)

    def test_rhs_zero_energy_floor(self) -> None:
        """W = 0 must return dW/dt = 0 (non-negative state floor, SPEC §4.6.5)."""
        params = self._make_default_params()
        dydt = plasma_energy_rhs_kernel(0.0, np.array([0.0]), params)
        assert dydt[0] == 0.0

    def test_rhs_negative_energy_floor(self) -> None:
        """W < 0 must return dW/dt = 0 (non-negative state floor)."""
        params = self._make_default_params()
        dydt = plasma_energy_rhs_kernel(0.0, np.array([-1.0]), params)
        assert dydt[0] == 0.0

    def test_external_heating_increases_dw_dt(self) -> None:
        """P_heat > 0 should shift dW/dt upward relative to P_heat=0."""
        W = 1e8
        params_no_heat   = self._make_default_params(P_heat=0.0)
        params_with_heat = self._make_default_params(P_heat=1e6)
        y = np.array([W], dtype=np.float64)
        dw_no_heat   = plasma_energy_rhs_kernel(0.0, y, params_no_heat)[0]
        dw_with_heat = plasma_energy_rhs_kernel(0.0, y, params_with_heat)[0]
        assert dw_with_heat == pytest.approx(dw_no_heat + 1e6, rel=1e-9)


# ---------------------------------------------------------------------------
# Full ODE integration tests
# ---------------------------------------------------------------------------


class TestHelionPulseIntegration:
    """Integration tests for integrate_helion_pulse — physics validation."""

    def _default_params(self, **overrides) -> HelionParameters:
        defaults = {
            "ion_temperature_kev": 20.0,
            "plasma_density_m3": 1e21,
            "compression_ratio": 10.0,
            "magnetic_field_t": 5.0,
            "plasma_volume_m3": 1.0,
            "pulse_duration_s": 1e-5,
        }
        defaults.update(overrides)
        return HelionParameters(**defaults)

    def test_integration_succeeds_default_params(self) -> None:
        result = integrate_helion_pulse(self._default_params())
        assert result.success, f"Integration failed: {result.message}"

    def test_result_arrays_have_consistent_lengths(self) -> None:
        result = integrate_helion_pulse(self._default_params(), n_t_points=50)
        n = len(result.t)
        assert len(result.W) == n
        assert len(result.T_kev) == n
        assert len(result.P_fusion) == n
        assert len(result.P_brem) == n
        assert len(result.P_cond) == n

    def test_result_w_initial_matches_compression(self) -> None:
        """W_initial in result must equal adiabatic compression output."""
        params = self._default_params()
        _, _, _, W_comp = apply_adiabatic_compression(
            params.ion_temperature_kev,
            params.plasma_density_m3,
            params.compression_ratio,
            params.plasma_volume_m3,
        )
        result = integrate_helion_pulse(params)
        assert result.W_initial == pytest.approx(W_comp, rel=1e-9)

    # ------------------------------------------------------------------
    # SPEC §7.3 Physics Validation: Bremsstrahlung dominance at T < 30 keV
    # ------------------------------------------------------------------

    def test_bremsstrahlung_dominates_fusion_below_30_kev(self) -> None:
        """SPEC §7.3: at T < 30 keV radiation losses exceed fusion power.

        At T=20 keV (post-compression with Rc=1, so T stays 20 keV),
        P_brem / P_fusion > 1 at n=1e21 m⁻³.
        This is independent of density because both scale as n².
        """
        T_kev = 20.0
        n     = 1.0e21
        V     = 1.0

        sv = sigma_v_dhe3(T_kev)
        P_fusion = (n ** 2 / 4.0) * sv * E_DHE3 * V
        P_brem   = bremsstrahlung_power(T_kev, n, V)

        assert P_brem > P_fusion, (
            f"Expected P_brem ({P_brem:.3e} W) > P_fusion ({P_fusion:.3e} W) "
            f"at T=20 keV"
        )

    def test_fusion_exceeds_bremsstrahlung_above_30_kev(self) -> None:
        """At T = 100 keV fusion power exceeds Bremsstrahlung (SPEC §7.3)."""
        T_kev = 100.0
        n     = 1.0e21
        V     = 1.0

        sv = sigma_v_dhe3(T_kev)
        P_fusion = (n ** 2 / 4.0) * sv * E_DHE3 * V
        P_brem   = bremsstrahlung_power(T_kev, n, V)

        assert P_fusion > P_brem, (
            f"Expected P_fusion ({P_fusion:.3e} W) > P_brem ({P_brem:.3e} W) "
            f"at T=100 keV"
        )

    # ------------------------------------------------------------------
    # SPEC §7.3 Physics Validation: Q boundary (sub-ignition vs ignition)
    # ------------------------------------------------------------------

    def test_q_less_than_unity_at_low_density(self) -> None:
        """SPEC §7.3: at sub-ignition conditions (low n), Q < 1."""
        params = self._default_params(
            plasma_density_m3=1e19,  # very low density
            ion_temperature_kev=20.0,
            compression_ratio=1.0,
        )
        result = integrate_helion_pulse(params)
        assert result.success
        assert result.Q < 1.0, f"Expected Q < 1 at low density; got Q = {result.Q:.4f}"

    def test_q_greater_than_unity_at_ignition_conditions(self) -> None:
        """SPEC §7.3: at high-density, high-compression conditions, Q > 1.

        Post-compression: n_final = 1e23 × 1000 = 1e26 m⁻³, T_final = 100 keV.
        At these parameters fusion energy dominates confinement losses over the pulse.
        """
        params = HelionParameters(
            ion_temperature_kev=1.0,      # 1 keV pre-compression
            plasma_density_m3=1e23,       # maximum allowed pre-compression density
            compression_ratio=1000.0,     # Rᶜ=1000 → T_final=100 keV, n_final=1e26
            magnetic_field_t=5.0,
            plasma_volume_m3=1.0,
            pulse_duration_s=1e-5,
        )
        result = integrate_helion_pulse(params)
        assert result.success
        assert result.Q > 1.0, f"Expected Q > 1 at ignition conditions; got Q = {result.Q:.4f}"

    # ------------------------------------------------------------------
    # SPEC §7.3 Physics Validation: Energy conservation within 1%
    # ------------------------------------------------------------------

    def test_energy_conservation_within_one_percent(self) -> None:
        """SPEC §7.3: fusion output + losses = input + fusion yield within 1%.

        Energy balance check:
            W_final + E_brem + E_cond = W_initial + E_fusion + E_heat
        All terms in Joules, integrated from the result time series.
        """
        params = self._default_params(
            ion_temperature_kev=50.0,
            compression_ratio=2.0,
            pulse_duration_s=1e-5,
        )
        result = integrate_helion_pulse(params, P_heat_w=0.0, n_t_points=500)
        assert result.success

        E_brem   = float(np.trapezoid(result.P_brem,   result.t))
        E_cond   = float(np.trapezoid(result.P_cond,   result.t))
        E_fusion = float(np.trapezoid(result.P_fusion, result.t))
        W_final  = float(result.W[-1])

        # W_initial + E_fusion = W_final + E_brem + E_cond  (energy conservation)
        lhs = result.W_initial + E_fusion
        rhs = W_final + E_brem + E_cond
        relative_error = abs(lhs - rhs) / max(abs(lhs), 1.0)
        assert relative_error < 0.01, (
            f"Energy conservation error {relative_error:.4f} exceeds 1% limit. "
            f"LHS={lhs:.4e} J, RHS={rhs:.4e} J"
        )

    # ------------------------------------------------------------------
    # Physical properties
    # ------------------------------------------------------------------

    def test_all_powers_nonnegative_over_pulse(self) -> None:
        """Bremsstrahlung, conduction loss, and fusion power must be non-negative."""
        result = integrate_helion_pulse(self._default_params(), n_t_points=100)
        assert result.success
        assert np.all(result.P_brem   >= 0.0), "P_brem went negative"
        assert np.all(result.P_cond   >= 0.0), "P_cond went negative"
        assert np.all(result.P_fusion >= 0.0), "P_fusion went negative"

    def test_w_nonnegative_over_pulse(self) -> None:
        """Plasma thermal energy W must never go negative."""
        result = integrate_helion_pulse(self._default_params(), n_t_points=100)
        assert result.success
        assert np.all(result.W >= 0.0), "W went negative during integration"

    def test_post_compression_temperature_reflected_in_t_kev(self) -> None:
        """T_kev[0] should equal the post-compression temperature."""
        params = self._default_params(
            ion_temperature_kev=10.0,
            compression_ratio=8.0,
        )
        T_expected = 10.0 * 8.0 ** (2.0 / 3.0)
        result = integrate_helion_pulse(params, n_t_points=50)
        assert result.success
        assert result.T_kev[0] == pytest.approx(T_expected, rel=0.01)


# ---------------------------------------------------------------------------
# Pydantic parameter validation (SPEC NFR-04, SR-02)
# ---------------------------------------------------------------------------


class TestHelionParameterBounds:
    """Verify Pydantic field bounds enforce physical constraints."""

    def test_valid_default_parameters(self) -> None:
        params = HelionParameters()
        assert params.ion_temperature_kev == 20.0

    def test_temperature_below_bosch_hale_lower_bound_rejected(self) -> None:
        with pytest.raises(Exception):
            HelionParameters(ion_temperature_kev=0.1)

    def test_temperature_above_bosch_hale_upper_bound_rejected(self) -> None:
        with pytest.raises(Exception):
            HelionParameters(ion_temperature_kev=200.0)

    def test_compression_ratio_below_1_rejected(self) -> None:
        with pytest.raises(Exception):
            HelionParameters(compression_ratio=0.5)

    def test_density_below_minimum_rejected(self) -> None:
        with pytest.raises(Exception):
            HelionParameters(plasma_density_m3=1e18)
