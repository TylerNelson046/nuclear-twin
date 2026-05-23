"""
Tests for physics/pwr/feedback.py — algebraic reactivity feedback calculators.

Software tests (CLAUDE.md §Testing Requirements, SPEC §7.2):
    Every equation implemented as a function is tested against a hand-calculated value.
    Operating range spans Cold Zero Power (CZP) through Hot Full Power (HFP).
    Edge cases include at-reference (zero delta) and unphysical inputs.

SPEC physics validation checks covered here (SPEC §7.3 PWR):
    - test_spec_doppler_hand_value: +100 K rise, α_D = -2.5 → -250 pcm (SPEC §7.2 example)
    - test_doppler_sign_convention: ΔT > 0 → ρ_D < 0 (inherent safety mechanism)
    - test_moderator_sign_convention: ΔT_c > 0 → ρ_m < 0 (second independent feedback)
    - CZP → HFP range tests cover the full PWR operating envelope
"""

from __future__ import annotations

import logging
import math

import pytest

from physics.pwr.feedback import (
    ALPHA_D_DEFAULT,
    ALPHA_M_DEFAULT,
    OMEGA_B_DEFAULT,
    boron_worth,
    doppler_feedback,
    moderator_feedback,
    pcm_to_dk_k,
    reactivity_balance,
)

# ---------------------------------------------------------------------------
# PWR operating condition constants (consistent with physics/pwr/parameters.py)
# ---------------------------------------------------------------------------

# Cold Zero Power: reactor cold, pressurised but not at operating temperature
T_FUEL_CZP: float = 560.0   # K  (~287 °C, below hot operating temp)
T_COOL_CZP: float = 560.0   # K  (fuel ≈ coolant at no power)

# Hot Zero Power: at operating pressure/temperature but zero power
T_FUEL_HZP: float = 565.0   # K  (~292 °C)
T_COOL_HZP: float = 565.0   # K

# Hot Full Power: nominal design operating point (parameters.py defaults)
T_FUEL_HFP: float = 900.0   # K  (~627 °C, nominal ceramic fuel temperature)
T_COOL_HFP: float = 590.0   # K  (~317 °C, nominal PWR coolant outlet)

# Nominal boron concentrations over the fuel cycle
BORON_BOC: float = 1200.0   # ppm — beginning of cycle (high, compensates fresh fuel)
BORON_MOC: float = 700.0    # ppm — mid-cycle
BORON_EOC: float = 50.0     # ppm — end of cycle (nearly all burned out)


# ===========================================================================
# Doppler (fuel temperature) feedback — SPEC Eq. 11
# ===========================================================================

class TestDopplerFeedback:
    """Tests for doppler_feedback(T_fuel_k, T_fuel_ref_k, alpha_D)."""

    def test_spec_hand_calculated_value(self):
        """SPEC §7.2 explicit example: +100 K rise, α_D = -2.5 pcm/K → -250 pcm."""
        T_ref = 900.0
        result = doppler_feedback(T_ref + 100.0, T_ref, alpha_D=-2.5)
        assert result == pytest.approx(-250.0, abs=1e-10)

    def test_zero_delta_returns_zero(self):
        """At the reference temperature the Doppler contribution is exactly 0 pcm."""
        assert doppler_feedback(T_FUEL_HFP, T_FUEL_HFP) == 0.0
        assert doppler_feedback(T_FUEL_CZP, T_FUEL_CZP) == 0.0

    def test_negative_sign_for_temperature_increase(self):
        """ΔT_f > 0 must yield negative ρ_D — the primary inherent safety mechanism."""
        result = doppler_feedback(T_FUEL_HFP + 50.0, T_FUEL_HFP)
        assert result < 0.0

    def test_positive_sign_for_temperature_decrease(self):
        """ΔT_f < 0 produces positive ρ_D (power increase upon cooldown)."""
        result = doppler_feedback(T_FUEL_HFP - 50.0, T_FUEL_HFP)
        assert result > 0.0

    def test_linearity_in_delta_T(self):
        """ρ_D is strictly linear in ΔT: doubling ΔT doubles ρ_D."""
        T_ref = 900.0
        rho_50 = doppler_feedback(T_ref + 50.0, T_ref)
        rho_100 = doppler_feedback(T_ref + 100.0, T_ref)
        assert rho_100 == pytest.approx(2.0 * rho_50, rel=1e-12)

    def test_czp_to_hfp_range(self):
        """CZP → HFP: ΔT = 340 K, α_D = -2.5 pcm/K → -850 pcm, within plausible range."""
        result = doppler_feedback(T_FUEL_HFP, T_FUEL_CZP)
        expected = ALPHA_D_DEFAULT * (T_FUEL_HFP - T_FUEL_CZP)
        assert result == pytest.approx(expected, rel=1e-12)
        # CZP → HFP Doppler worth must be large and negative (hundreds of pcm)
        assert result < -500.0

    def test_hzp_to_hfp_range(self):
        """HZP → HFP: smaller power-rise ΔT still produces correct negative worth."""
        result = doppler_feedback(T_FUEL_HFP, T_FUEL_HZP)
        expected = ALPHA_D_DEFAULT * (T_FUEL_HFP - T_FUEL_HZP)
        assert result == pytest.approx(expected, rel=1e-12)
        assert result < 0.0

    def test_custom_alpha_D_lower_bound(self):
        """α_D = -2 pcm/K (lower physical bound per CLAUDE.md): correct value."""
        result = doppler_feedback(1000.0, 900.0, alpha_D=-2.0)
        assert result == pytest.approx(-200.0, abs=1e-10)

    def test_custom_alpha_D_upper_bound(self):
        """α_D = -3 pcm/K (upper physical bound per CLAUDE.md): correct value."""
        result = doppler_feedback(1000.0, 900.0, alpha_D=-3.0)
        assert result == pytest.approx(-300.0, abs=1e-10)

    # --- Unphysical / edge cases ---

    def test_absolute_zero_fuel_is_valid(self):
        """T_fuel = 0 K is the physical lower bound; must NOT trigger NaN."""
        result = doppler_feedback(0.0, 900.0)
        assert not math.isnan(result)
        assert result == pytest.approx(ALPHA_D_DEFAULT * (0.0 - 900.0), rel=1e-12)

    def test_negative_fuel_temperature_returns_nan(self):
        """T_fuel < 0 K is physically impossible; function must return NaN."""
        result = doppler_feedback(-1.0, 900.0)
        assert math.isnan(result)

    def test_negative_reference_temperature_returns_nan(self):
        """T_fuel_ref < 0 K is physically impossible; function must return NaN."""
        result = doppler_feedback(900.0, -1.0)
        assert math.isnan(result)

    def test_negative_fuel_temperature_logs_warning(self, caplog):
        """Unphysical fuel temperature triggers a WARNING-level log message."""
        with caplog.at_level(logging.WARNING, logger="physics.pwr.feedback"):
            doppler_feedback(-10.0, 900.0)
        assert len(caplog.records) >= 1
        assert any("unphysical" in r.message.lower() for r in caplog.records)

    def test_both_negative_inputs_returns_nan(self):
        """Both inputs negative: still returns NaN without raising an exception."""
        result = doppler_feedback(-100.0, -200.0)
        assert math.isnan(result)

    @pytest.mark.parametrize("T_fuel", [550.0, 700.0, 900.0, 1100.0, 1400.0])
    def test_parametrized_operating_range(self, T_fuel):
        """Across the full Pydantic-bounded fuel temperature range (550–1500 K):
        result equals α_D · (T_fuel − T_ref) for any T_ref = 900 K."""
        T_ref = 900.0
        result = doppler_feedback(T_fuel, T_ref)
        expected = ALPHA_D_DEFAULT * (T_fuel - T_ref)
        assert result == pytest.approx(expected, rel=1e-12)


# ===========================================================================
# Moderator temperature feedback — SPEC Eq. 12
# ===========================================================================

class TestModeratorFeedback:
    """Tests for moderator_feedback(T_cool_k, T_cool_ref_k, alpha_m)."""

    def test_hand_calculated_value_50k_rise(self):
        """ΔT_c = +50 K, α_m = -35 pcm/K → -1750 pcm."""
        result = moderator_feedback(590.0 + 50.0, 590.0, alpha_m=-35.0)
        assert result == pytest.approx(-1750.0, abs=1e-10)

    def test_hand_calculated_value_minus_20k(self):
        """ΔT_c = -20 K, α_m = -35 pcm/K → +700 pcm."""
        result = moderator_feedback(590.0 - 20.0, 590.0, alpha_m=-35.0)
        assert result == pytest.approx(700.0, abs=1e-10)

    def test_zero_delta_returns_zero(self):
        """At reference coolant temperature, moderator contribution is exactly 0 pcm."""
        assert moderator_feedback(T_COOL_HFP, T_COOL_HFP) == 0.0
        assert moderator_feedback(T_COOL_CZP, T_COOL_CZP) == 0.0

    def test_negative_sign_for_temperature_increase(self):
        """ΔT_c > 0 must yield negative ρ_m (second independent inherent safety)."""
        result = moderator_feedback(T_COOL_HFP + 20.0, T_COOL_HFP)
        assert result < 0.0

    def test_positive_sign_for_temperature_decrease(self):
        """ΔT_c < 0 produces positive ρ_m."""
        result = moderator_feedback(T_COOL_HFP - 20.0, T_COOL_HFP)
        assert result > 0.0

    def test_linearity_in_delta_T(self):
        """ρ_m is strictly linear in ΔT_c."""
        T_ref = 590.0
        rho_20 = moderator_feedback(T_ref + 20.0, T_ref)
        rho_40 = moderator_feedback(T_ref + 40.0, T_ref)
        assert rho_40 == pytest.approx(2.0 * rho_20, rel=1e-12)

    def test_czp_to_hfp_range(self):
        """CZP → HFP coolant rise: ΔT = 30 K, α_m = -35 pcm/K → -1050 pcm."""
        result = moderator_feedback(T_COOL_HFP, T_COOL_CZP)
        expected = ALPHA_M_DEFAULT * (T_COOL_HFP - T_COOL_CZP)
        assert result == pytest.approx(expected, rel=1e-12)

    def test_alpha_m_at_lower_physical_bound(self):
        """α_m = -20 pcm/K (lower bound per CLAUDE.md) produces correct result."""
        result = moderator_feedback(610.0, 590.0, alpha_m=-20.0)
        assert result == pytest.approx(-400.0, abs=1e-10)

    def test_alpha_m_at_upper_physical_bound(self):
        """α_m = -50 pcm/K (upper bound per CLAUDE.md) produces correct result."""
        result = moderator_feedback(610.0, 590.0, alpha_m=-50.0)
        assert result == pytest.approx(-1000.0, abs=1e-10)

    # --- Unphysical / edge cases ---

    def test_absolute_zero_coolant_is_valid(self):
        """T_cool = 0 K is the physical boundary; must NOT trigger NaN."""
        result = moderator_feedback(0.0, 590.0)
        assert not math.isnan(result)
        assert result == pytest.approx(ALPHA_M_DEFAULT * (0.0 - 590.0), rel=1e-12)

    def test_negative_coolant_temperature_returns_nan(self):
        """T_cool < 0 K is physically impossible; function must return NaN."""
        result = moderator_feedback(-1.0, 590.0)
        assert math.isnan(result)

    def test_negative_reference_temperature_returns_nan(self):
        """T_cool_ref < 0 K is physically impossible; function must return NaN."""
        result = moderator_feedback(590.0, -1.0)
        assert math.isnan(result)

    def test_negative_coolant_temperature_logs_warning(self, caplog):
        """Unphysical coolant temperature triggers a WARNING-level log message."""
        with caplog.at_level(logging.WARNING, logger="physics.pwr.feedback"):
            moderator_feedback(-5.0, 590.0)
        assert len(caplog.records) >= 1
        assert any("unphysical" in r.message.lower() for r in caplog.records)

    @pytest.mark.parametrize("T_cool", [540.0, 560.0, 590.0, 610.0, 645.0])
    def test_parametrized_operating_range(self, T_cool):
        """Across the Pydantic-bounded coolant range (540–650 K): correct value."""
        T_ref = 590.0
        result = moderator_feedback(T_cool, T_ref)
        expected = ALPHA_M_DEFAULT * (T_cool - T_ref)
        assert result == pytest.approx(expected, rel=1e-12)


# ===========================================================================
# Boron worth calculator — SPEC Eq. 13
# ===========================================================================

class TestBoronWorth:
    """Tests for boron_worth(boron_ppm, omega_B)."""

    def test_zero_boron_returns_zero(self):
        """Zero boron concentration → zero reactivity worth (no absorber present)."""
        assert boron_worth(0.0) == 0.0

    def test_hand_calculated_1000ppm(self):
        """1000 ppm, ω_B = -10 pcm/ppm → -10 000 pcm."""
        result = boron_worth(1000.0, omega_B=-10.0)
        assert result == pytest.approx(-10_000.0, abs=1e-10)

    def test_hand_calculated_500ppm(self):
        """500 ppm, ω_B = -10 pcm/ppm → -5 000 pcm."""
        result = boron_worth(500.0, omega_B=-10.0)
        assert result == pytest.approx(-5_000.0, abs=1e-10)

    def test_hand_calculated_2500ppm_max(self):
        """2500 ppm (Pydantic upper bound), ω_B = -10 pcm/ppm → -25 000 pcm."""
        result = boron_worth(2500.0, omega_B=-10.0)
        assert result == pytest.approx(-25_000.0, abs=1e-10)

    def test_negative_sign_for_positive_boron(self):
        """Any positive boron concentration must yield strictly negative worth."""
        for ppm in [1.0, 100.0, 1000.0, 2500.0]:
            assert boron_worth(ppm) < 0.0

    def test_linearity(self):
        """ρ_B is strictly linear in boron_ppm: doubling ppm doubles worth."""
        rho_500 = boron_worth(500.0)
        rho_1000 = boron_worth(1000.0)
        assert rho_1000 == pytest.approx(2.0 * rho_500, rel=1e-12)

    def test_beginning_of_cycle_worth(self):
        """BOC boron (1200 ppm) produces large negative worth for startup control."""
        result = boron_worth(BORON_BOC)
        assert result == pytest.approx(OMEGA_B_DEFAULT * BORON_BOC, rel=1e-12)
        assert result < -10_000.0

    def test_end_of_cycle_worth(self):
        """EOC boron (50 ppm) produces small negative worth as burnup compensated."""
        result = boron_worth(BORON_EOC)
        assert result == pytest.approx(OMEGA_B_DEFAULT * BORON_EOC, rel=1e-12)
        assert abs(result) < 1000.0

    def test_custom_omega_B(self):
        """Custom ω_B is applied correctly."""
        result = boron_worth(100.0, omega_B=-9.5)
        assert result == pytest.approx(-950.0, abs=1e-10)

    # --- Unphysical / edge cases ---

    def test_negative_boron_returns_nan(self):
        """Negative boron concentration is physically impossible; must return NaN."""
        result = boron_worth(-1.0)
        assert math.isnan(result)

    def test_negative_boron_logs_warning(self, caplog):
        """Negative boron triggers a WARNING-level log message."""
        with caplog.at_level(logging.WARNING, logger="physics.pwr.feedback"):
            boron_worth(-0.5)
        assert len(caplog.records) >= 1
        assert any("unphysical" in r.message.lower() for r in caplog.records)

    @pytest.mark.parametrize("boron", [0.0, 50.0, 500.0, 1000.0, 1500.0, 2500.0])
    def test_parametrized_fuel_cycle_range(self, boron):
        """Boron worth is correct across the full Pydantic-bounded range (0–2500 ppm)."""
        result = boron_worth(boron)
        assert result == pytest.approx(OMEGA_B_DEFAULT * boron, rel=1e-12)


# ===========================================================================
# Reactivity balance — ARCHITECTURE.md §4.1
# ===========================================================================

class TestReactivityBalance:
    """Tests for reactivity_balance(rod, doppler, moderator, boron, xenon)."""

    def test_sum_equals_arithmetic_total(self):
        """ρ_total equals the exact arithmetic sum of all five components."""
        rod, dopp, mod, bor, xen = 50.0, -120.0, -80.0, -500.0, -200.0
        result = reactivity_balance(rod, dopp, mod, bor, xen)
        assert result == pytest.approx(rod + dopp + mod + bor + xen, abs=1e-10)

    def test_all_zeros_returns_zero(self):
        """All-zero inputs → exactly 0 pcm (perfect criticality)."""
        assert reactivity_balance(0.0, 0.0, 0.0, 0.0, 0.0) == 0.0

    def test_sign_with_dominant_positive_rod(self):
        """Large positive rod insertion dominates and produces positive ρ_total."""
        result = reactivity_balance(1000.0, -50.0, -30.0, -100.0, -50.0)
        assert result > 0.0

    def test_sign_with_dominant_negative_boron(self):
        """High boron dominates and produces large negative ρ_total."""
        result = reactivity_balance(100.0, -50.0, -30.0, -5000.0, -100.0)
        assert result < 0.0

    def test_typical_steady_state_near_critical(self):
        """Near-critical HFP: positive rod worth balanced by negative feedback sum."""
        rho_rod = 500.0    # pcm rod withdrawal
        rho_D = -400.0     # pcm Doppler (fuel at ~900 K vs CZP ~560 K)
        rho_m = -80.0      # pcm moderator (~30 K rise)
        rho_B = -5000.0    # pcm boron (500 ppm)
        rho_Xe = -2700.0   # pcm equilibrium xenon
        result = reactivity_balance(rho_rod, rho_D, rho_m, rho_B, rho_Xe)
        assert result == pytest.approx(rho_rod + rho_D + rho_m + rho_B + rho_Xe, abs=1e-10)

    def test_rod_ejection_scenario_is_positive(self):
        """Step rod ejection (+50 pcm) with frozen feedback produces positive ρ_total."""
        result = reactivity_balance(50.0, 0.0, 0.0, 0.0, 0.0)
        assert result == pytest.approx(50.0, abs=1e-10)

    def test_returns_float(self):
        """Return type is always a float, never None or NaN for valid inputs."""
        result = reactivity_balance(0.0, -100.0, -50.0, -200.0, -100.0)
        assert isinstance(result, float)
        assert not math.isnan(result)


# ===========================================================================
# Unit conversion — SPEC §4.6.4
# ===========================================================================

class TestPcmToDkk:
    """Tests for pcm_to_dk_k(rho_pcm)."""

    def test_one_pcm(self):
        """1 pcm = 1 × 10⁻⁵ Δk/k (SPEC §4.6.4 definition)."""
        assert pcm_to_dk_k(1.0) == pytest.approx(1.0e-5, rel=1e-12)

    def test_hundred_pcm(self):
        """100 pcm = 1 × 10⁻³ Δk/k."""
        assert pcm_to_dk_k(100.0) == pytest.approx(1.0e-3, rel=1e-12)

    def test_beta_eff_in_pcm(self):
        """β_eff ≈ 650 pcm for U-235; in Δk/k this is 6.5 × 10⁻³."""
        assert pcm_to_dk_k(650.0) == pytest.approx(6.5e-3, rel=1e-12)

    def test_zero_reactivity(self):
        """0 pcm → exactly 0.0 Δk/k (criticality)."""
        assert pcm_to_dk_k(0.0) == 0.0

    def test_negative_reactivity(self):
        """-500 pcm → -5 × 10⁻³ Δk/k (subcritical)."""
        assert pcm_to_dk_k(-500.0) == pytest.approx(-5.0e-3, rel=1e-12)

    def test_round_trip_consistency(self):
        """Converting pcm → Δk/k then multiplying back recovers the original value."""
        rho_pcm = -2700.0
        rho_dkk = pcm_to_dk_k(rho_pcm)
        recovered = rho_dkk / 1.0e-5
        assert recovered == pytest.approx(rho_pcm, rel=1e-12)
