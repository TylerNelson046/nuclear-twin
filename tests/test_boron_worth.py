"""
tests/test_boron_worth.py — Dedicated unit-test suite for the boron worth calculator.

Targets: physics/pwr/feedback.py::boron_worth  (SPEC Eq. 13)
         physics/pwr/parameters.py::PWRParameters (Pydantic boron_ppm field)

Physics background (SPEC §4.2, §4.5.1):
    Dissolved boric acid (H₃BO₃) absorbs thermal neutrons in proportion to
    concentration.  The net effect is a linear reactivity insertion:

        ρ_B = ω_B · C_B         (SPEC Eq. 13)

    where ω_B ≈ −10 pcm/ppm (CLAUDE.md §Physics Constants).

Coverage strategy:
    1. Core algebraic correctness  — hand-calculated reference values
    2. Fuel-cycle operating points — BOL / MOC / EOC critical boron concentrations
    3. Temperature-dependent interface — ω_B can be varied by the caller to model
       temperature-dependent worth without changing the function (SPEC says linear
       model; richer corrections are injected through the ω_B parameter).
    4. Input validation           — negative ppm returns NaN and logs warning (SR-01)
    5. Boron dilution scenario    — progressive concentration decrease produces
       monotonically increasing (less negative) reactivity, enabling Week 5 dilution
       scenario to trivially call boron_worth(concentration) at each step.
    6. Pydantic integration       — Pydantic schema rejects out-of-range boron_ppm
       before any physics function is reached (NFR-04).
"""

from __future__ import annotations

import logging
import math

import pytest
from pydantic import ValidationError

from physics.pwr.feedback import OMEGA_B_DEFAULT, boron_worth
from physics.pwr.parameters import PWRParameters

# ---------------------------------------------------------------------------
# Reference fuel-cycle boron concentrations (SPEC §3.2, SPEC §4.6.4)
# ---------------------------------------------------------------------------
# Hot full-power critical boron for a generic large commercial PWR (1000 MWe class).
# Concentrations are representative; exact values depend on enrichment and burnup.
BORON_BOL_PPM: float = 1500.0   # ppm — Beginning of Life (fresh fuel, high excess reactivity)
BORON_BOC_PPM: float = 1200.0   # ppm — Beginning of Cycle (first cycle, nominal startup)
BORON_MOC_PPM: float = 700.0    # ppm — Middle of Cycle
BORON_EOC_PPM: float = 50.0     # ppm — End of Cycle (near burnout, minimal boron needed)
BORON_ZERO_PPM: float = 0.0     # ppm — Unborated water (used in special tests only)

# Physical omega_B range per CLAUDE.md (−10 pcm/ppm is the canonical default;
# real plants vary ~−8 to −11 pcm/ppm depending on moderator temperature, enrichment).
OMEGA_B_COLD: float = -8.5      # pcm/ppm — representative cold-temperature worth
OMEGA_B_HOT: float = -10.5     # pcm/ppm — representative hot-temperature worth


# ===========================================================================
# 1. Core algebraic correctness — SPEC Eq. 13
# ===========================================================================

class TestBoronWorthCore:
    """Verify ρ_B = ω_B · C_B with hand-calculated reference values."""

    def test_spec_eq13_hand_calc_1000ppm(self):
        """1000 ppm × (−10 pcm/ppm) = −10 000 pcm  (SPEC Eq. 13 explicit example)."""
        assert boron_worth(1000.0, omega_B=-10.0) == pytest.approx(-10_000.0, abs=1e-10)

    def test_spec_eq13_hand_calc_500ppm(self):
        """500 ppm × (−10 pcm/ppm) = −5 000 pcm."""
        assert boron_worth(500.0, omega_B=-10.0) == pytest.approx(-5_000.0, abs=1e-10)

    def test_spec_eq13_hand_calc_2500ppm(self):
        """2500 ppm (Pydantic upper bound) × (−10 pcm/ppm) = −25 000 pcm."""
        assert boron_worth(2500.0, omega_B=-10.0) == pytest.approx(-25_000.0, abs=1e-10)

    def test_zero_ppm_returns_exactly_zero(self):
        """Zero boron concentration → zero reactivity contribution (no absorber present)."""
        assert boron_worth(0.0) == 0.0
        assert boron_worth(0.0, omega_B=-10.0) == 0.0

    def test_default_omega_b_applied_correctly(self):
        """Default ω_B = OMEGA_B_DEFAULT (−10 pcm/ppm) is applied when not specified."""
        result = boron_worth(100.0)
        assert result == pytest.approx(OMEGA_B_DEFAULT * 100.0, rel=1e-12)

    def test_negative_sign_for_any_positive_concentration(self):
        """ρ_B < 0 for all positive boron concentrations (boron is always a neutron poison)."""
        for ppm in [0.1, 1.0, 50.0, 500.0, 1000.0, 2500.0]:
            assert boron_worth(ppm) < 0.0, f"Expected negative worth at {ppm} ppm"

    def test_strictly_linear_in_concentration(self):
        """ρ_B scales linearly: doubling C_B exactly doubles |ρ_B| (SPEC Eq. 13 is linear)."""
        rho_500 = boron_worth(500.0)
        rho_1000 = boron_worth(1000.0)
        assert rho_1000 == pytest.approx(2.0 * rho_500, rel=1e-12)

    def test_linearity_holds_at_multiple_scales(self):
        """Linearity verified across orders of magnitude: 10 × 10 = 100, etc."""
        base = boron_worth(100.0)
        assert boron_worth(1000.0) == pytest.approx(10.0 * base, rel=1e-12)

    def test_proportional_to_omega_b(self):
        """Doubling ω_B (with same concentration) doubles ρ_B."""
        rho_default = boron_worth(500.0, omega_B=-10.0)
        rho_double = boron_worth(500.0, omega_B=-20.0)
        assert rho_double == pytest.approx(2.0 * rho_default, rel=1e-12)

    def test_return_type_is_float(self):
        """boron_worth always returns a float, never None, int, or NaN for valid inputs."""
        result = boron_worth(1000.0)
        assert isinstance(result, float)
        assert not math.isnan(result)


# ===========================================================================
# 2. Fuel-cycle operating points — BOL / MOC / EOC
# ===========================================================================

class TestBoronWorthFuelCyclePoints:
    """Verify reactivity at representative critical boron concentrations through the fuel cycle.

    Physical context (SPEC §4.2, §3.2):
        A PWR manages long-term reactivity by adjusting dissolved boron.  At BOL,
        fresh fuel has excess k_eff; high boron holds it down.  As fuel burns up,
        boron is diluted until the EOC point when boron is nearly exhausted.
    """

    def test_bol_large_negative_worth(self):
        """BOL (~1500 ppm): large negative worth to suppress fresh-fuel excess reactivity."""
        result = boron_worth(BORON_BOL_PPM)
        expected = OMEGA_B_DEFAULT * BORON_BOL_PPM
        assert result == pytest.approx(expected, rel=1e-12)
        assert result < -10_000.0, "BOL boron worth should be > 10 000 pcm in magnitude"

    def test_boc_worth_matches_formula(self):
        """BOC (1200 ppm): ρ_B = −12 000 pcm at default ω_B = −10 pcm/ppm."""
        result = boron_worth(BORON_BOC_PPM, omega_B=-10.0)
        assert result == pytest.approx(-12_000.0, abs=1e-10)

    def test_moc_worth(self):
        """MOC (700 ppm): ρ_B = −7 000 pcm at default ω_B."""
        result = boron_worth(BORON_MOC_PPM, omega_B=-10.0)
        assert result == pytest.approx(-7_000.0, abs=1e-10)

    def test_eoc_worth_small_magnitude(self):
        """EOC (~50 ppm): reactivity worth small — core is near its natural criticality."""
        result = boron_worth(BORON_EOC_PPM, omega_B=-10.0)
        assert result == pytest.approx(-500.0, abs=1e-10)
        assert abs(result) < 1_000.0, "EOC boron worth should be < 1000 pcm in magnitude"

    def test_bol_to_eoc_reactivity_increases_monotonically(self):
        """ρ_B increases (becomes less negative) from BOL to EOC as boron is diluted."""
        concentrations = [BORON_BOL_PPM, BORON_BOC_PPM, BORON_MOC_PPM, BORON_EOC_PPM, 0.0]
        worths = [boron_worth(c) for c in concentrations]
        for i in range(len(worths) - 1):
            assert worths[i] < worths[i + 1], (
                f"ρ_B must increase as concentration drops: {concentrations[i]} → {concentrations[i+1]}"
            )

    def test_bol_eoc_delta_reactivity(self):
        """BOL → EOC dilution adds back the full boron reactivity difference."""
        rho_bol = boron_worth(BORON_BOL_PPM)
        rho_eoc = boron_worth(BORON_EOC_PPM)
        delta = rho_eoc - rho_bol
        expected_delta = OMEGA_B_DEFAULT * (BORON_EOC_PPM - BORON_BOL_PPM)
        assert delta == pytest.approx(expected_delta, rel=1e-12)


# ===========================================================================
# 3. Temperature-dependent worth interface
# ===========================================================================

class TestBoronWorthTemperatureInterface:
    """Verify that ω_B can be varied to simulate temperature-dependent boron worth.

    SPEC Eq. 13 is a linear model (no built-in temperature correction).  Real plants
    see a slight variation in boron worth with moderator temperature (~10–15% across the
    operating range) because temperature affects moderator density and the neutron spectrum.
    The clean interface hook is the ω_B parameter — callers supply a temperature-corrected
    value rather than modifying the function itself.
    """

    def test_cold_omega_b_less_negative_than_default(self):
        """Cold ω_B (−8.5 pcm/ppm) produces smaller |ρ_B| than hot ω_B (−10.5 pcm/ppm)."""
        ppm = 1000.0
        rho_cold = boron_worth(ppm, omega_B=OMEGA_B_COLD)
        rho_hot = boron_worth(ppm, omega_B=OMEGA_B_HOT)
        assert abs(rho_cold) < abs(rho_hot)

    def test_cold_hot_worth_ratio_matches_omega_ratio(self):
        """ρ_B ratio equals ω_B ratio (linearity in ω_B)."""
        ppm = 1000.0
        rho_cold = boron_worth(ppm, omega_B=OMEGA_B_COLD)
        rho_hot = boron_worth(ppm, omega_B=OMEGA_B_HOT)
        omega_ratio = OMEGA_B_COLD / OMEGA_B_HOT
        assert rho_cold / rho_hot == pytest.approx(omega_ratio, rel=1e-12)

    def test_zero_ppm_temperature_independent(self):
        """At zero concentration, any ω_B gives zero worth — temperature correction irrelevant."""
        assert boron_worth(0.0, omega_B=OMEGA_B_COLD) == 0.0
        assert boron_worth(0.0, omega_B=OMEGA_B_HOT) == 0.0

    def test_temperature_corrected_boc_range(self):
        """BOC boron (1200 ppm) with cold ω_B: magnitude within 8500–12000 pcm range."""
        result_cold = boron_worth(BORON_BOC_PPM, omega_B=OMEGA_B_COLD)
        result_hot = boron_worth(BORON_BOC_PPM, omega_B=OMEGA_B_HOT)
        for result in (result_cold, result_hot):
            assert -12_600.0 <= result <= -8_500.0, (
                f"Temperature-corrected BOC boron worth {result} pcm outside expected range"
            )

    @pytest.mark.parametrize("omega_b", [-8.0, -9.0, -10.0, -11.0])
    def test_parametrized_omega_b_values(self, omega_b):
        """boron_worth(1000 ppm, ω_B) == ω_B × 1000 for any ω_B in the physical range."""
        result = boron_worth(1000.0, omega_B=omega_b)
        assert result == pytest.approx(omega_b * 1000.0, abs=1e-10)


# ===========================================================================
# 4. Input validation — SR-01, SPEC §4.6.4
# ===========================================================================

class TestBoronWorthValidation:
    """Verify graceful handling of unphysical inputs per SR-01 and CLAUDE.md rule 4.

    Physics functions must never raise unhandled exceptions.  Instead they return
    NaN and emit a WARNING-level log entry so the calling engine can detect and
    respond to the failure state.
    """

    def test_negative_concentration_returns_nan(self):
        """Negative ppm is physically impossible; function must return NaN, not a value."""
        assert math.isnan(boron_worth(-0.001))
        assert math.isnan(boron_worth(-1.0))
        assert math.isnan(boron_worth(-1000.0))

    def test_negative_concentration_logs_warning(self, caplog):
        """Negative boron triggers a WARNING-level log message describing the fault."""
        with caplog.at_level(logging.WARNING, logger="physics.pwr.feedback"):
            boron_worth(-5.0)
        assert len(caplog.records) >= 1
        assert any("unphysical" in r.message.lower() for r in caplog.records)

    def test_large_negative_concentration_still_nan(self, caplog):
        """Extremely negative ppm still returns NaN (not a computed value or exception)."""
        with caplog.at_level(logging.WARNING, logger="physics.pwr.feedback"):
            result = boron_worth(-99_999.0)
        assert math.isnan(result)

    def test_small_positive_epsilon_is_valid(self):
        """Near-zero positive concentration (epsilon above zero) returns a valid result."""
        result = boron_worth(1e-9)
        assert not math.isnan(result)
        assert result == pytest.approx(OMEGA_B_DEFAULT * 1e-9, rel=1e-6)

    def test_exactly_zero_is_valid(self):
        """Exactly zero concentration (boundary case) returns exactly 0.0, not NaN."""
        result = boron_worth(0.0)
        assert result == 0.0
        assert not math.isnan(result)

    def test_does_not_raise_exception_for_negative_input(self):
        """Negative ppm must not raise any exception — return path is NaN only."""
        try:
            boron_worth(-50.0)
        except Exception as exc:
            pytest.fail(f"boron_worth(-50) raised an unexpected exception: {exc}")

    @pytest.mark.parametrize("ppm", [0.0, 1.0, 100.0, 500.0, 1000.0, 1500.0, 2000.0, 2500.0])
    def test_valid_range_never_returns_nan(self, ppm):
        """All concentrations in the Pydantic-valid range [0, 2500] ppm return non-NaN."""
        result = boron_worth(ppm)
        assert not math.isnan(result), f"Unexpected NaN at {ppm} ppm"


# ===========================================================================
# 5. Boron dilution scenario — Week 5 readiness
# ===========================================================================

class TestBoronWorthDilutionScenario:
    """Verify monotonic reactivity increase during progressive boron dilution.

    Week 5 scenario context (CLAUDE.md §Milestone, Week 5):
        The boron dilution scenario decreases C_B over time (simulating coolant
        dilution by unborated water injection).  The physics requires that each
        concentration step adds a positive Δρ relative to the previous step.
        These tests confirm that boron_worth() is ready to serve as the
        concentration → reactivity mapping in that scenario loop.
    """

    # Dilution sequence: high → low boron concentration
    DILUTION_SEQUENCE = [1200.0, 1000.0, 800.0, 600.0, 400.0, 200.0, 100.0, 10.0, 0.0]

    def test_dilution_sequence_is_monotonically_increasing(self):
        """Each dilution step increases ρ_B (reduces |ρ_B|), moving core toward criticality."""
        worths = [boron_worth(c) for c in self.DILUTION_SEQUENCE]
        for i in range(len(worths) - 1):
            assert worths[i] < worths[i + 1], (
                f"ρ_B must increase at each dilution step: "
                f"C_B={self.DILUTION_SEQUENCE[i]} → {self.DILUTION_SEQUENCE[i+1]}"
            )

    def test_dilution_delta_reactivity_is_positive(self):
        """Each dilution step contributes positive Δρ (as concentration decreases)."""
        for i in range(len(self.DILUTION_SEQUENCE) - 1):
            c_before = self.DILUTION_SEQUENCE[i]
            c_after = self.DILUTION_SEQUENCE[i + 1]
            delta_rho = boron_worth(c_after) - boron_worth(c_before)
            assert delta_rho > 0.0, (
                f"Δρ must be positive for dilution {c_before} → {c_after} ppm"
            )

    def test_total_reactivity_insertion_from_full_dilution(self):
        """Full dilution from 1200 → 0 ppm inserts exactly |ω_B × 1200| pcm."""
        rho_start = boron_worth(1200.0)
        rho_end = boron_worth(0.0)
        total_insertion = rho_end - rho_start
        expected = -OMEGA_B_DEFAULT * 1200.0   # negative × negative = positive
        assert total_insertion == pytest.approx(expected, rel=1e-12)
        assert total_insertion > 0.0, "Full dilution must be a positive reactivity insertion"

    def test_dilution_step_proportional_to_concentration_change(self):
        """Reactivity change between any two steps equals ω_B × ΔC_B (linearity)."""
        c1, c2 = 800.0, 600.0
        delta_rho = boron_worth(c2) - boron_worth(c1)
        expected = OMEGA_B_DEFAULT * (c2 - c1)
        assert delta_rho == pytest.approx(expected, rel=1e-12)

    def test_dilution_to_zero_removes_all_boron_worth(self):
        """Diluting to 0 ppm leaves zero boron reactivity — no absorber remaining."""
        assert boron_worth(0.0) == 0.0

    def test_rapid_dilution_accumulates_reactivity_linearly(self):
        """Rapid dilution from 1000 → 500 → 0 ppm accumulates the same Δρ as direct 1000 → 0."""
        delta_stepwise = (boron_worth(500.0) - boron_worth(1000.0)) + (
            boron_worth(0.0) - boron_worth(500.0)
        )
        delta_direct = boron_worth(0.0) - boron_worth(1000.0)
        assert delta_stepwise == pytest.approx(delta_direct, rel=1e-12)


# ===========================================================================
# 6. Pydantic schema integration — NFR-04
# ===========================================================================

class TestBoronWorthPydanticIntegration:
    """Confirm the PWRParameters Pydantic schema rejects out-of-range boron_ppm values.

    The schema is the input boundary (CLAUDE.md architecture rule 3, NFR-04):
    physics functions never see invalid inputs in production code paths.
    These tests verify that guarantee holds.
    """

    def test_pydantic_accepts_zero_boron(self):
        """boron_ppm = 0.0 is valid (unborated, e.g. end-of-cycle or shutdown)."""
        p = PWRParameters(boron_ppm=0.0)
        assert p.boron_ppm == 0.0

    def test_pydantic_accepts_nominal_boron(self):
        """boron_ppm = 1000.0 (default nominal operating point) is valid."""
        p = PWRParameters(boron_ppm=1000.0)
        assert p.boron_ppm == 1000.0

    def test_pydantic_accepts_maximum_boron(self):
        """boron_ppm = 2500.0 (Pydantic upper bound) is accepted."""
        p = PWRParameters(boron_ppm=2500.0)
        assert p.boron_ppm == 2500.0

    def test_pydantic_rejects_negative_boron(self):
        """boron_ppm < 0 must raise Pydantic ValidationError — never reaches physics layer."""
        with pytest.raises(ValidationError):
            PWRParameters(boron_ppm=-1.0)

    def test_pydantic_rejects_boron_above_max(self):
        """boron_ppm > 2500 must raise Pydantic ValidationError."""
        with pytest.raises(ValidationError):
            PWRParameters(boron_ppm=2501.0)

    def test_pydantic_validated_boron_gives_correct_reactivity(self):
        """End-to-end: Pydantic validates → physics function computes correct worth."""
        p = PWRParameters(boron_ppm=BORON_BOC_PPM)
        rho = boron_worth(p.boron_ppm, omega_B=-10.0)
        assert rho == pytest.approx(-12_000.0, abs=1e-10)

    @pytest.mark.parametrize("ppm", [0.0, 50.0, 500.0, 1000.0, 1500.0, 2000.0, 2500.0])
    def test_parametrized_valid_range_passes_pydantic_and_physics(self, ppm):
        """Every valid Pydantic boron_ppm value passes schema and produces correct worth."""
        p = PWRParameters(boron_ppm=ppm)
        result = boron_worth(p.boron_ppm)
        assert result == pytest.approx(OMEGA_B_DEFAULT * ppm, rel=1e-12)
        assert not math.isnan(result)
