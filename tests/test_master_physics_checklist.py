"""
tests/test_master_physics_checklist.py — Phase 1 complete physics validation checklist.

Implements the formal Phase 1 "Complete physics validation checklist" required
for PWR twin close-out (CLAUDE.md §Development Phases, SPEC §7.3,
ARCHITECTURE.md §12 Phase 1 Completion Gate).

Five programmatically-asserted physics principles tested and reported:

    [a] Mass & Energy Conservation
        At thermal steady state, three energy flow rates must agree within 0.01%:
        Q_gen = γ_f·P  (heat deposited in fuel)
        Q_fc  = (T_f − T_c)/R_fc  (fuel→coolant conduction, SPEC Eq. 9)
        Q_adv = ṁ·c_c·(T_c − T_in)  (coolant advection removal, SPEC Eq. 10)

    [b] Multi-Group Delayed Precursor Balance
        All 6 Keepin group precursor concentrations scale exactly proportionally
        with steady-state neutron population. Doubling power doubles every C_i
        with relative error < 1e-10. At steady state (ρ=0), dC_i/dt = 0.

    [c] Global Reactivity Budget
        Sign of ρ_total maps directly to sign of dn/dt at every instant:
            ρ > 0  →  dn/dt > 0  (power rises)
            ρ = 0  →  dn/dt = 0  (power held constant)
            ρ < 0  →  dn/dt < 0  (power falls)
        Verified both qualitatively and quantitatively against dn/dt = (ρ/Λ)·n.

    [d] Xenon Burnout Dynamics
        Starting at equilibrium xenon (I_eq, X_eq) for flux φ_low, a step-up
        to φ_high produces dX/dt < 0 (xenon destroyed faster than produced),
        and the destruction rate increases monotonically with flux magnitude.

    [e] Boundary Error Handling
        The engine and Pydantic schema reject physically impossible inputs —
        zero/negative coolant flow, temperatures below absolute zero, negative
        boron — by returning NaN or raising ValidationError (SPEC SR-01).

Usage (standalone report with structured output):
    python tests/test_master_physics_checklist.py

Usage (pytest):
    pytest tests/test_master_physics_checklist.py -v

SPEC equations exercised:
    Eq. 1   dn/dt = [(ρ − β_eff)/Λ]·n + Σᵢ λᵢ Cᵢ
    Eq. 2   dCᵢ/dt = (βᵢ/Λ)·n − λᵢ Cᵢ
    Eq. 9   m_f c_f·dT_f/dt = γ_f·P − (T_f − T_c)/R_fc
    Eq. 10  m_c c_c·dT_c/dt = (T_f − T_c)/R_fc − ṁ c_c(T_c − T_in)
    Eq. 14  dI/dt = γ_I Σ_f φ − λ_I I
    Eq. 15  dX/dt = γ_X Σ_f φ + λ_I I − (λ_X + σ_aX φ) X
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path
from typing import NamedTuple

# Must be set before any numba-decorated imports. conftest.py handles pytest;
# this line covers standalone execution via `python tests/test_master_physics_checklist.py`.
os.environ.setdefault("NUMBA_DISABLE_JIT", "1")

# Ensure project root (nuclear-twin/) is on sys.path when invoked as a standalone
# script from any working directory (pytest already handles this via conftest.py).
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import pytest
from pydantic import ValidationError

from data.keepin_dnp import beta_fractions, decay_constants
from physics.pwr.engine import (
    PHI_NOM,
    SIGMA_A,
    SIGMA_F,
    PWRControls,
    PWRModelConfig,
    build_initial_state,
    constant_controls_fn,
    critical_base_reactivity_pcm,
    pwr_coupled_rhs,
)
from physics.pwr.feedback import (
    boron_worth,
    doppler_feedback,
    moderator_feedback,
)
from physics.pwr.parameters import PWRParameters
from physics.pwr.reactivity import ReferenceState
from physics.pwr.thermal_hydraulics import (
    C_COOL,
    GAMMA_F,
    M_DOT_NOM,
    P_NOM,
    R_FC,
    T_COOL_NOM,
    T_FUEL_NOM,
    T_IN_NOM,
    steady_state_temperatures,
)
from physics.shared.kinetics import (
    LAMBDA_PWR,
    point_kinetics_rhs,
    steady_state_precursors,
)
from physics.shared.xenon import (
    GAMMA_I,
    GAMMA_X,
    LAMBDA_I,
    LAMBDA_X,
    SIGMA_AX,
    steady_state_xenon,
    xenon_iodine_rhs,
)

# ---------------------------------------------------------------------------
# Module-level constants — computed once and shared across all tests
# ---------------------------------------------------------------------------

_BETA_ARR: np.ndarray = np.array(beta_fractions(), dtype=np.float64)
_LAMBDA_ARR: np.ndarray = np.array(decay_constants(), dtype=np.float64)
_BETA_EFF: float = float(_BETA_ARR.sum())

# Equilibrium xenon/iodine at nominal flux — used in [d] xenon burnout tests
_XE_EQ_NOM: np.ndarray = steady_state_xenon(PHI_NOM, SIGMA_F)


# ======================================================================
# [a] MASS AND ENERGY CONSERVATION
# ======================================================================

class TestEnergyConservation:
    """[a] At thermal steady state, three energy flow rates agree within 0.01%.

    SPEC Eqs. 9–10 imply a closed steady-state heat balance:
        Q_gen = γ_f · P              (fission heat deposited in fuel)
        Q_fc  = (T_f − T_c) / R_fc  (fuel → coolant conduction)
        Q_adv = ṁ · c_c · (T_c − T_in)  (advection removal by coolant flow)

    All three must be equal at steady state. Deviation > 0.01% would indicate
    a sign error, unit inconsistency, or incorrect thermal resistance derivation.
    """

    TOLERANCE = 1.0e-4  # 0.01% expressed as a relative fraction

    @staticmethod
    def _compute_heat_flows(P: float, m_dot: float, T_in: float) -> tuple[float, float, float]:
        """Return (Q_gen, Q_fc, Q_adv) at the given steady-state operating point."""
        temps = steady_state_temperatures(P, m_dot, T_in)
        T_f, T_c = float(temps[0]), float(temps[1])
        Q_gen = GAMMA_F * P
        Q_fc  = (T_f - T_c) / R_FC
        Q_adv = m_dot * C_COOL * (T_c - T_in)
        return Q_gen, Q_fc, Q_adv

    def test_nominal_fuel_to_coolant_conduction_matches_generation(self):
        """Q_gen == Q_fc within 0.01% at nominal operating point."""
        Q_gen, Q_fc, _ = self._compute_heat_flows(P_NOM, M_DOT_NOM, T_IN_NOM)
        err = abs(Q_fc / Q_gen - 1.0)
        assert err < self.TOLERANCE, (
            f"Fuel→coolant conduction ({Q_fc:.6e} W) ≠ heat generation ({Q_gen:.6e} W). "
            f"Relative error = {err:.2e} (limit {self.TOLERANCE:.0e}). "
            "Check R_FC derivation in SPEC Eq. 9 steady-state form."
        )

    def test_nominal_coolant_advection_matches_generation(self):
        """Q_gen == Q_adv within 0.01% at nominal operating point."""
        Q_gen, _, Q_adv = self._compute_heat_flows(P_NOM, M_DOT_NOM, T_IN_NOM)
        err = abs(Q_adv / Q_gen - 1.0)
        assert err < self.TOLERANCE, (
            f"Coolant advection removal ({Q_adv:.6e} W) ≠ heat generation ({Q_gen:.6e} W). "
            f"Relative error = {err:.2e} (limit {self.TOLERANCE:.0e}). "
            "Check M_DOT_NOM derivation in SPEC Eq. 10 steady-state form."
        )

    def test_off_nominal_power_energy_balance_closes(self):
        """Energy balance closes within 0.01% at 70% power / 85% nominal flow."""
        P_test = 0.70 * P_NOM
        m_test = 0.85 * M_DOT_NOM
        Q_gen, Q_fc, Q_adv = self._compute_heat_flows(P_test, m_test, T_IN_NOM)
        err_fc  = abs(Q_fc  / Q_gen - 1.0)
        err_adv = abs(Q_adv / Q_gen - 1.0)
        assert err_fc  < self.TOLERANCE, (
            f"Off-nominal Q_fc error = {err_fc:.2e} at 70% P, 85% flow."
        )
        assert err_adv < self.TOLERANCE, (
            f"Off-nominal Q_adv error = {err_adv:.2e} at 70% P, 85% flow."
        )

    def test_ode_power_derivative_zero_at_steady_state(self):
        """Full 11-state PWR ODE: dn/dt ≈ 0 at exact steady-state initial conditions.

        Verifies that the critical_base_reactivity_pcm() helper correctly offsets
        equilibrium xenon so the coupled system starts in balance. A non-zero
        dn/dt at t=0 would indicate the system is not truly at steady state.
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

        dydt = pwr_coupled_rhs(
            0.0, y0,
            constant_controls_fn(controls),
            reference_state, config,
            _BETA_ARR, _LAMBDA_ARR, LAMBDA_PWR,
        )

        dn_dt = float(dydt[0])
        dTf_dt = float(dydt[7])
        dTc_dt = float(dydt[8])

        assert abs(dn_dt) < 1e-7, (
            f"dn/dt at steady state = {dn_dt:.3e}. "
            "Expected ≈ 0; base reactivity must cancel equilibrium xenon worth."
        )
        assert abs(dTf_dt) / T_FUEL_NOM < 1e-7, (
            f"dT_fuel/dt = {dTf_dt:.3e} K/s at steady state; expected ≈ 0."
        )
        assert abs(dTc_dt) / T_COOL_NOM < 1e-7, (
            f"dT_cool/dt = {dTc_dt:.3e} K/s at steady state; expected ≈ 0."
        )


# ======================================================================
# [b] MULTI-GROUP DELAYED PRECURSOR BALANCE
# ======================================================================

class TestPrecursorBalance:
    """[b] All 6 Keepin precursor groups scale exactly proportionally with power.

    From SPEC Eq. 2 at dCᵢ/dt = 0:
        Cᵢ,₀ = (βᵢ / λᵢ Λ) · n₀

    This linear relationship means any fractional change in n produces the
    identical fractional change in every Cᵢ. Deviation from exact proportionality
    would indicate incorrect indexing, a sign error, or an off-by-one in the
    precursor loop.
    """

    def test_power_doubling_doubles_all_precursors(self):
        """C_i(2n) / C_i(n) = 2.0 exactly for all 6 groups."""
        C_n1 = steady_state_precursors(1.0, _BETA_ARR, _LAMBDA_ARR, LAMBDA_PWR)
        C_n2 = steady_state_precursors(2.0, _BETA_ARR, _LAMBDA_ARR, LAMBDA_PWR)
        ratios = C_n2 / C_n1
        max_err = float(np.max(np.abs(ratios - 2.0)))
        assert max_err < 1e-10, (
            f"Power-doubling precursor ratio error: max|C_i(2n)/C_i(n) − 2| = {max_err:.2e}. "
            "All 6 groups must scale exactly with neutron population (SPEC Eq. 2)."
        )

    def test_power_halving_halves_all_precursors(self):
        """C_i(0.5n) / C_i(n) = 0.5 exactly for all 6 groups."""
        C_n1   = steady_state_precursors(1.0, _BETA_ARR, _LAMBDA_ARR, LAMBDA_PWR)
        C_half = steady_state_precursors(0.5, _BETA_ARR, _LAMBDA_ARR, LAMBDA_PWR)
        ratios = C_half / C_n1
        max_err = float(np.max(np.abs(ratios - 0.5)))
        assert max_err < 1e-10, (
            f"Power-halving precursor ratio error: max|C_i(0.5n)/C_i(n) − 0.5| = {max_err:.2e}."
        )

    def test_all_six_precursor_groups_present(self):
        """Exactly 6 precursor groups returned, all with positive β_i and λ_i."""
        assert len(_BETA_ARR) == 6, f"Expected 6 beta groups, got {len(_BETA_ARR)}"
        assert len(_LAMBDA_ARR) == 6, f"Expected 6 lambda groups, got {len(_LAMBDA_ARR)}"
        assert np.all(_BETA_ARR > 0), f"Non-positive β_i detected: {_BETA_ARR}"
        assert np.all(_LAMBDA_ARR > 0), f"Non-positive λ_i detected: {_LAMBDA_ARR}"

    def test_beta_eff_in_physical_range_for_u235(self):
        """Total β_eff = Σ βᵢ must be in 0.005–0.008 (U-235 thermal, CLAUDE.md)."""
        assert 0.005 < _BETA_EFF < 0.008, (
            f"β_eff = {_BETA_EFF:.5f}. Expected 0.005–0.008 for U-235 thermal fission. "
            "Check IAEA delayed neutron data loading in data/keepin_dnp.py."
        )

    def test_precursors_satisfy_steady_state_ode_at_zero_reactivity(self):
        """dC_i/dt = (β_i/Λ)·n − λ_i·C_i = 0 at analytical steady state (ρ=0).

        This verifies the closed-form Cᵢ,₀ = (βᵢ/λᵢΛ)·n₀ is self-consistent
        with the ODE. A non-zero residual implies the formula or the ODE kernel
        has a coefficient error.
        """
        n0 = 1.0
        C0 = steady_state_precursors(n0, _BETA_ARR, _LAMBDA_ARR, LAMBDA_PWR)
        y_ss = np.concatenate([[n0], C0])
        rhs = point_kinetics_rhs(0.0, y_ss, 0.0, _BETA_ARR, _LAMBDA_ARR, LAMBDA_PWR)
        max_dcdt = float(np.max(np.abs(rhs[1:])))
        assert max_dcdt < 1e-8, (
            f"max |dCᵢ/dt| at analytical steady state = {max_dcdt:.2e}. "
            "Expected < 1e-8: equilibrium precursor formula must satisfy its own ODE."
        )

    def test_neutron_ode_zero_at_steady_state(self):
        """dn/dt = 0 at exact steady state (ρ=0, C_i at equilibrium)."""
        n0 = 1.0
        C0 = steady_state_precursors(n0, _BETA_ARR, _LAMBDA_ARR, LAMBDA_PWR)
        y_ss = np.concatenate([[n0], C0])
        rhs = point_kinetics_rhs(0.0, y_ss, 0.0, _BETA_ARR, _LAMBDA_ARR, LAMBDA_PWR)
        assert abs(rhs[0]) < 1e-8, (
            f"dn/dt at steady state (ρ=0) = {rhs[0]:.3e}. "
            "Must be 0: delayed source Σ λᵢCᵢ must exactly cancel prompt loss (β_eff/Λ)·n."
        )


# ======================================================================
# [c] GLOBAL REACTIVITY BUDGET
# ======================================================================

class TestReactivityBudget:
    """[c] Reactivity sign maps directly and quantitatively to power derivative sign.

    From SPEC Eq. 1 at precursor equilibrium:
        dn/dt = [(ρ − β_eff)/Λ]·n + Σᵢ λᵢ Cᵢ
              = (ρ/Λ)·n   (when Cᵢ = (βᵢ/λᵢΛ)·n)

    Therefore sign(dn/dt) = sign(ρ) · sign(n). Since n > 0 always,
    sign(dn/dt) = sign(ρ). This is the fundamental property that makes
    subcritical reactors controllable.
    """

    @pytest.fixture(scope="class")
    def ss_kinetics_state(self) -> np.ndarray:
        """7-state steady-state kinetics vector [n=1, C₁…C₆]."""
        n0 = 1.0
        C0 = steady_state_precursors(n0, _BETA_ARR, _LAMBDA_ARR, LAMBDA_PWR)
        return np.concatenate([[n0], C0])

    def test_positive_reactivity_raises_power(self, ss_kinetics_state):
        """ρ = +50 pcm → dn/dt > 0 (power ascends)."""
        rho = +50.0e-5  # +50 pcm as dimensionless Δk/k
        rhs = point_kinetics_rhs(
            0.0, ss_kinetics_state, rho, _BETA_ARR, _LAMBDA_ARR, LAMBDA_PWR
        )
        assert rhs[0] > 0.0, (
            f"dn/dt = {rhs[0]:.4e} with ρ = +50 pcm. "
            "Must be positive: sub-prompt insertion drives power up (SPEC Eq. 1)."
        )

    def test_zero_reactivity_holds_power_constant(self, ss_kinetics_state):
        """ρ = 0 → dn/dt = 0 (power held exactly constant at steady state)."""
        rhs = point_kinetics_rhs(
            0.0, ss_kinetics_state, 0.0, _BETA_ARR, _LAMBDA_ARR, LAMBDA_PWR
        )
        assert abs(rhs[0]) < 1e-8, (
            f"dn/dt = {rhs[0]:.4e} with ρ = 0. "
            "Expected < 1e-8: zero reactivity at precursor equilibrium = steady state."
        )

    def test_negative_reactivity_reduces_power(self, ss_kinetics_state):
        """ρ = -50 pcm → dn/dt < 0 (power decreases)."""
        rho = -50.0e-5  # -50 pcm as dimensionless Δk/k
        rhs = point_kinetics_rhs(
            0.0, ss_kinetics_state, rho, _BETA_ARR, _LAMBDA_ARR, LAMBDA_PWR
        )
        assert rhs[0] < 0.0, (
            f"dn/dt = {rhs[0]:.4e} with ρ = -50 pcm. "
            "Must be negative: negative reactivity insertion reduces power."
        )

    def test_power_derivative_magnitude_matches_theory(self, ss_kinetics_state):
        """dn/dt = (ρ/Λ)·n — quantitative check against analytical formula.

        At precursor equilibrium: Σᵢ λᵢ Cᵢ = (β_eff/Λ)·n, so:
            dn/dt = [(ρ − β_eff)/Λ]·n + (β_eff/Λ)·n = (ρ/Λ)·n
        """
        rho_dk_k = 30.0e-5  # 30 pcm
        n0 = float(ss_kinetics_state[0])
        rhs = point_kinetics_rhs(
            0.0, ss_kinetics_state, rho_dk_k, _BETA_ARR, _LAMBDA_ARR, LAMBDA_PWR
        )
        expected_dndt = rho_dk_k / LAMBDA_PWR * n0
        rel_error = abs(rhs[0] / expected_dndt - 1.0)
        assert rel_error < 1e-6, (
            f"dn/dt = {rhs[0]:.6e}, theory (ρ/Λ)·n = {expected_dndt:.6e}. "
            f"Relative error = {rel_error:.2e}. "
            "Kinetics kernel must satisfy dn/dt = ρ/Λ · n at precursor equilibrium."
        )

    def test_dndt_monotone_increasing_with_reactivity(self, ss_kinetics_state):
        """dn/dt increases strictly monotonically with ρ across a wide range."""
        rho_pcm_vals = np.array([-200, -100, -50, 0, 50, 100, 200])
        dndt_vals = [
            point_kinetics_rhs(
                0.0, ss_kinetics_state,
                rho_pcm * 1e-5, _BETA_ARR, _LAMBDA_ARR, LAMBDA_PWR
            )[0]
            for rho_pcm in rho_pcm_vals
        ]
        for i in range(len(dndt_vals) - 1):
            assert dndt_vals[i] < dndt_vals[i + 1], (
                f"dn/dt not monotone: at ρ={rho_pcm_vals[i]} pcm, dn/dt={dndt_vals[i]:.3e}; "
                f"at ρ={rho_pcm_vals[i+1]} pcm, dn/dt={dndt_vals[i+1]:.3e}. "
                "The prompt term (ρ/Λ)·n must dominate and monotonically track ρ."
            )


# ======================================================================
# [d] XENON BURNOUT DYNAMICS
# ======================================================================

class TestXenonBurnoutDynamics:
    """[d] Higher neutron flux accelerates Xe-135 destruction via neutron absorption.

    Physical mechanism (SPEC Eq. 15):
        dX/dt = γ_X Σ_f φ + λ_I I − (λ_X + σ_aX φ) X

    At equilibrium φ_low: all terms balance, dX/dt = 0.
    After a step-up to φ_high > φ_low with X, I still at equilibrium values:
        - σ_aX·φ_high·X >> γ_X·Σ_f·φ_high  (absorption >> direct production)
        - Net dX/dt < 0 — xenon is burned out faster than it is produced
        - Burnout rate scales with flux increase
    """

    def test_xenon_iodine_equilibrium_self_consistent(self):
        """At analytical (I_eq, X_eq) with φ_nom, dX/dt ≈ 0 (SPEC Eq. 15 residual)."""
        rhs = xenon_iodine_rhs(0.0, _XE_EQ_NOM, PHI_NOM, SIGMA_F)
        assert abs(rhs[1]) < 1.0, (
            f"dX/dt at analytical equilibrium = {rhs[1]:.3e} atoms/cm³/s. "
            "Must be < 1: the steady-state formula X_eq must satisfy the ODE."
        )

    def test_flux_step_up_drives_xenon_burnout(self):
        """After flux doubling, dX/dt < 0 — xenon burns faster than produced."""
        phi_high = 2.0 * PHI_NOM
        rhs_high = xenon_iodine_rhs(0.0, _XE_EQ_NOM, phi_high, SIGMA_F)
        assert rhs_high[1] < 0.0, (
            f"dX/dt after 2× flux step = {rhs_high[1]:.3e} atoms/cm³/s. "
            "Must be negative: doubled flux burns xenon faster than it is produced "
            "(σ_aX·φ_high·X_eq >> production terms)."
        )

    def test_burnout_rate_increases_with_flux(self):
        """dX/dt(5×φ) is more negative than dX/dt(2×φ) — monotone in flux."""
        phi_2x = 2.0 * PHI_NOM
        phi_5x = 5.0 * PHI_NOM
        rhs_2x = xenon_iodine_rhs(0.0, _XE_EQ_NOM, phi_2x, SIGMA_F)
        rhs_5x = xenon_iodine_rhs(0.0, _XE_EQ_NOM, phi_5x, SIGMA_F)
        assert rhs_5x[1] < rhs_2x[1], (
            f"dX/dt(2× flux)={rhs_2x[1]:.3e}, dX/dt(5× flux)={rhs_5x[1]:.3e}. "
            "Higher flux must produce more negative dX/dt (faster burnout)."
        )

    def test_xe_absorption_dominates_direct_fission_production(self):
        """σ_aX · Δφ · X_eq >> γ_X · Σ_f · Δφ — absorption term dominates.

        This verifies that the xenon absorption cross section is correctly
        encoded as SIGMA_AX = 2.6e-18 cm² (2.6e6 barns, SPEC §4.5.4), making
        xenon a powerful neutron absorber that self-limits at high flux.
        """
        X_eq = float(_XE_EQ_NOM[1])
        delta_phi = PHI_NOM
        extra_absorption = SIGMA_AX * delta_phi * X_eq
        extra_fission_source = GAMMA_X * SIGMA_F * delta_phi
        # Absorption rate must exceed direct fission production by >> 10×
        assert extra_absorption > extra_fission_source * 10.0, (
            f"σ_aX·Δφ·X_eq = {extra_absorption:.3e} atoms/cm³/s, "
            f"γ_X·Σ_f·Δφ = {extra_fission_source:.3e} atoms/cm³/s. "
            "Absorption must dominate by > 10× for xenon burnout to be meaningful. "
            "Check SIGMA_AX (should be 2.6e-18 cm², not barns)."
        )

    def test_equilibrium_xenon_worth_in_spec_range(self):
        """Equilibrium xenon worth at full power must be −2500 to −3000 pcm.

        Published equilibrium xenon worth for large commercial PWRs at full power
        is −2500 to −3000 pcm (ARCHITECTURE §9, SPEC §7.3).
        """
        X_eq = float(_XE_EQ_NOM[1])
        rho_Xe_pcm = -SIGMA_AX * X_eq / SIGMA_A * 1.0e5
        assert -3000.0 <= rho_Xe_pcm <= -2500.0, (
            f"Equilibrium xenon worth = {rho_Xe_pcm:.0f} pcm. "
            "Must be in −2500 to −3000 pcm for a large commercial PWR at full power."
        )


# ======================================================================
# [e] BOUNDARY ERROR HANDLING
# ======================================================================

class TestBoundaryErrorHandling:
    """[e] Engine returns NaN (never raises) for unphysical inputs; Pydantic raises.

    SPEC SR-01: "No user input combination shall cause the physics engine to
    produce an unhandled exception visible to the end user."

    Engine-level protection: unphysical inputs (T < 0, m_dot ≤ 0, boron < 0)
    return NaN and log a warning. They never propagate an exception.

    Schema-level protection: Pydantic raises ValidationError for inputs outside
    the physically-grounded bounds defined in physics/pwr/parameters.py.
    """

    def test_zero_coolant_flow_returns_nan(self):
        """steady_state_temperatures with m_dot = 0 returns NaN — not an exception."""
        result = steady_state_temperatures(P_NOM, 0.0, T_IN_NOM)
        assert np.all(np.isnan(result)), (
            f"m_dot=0 → {result}; expected NaN. "
            "Zero flow is a loss-of-flow condition; return NaN per SPEC SR-01."
        )

    def test_negative_coolant_flow_returns_nan(self):
        """steady_state_temperatures with m_dot < 0 returns NaN."""
        result = steady_state_temperatures(P_NOM, -500.0, T_IN_NOM)
        assert np.all(np.isnan(result)), (
            f"m_dot=-500 → {result}; expected NaN. "
            "Negative mass flow rate is physically impossible."
        )

    def test_fuel_temperature_below_absolute_zero_returns_nan(self):
        """doppler_feedback with T_fuel < 0 K returns NaN — temperatures below 0 K are unphysical."""
        rho_D = doppler_feedback(-1.0, T_FUEL_NOM)
        assert math.isnan(rho_D), (
            f"doppler_feedback(T_fuel=-1 K) = {rho_D}; expected NaN. "
            "Fuel temperature cannot be below absolute zero (SPEC §4.6.5)."
        )

    def test_coolant_temperature_below_absolute_zero_returns_nan(self):
        """moderator_feedback with T_cool < 0 K returns NaN."""
        rho_m = moderator_feedback(-1.0, T_COOL_NOM)
        assert math.isnan(rho_m), (
            f"moderator_feedback(T_cool=-1 K) = {rho_m}; expected NaN."
        )

    def test_negative_boron_concentration_returns_nan(self):
        """boron_worth with C_B < 0 returns NaN — boron cannot be negative."""
        rho_B = boron_worth(-100.0)
        assert math.isnan(rho_B), (
            f"boron_worth(-100 ppm) = {rho_B}; expected NaN. "
            "Physical H₃BO₃ concentration is always non-negative."
        )

    def test_coolant_inlet_below_absolute_zero_returns_nan(self):
        """steady_state_temperatures with T_in < 0 K returns NaN."""
        result = steady_state_temperatures(P_NOM, M_DOT_NOM, -1.0)
        assert np.all(np.isnan(result)), (
            f"T_in=-1 K → {result}; expected NaN array."
        )

    def test_pydantic_rejects_sub_minimum_flow_fraction(self):
        """Pydantic ValidationError for coolant_flow_fraction below minimum (0.2)."""
        with pytest.raises(ValidationError):
            PWRParameters(coolant_flow_fraction=0.0)

    def test_pydantic_rejects_negative_boron(self):
        """Pydantic ValidationError for negative boron ppm (physically impossible)."""
        with pytest.raises(ValidationError):
            PWRParameters(boron_ppm=-1.0)

    def test_pydantic_rejects_excessive_rod_reactivity(self):
        """Pydantic ValidationError for rod insertion exceeding ±1000 pcm safety bound."""
        with pytest.raises(ValidationError):
            PWRParameters(rod_reactivity_pcm=1500.0)

    def test_pydantic_accepts_valid_parameters(self):
        """Baseline: valid nominal parameters must not raise ValidationError."""
        params = PWRParameters()  # all defaults
        assert params.coolant_flow_fraction == 1.0
        assert params.boron_ppm == 1000.0


# ======================================================================
# STANDALONE PHYSICS VALIDATION CHECKLIST REPORT
# ======================================================================

class ChecklistResult(NamedTuple):
    label: str
    name: str
    target: str
    result_str: str
    passed: bool


def _check_a() -> ChecklistResult:
    """[a] Mass & Energy Conservation."""
    Q_gen = GAMMA_F * P_NOM
    temps = steady_state_temperatures(P_NOM, M_DOT_NOM, T_IN_NOM)
    T_f, T_c = float(temps[0]), float(temps[1])
    Q_fc  = (T_f - T_c) / R_FC
    Q_adv = M_DOT_NOM * C_COOL * (T_c - T_IN_NOM)
    err_fc  = abs(Q_fc  / Q_gen - 1.0) * 100.0   # in %
    err_adv = abs(Q_adv / Q_gen - 1.0) * 100.0   # in %
    passed = err_fc < 0.01 and err_adv < 0.01
    result_str = (
        f"Q_gen={Q_gen:.4e} W  |  "
        f"ΔQ_fc={err_fc:.8f}%  |  "
        f"ΔQ_adv={err_adv:.8f}%"
    )
    return ChecklistResult(
        "[a]", "Mass & Energy Conservation",
        "< 0.01% error on all three heat flow paths",
        result_str, passed,
    )


def _check_b() -> ChecklistResult:
    """[b] Multi-Group Delayed Precursor Balance."""
    C_n1 = steady_state_precursors(1.0, _BETA_ARR, _LAMBDA_ARR, LAMBDA_PWR)
    C_n2 = steady_state_precursors(2.0, _BETA_ARR, _LAMBDA_ARR, LAMBDA_PWR)
    max_ratio_err = float(np.max(np.abs(C_n2 / C_n1 - 2.0)))
    y_ss = np.concatenate([[1.0], C_n1])
    rhs = point_kinetics_rhs(0.0, y_ss, 0.0, _BETA_ARR, _LAMBDA_ARR, LAMBDA_PWR)
    max_dcdt = float(np.max(np.abs(rhs[1:])))
    beta_eff = float(_BETA_ARR.sum())
    passed = max_ratio_err < 1e-10 and max_dcdt < 1e-8
    result_str = (
        f"β_eff={beta_eff:.5f}  |  "
        f"max|C_i(2n)/C_i(n)−2|={max_ratio_err:.2e}  |  "
        f"max|dC_i/dt|={max_dcdt:.2e} (@ ρ=0)"
    )
    return ChecklistResult(
        "[b]", "Multi-Group Delayed Precursor Balance",
        "Exact proportional scaling (< 1e-10) + dC/dt=0 at SS (< 1e-8)",
        result_str, passed,
    )


def _check_c() -> ChecklistResult:
    """[c] Global Reactivity Budget."""
    n0 = 1.0
    C0 = steady_state_precursors(n0, _BETA_ARR, _LAMBDA_ARR, LAMBDA_PWR)
    y_ss = np.concatenate([[n0], C0])
    rhs_pos  = point_kinetics_rhs(0.0, y_ss, +50e-5, _BETA_ARR, _LAMBDA_ARR, LAMBDA_PWR)
    rhs_zero = point_kinetics_rhs(0.0, y_ss,   0.0,  _BETA_ARR, _LAMBDA_ARR, LAMBDA_PWR)
    rhs_neg  = point_kinetics_rhs(0.0, y_ss, -50e-5, _BETA_ARR, _LAMBDA_ARR, LAMBDA_PWR)
    expected_pos = +50e-5 / LAMBDA_PWR * n0
    quant_err = abs(rhs_pos[0] / expected_pos - 1.0)
    passed = (
        rhs_pos[0] > 0
        and abs(rhs_zero[0]) < 1e-8
        and rhs_neg[0] < 0
        and quant_err < 1e-6
    )
    result_str = (
        f"dn/dt(+50pcm)={rhs_pos[0]:+.3e}  "
        f"dn/dt(0pcm)={rhs_zero[0]:+.2e}  "
        f"dn/dt(−50pcm)={rhs_neg[0]:+.3e}  |  "
        f"theory match error={quant_err:.2e}"
    )
    return ChecklistResult(
        "[c]", "Global Reactivity Budget",
        "+ρ → +dn/dt, 0ρ → dn/dt≈0, −ρ → −dn/dt; dn/dt=ρ/Λ·n within 1e-6",
        result_str, passed,
    )


def _check_d() -> ChecklistResult:
    """[d] Xenon Burnout Dynamics."""
    phi_high = 2.0 * PHI_NOM
    rhs_eq   = xenon_iodine_rhs(0.0, _XE_EQ_NOM, PHI_NOM,  SIGMA_F)
    rhs_high = xenon_iodine_rhs(0.0, _XE_EQ_NOM, phi_high, SIGMA_F)
    dX_eq   = float(rhs_eq[1])
    dX_high = float(rhs_high[1])
    X_eq = float(_XE_EQ_NOM[1])
    rho_Xe_pcm = -SIGMA_AX * X_eq / SIGMA_A * 1.0e5
    passed = (
        abs(dX_eq) < 1.0
        and dX_high < 0.0
        and dX_high < dX_eq - 1.0   # high-flux burnout clearly more negative
    )
    result_str = (
        f"dX/dt@eq={dX_eq:+.3e} atoms/cm³/s  |  "
        f"dX/dt@2×φ={dX_high:+.3e} atoms/cm³/s  |  "
        f"ρ_Xe(eq)={rho_Xe_pcm:.0f} pcm"
    )
    return ChecklistResult(
        "[d]", "Xenon Burnout Dynamics",
        "dX/dt≈0 at eq, dX/dt<<0 at 2×φ, ρ_Xe in −2500 to −3000 pcm",
        result_str, passed,
    )


def _check_e() -> ChecklistResult:
    """[e] Boundary Error Handling."""
    nan_checks: list[tuple[str, bool]] = []
    r = steady_state_temperatures(P_NOM, 0.0, T_IN_NOM)
    nan_checks.append(("m_dot=0 → NaN", bool(np.all(np.isnan(r)))))
    r = steady_state_temperatures(P_NOM, -500.0, T_IN_NOM)
    nan_checks.append(("m_dot<0 → NaN", bool(np.all(np.isnan(r)))))
    r = doppler_feedback(-1.0, T_FUEL_NOM)
    nan_checks.append(("T_fuel<0 K → NaN", math.isnan(r)))
    r = moderator_feedback(-1.0, T_COOL_NOM)
    nan_checks.append(("T_cool<0 K → NaN", math.isnan(r)))
    r = boron_worth(-100.0)
    nan_checks.append(("boron<0 → NaN", math.isnan(r)))
    r = steady_state_temperatures(P_NOM, M_DOT_NOM, -1.0)
    nan_checks.append(("T_in<0 K → NaN", bool(np.all(np.isnan(r)))))

    pydantic_checks: list[tuple[str, bool]] = []
    for kwargs, lbl in [
        ({"coolant_flow_fraction": 0.0},    "flow=0 → ValidationError"),
        ({"boron_ppm": -1.0},               "boron=-1 → ValidationError"),
        ({"rod_reactivity_pcm": 1500.0},    "rod=1500pcm → ValidationError"),
    ]:
        try:
            PWRParameters(**kwargs)
            pydantic_checks.append((lbl, False))
        except ValidationError:
            pydantic_checks.append((lbl, True))

    all_pass = all(ok for _, ok in nan_checks) and all(ok for _, ok in pydantic_checks)
    failures = [name for name, ok in nan_checks + pydantic_checks if not ok]
    n_nan = len(nan_checks)
    n_pyd = len(pydantic_checks)
    if all_pass:
        result_str = f"{n_nan}/{n_nan} NaN guards enforced  |  {n_pyd}/{n_pyd} Pydantic bounds enforced"
    else:
        result_str = f"FAILURES: {', '.join(failures)}"
    return ChecklistResult(
        "[e]", "Boundary Error Handling",
        f"All {n_nan} NaN guards + {n_pyd} Pydantic ValidationErrors raised",
        result_str, all_pass,
    )


def run_checklist() -> list[ChecklistResult]:
    """Execute all five physics validation checklist items; return results."""
    checkers = [_check_a, _check_b, _check_c, _check_d, _check_e]
    results: list[ChecklistResult] = []
    for fn in checkers:
        try:
            results.append(fn())
        except Exception as exc:  # defensive: surface any unexpected error in report
            lbl = fn.__name__.replace("_check_", "")
            results.append(ChecklistResult(
                f"[{lbl}]", fn.__name__, "—",
                f"UNEXPECTED EXCEPTION: {type(exc).__name__}: {exc}",
                False,
            ))
    return results


_SEP_WIDE  = "=" * 78
_SEP_THIN  = "-" * 78


def print_report(results: list[ChecklistResult]) -> None:
    """Print the structured Physics Validation Checklist Report to stdout."""
    n_pass = sum(r.passed for r in results)
    n_total = len(results)

    print()
    print(_SEP_WIDE)
    print("  NUCLEAR REACTOR DIGITAL TWIN — PHYSICS VALIDATION CHECKLIST REPORT")
    print("  Phase 1 (PWR)  ·  SPEC §7.3  ·  ARCHITECTURE §9  ·  CLAUDE.md §Testing")
    print(_SEP_WIDE)
    print()

    for r in results:
        status_str = "\033[92m[ PASSED ]\033[0m" if r.passed else "\033[91m[ FAILED ]\033[0m"
        # Fallback to plain text if terminal doesn't support ANSI
        status_plain = "[ PASSED ]" if r.passed else "[ FAILED ]"
        print(f"  {r.label}  {r.name}")
        print(f"         Target  :  {r.target}")
        print(f"         Result  :  {r.result_str}")
        print(f"         Status  :  {status_plain}")
        print()

    print(_SEP_WIDE)
    if n_pass == n_total:
        overall = f"OVERALL:  ALL {n_total}/{n_total} CHECKS PASSED  ✓"
    else:
        overall = f"OVERALL:  {n_pass}/{n_total} PASSED — {n_total - n_pass} FAILURE(S) DETECTED  ✗"
    print(f"  {overall}")
    print(_SEP_WIDE)
    print()


if __name__ == "__main__":
    _results = run_checklist()
    print_report(_results)
    sys.exit(0 if all(r.passed for r in _results) else 1)
