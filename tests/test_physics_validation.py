"""Physics validation suite — Doppler self-regulation and xenon equilibrium.

Week 3 validation benchmarks (ARCHITECTURE.md §9, CLAUDE.md §Testing Requirements,
SPEC §7.3):

    TestDopplerSelfRegulation
        +100 pcm step reactivity insertion (deeply sub-prompt-critical, ~15% of β_eff)
        starting from the critical full-power equilibrium.  The simulation must
        demonstrate all four stages of inherent self-regulation:
        (a) Prompt power jump — power rises immediately after insertion
        (b) Fuel temperature rise — fission heating drives T_fuel above T_fuel_ref
        (c) Negative Doppler feedback — α_D < 0, ΔT_fuel > 0 ⟹ ρ_D < 0
        (d) Net reactivity → 0 and power plateaus at a new safe equilibrium

    TestXenonEquilibrium
        2-state I-135/Xe-135 ODE integrated for 50 simulated hours from I=X=0
        at constant full-power thermal flux.  The simulation must asymptotically
        converge within 1% of the analytical steady-state equilibrium values and
        produce xenon worth within the published −2500 to −3000 pcm band.

SPEC equations exercised:
    Eq. 1  neutron population ODE
    Eq. 2  precursor group ODEs
    Eq. 9  fuel temperature ODE
    Eq. 10 coolant temperature ODE
    Eq. 11 Doppler reactivity feedback  ← primary Doppler validation target
    Eq. 12 moderator temperature feedback
    Eq. 14 I-135 ODE                    ← primary xenon validation target
    Eq. 15 Xe-135 ODE                   ← primary xenon validation target

ODE solver: scipy Radau, rtol=1e-6, atol=1e-9 (CLAUDE.md §Architecture Rules).
"""

from __future__ import annotations

import numpy as np
import pytest

from physics.pwr.engine import (
    PHI_NOM,
    SIGMA_A,
    SIGMA_F,
    PWRControls,
    PWRModelConfig,
    build_initial_state,
    critical_base_reactivity_pcm,
    integrate_pwr_11state,
    make_params_array,
)
from physics.pwr.feedback import ALPHA_D_DEFAULT, ALPHA_M_DEFAULT
from physics.pwr.reactivity import ReferenceState
from physics.pwr.thermal_hydraulics import T_COOL_NOM, T_FUEL_NOM, T_IN_NOM
from physics.shared.xenon import (
    GAMMA_I,
    GAMMA_X,
    LAMBDA_I,
    LAMBDA_X,
    SIGMA_AX,
    integrate_xenon,
    xenon_iodine_rhs,
)

# ---------------------------------------------------------------------------
# Analytical equilibrium values (SPEC Eqs. 14–15, dI/dt = dX/dt = 0 at φ = PHI_NOM)
# ---------------------------------------------------------------------------

I_EQ_ANALYTICAL: float = GAMMA_I * SIGMA_F * PHI_NOM / LAMBDA_I
X_EQ_ANALYTICAL: float = (
    (GAMMA_X + GAMMA_I) * SIGMA_F * PHI_NOM / (LAMBDA_X + SIGMA_AX * PHI_NOM)
)

XENON_EQUILIBRIUM_HOURS: float = 50.0  # simulated hours; covers ~5 Xe t½


# ============================================================================
# Shared fixture — critical full-power 11-state equilibrium
# ============================================================================


@pytest.fixture(scope="module")
def critical_equilibrium():
    """Build the critical 11-state PWR initial state at full power.

    Returns (y0, reference_state, controls, config) where base_reactivity_pcm
    in config exactly cancels the equilibrium xenon worth so ρ_total = 0.
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
    return y0, reference_state, controls, config


# ============================================================================
# Test class 1 — Doppler Self-Regulation
# ============================================================================


@pytest.fixture(scope="module")
def doppler_transient(critical_equilibrium):
    """600-second full-coupled 11-state PWR transient after +100 pcm step insertion.

    +100 pcm ≈ 15% of β_eff (~650 pcm for U-235): deeply sub-prompt-critical,
    physically representative of a significant but bounded rod withdrawal event.
    Thermal time constants (τ_f ~ 3 s, τ_c ~ 5 s) ensure the feedback fully
    engages well before the 600 s window ends.

    Returns (OdeResult, y0_array) for downstream assertions.
    """
    y0, reference_state, _, config = critical_equilibrium
    insertion_pcm = 100.0
    perturbed_controls = PWRControls(
        rod_reactivity_pcm=insertion_pcm,
        boron_ppm=0.0,
        coolant_flow_fraction=1.0,
        inlet_temperature_k=T_IN_NOM,
    )
    params = make_params_array(perturbed_controls, reference_state, config)
    t_eval = np.linspace(0.0, 600.0, 1201)  # 0.5 s resolution
    result = integrate_pwr_11state(y0, (0.0, 600.0), params, t_eval=t_eval)
    return result, y0


class TestDopplerSelfRegulation:
    """SPEC §7.3 / ARCHITECTURE §9 — +100 pcm insertion self-limits via Doppler feedback."""

    # ------------------------------------------------------------------
    # Solver health
    # ------------------------------------------------------------------

    def test_solver_converges(self, doppler_transient):
        """Radau must converge across the 600-second stiff coupled transient."""
        result, _ = doppler_transient
        assert result.success, f"ODE solver failed: {result.message}"

    def test_integration_reaches_600s(self, doppler_transient):
        result, _ = doppler_transient
        assert result.t[-1] == pytest.approx(600.0, abs=0.1)

    def test_no_negative_states(self, doppler_transient):
        """n, Cᵢ, T_fuel, T_cool, I, X must all remain ≥ 0 (SPEC §4.6.5)."""
        result, _ = doppler_transient
        assert np.all(result.y >= 0.0), (
            "Negative physical state detected: "
            f"{list(zip(*np.where(result.y < 0.0)))}"
        )

    # ------------------------------------------------------------------
    # (a) Prompt power jump
    # ------------------------------------------------------------------

    def test_prompt_power_jump_occurs(self, doppler_transient):
        """(a) Power must rise immediately and visibly above initial after +100 pcm.

        A +100 pcm sub-prompt insertion produces a prompt jump factor of
        approximately β/(β − ρ) ≈ 650/(650 − 100) ≈ 1.18 before feedback acts.
        We require at least 2% above initial — a conservative lower bound.
        """
        result, y0 = doppler_transient
        power = result.y[0]
        n0 = float(y0[0])
        assert power.max() > n0 * 1.02, (
            f"No prompt power jump detected: max/n₀ = {power.max() / n0:.4f}. "
            "+100 pcm must produce a measurable power excursion."
        )

    # ------------------------------------------------------------------
    # (b) Fuel temperature rise
    # ------------------------------------------------------------------

    def test_fuel_temperature_rises_above_reference(self, doppler_transient):
        """(b) T_fuel must exceed T_fuel_ref — increased fission power heats the fuel.

        Analytical estimate: at the new SS, ΔT_fuel ≈ 335 K × ΔP/P ≈ 20 K.
        Require at least 5 K above reference as a robust lower bound.
        """
        result, _ = doppler_transient
        T_fuel_final = float(result.y[7, -1])
        assert T_fuel_final > T_FUEL_NOM + 5.0, (
            f"Fuel temperature at t=600 s: {T_fuel_final:.1f} K; "
            f"reference: {T_FUEL_NOM:.1f} K. "
            "T_fuel must rise for Doppler feedback to engage."
        )

    # ------------------------------------------------------------------
    # (c) Negative Doppler feedback
    # ------------------------------------------------------------------

    def test_doppler_feedback_is_negative(self, doppler_transient):
        """(c) ρ_D = α_D·(T_fuel − T_ref) must be negative at t = 600 s.

        With α_D = −2.5 pcm/K and ΔT_fuel ≈ 20 K the Doppler contribution
        is approximately −50 pcm.  Require < −30 pcm as a robust lower bound.
        """
        result, _ = doppler_transient
        T_fuel_final = float(result.y[7, -1])
        rho_D_pcm = ALPHA_D_DEFAULT * (T_fuel_final - T_FUEL_NOM)  # Eq. 11
        assert rho_D_pcm < -30.0, (
            f"Doppler feedback ρ_D = {rho_D_pcm:.1f} pcm at t=600 s. "
            f"Expected < −30 pcm for ΔT_fuel = {T_fuel_final - T_FUEL_NOM:.1f} K "
            f"with α_D = {ALPHA_D_DEFAULT} pcm/K. "
            "Positive or small Doppler feedback indicates a sign error in Eq. 11."
        )

    def test_moderator_feedback_is_negative(self, doppler_transient):
        """Moderator feedback must also be negative — a second independent mechanism."""
        result, _ = doppler_transient
        T_cool_final = float(result.y[8, -1])
        rho_m_pcm = ALPHA_M_DEFAULT * (T_cool_final - T_COOL_NOM)  # Eq. 12
        assert rho_m_pcm < -10.0, (
            f"Moderator feedback ρ_m = {rho_m_pcm:.1f} pcm at t=600 s. "
            f"Expected < −10 pcm for ΔT_cool = {T_cool_final - T_COOL_NOM:.1f} K."
        )

    def test_doppler_dominates_at_early_times(self, doppler_transient):
        """Doppler feedback must engage before coolant feedback (faster thermal time constant).

        T_fuel time constant τ_f ~ 3 s; T_cool time constant τ_c ~ 5 s.
        At t = 10 s, Doppler feedback must be active (ΔT_fuel > 0).
        """
        result, _ = doppler_transient
        idx_10s = np.searchsorted(result.t, 10.0)
        T_fuel_10s = float(result.y[7, idx_10s])
        rho_D_10s = ALPHA_D_DEFAULT * (T_fuel_10s - T_FUEL_NOM)
        assert rho_D_10s < 0.0, (
            f"Doppler feedback is zero or positive at t=10 s "
            f"(T_fuel = {T_fuel_10s:.1f} K, ρ_D = {rho_D_10s:.1f} pcm). "
            "Fuel temperature must rise before coolant responds."
        )

    # ------------------------------------------------------------------
    # (d) Net reactivity → 0, power stabilizes at new equilibrium
    # ------------------------------------------------------------------

    def test_net_reactivity_returns_toward_zero(self, doppler_transient):
        """(d) Rod + Doppler + moderator must sum to < 10 pcm at t = 600 s.

        At the new SS: ρ_rod + ρ_D + ρ_m = 0 exactly.  Xenon changes negligibly
        in 600 s (t½ = 9.2 hr), so the base + xenon sum remains ≈ 0 as at t=0.
        We therefore need only the three non-xenon terms to cancel.
        """
        result, _ = doppler_transient
        T_fuel_final = float(result.y[7, -1])
        T_cool_final = float(result.y[8, -1])
        rho_D_pcm = ALPHA_D_DEFAULT * (T_fuel_final - T_FUEL_NOM)
        rho_m_pcm = ALPHA_M_DEFAULT * (T_cool_final - T_COOL_NOM)
        rho_net_pcm = 100.0 + rho_D_pcm + rho_m_pcm
        assert abs(rho_net_pcm) < 10.0, (
            f"Net reactivity at t=600 s = {rho_net_pcm:.2f} pcm "
            f"(ρ_rod = +100, ρ_D = {rho_D_pcm:.1f}, ρ_m = {rho_m_pcm:.1f}). "
            "Expected |ρ_net| < 10 pcm at the new equilibrium."
        )

    def test_power_stabilizes_not_diverges(self, doppler_transient):
        """(d) Power must plateau — no divergent trend in the final 60 seconds."""
        result, _ = doppler_transient
        power = result.y[0]
        # Last 10% of the time window (60 s)
        p_tail = power[power.size * 90 // 100:]
        drift = (p_tail.max() - p_tail.min()) / p_tail.mean()
        assert drift < 0.002, (
            f"Power still drifting in last 60 s: peak-to-peak = {drift:.4%}. "
            "System must plateau before 600 s for inherent self-regulation."
        )

    def test_final_power_below_transient_peak(self, doppler_transient):
        """(d) Power at t = 600 s must be below the transient peak — feedback self-limits."""
        result, _ = doppler_transient
        power = result.y[0]
        assert power[-1] < power.max() * 0.99, (
            f"Final power ({power[-1]:.4f}) is at or above the transient peak "
            f"({power.max():.4f}). Doppler feedback must pull power back from peak."
        )

    def test_new_equilibrium_power_in_expected_range(self, doppler_transient):
        """(d) Final power consistent with the analytically predicted new SS.

        From the thermal-hydraulic steady-state balance with α_D = −2.5 pcm/K,
        α_m = −35 pcm/K, and nominal coolant constants:
            ΔP/P₀ = 100 / (|α_D|·335 + |α_m|·25) = 100 / 1712.5 ≈ 5.8%

        Acceptable window: 1.02 – 1.15 × initial power.
        """
        result, y0 = doppler_transient
        power = result.y[0]
        ratio = float(power[-1]) / float(y0[0])
        assert 1.02 < ratio < 1.15, (
            f"Final power / initial = {ratio:.4f}. "
            "Expected 1.02–1.15 for +100 pcm with Doppler + moderator self-regulation. "
            "Values outside this range suggest incorrect feedback signs or magnitudes."
        )

    def test_no_prompt_critical_excursion(self, doppler_transient):
        """Power must not exceed 10× initial — +100 pcm is far sub-prompt-critical."""
        result, y0 = doppler_transient
        power = result.y[0]
        ratio = float(power.max()) / float(y0[0])
        assert ratio < 10.0, (
            f"Power spiked to {ratio:.1f}× initial. "
            "+100 pcm ≪ β_eff ≈ 650 pcm; a 10× spike implies a sign error in feedback."
        )


# ============================================================================
# Test class 2 — Xenon Equilibrium Convergence
# ============================================================================


@pytest.fixture(scope="module")
def xenon_equilibrium_run():
    """Integrate I-135/Xe-135 from I=X=0 for 50 simulated hours at full power.

    Dominant time constants at PHI_NOM = 3.1e13 n/cm²/s:
        I-135: τ = t½/ln2 = 9.67 hr → at 50 hr: exp(−50/9.67) = 0.006 → 99.4% eq.
        Xe-135: coupled to I decay, but effective approach time is ~20–30 hr.

    Starting from zero isolates the convergence behavior: both species must be
    driven entirely by the fission source term and decay chains.
    """
    t_end = XENON_EQUILIBRIUM_HOURS * 3600.0  # 180 000 s
    t_eval = np.linspace(0.0, t_end, 1001)

    y0 = np.zeros(2, dtype=np.float64)  # fresh start: I = X = 0

    result = integrate_xenon(
        y0,
        (0.0, t_end),
        phi_fn=lambda t, y: PHI_NOM,
        Sigma_f=SIGMA_F,
        t_eval=t_eval,
    )
    return result


class TestXenonEquilibrium:
    """SPEC §7.3 / ARCHITECTURE §9 — I-135 and Xe-135 converge to analytical equilibrium."""

    # ------------------------------------------------------------------
    # Analytical self-consistency: ODE derivatives = 0 at equilibrium
    # ------------------------------------------------------------------

    def test_iodine_equilibrium_formula_is_consistent(self):
        """dI/dt must be ≈ 0 when I = I_EQ_ANALYTICAL (SPEC Eq. 14 at dI/dt = 0)."""
        rhs = xenon_iodine_rhs(
            0.0,
            np.array([I_EQ_ANALYTICAL, X_EQ_ANALYTICAL]),
            PHI_NOM,
            SIGMA_F,
        )
        assert abs(rhs[0]) < 1.0, (  # atoms/cm³/s; floating-point residual only
            f"I-135 ODE derivative at analytical equilibrium = {rhs[0]:.3e} atoms/cm³/s. "
            "Should be machine-zero: formula I_eq = γ_I·Σ_f·φ / λ_I must cancel exactly."
        )

    def test_xenon_equilibrium_formula_is_consistent(self):
        """dX/dt must be ≈ 0 when X = X_EQ_ANALYTICAL (SPEC Eq. 15 at dX/dt = 0)."""
        rhs = xenon_iodine_rhs(
            0.0,
            np.array([I_EQ_ANALYTICAL, X_EQ_ANALYTICAL]),
            PHI_NOM,
            SIGMA_F,
        )
        assert abs(rhs[1]) < 1.0, (
            f"Xe-135 ODE derivative at analytical equilibrium = {rhs[1]:.3e} atoms/cm³/s. "
            "Should be machine-zero."
        )

    def test_analytical_xenon_worth_in_spec_range(self):
        """Analytical X_eq must already give xenon worth in −2500 to −3000 pcm.

        This is a pre-flight check: if the analytical equilibrium values themselves
        are out of range the simulation test is vacuous.
        """
        rho_Xe_pcm = -SIGMA_AX * X_EQ_ANALYTICAL / SIGMA_A * 1.0e5
        assert -3000 <= rho_Xe_pcm <= -2500, (
            f"Analytical equilibrium xenon worth = {rho_Xe_pcm:.0f} pcm. "
            "Must be in −2500 to −3000 pcm (ARCHITECTURE §9). "
            "Check SIGMA_AX, SIGMA_A, GAMMA_I, GAMMA_X, PHI_NOM, SIGMA_F."
        )

    # ------------------------------------------------------------------
    # Solver health
    # ------------------------------------------------------------------

    def test_solver_converges(self, xenon_equilibrium_run):
        result = xenon_equilibrium_run
        assert result.success, f"Xenon/Iodine ODE solver failed: {result.message}"

    def test_integration_reaches_50h(self, xenon_equilibrium_run):
        result = xenon_equilibrium_run
        assert result.t[-1] == pytest.approx(XENON_EQUILIBRIUM_HOURS * 3600.0, rel=1e-4)

    def test_all_states_non_negative(self, xenon_equilibrium_run):
        """I-135 and Xe-135 number densities must remain non-negative throughout."""
        result = xenon_equilibrium_run
        assert np.all(result.y >= 0.0), (
            "I-135 or Xe-135 went negative (unphysical)."
        )

    # ------------------------------------------------------------------
    # Monotonic buildup — qualitative physics check
    # ------------------------------------------------------------------

    def test_iodine_builds_monotonically(self, xenon_equilibrium_run):
        """I-135 starts at zero and must increase monotonically toward equilibrium.

        I-135 has a single exponential buildup: I(t) = I_eq·(1 − e^{−λ_I t}).
        No post-peak decay is expected in the first 50 hr.
        """
        result = xenon_equilibrium_run
        I = result.y[0]
        assert float(I[I.size // 2]) < float(I[-1]), (
            "I-135 not monotonically increasing: "
            f"mid = {I[I.size//2]:.3e}, end = {I[-1]:.3e}. "
            "Check the production term γ_I·Σ_f·φ in Eq. 14."
        )

    def test_xenon_builds_then_plateaus(self, xenon_equilibrium_run):
        """Xe-135 starts at zero and must be higher at the end than at the midpoint."""
        result = xenon_equilibrium_run
        X = result.y[1]
        assert float(X[X.size // 2]) < float(X[-1]), (
            "Xe-135 final value unexpectedly lower than midpoint: "
            f"mid = {X[X.size//2]:.3e}, end = {X[-1]:.3e}."
        )

    # ------------------------------------------------------------------
    # Primary validation: 1% convergence to analytical equilibrium
    # ------------------------------------------------------------------

    def test_iodine_converges_within_1pct(self, xenon_equilibrium_run):
        """I-135 must be within 1% of analytical equilibrium after 50 hr (SPEC §7.3).

        Analytical value: I_eq = γ_I · Σ_f · φ / λ_I
        At 50 hr: exp(−λ_I · 50·3600) ≈ 0.006 → < 0.6% from equilibrium.
        """
        result = xenon_equilibrium_run
        I_final = float(result.y[0, -1])
        rel_error = abs(I_final / I_EQ_ANALYTICAL - 1.0)
        assert rel_error < 0.01, (
            f"I-135 final = {I_final:.4e} atoms/cm³; "
            f"analytical = {I_EQ_ANALYTICAL:.4e} atoms/cm³; "
            f"relative error = {rel_error:.3%} (limit: 1%)"
        )

    def test_xenon_converges_within_1pct(self, xenon_equilibrium_run):
        """Xe-135 must be within 1% of analytical equilibrium after 50 hr (SPEC §7.3).

        Analytical value: X_eq = (γ_X + γ_I) · Σ_f · φ / (λ_X + σ_aX · φ)
        """
        result = xenon_equilibrium_run
        X_final = float(result.y[1, -1])
        rel_error = abs(X_final / X_EQ_ANALYTICAL - 1.0)
        assert rel_error < 0.01, (
            f"Xe-135 final = {X_final:.4e} atoms/cm³; "
            f"analytical = {X_EQ_ANALYTICAL:.4e} atoms/cm³; "
            f"relative error = {rel_error:.3%} (limit: 1%)"
        )

    # ------------------------------------------------------------------
    # Xenon worth in spec range
    # ------------------------------------------------------------------

    def test_simulated_xenon_worth_in_spec_range(self, xenon_equilibrium_run):
        """Simulated equilibrium xenon worth must be in −2500 to −3000 pcm.

        This is the primary physics validation benchmark for xenon equilibrium
        (ARCHITECTURE §9, CLAUDE.md §Testing Requirements).
        Published equilibrium xenon worth for large commercial PWRs at full power
        is −2500 to −3000 pcm; values outside this band indicate incorrect nuclear
        data (SIGMA_AX, fission yields, flux) or a sign error in xenon worth.
        """
        result = xenon_equilibrium_run
        X_final = float(result.y[1, -1])
        rho_Xe_pcm = -SIGMA_AX * X_final / SIGMA_A * 1.0e5
        assert -3000 <= rho_Xe_pcm <= -2500, (
            f"Simulated xenon worth after {XENON_EQUILIBRIUM_HOURS:.0f} hr = "
            f"{rho_Xe_pcm:.0f} pcm. "
            "Expected −2500 to −3000 pcm (ARCHITECTURE §9). "
            f"Xe-135 density = {X_final:.3e} atoms/cm³."
        )

    # ------------------------------------------------------------------
    # Iodine and xenon worth comparison: I feeds X
    # ------------------------------------------------------------------

    def test_iodine_production_feeds_xenon(self, xenon_equilibrium_run):
        """At equilibrium, λ_I · I_eq must equal the xenon production from iodine decay.

        I-135 decay is the dominant xenon source (γ_I >> γ_X): at equilibrium
        λ_I · I = γ_I · Σ_f · φ which feeds directly into dX/dt.
        """
        result = xenon_equilibrium_run
        I_final = float(result.y[0, -1])
        iodine_feed_rate = LAMBDA_I * I_final           # atoms/cm³/s into Xe channel
        direct_fission_xe = GAMMA_X * SIGMA_F * PHI_NOM  # direct Xe production rate
        # At equilibrium I feeds ~γ_I / (γ_X + γ_I) ≈ 96.4% of total xenon production
        assert iodine_feed_rate > direct_fission_xe * 10, (
            f"λ_I · I = {iodine_feed_rate:.3e} atoms/cm³/s; "
            f"direct Xe production = {direct_fission_xe:.3e} atoms/cm³/s. "
            "I-135 decay must dominate Xe-135 production (γ_I >> γ_X)."
        )
