"""
Integration test: Week 2 coupled steady-state stability.

Instantiates and couples the full Week 2 ecosystem:
    - 6-group point kinetics engine  (physics/shared/kinetics.py)
    - Xe-135 / I-135 ODE system      (physics/shared/xenon.py)
    - Algebraic feedback calculators  (physics/pwr/feedback.py)
    - Reactivity balance summation    (physics/pwr/feedback.py::reactivity_balance)

Steady-state operating point (SPEC §4.6.2, §4.5.4):
    PHI_0      = 3.1e13 n/cm²/s   full-power thermal flux
    SIGMA_F    = 0.30 cm⁻¹        macroscopic fission XS
    SIGMA_A    = 0.55 cm⁻¹        total absorption XS (for xenon worth, 1-group)
    T_FUEL_REF = 900.0 K           reference fuel temperature (zero Doppler feedback)
    T_COOL_REF = 590.0 K           reference coolant temperature (zero moderator feedback)
    boron_ppm  = 0.0               no dissolved boron (rod provides all excess hold-down)
    n0         = 1.0               normalized neutron population
    rho_rod    = -rho_Xe_eq        rod withdrawn to exactly cancel equilibrium xenon worth

At this operating point every time derivative is analytically zero (SPEC §4.6.2),
so the coupled 9-state system must remain perfectly static under integration.

SPEC physics validation benchmark (ARCHITECTURE.md §9, CLAUDE.md §Testing Requirements):
    test_steady_state_stability: power drift < 0.1% over 1000 simulated seconds.
    This test tightens that criterion to delta < 1e-5 (10× stricter).

ODE solver: scipy Radau, rtol=1e-6, atol=1e-9 (CLAUDE.md §Architecture Rules).
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from data.keepin_dnp import beta_fractions, decay_constants
from physics.pwr.feedback import pcm_to_dk_k, reactivity_balance
from physics.shared.kinetics import (
    LAMBDA_PWR,
    point_kinetics_rhs,
    steady_state_precursors,
)
from physics.shared.xenon import (
    SIGMA_AX,
    steady_state_xenon,
    xenon_iodine_rhs,
)

# ---------------------------------------------------------------------------
# Shared reactor operating parameters (SPEC §4.5.4, test_xenon.py convention)
# ---------------------------------------------------------------------------

PHI_0: float = 3.1e13   # full-power thermal neutron flux (n/cm²/s)
SIGMA_F: float = 0.30   # macroscopic fission XS (cm⁻¹)
SIGMA_A: float = 0.55   # one-group total absorption XS (cm⁻¹); for xenon worth only

# Reference temperatures at zero feedback (temperatures equal reference → ΔT = 0)
T_FUEL_REF: float = 900.0   # K
T_COOL_REF: float = 590.0   # K

BORON_PPM: float = 0.0      # no dissolved boron; rod holds down xenon worth entirely

# Stability tolerance: relative drift allowed over the entire integration window.
# 1e-5 is 10× stricter than the SPEC 0.1% (1e-3) criterion.
STABILITY_TOL: float = 1e-5

# Net-reactivity tolerance (dimensionless Δk/k): 1e-7 ≈ 0.01 pcm, well below any
# physical significance threshold and consistent with Radau rtol=1e-6.
REACTIVITY_TOL: float = 1e-7


# ---------------------------------------------------------------------------
# Module-scoped fixture: build the coupled 9-state steady-state initial vector
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def coupled_steady_state():
    """Return all objects needed to run the coupled 9-state integration test.

    9-state vector layout:
        y[0]   = n      neutron population (normalised)
        y[1:7] = C₁…C₆ delayed-neutron precursor concentrations
        y[7]   = I      I-135 number density (atoms/cm³)
        y[8]   = X      Xe-135 number density (atoms/cm³)

    Steady-state initialisation (SPEC §4.6.2):
        Cᵢ,₀     = (βᵢ / λᵢ Λ) · n₀
        I₀        = γ_I · Σ_f · φ₀ / λ_I
        X₀        = (γ_X + γ_I) · Σ_f · φ₀ / (λ_X + σ_aX · φ₀)
        ρ_rod_pcm = +σ_aX · X₀ / Σ_A × 10⁵   (cancels equilibrium xenon worth)

    Returns:
        (y0, beta_arr, lambda_arr, Lambda, n0, I0, X0, rho_rod_pcm)
    """
    beta_arr = np.array(beta_fractions(), dtype=np.float64)
    lambda_arr = np.array(decay_constants(), dtype=np.float64)
    Lambda = LAMBDA_PWR

    n0 = 1.0
    C0 = steady_state_precursors(n0, beta_arr, lambda_arr, Lambda)

    IX0 = steady_state_xenon(PHI_0, SIGMA_F)
    I0, X0 = float(IX0[0]), float(IX0[1])

    # Equilibrium xenon worth in pcm — SPEC §4.5.4 / ARCHITECTURE §9 (−2500 to −3000 pcm)
    rho_Xe_eq_pcm = -SIGMA_AX * X0 / SIGMA_A * 1e5

    # Rod reactivity exactly cancels equilibrium xenon so ρ_total = 0 at t=0
    rho_rod_pcm = -rho_Xe_eq_pcm

    y0 = np.concatenate(([n0], C0, [I0, X0]))

    return y0, beta_arr, lambda_arr, Lambda, n0, I0, X0, rho_rod_pcm


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_coupled_rhs(beta_arr, lambda_arr, Lambda, n0, rho_rod_pcm):
    """Return the coupled 9-state ODE RHS callable for scipy solve_ivp.

    Coupling at each solver step:
        1. φ(t) = (n / n₀) · φ₀         — flux proportional to neutron population
        2. ρ_Xe = −σ_aX · X / Σ_A       — xenon reactivity (SPEC §4.5.1)
        3. ρ_total = ρ_rod + ρ_Xe        — balance (temperatures at ref → 0 pcm each)
        4. kinetics_rhs = f(ρ_total, y[:7])
        5. xe_rhs = g(φ, y[7:9])
    """
    def coupled_rhs(t: float, y: np.ndarray) -> np.ndarray:
        n = y[0]
        X = y[8]

        # Neutron flux proportional to normalised population
        phi = (n / n0) * PHI_0

        # Xenon reactivity — one-group perturbation theory (SPEC §4.5.4)
        rho_Xe_pcm = -SIGMA_AX * X / SIGMA_A * 1e5

        # Total reactivity: rod balances xenon; Doppler, moderator, boron all zero
        # because temperatures are held at their reference values for Week 2 coupling.
        rho_total_pcm = reactivity_balance(
            rho_rod_pcm,    # rod
            0.0,            # Doppler   — T_fuel = T_fuel_ref → 0 pcm
            0.0,            # moderator — T_cool = T_cool_ref → 0 pcm
            0.0,            # boron     — boron_ppm = 0
            rho_Xe_pcm,     # xenon
        )
        rho_total = pcm_to_dk_k(rho_total_pcm)

        kin_rhs = point_kinetics_rhs(t, y[:7], rho_total, beta_arr, lambda_arr, Lambda)
        xe_rhs = xenon_iodine_rhs(t, y[7:9], phi, SIGMA_F)

        return np.concatenate((kin_rhs, xe_rhs))

    return coupled_rhs


# ---------------------------------------------------------------------------
# Initialisation correctness — verify the system truly starts at steady state
# ---------------------------------------------------------------------------


def test_coupled_rhs_is_zero_at_initial_conditions(coupled_steady_state) -> None:
    """All 9 derivatives must be ≈ 0 at the analytical steady-state ICs.

    This is the mathematical precondition for the stability integration test:
    if the RHS is not zero at t=0 there is no steady state to perserve.

    Expected bounds:
        Kinetics RHS: |dy/dt| < 1e-10 (SPEC Eq. 1, Eq. 2 at ρ=0 equilibrium)
        Xenon RHS:    |dy/dt| < 1e-3  (SPEC Eq. 14–15 at equilibrium flux;
                      units are atoms/cm³/s — large absolute value, tiny relative)
    """
    y0, beta_arr, lambda_arr, Lambda, n0, I0, X0, rho_rod_pcm = coupled_steady_state
    rhs = _build_coupled_rhs(beta_arr, lambda_arr, Lambda, n0, rho_rod_pcm)

    dydt = rhs(0.0, y0)

    assert dydt.shape == (9,), "RHS must return a 9-element vector"

    # Kinetics block: analytically zero at equilibrium ICs + ρ=0 (SPEC Eq. 1–2)
    np.testing.assert_allclose(
        dydt[:7], 0.0, atol=1e-10,
        err_msg="Kinetics RHS block non-zero at steady-state ICs"
    )

    # Xenon block: zero when flux is at equilibrium value (SPEC Eq. 14–15)
    np.testing.assert_allclose(
        dydt[7:], 0.0, atol=1e-3,
        err_msg="Xenon/Iodine RHS block non-zero at steady-state ICs"
    )


def test_equilibrium_reactivity_is_zero(coupled_steady_state) -> None:
    """Net reactivity must be exactly zero at the analytically initialised state.

    ρ_total = ρ_rod + ρ_Xe = 0 is the criticality condition (SPEC §4.5.1).
    """
    _, _, _, _, _, _, X0, rho_rod_pcm = coupled_steady_state

    rho_Xe_pcm = -SIGMA_AX * X0 / SIGMA_A * 1e5
    rho_total_pcm = reactivity_balance(rho_rod_pcm, 0.0, 0.0, 0.0, rho_Xe_pcm)
    rho_total = pcm_to_dk_k(rho_total_pcm)

    assert abs(rho_total) < 1e-15, (
        f"Initial reactivity ρ_total = {rho_total:.2e} (dimensionless); "
        f"must be 0 at steady state"
    )


def test_xenon_worth_at_equilibrium_in_spec_range(coupled_steady_state) -> None:
    """Equilibrium xenon worth must fall within −2500 to −3000 pcm (ARCHITECTURE §9).

    This bounds-check confirms the operating point is physically valid before
    running the full stability integration.
    """
    _, _, _, _, _, _, X0, _ = coupled_steady_state
    rho_Xe_eq_pcm = -SIGMA_AX * X0 / SIGMA_A * 1e5

    assert -3000 <= rho_Xe_eq_pcm <= -2500, (
        f"Equilibrium xenon worth = {rho_Xe_eq_pcm:.0f} pcm; "
        f"must be within −2500 to −3000 pcm (ARCHITECTURE.md §9)"
    )


# ---------------------------------------------------------------------------
# Steady-state stability — primary integration benchmark
# ---------------------------------------------------------------------------


def test_steady_state_stability_1000s(coupled_steady_state) -> None:
    """SPEC §9.3 PWR #1 (tightened): all states stable within 1e-5 over 1000 s.

    The coupled 9-state system (kinetics + xenon/iodine) is integrated from
    the analytical steady-state ICs with no external perturbation for 1000
    simulated seconds.  Every state variable must remain within a relative
    tolerance of 1e-5 of its initial value.

    Checked variables:
        n         — neutron population / thermal power proxy
        C₁ … C₆  — delayed-neutron precursor concentrations
        I-135     — number density
        Xe-135    — number density
        ρ_total   — net core reactivity (must stay ≈ 0)

    Stability criterion (ARCHITECTURE.md §9, CLAUDE.md §Testing Requirements):
        |x_final / x₀ − 1| < 1e-5   for each state variable x
        |ρ_total|           < 1e-7   (dimensionless) at t = 1000 s
    """
    y0, beta_arr, lambda_arr, Lambda, n0, I0, X0, rho_rod_pcm = coupled_steady_state
    C0 = y0[1:7]

    t_end = 1000.0  # s — SPEC §9.3 benchmark duration
    t_eval = np.linspace(0.0, t_end, 201)  # 5-s grid for detailed diagnostics

    rhs = _build_coupled_rhs(beta_arr, lambda_arr, Lambda, n0, rho_rod_pcm)

    result = solve_ivp(
        rhs,
        (0.0, t_end),
        y0,
        method="Radau",
        rtol=1e-6,
        atol=1e-9,
        t_eval=t_eval,
        dense_output=False,
    )

    assert result.success, f"Coupled ODE integration failed: {result.message}"
    assert result.y.shape[0] == 9
    assert result.t[-1] == pytest.approx(t_end)

    n_all = result.y[0]
    C_all = result.y[1:7]
    I_all = result.y[7]
    X_all = result.y[8]

    # --- Neutron population (power) ---
    n_final = n_all[-1]
    n_drift = abs(n_final / n0 - 1.0)
    assert n_drift < STABILITY_TOL, (
        f"Neutron population drifted {n_drift:.2e} from n₀ at t=1000 s "
        f"(limit {STABILITY_TOL:.0e})"
    )

    # Maximum drift across all output times (not just final)
    n_max_drift = float(np.max(np.abs(n_all / n0 - 1.0)))
    assert n_max_drift < STABILITY_TOL, (
        f"Neutron population peak drift = {n_max_drift:.2e} across 1000 s "
        f"(limit {STABILITY_TOL:.0e})"
    )

    # --- Delayed-neutron precursor groups ---
    for i in range(6):
        C_drift = abs(C_all[i, -1] / C0[i] - 1.0)
        assert C_drift < STABILITY_TOL, (
            f"Precursor group {i+1} drifted {C_drift:.2e} at t=1000 s "
            f"(limit {STABILITY_TOL:.0e})"
        )
        C_max_drift = float(np.max(np.abs(C_all[i] / C0[i] - 1.0)))
        assert C_max_drift < STABILITY_TOL, (
            f"Precursor group {i+1} peak drift = {C_max_drift:.2e} across 1000 s "
            f"(limit {STABILITY_TOL:.0e})"
        )

    # --- Iodine-135 ---
    I_final = I_all[-1]
    I_drift = abs(I_final / I0 - 1.0)
    assert I_drift < STABILITY_TOL, (
        f"I-135 drifted {I_drift:.2e} at t=1000 s (limit {STABILITY_TOL:.0e})"
    )

    # --- Xenon-135 ---
    X_final = X_all[-1]
    X_drift = abs(X_final / X0 - 1.0)
    assert X_drift < STABILITY_TOL, (
        f"Xe-135 drifted {X_drift:.2e} at t=1000 s (limit {STABILITY_TOL:.0e})"
    )

    # --- Net reactivity must stay near zero throughout ---
    rho_Xe_all_pcm = -SIGMA_AX * X_all / SIGMA_A * 1e5
    rho_total_all_pcm = rho_rod_pcm + rho_Xe_all_pcm   # Doppler/mod/boron all zero
    rho_total_all = rho_total_all_pcm * 1e-5

    max_rho = float(np.max(np.abs(rho_total_all)))
    assert max_rho < REACTIVITY_TOL, (
        f"|ρ_total|_max = {max_rho:.2e} across 1000 s "
        f"(limit {REACTIVITY_TOL:.0e})"
    )


# ---------------------------------------------------------------------------
# Physical constraints — states must stay non-negative
# ---------------------------------------------------------------------------


def test_all_states_remain_non_negative(coupled_steady_state) -> None:
    """Physical constraint: n, Cᵢ, I, X must remain ≥ 0 (SPEC §4.6.5).

    Negative concentrations or populations are unphysical. This test verifies
    the solver does not produce any negative values at any output timestep.
    """
    y0, beta_arr, lambda_arr, Lambda, n0, I0, X0, rho_rod_pcm = coupled_steady_state

    rhs = _build_coupled_rhs(beta_arr, lambda_arr, Lambda, n0, rho_rod_pcm)

    result = solve_ivp(
        rhs,
        (0.0, 1000.0),
        y0,
        method="Radau",
        rtol=1e-6,
        atol=1e-9,
        t_eval=np.linspace(0.0, 1000.0, 201),
        dense_output=False,
    )

    assert result.success, f"Solver failed: {result.message}"
    assert np.all(result.y >= 0.0), (
        "One or more state variables went negative during integration; "
        "first negative at indices: "
        f"{list(zip(*np.where(result.y < 0.0)))}"
    )


# ---------------------------------------------------------------------------
# Equilibrium xenon worth (re-confirmed after integration)
# ---------------------------------------------------------------------------


def test_xenon_worth_stable_after_integration(coupled_steady_state) -> None:
    """Xenon reactivity worth must remain within −2500 to −3000 pcm throughout 1000 s.

    Since xenon time constants (t½ ≈ 9.2 hr) greatly exceed the 1000 s window,
    the xenon worth at t=1000 s must be essentially identical to its initial value.
    The ARCHITECTURE §9 band (−2500 to −3000 pcm) must never be violated.
    """
    y0, beta_arr, lambda_arr, Lambda, n0, I0, X0, rho_rod_pcm = coupled_steady_state

    rhs = _build_coupled_rhs(beta_arr, lambda_arr, Lambda, n0, rho_rod_pcm)

    result = solve_ivp(
        rhs,
        (0.0, 1000.0),
        y0,
        method="Radau",
        rtol=1e-6,
        atol=1e-9,
        dense_output=False,
    )

    assert result.success, f"Solver failed: {result.message}"

    X_final = result.y[8, -1]
    rho_Xe_final_pcm = -SIGMA_AX * X_final / SIGMA_A * 1e5

    assert -3000 <= rho_Xe_final_pcm <= -2500, (
        f"Xenon worth at t=1000 s = {rho_Xe_final_pcm:.0f} pcm; "
        f"must remain within −2500 to −3000 pcm"
    )
