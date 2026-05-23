"""
physics/msr/engine.py — MSR modified point kinetics engine with precursor drift.

The MSR engine wraps the shared 6-group point kinetics core (physics/shared/kinetics.py)
and injects precursor drift modification terms unique to liquid-fueled reactors.
It does NOT reimplement kinetics — that math lives once in the shared module
(ARCHITECTURE.md §5, CLAUDE.md Architecture Rule 2).

State vector layout (8 ODE states):
    y[0]    = n         neutron population (normalised; n=1 ↔ P = P_ref)
    y[1:7]  = C₁–C₆   in-core delayed-neutron precursor concentrations
    y[7]    = T_salt    unified salt temperature (K)

SPEC equations implemented here:
    Eq. 1   dn/dt   = [(ρ − β_eff) / Λ] · n  +  Σᵢ λᵢ Cᵢ           (via shared core)
    Eq. 22  dCᵢ/dt = (βᵢ/Λ)n − λᵢCᵢ − (1/τ_core)Cᵢ + Cᵢ^return(t) (shared + drift)
    Eq. 23  Cᵢ^return(t) = (1/τ_core)·Cᵢ(t−τ_loop)·e^{−λᵢτ_loop}   (via drift.py)
    Eq. 24  β_eff,flow                                                 (via drift.py)
    Salt thermal (SPEC §4.4)                                           (via thermal.py)

Delay-differential equation (DDE) integration strategy (SPEC §4.6.3):
    The standard solve_ivp interface cannot handle DDEs natively. This engine uses
    the method-of-steps approach: integrate_msr calls solve_ivp in chunks of at
    most τ_loop / 2 seconds, updating the PrecursorHistory buffer between chunks.
    Within each chunk every history lookup C(t − τ_loop) references already-accepted
    history — no speculative future values are used. Radau with rtol=1e-6, atol=1e-9
    is used for each chunk (CLAUDE.md Architecture Rule 6).

Architecture rules (CLAUDE.md):
    - Pure physics: no Dash, UI, or orchestrator imports.
    - Keepin U-235 6-group parameters imported from data.keepin_dnp.
    - solve_ivp(method='Radau', rtol=1e-6, atol=1e-9) — never change solver or tolerances.
    - All hot ODE kernels operate on float64 numpy arrays only.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import numpy as np
from scipy.integrate import solve_ivp

from data.keepin_dnp import beta_fractions, decay_constants
from physics.msr.drift import (
    PrecursorHistory,
    compute_beta_eff_flow,
    msr_precursor_drift_rhs,
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
from physics.shared.kinetics import (
    LAMBDA_PWR,
    point_kinetics_rhs,
    steady_state_precursors,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Reference flux/XS constants — same order as PWR for cross-validation
# ---------------------------------------------------------------------------

MSR_PHI_NOM: float = 3.1e13   # thermal neutron flux at n = 1 (n/cm²/s)
MSR_SIGMA_F: float = 0.30     # macroscopic fission cross section (cm⁻¹)
MSR_SIGMA_A: float = 0.55     # one-group absorption XS for xenon worth denominator

# Superprompt-critical threshold (SPEC §4.6.5)
_SUPERCRITICAL_RHO_THRESHOLD: float = 0.01


# ---------------------------------------------------------------------------
# Configuration and control dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MSRModelConfig:
    """Physics constants and geometry for one MSR simulation run.

    τ_core and τ_loop are fixed for the run lifetime (SPEC §4.6.3 — variable-τ
    scenarios require re-initialising the history buffer and are Risk R-03).
    """

    # Core/loop transit times — primary DDE parameters (SPEC Eq. 22-23).
    # τ_core = L_core / v_salt — residence time of salt (and precursors) in the active core.
    tau_core: float = TAU_CORE_NOM  # s; must be > 0

    # τ_loop = L_loop / v_salt — transit time through the external loop.
    tau_loop: float = 20.0          # s; 0 means precursors return instantly (≡ PWR)

    # Combined salt temperature feedback coefficient (Doppler + moderator combined,
    # since fuel and coolant are the same material in an MSR — SPEC §4.4).
    # Negative for inherent self-regulation; generic graphite-moderated MSR range: −5 pcm/K.
    alpha_salt_pcm_per_k: float = -5.0  # pcm/K

    # Reference salt temperature for zero feedback (initial steady-state value).
    T_salt_ref_k: float = T_SALT_NOM    # K

    # Salt inlet temperature from the primary heat exchanger at nominal flow.
    T_salt_inlet_k: float = T_SALT_INLET_NOM  # K

    # Power reference: fission power (W) when n = 1.0 (normalised population).
    P_ref: float = P_NOM_MSR            # W

    # Neutron flux reference: thermal flux at n = 1.0 (not used by the engine RHS
    # directly but available for xenon coupling if added in Phase 3 Week 11+).
    phi_ref: float = MSR_PHI_NOM        # n/cm²/s

    # Prompt neutron generation time (s) — same value used by the shared kinetics core.
    Lambda: float = LAMBDA_PWR          # s

    # Base reactivity to compensate for flowing-precursor deficit and maintain criticality.
    # Set to msr_critical_base_reactivity_pcm(...) for a steady-state simulation;
    # leave at 0.0 for transient analysis where absolute criticality is not required.
    base_reactivity_pcm: float = 0.0    # pcm


@dataclass(frozen=True)
class MSRControls:
    """External controls for one MSR ODE evaluation step.

    external_reactivity_pcm combines rod reactivity and any other operator-driven
    reactivity insertion. The salt temperature feedback is computed internally
    from T_salt and alpha_salt_pcm_per_k.
    """

    external_reactivity_pcm: float = 0.0   # pcm; rod + other operator-driven reactivity
    salt_flow_fraction: float = 1.0         # scales ṁ_salt for thermal model; does NOT
                                             # change τ_core/τ_loop (fixed-run geometry)


ControlsFn = Callable[[float, np.ndarray], MSRControls]


def constant_controls_fn(controls: MSRControls) -> ControlsFn:
    """Return a controls callback that returns the same controls at every timestep."""
    def _fn(_t: float, _y: np.ndarray) -> MSRControls:
        return controls
    return _fn


# ---------------------------------------------------------------------------
# Flowing steady-state helpers
# ---------------------------------------------------------------------------


def flowing_steady_state_precursors(
    n0: float,
    beta_arr: np.ndarray,
    lambda_arr: np.ndarray,
    Lambda: float,
    tau_core: float,
    tau_loop: float,
) -> np.ndarray:
    """Return flowing-equilibrium precursor concentrations for an MSR at power n0.

    Derived from SPEC Eq. 22 with dCᵢ/dt = 0 and C_return using the same Cᵢ
    (the reactor has been at flowing steady state for t ≪ 0):

        0 = (βᵢ/Λ)n − λᵢCᵢ − (1/τ_core)Cᵢ + (1/τ_core)Cᵢ·e^{−λᵢτ_loop}
        Cᵢ,flow = (βᵢ/Λ)·n / [λᵢ + (1/τ_core)·(1 − e^{−λᵢτ_loop})]

    Limiting cases:
      - τ_core → ∞ or τ_loop = 0: Cᵢ,flow → βᵢ/(λᵢΛ)·n = Cᵢ,static  ✓
      - τ_loop → ∞:  Cᵢ,flow → (βᵢ/Λ)·n / (λᵢ + 1/τ_core)            (maximum loss)

    Args:
        n0:         Steady-state neutron population.
        beta_arr:   Delayed-neutron group fractions, shape (6,).
        lambda_arr: Precursor decay constants (s⁻¹), shape (6,).
        Lambda:     Prompt neutron generation time (s).
        tau_core:   Core transit time (s).
        tau_loop:   External loop transit time (s).

    Returns:
        Cᵢ,flow array, shape (6,), float64.
    """
    if tau_core >= 1.0e8:
        # Zero-flow limit: flowing SS equals static SS
        return steady_state_precursors(n0, beta_arr, lambda_arr, Lambda)
    alpha_i = (1.0 / tau_core) * (1.0 - np.exp(-lambda_arr * tau_loop))
    return beta_arr * n0 / (Lambda * (lambda_arr + alpha_i))


def msr_critical_base_reactivity_pcm(
    beta_arr: np.ndarray,
    lambda_arr: np.ndarray,
    Lambda: float,
    tau_core: float,
    tau_loop: float,
) -> float:
    """Return the base reactivity (pcm) needed to make the flowing MSR exactly critical.

    At flowing steady state the effective delayed-neutron source is reduced by
    precursor loss outside the core. The neutron balance (Eq. 1 at dn/dt = 0) then
    requires a compensating positive reactivity:

        ρ_crit = β_eff_static − Σᵢ βᵢλᵢ/(λᵢ + αᵢ)   [dimensionless]

    where αᵢ = (1/τ_core)·(1 − e^{−λᵢτ_loop}) is the net flow-loss rate per precursor.

    At zero salt velocity (τ_core → ∞, αᵢ → 0): ρ_crit → 0.
    At finite velocity: ρ_crit > 0 (must compensate precursor deficit with rod withdrawal).

    Args:
        beta_arr:   Delayed-neutron group fractions, shape (6,).
        lambda_arr: Precursor decay constants (s⁻¹), shape (6,).
        Lambda:     Prompt neutron generation time (s) — unused in formula but kept
                    for API consistency with flowing_steady_state_precursors.
        tau_core:   Core transit time (s).
        tau_loop:   External loop transit time (s).

    Returns:
        ρ_crit in pcm. Non-negative; 0 at zero salt velocity.
    """
    if tau_core >= 1.0e8:
        return 0.0
    alpha_i = (1.0 / tau_core) * (1.0 - np.exp(-lambda_arr * tau_loop))
    beta_eff_static = float(np.sum(beta_arr))
    effective_delayed = float(np.sum(beta_arr * lambda_arr / (lambda_arr + alpha_i)))
    rho_crit_dk_k = beta_eff_static - effective_delayed
    return rho_crit_dk_k * 1.0e5  # convert to pcm


# ---------------------------------------------------------------------------
# Initial state construction
# ---------------------------------------------------------------------------


def build_initial_state(
    *,
    n0: float = 1.0,
    config: MSRModelConfig = MSRModelConfig(),
    beta_arr: np.ndarray | None = None,
    lambda_arr: np.ndarray | None = None,
) -> np.ndarray:
    """Construct the 8-element steady-state initial state vector y₀.

    Analytical steady-state initialization (SPEC §4.6.2):
        Cᵢ,₀ = (βᵢ / λᵢ Λ) · n₀           (from dCᵢ/dt = 0 at constant power)
        T_salt,₀ = T_inlet + P₀ / (ṁ_nom · c_salt)  (from dT/dt = 0)

    The precursor concentrations are initialised at the FLOWING steady-state
    values (SPEC Eq. 22 with dCᵢ/dt = 0), which differ from the static formula
    whenever salt is flowing. This ensures the system starts at true flowing
    equilibrium with dCᵢ/dt ≈ 0 at t = 0, avoiding an artificial start-up
    transient. The DDE history buffer is pre-filled with the same values.

    Pair this with config.base_reactivity_pcm = msr_critical_base_reactivity_pcm(...)
    to ensure dn/dt = 0 at t = 0 as well. The helper make_critical_msr_config()
    bundles both into one call.

    Args:
        n0:         Initial normalised neutron population (1.0 = rated power).
        config:     MSR model configuration.
        beta_arr:   Optional β fractions; defaults to IAEA U-235 values.
        lambda_arr: Optional λᵢ; defaults to IAEA U-235 values.

    Returns:
        y₀, shape (8,), float64: [n₀, C₁,flow,₀, …, C₆,flow,₀, T_salt,₀].
    """
    if beta_arr is None:
        beta_arr = np.array(beta_fractions(), dtype=np.float64)
    if lambda_arr is None:
        lambda_arr = np.array(decay_constants(), dtype=np.float64)

    C0 = flowing_steady_state_precursors(
        n0, beta_arr, lambda_arr, config.Lambda, config.tau_core, config.tau_loop
    )

    power0 = n0 * config.P_ref
    m_dot0 = m_dot_salt_from_tau_core(config.tau_core)
    T_salt0 = steady_state_salt_temperature(power0, m_dot0, config.T_salt_inlet_k)
    # Guard against the zero-flow consistency-test edge case: at very large tau_core
    # the salt is essentially stagnant, ṁ → 0 makes P/(ṁ c_p) diverge, and the SS
    # formula returns an unphysical temperature. The MSR thermal model assumes flowing
    # salt; for the v_salt → 0 PWR-equivalence test the salt thermal node is not the
    # variable being compared, so falling back to T_salt_ref_k is the physically correct
    # initialization. The 3000 K bound conservatively brackets any operable MSR salt.
    if np.isnan(T_salt0) or T_salt0 > 3000.0 or T_salt0 < 0.0:
        logger.warning(
            "build_initial_state: steady-state T_salt = %.3e K is non-physical "
            "(τ_core = %.3e s ⇒ ṁ ≈ 0); defaulting to T_salt_ref_k = %.2f K.",
            T_salt0, config.tau_core, config.T_salt_ref_k,
        )
        T_salt0 = config.T_salt_ref_k

    return np.concatenate((
        np.array([n0], dtype=np.float64),
        C0,
        np.array([T_salt0], dtype=np.float64),
    ))


# ---------------------------------------------------------------------------
# ODE right-hand side
# ---------------------------------------------------------------------------


def msr_coupled_rhs(
    t: float,
    y: np.ndarray,
    controls_fn: ControlsFn,
    history: PrecursorHistory,
    config: MSRModelConfig,
    beta_arr: np.ndarray,
    lambda_arr: np.ndarray,
) -> np.ndarray:
    """Evaluate the full 8-state closed-loop MSR ODE right-hand side.

    Coupling chain at each solver call:
        1. T_salt drives salt temperature feedback → ρ_temp.
        2. Total ρ drives neutron kinetics (Eq. 1, shared core).
        3. History lookup C(t − τ_loop) provides the precursor return term (Eq. 23).
        4. Drift terms are added to dCᵢ/dt (Eq. 22).
        5. Power P = n · P_ref drives the salt temperature ODE.

    Args:
        t:           Current time (s).
        y:           8-state vector [n, C₁–C₆, T_salt], float64.
        controls_fn: Callable(t, y) → MSRControls. Called once per evaluation.
        history:     PrecursorHistory buffer supplying C(t − τ_loop).
        config:      MSRModelConfig with geometry and feedback parameters.
        beta_arr:    β fractions, shape (6,), float64.
        lambda_arr:  λᵢ (s⁻¹), shape (6,), float64.

    Returns:
        dy/dt, shape (8,), float64.
    """
    controls = controls_fn(t, y)

    n = y[0]
    C = y[1:7]
    T_salt = y[7]

    # --- Algebraic reactivity ---
    rho_base = config.base_reactivity_pcm * 1.0e-5        # criticality compensation
    rho_ext = controls.external_reactivity_pcm * 1.0e-5
    rho_temp = config.alpha_salt_pcm_per_k * (T_salt - config.T_salt_ref_k) * 1.0e-5
    rho_total = rho_base + rho_ext + rho_temp

    # --- Kinetics block (SPEC Eq. 1 + Eq. 22 base) — shared core ---
    kin_rhs = point_kinetics_rhs(t, y[:7], rho_total, beta_arr, lambda_arr, config.Lambda)
    # kin_rhs: shape (7,) = [dn/dt, dC₁/dt, …, dC₆/dt] WITHOUT drift terms yet

    # --- Precursor drift terms (SPEC Eqs. 22-23) ---
    # Skip drift when tau_core is very large (effectively zero salt velocity);
    # also handles tau_loop=0 (instant return, drift cancels to zero).
    if config.tau_core < 1.0e8:
        C_delayed = history.get(t - config.tau_loop)
        drift = msr_precursor_drift_rhs(
            C, C_delayed, lambda_arr, config.tau_core, config.tau_loop
        )
        # Drift terms add to precursor equations only (indices 1-6); dn/dt unchanged
        dydt = np.empty(8)
        dydt[0] = kin_rhs[0]
        dydt[1] = kin_rhs[1] + drift[0]
        dydt[2] = kin_rhs[2] + drift[1]
        dydt[3] = kin_rhs[3] + drift[2]
        dydt[4] = kin_rhs[4] + drift[3]
        dydt[5] = kin_rhs[5] + drift[4]
        dydt[6] = kin_rhs[6] + drift[5]
    else:
        # Zero-flow limit: drift terms vanish; MSR ≡ static-fuel point kinetics
        dydt = np.empty(8)
        dydt[:7] = kin_rhs

    # --- Salt thermal block ---
    power_w = n * config.P_ref
    m_dot = m_dot_salt_from_tau_core(config.tau_core) * controls.salt_flow_fraction
    th_rhs = salt_thermal_rhs(t, y[7:8], power_w, m_dot, config.T_salt_inlet_k)
    dydt[7] = th_rhs[0]

    return dydt


# ---------------------------------------------------------------------------
# Method-of-steps DDE integration
# ---------------------------------------------------------------------------


def integrate_msr(
    y0: np.ndarray,
    t_span: tuple[float, float],
    controls: MSRControls | ControlsFn,
    config: MSRModelConfig = MSRModelConfig(),
    *,
    beta_arr: np.ndarray | None = None,
    lambda_arr: np.ndarray | None = None,
    t_eval: np.ndarray | None = None,
    dt_history: float | None = None,
) -> Any:
    """Integrate the 8-state MSR DDE system using the method-of-steps approach.

    The precursor return term (SPEC Eq. 23) makes the MSR a delay-differential
    equation (DDE) system that scipy cannot handle in a single solve_ivp call.
    This function integrates in chunks of at most ``dt_history`` seconds and
    updates the PrecursorHistory buffer between chunks, so every history lookup
    within a chunk references already-accepted state values.

    Chunk size guarantee (SPEC §4.6.3): dt_history ≤ τ_loop / 2 ensures that
    within each chunk [t_i, t_{i+1}], all lookups C(t − τ_loop) reference
    t_i − τ_loop, which is at least τ_loop/2 in the past — always in history.

    For τ_loop = 0 (instant precursor return, drift cancels), dt_history defaults
    to 10 s (no delay to enforce, so step size is set by accuracy only).

    Args:
        y0:          Initial 8-state vector, shape (8,), float64.
        t_span:      (t_start, t_end) in seconds.
        controls:    Either a MSRControls instance (constant) or a ControlsFn callable.
        config:      MSRModelConfig; τ_core and τ_loop are fixed for the full run.
        beta_arr:    Optional β fractions; defaults to IAEA U-235 values.
        lambda_arr:  Optional λᵢ; defaults to IAEA U-235 values.
        t_eval:      Optional array of output times (s). None lets the solver choose.
        dt_history:  Chunk size (s) for method-of-steps. Defaults to τ_loop / 2
                     (or 10 s when τ_loop = 0). Override for long xenon-timescale runs.

    Returns:
        SimpleNamespace with attributes:
            .t       (n,) array of output times
            .y       (8, n) array of output states
            .success True if all chunks converged
            .message Status message
    """
    if beta_arr is None:
        beta_arr = np.array(beta_fractions(), dtype=np.float64)
    if lambda_arr is None:
        lambda_arr = np.array(decay_constants(), dtype=np.float64)

    controls_fn: ControlsFn = (
        controls if callable(controls) else constant_controls_fn(controls)
    )

    t_start, t_end = float(t_span[0]), float(t_span[1])

    # --- Determine chunk size ---
    tau_loop = config.tau_loop
    if dt_history is None:
        dt_history = tau_loop / 2.0 if tau_loop > 0.0 else 10.0
    dt_history = float(dt_history)

    # Warn for superprompt-critical initial reactivity (SPEC §4.6.5)
    ctrl0 = controls_fn(t_start, y0)
    rho0 = (
        ctrl0.external_reactivity_pcm * 1.0e-5
        + config.alpha_salt_pcm_per_k * (y0[7] - config.T_salt_ref_k) * 1.0e-5
    )
    if abs(rho0) > _SUPERCRITICAL_RHO_THRESHOLD:
        logger.warning(
            "MSR: |ρ₀| = %.5f (%.0f pcm) exceeds 1000 pcm superprompt threshold.",
            rho0, rho0 * 1.0e5,
        )

    # --- Initialise history buffer ---
    # Pre-fill with steady-state precursor concentrations (SPEC §4.6.3).
    C_init = y0[1:7].copy()
    history = PrecursorHistory(t_start=t_start, C_init=C_init)
    history.add(t_start, C_init)

    # --- Filter t_eval to the requested time span ---
    if t_eval is not None:
        t_eval_use = t_eval[(t_eval >= t_start) & (t_eval <= t_end)]
    else:
        t_eval_use = None

    # --- Method-of-steps integration loop ---
    all_t: list[np.ndarray] = []
    all_y: list[np.ndarray] = []
    success = True
    message = "Integration completed successfully."

    t_cur = t_start
    y_cur = y0.astype(np.float64).copy()

    while t_cur < t_end - 1.0e-12:
        t_chunk_end = min(t_cur + dt_history, t_end)

        # Build t_eval slice for this chunk
        if t_eval_use is not None:
            chunk_eval = t_eval_use[
                (t_eval_use > t_cur - 1.0e-12) & (t_eval_use <= t_chunk_end + 1.0e-12)
            ]
            chunk_eval = chunk_eval[chunk_eval > t_cur]  # exclude t_cur (already recorded)
            if len(chunk_eval) == 0:
                chunk_eval = np.array([t_chunk_end])
            elif chunk_eval[-1] < t_chunk_end - 1.0e-12:
                chunk_eval = np.append(chunk_eval, t_chunk_end)
        else:
            chunk_eval = None

        # Build the RHS closure capturing current history snapshot
        def _rhs(
            t: float,
            y: np.ndarray,
            _history: PrecursorHistory = history,
            _controls_fn: ControlsFn = controls_fn,
            _config: MSRModelConfig = config,
            _beta: np.ndarray = beta_arr,
            _lam: np.ndarray = lambda_arr,
        ) -> np.ndarray:
            return msr_coupled_rhs(t, y, _controls_fn, _history, _config, _beta, _lam)

        result = solve_ivp(
            _rhs,
            (t_cur, t_chunk_end),
            y_cur,
            method="Radau",
            rtol=1.0e-6,
            atol=1.0e-9,
            t_eval=chunk_eval,
            dense_output=False,
        )

        if not result.success:
            logger.error(
                "MSR ODE integration failed at t=%.4f–%.4f: %s",
                t_cur, t_chunk_end, result.message,
            )
            success = False
            message = f"Integration failed at t={t_cur:.4f} s: {result.message}"
            # Append partial results so caller can inspect state at failure
            if result.t.size > 0:
                all_t.append(result.t)
                all_y.append(result.y)
            break

        # Update history with accepted output points from this chunk
        for k in range(result.y.shape[1]):
            history.add(result.t[k], result.y[1:7, k])

        # Prune history entries older than τ_loop + margin to bound memory
        history.prune_before(t_chunk_end - tau_loop - dt_history)

        all_t.append(result.t)
        all_y.append(result.y)

        t_cur = result.t[-1]
        y_cur = result.y[:, -1]

    # --- Combine all chunks ---
    # Always prepend the initial state so t_start is present in the output
    # even when t_eval includes t_start (the chunk filter skips it).
    t_init = np.array([t_start])
    y_init = y0.astype(np.float64).reshape(-1, 1)

    if not all_t:
        return SimpleNamespace(
            t=t_init,
            y=y_init,
            success=False,
            message="No integration steps completed.",
        )

    t_combined = np.concatenate([t_init] + all_t)
    y_combined = np.concatenate([y_init] + all_y, axis=1)

    # Remove duplicate time points (chunk boundaries and repeated t_start)
    _, unique_idx = np.unique(t_combined, return_index=True)
    t_out = t_combined[unique_idx]
    y_out = y_combined[:, unique_idx]

    return SimpleNamespace(t=t_out, y=y_out, success=success, message=message)


# ---------------------------------------------------------------------------
# Convenience constructor for critical flowing-steady-state simulations
# ---------------------------------------------------------------------------


def make_critical_msr_config(
    tau_core: float = TAU_CORE_NOM,
    tau_loop: float = 20.0,
    alpha_salt_pcm_per_k: float = -5.0,
    T_salt_ref_k: float = T_SALT_NOM,
    T_salt_inlet_k: float = T_SALT_INLET_NOM,
    P_ref: float = P_NOM_MSR,
    phi_ref: float = MSR_PHI_NOM,
    Lambda: float = LAMBDA_PWR,
    beta_arr: np.ndarray | None = None,
    lambda_arr: np.ndarray | None = None,
) -> MSRModelConfig:
    """Return MSRModelConfig with base_reactivity_pcm set for flowing criticality.

    Computes msr_critical_base_reactivity_pcm() automatically and embeds it in
    the config so that a simulation starting from build_initial_state() with
    MSRControls(external_reactivity_pcm=0.0) will be at exact flowing equilibrium
    (dn/dt = 0, dCᵢ/dt = 0, dT_salt/dt = 0) at t = 0.

    Args:
        tau_core, tau_loop:  Core and loop transit times (s).
        alpha_salt_pcm_per_k: Salt temperature feedback (pcm/K). Negative.
        T_salt_ref_k:        Reference (zero-feedback) salt temperature (K).
        T_salt_inlet_k:      Salt inlet temperature (K).
        P_ref, phi_ref:      Reference power (W) and flux (n/cm²/s) at n=1.
        Lambda:              Prompt neutron generation time (s).
        beta_arr:            Optional β fractions; defaults to IAEA U-235.
        lambda_arr:          Optional λᵢ; defaults to IAEA U-235.

    Returns:
        MSRModelConfig with base_reactivity_pcm = ρ_crit.
    """
    if beta_arr is None:
        beta_arr = np.array(beta_fractions(), dtype=np.float64)
    if lambda_arr is None:
        lambda_arr = np.array(decay_constants(), dtype=np.float64)

    rho_crit_pcm = msr_critical_base_reactivity_pcm(
        beta_arr, lambda_arr, Lambda, tau_core, tau_loop
    )
    return MSRModelConfig(
        tau_core=tau_core,
        tau_loop=tau_loop,
        alpha_salt_pcm_per_k=alpha_salt_pcm_per_k,
        T_salt_ref_k=T_salt_ref_k,
        T_salt_inlet_k=T_salt_inlet_k,
        P_ref=P_ref,
        phi_ref=phi_ref,
        Lambda=Lambda,
        base_reactivity_pcm=rho_crit_pcm,
    )


# ---------------------------------------------------------------------------
# JIT pre-warm
# ---------------------------------------------------------------------------


def prewarm_jit_msr(
    beta_arr: np.ndarray | None = None,
    lambda_arr: np.ndarray | None = None,
) -> None:
    """Pre-warm Numba JIT caches for MSR kernels.

    Triggers AOT compilation of point_kinetics_rhs, msr_precursor_drift_rhs, and
    salt_thermal_rhs at startup to eliminate first-call compile latency
    (ARCHITECTURE.md §7 Risk R-04, CLAUDE.md §Numba Rules).

    Args:
        beta_arr:   Optional β fractions; defaults to IAEA U-235 values.
        lambda_arr: Optional λᵢ; defaults to IAEA U-235 values.
    """
    if beta_arr is None:
        beta_arr = np.array(beta_fractions(), dtype=np.float64)
    if lambda_arr is None:
        lambda_arr = np.array(decay_constants(), dtype=np.float64)

    y_dummy = np.ones(8, dtype=np.float64)
    y_dummy[7] = T_SALT_NOM

    config = MSRModelConfig()
    history = PrecursorHistory(t_start=0.0, C_init=y_dummy[1:7])
    history.add(0.0, y_dummy[1:7])
    controls_fn = constant_controls_fn(MSRControls())

    # One RHS evaluation warms all @njit sub-kernels
    msr_coupled_rhs(0.0, y_dummy, controls_fn, history, config, beta_arr, lambda_arr)
