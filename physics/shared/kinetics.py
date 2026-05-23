"""
physics/shared/kinetics.py — 6-group point kinetics ODE system.

Shared core for PWR (Phase 1) and MSR (Phase 3). MSR wraps this module
and injects precursor drift terms; it never duplicates these equations.

SPEC equations implemented:
    Eq. 1  dn/dt   = [(ρ − β_eff) / Λ] · n  +  Σᵢ λᵢ Cᵢ
    Eq. 2  dCᵢ/dt = (βᵢ / Λ) · n  −  λᵢ Cᵢ     i = 1 … 6

State vector layout (7 states):
    y[0]   = n       neutron population (normalised; proportional to thermal power)
    y[1:7] = C₁…C₆  delayed-neutron precursor group concentrations

Unit conventions (SPEC §4.6.4):
    - Reactivity ρ is dimensionless (Δk/k).  1 pcm = 1 × 10⁻⁵.
    - Λ in seconds.  λᵢ in s⁻¹.

Architecture rules (CLAUDE.md):
    - Pure physics: no Dash, UI, or orchestrator imports anywhere here.
    - Keepin U-235 6-group parameters imported from data.keepin_dnp — never hardcoded.
    - @numba.njit on the hot inner kernel (point_kinetics_rhs).
    - solve_ivp called with method='Radau', rtol=1e-6, atol=1e-9.
"""

from __future__ import annotations

import logging
from typing import Callable

import numba
import numpy as np
from scipy.integrate import solve_ivp

from data.keepin_dnp import beta_fractions, decay_constants

logger = logging.getLogger(__name__)

# Default PWR prompt neutron generation time (SPEC §4.5.4, Table of Variables)
LAMBDA_PWR: float = 1.0e-5  # s

# |ρ| above this triggers a superprompt-critical warning (SPEC §4.6.5, 1000 pcm ≈ 1.5β)
_SUPERCRITICAL_RHO_THRESHOLD: float = 0.01


@numba.njit(cache=True)
def point_kinetics_rhs(
    t: float,
    y: np.ndarray,
    rho: float,
    beta_arr: np.ndarray,
    lambda_arr: np.ndarray,
    Lambda: float,
) -> np.ndarray:
    """Numba JIT kernel: 7-state PKE right-hand side for one solver call.

    SPEC Eq. 1:  dy[0]/dt = [(ρ − β_eff) / Λ] · n  +  Σᵢ λᵢ Cᵢ
    SPEC Eq. 2:  dy[i]/dt = (βᵢ / Λ) · n  −  λᵢ Cᵢ    i = 1…6

    Args:
        t: Current time (s) — required by scipy ODE driver signature.
        y: State vector [n, C₁, …, C₆], shape (7,), float64.
        rho: Total reactivity ρ (dimensionless). Caller computes feedback before call.
        beta_arr: Delayed-neutron group fractions βᵢ, shape (6,), float64.
        lambda_arr: Precursor decay constants λᵢ (s⁻¹), shape (6,), float64.
        Lambda: Prompt neutron generation time Λ (s).

    Returns:
        dy/dt, shape (7,), float64.
    """
    n = y[0]

    # β_eff = Σᵢ βᵢ (computed each call so beta_arr can vary for future MSR extension)
    beta_eff = 0.0
    for i in range(6):
        beta_eff += beta_arr[i]

    # Delayed neutron source: Σᵢ λᵢ Cᵢ
    delayed_source = 0.0
    for i in range(6):
        delayed_source += lambda_arr[i] * y[i + 1]

    dydt = np.empty(7)

    # Eq. 1 — neutron population
    dydt[0] = ((rho - beta_eff) / Lambda) * n + delayed_source

    # Eq. 2 — one equation per precursor group
    for i in range(6):
        dydt[i + 1] = (beta_arr[i] / Lambda) * n - lambda_arr[i] * y[i + 1]

    return dydt


def steady_state_precursors(
    n0: float,
    beta_arr: np.ndarray,
    lambda_arr: np.ndarray,
    Lambda: float,
) -> np.ndarray:
    """Return equilibrium precursor concentrations for steady-state power n0.

    Derived from SPEC Eq. 2 by setting dCᵢ/dt = 0 (SPEC §4.6.2):
        Cᵢ,₀ = (βᵢ / λᵢ Λ) · n₀

    Args:
        n0: Steady-state neutron population (same normalisation as y[0]).
        beta_arr: Delayed-neutron group fractions βᵢ, shape (6,), float64.
        lambda_arr: Precursor decay constants λᵢ (s⁻¹), shape (6,), float64.
        Lambda: Prompt neutron generation time Λ (s).

    Returns:
        Cᵢ,₀ array, shape (6,), float64.
    """
    return (beta_arr / (lambda_arr * Lambda)) * n0


def build_initial_state(
    n0: float,
    beta_arr: np.ndarray,
    lambda_arr: np.ndarray,
    Lambda: float,
) -> np.ndarray:
    """Construct the 7-element steady-state initial state vector y₀.

    y₀ = [n₀, C₁,₀, C₂,₀, C₃,₀, C₄,₀, C₅,₀, C₆,₀]

    Args:
        n0: Initial (steady-state) neutron population.
        beta_arr: Delayed-neutron group fractions βᵢ, shape (6,), float64.
        lambda_arr: Precursor decay constants λᵢ (s⁻¹), shape (6,), float64.
        Lambda: Prompt neutron generation time Λ (s).

    Returns:
        y₀, shape (7,), float64.
    """
    C0 = steady_state_precursors(n0, beta_arr, lambda_arr, Lambda)
    return np.concatenate(([n0], C0))


def integrate_kinetics(
    y0: np.ndarray,
    t_span: tuple[float, float],
    rho_fn: Callable[[float, np.ndarray], float],
    beta_arr: np.ndarray,
    lambda_arr: np.ndarray,
    Lambda: float,
    t_eval: np.ndarray | None = None,
):
    """Integrate the 7-state point kinetics ODE over t_span.

    Uses scipy Radau (stiff implicit Runge-Kutta) with tolerances fixed per
    CLAUDE.md §Architecture Rules (rtol=1e-6, atol=1e-9).

    The inner hot loop is compiled by Numba; Python overhead is limited to
    one rho_fn evaluation and one @njit call per solver step.

    Args:
        y0: Initial state [n₀, C₁,…,C₆], shape (7,), float64.
        t_span: (t_start, t_end) in seconds.
        rho_fn: Callable(t, y) → ρ_total (dimensionless). Evaluated at each step.
                For constant reactivity: ``lambda t, y: rho_value``.
        beta_arr: Delayed-neutron group fractions βᵢ, shape (6,), float64.
        lambda_arr: Precursor decay constants λᵢ (s⁻¹), shape (6,), float64.
        Lambda: Prompt neutron generation time Λ (s).
        t_eval: Optional array of output times (s). None lets the solver choose.

    Returns:
        scipy OdeResult with .t (shape n,), .y (shape 7×n), .success, .message.
    """
    # Emit a warning for superprompt-critical conditions (SPEC §4.6.5)
    rho_initial = rho_fn(t_span[0], y0)
    if abs(rho_initial) > _SUPERCRITICAL_RHO_THRESHOLD:
        logger.warning(
            "Superprompt-critical condition: |ρ| = %.5f (%.0f pcm) exceeds 1000 pcm. "
            "Solver continues; inspect results carefully.",
            rho_initial,
            rho_initial * 1e5,
        )

    def _rhs(t: float, y: np.ndarray) -> np.ndarray:
        rho = rho_fn(t, y)
        return point_kinetics_rhs(t, y, rho, beta_arr, lambda_arr, Lambda)

    result = solve_ivp(
        _rhs,
        t_span,
        y0,
        method="Radau",
        rtol=1e-6,
        atol=1e-9,
        t_eval=t_eval,
        dense_output=False,
    )

    if not result.success:
        logger.error(
            "Point kinetics ODE integration failed: %s", result.message
        )

    return result


def prewarm_jit(
    beta_arr: np.ndarray | None = None,
    lambda_arr: np.ndarray | None = None,
    Lambda: float = LAMBDA_PWR,
) -> None:
    """Pre-warm the Numba JIT cache to eliminate first-call compile latency.

    Must be called once at application startup before any UI interaction
    (CLAUDE.md §Numba Rules, ARCHITECTURE.md §7 Risk R-04).

    Args:
        beta_arr: Optional beta fractions; defaults to IAEA U-235 values.
        lambda_arr: Optional decay constants; defaults to IAEA U-235 values.
        Lambda: Prompt neutron generation time (s).
    """
    if beta_arr is None:
        beta_arr = np.array(beta_fractions(), dtype=np.float64)
    if lambda_arr is None:
        lambda_arr = np.array(decay_constants(), dtype=np.float64)

    y_dummy = np.ones(7, dtype=np.float64)
    point_kinetics_rhs(0.0, y_dummy, 0.0, beta_arr, lambda_arr, Lambda)
