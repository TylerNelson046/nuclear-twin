"""
physics/shared/xenon.py — Iodine-135 / Xenon-135 coupled ODE system.

Implements fission-product poisoning kinetics for Xe-135 and I-135,
the largest time-dependent reactivity effect in an operating PWR.

SPEC equations implemented:
    Eq. 14  dI/dt = γ_I · Σ_f · φ − λ_I · I
    Eq. 15  dX/dt = γ_X · Σ_f · φ + λ_I · I − (λ_X + σ_aX · φ) · X

State vector layout (2 states):
    y[0]  = I   I-135 number density  (atoms/cm³)
    y[1]  = X   Xe-135 number density (atoms/cm³)

Coupling point:
    phi (neutrons/cm²/s) is received as a dynamic variable from the point
    kinetics engine at each ODE timestep. In the coupled 11-state PWR system,
    phi = (n / n0) * phi_0, where n is the current neutron population.

Unit conventions (SPEC §4.6.4):
    CGS units throughout: cm² for σ, cm⁻¹ for Σ, neutrons/cm²/s for φ.

Architecture rules (CLAUDE.md):
    - Pure physics: no Dash, UI, or orchestrator imports.
    - solve_ivp called with method='Radau', rtol=1e-6, atol=1e-9.
    - @njit on the hot inner kernel (xenon_iodine_rhs).
"""

from __future__ import annotations

import logging
from typing import Callable

import numba
import numpy as np
from scipy.integrate import solve_ivp

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Physical constants — source of truth: SPEC §4.5.4
# ---------------------------------------------------------------------------

GAMMA_I: float = 0.0639     # I-135 cumulative fission yield (dimensionless)
GAMMA_X: float = 0.00237    # Xe-135 direct fission yield (dimensionless)

LAMBDA_I: float = 2.87e-5   # I-135 decay constant (s⁻¹); t½ ≈ 6.7 hr
LAMBDA_X: float = 2.09e-5   # Xe-135 decay constant (s⁻¹); t½ ≈ 9.2 hr

# Xe-135 microscopic neutron absorption cross section (SPEC §4.5.4):
# 2.6×10⁶ barns × 1×10⁻²⁴ cm²/barn = 2.6×10⁻¹⁸ cm²
SIGMA_AX: float = 2.6e-18   # cm²


@numba.njit(cache=True)
def xenon_iodine_rhs(
    t: float,
    y: np.ndarray,
    phi: float,
    Sigma_f: float,
) -> np.ndarray:
    """Numba JIT kernel: 2-state I-135/Xe-135 ODE right-hand side.

    SPEC Eq. 14:  dI/dt = γ_I · Σ_f · φ − λ_I · I
    SPEC Eq. 15:  dX/dt = γ_X · Σ_f · φ + λ_I · I − (λ_X + σ_aX · φ) · X

    Args:
        t: Current time (s) — required by scipy ODE driver signature.
        y: State vector [I, X] (atoms/cm³), shape (2,), float64.
        phi: Thermal neutron flux (neutrons/cm²/s). Dynamic coupling variable
             received from the point kinetics engine each solver step.
        Sigma_f: Macroscopic fission cross section (cm⁻¹).

    Returns:
        dy/dt = [dI/dt, dX/dt], shape (2,), float64.
    """
    I = y[0]
    X = y[1]

    fission_rate = Sigma_f * phi  # fissions cm⁻³ s⁻¹

    dI_dt = GAMMA_I * fission_rate - LAMBDA_I * I
    dX_dt = GAMMA_X * fission_rate + LAMBDA_I * I - (LAMBDA_X + SIGMA_AX * phi) * X

    dydt = np.empty(2)
    dydt[0] = dI_dt
    dydt[1] = dX_dt
    return dydt


def steady_state_xenon(phi: float, Sigma_f: float) -> np.ndarray:
    """Return equilibrium [I₀, X₀] at constant flux φ.

    Derived from SPEC Eqs. 14–15 with dI/dt = dX/dt = 0 (SPEC §4.6.2):
        I₀ = γ_I · Σ_f · φ / λ_I
        X₀ = (γ_X + γ_I) · Σ_f · φ / (λ_X + σ_aX · φ)

    At φ = 0 both concentrations are zero (no fission source).

    Args:
        phi: Thermal neutron flux (neutrons/cm²/s). May be zero.
        Sigma_f: Macroscopic fission cross section (cm⁻¹).

    Returns:
        [I₀, X₀] in atoms/cm³, shape (2,), float64.
    """
    if phi == 0.0:
        return np.zeros(2, dtype=np.float64)

    fission_rate = Sigma_f * phi
    I0 = GAMMA_I * fission_rate / LAMBDA_I
    X0 = (GAMMA_X + GAMMA_I) * fission_rate / (LAMBDA_X + SIGMA_AX * phi)
    return np.array([I0, X0], dtype=np.float64)


def integrate_xenon(
    y0: np.ndarray,
    t_span: tuple[float, float],
    phi_fn: Callable[[float, np.ndarray], float],
    Sigma_f: float,
    t_eval: np.ndarray | None = None,
):
    """Integrate the 2-state I-135/Xe-135 ODE over t_span.

    Uses scipy Radau (stiff implicit Runge-Kutta) with tolerances fixed per
    CLAUDE.md §Architecture Rules (rtol=1e-6, atol=1e-9).

    The inner hot loop is compiled by Numba; Python overhead per solver step
    is limited to one phi_fn call and one @njit dispatch.

    Args:
        y0: Initial state [I₀, X₀] (atoms/cm³), shape (2,), float64.
        t_span: (t_start, t_end) in seconds.
        phi_fn: Callable(t, y) → φ (neutrons/cm²/s). Evaluated each solver step;
                this is the coupling injection point from the kinetics engine.
                For constant flux:  ``lambda t, y: phi_value``
                For shutdown at t_s: ``lambda t, y: phi0 if t < t_s else 0.0``
        Sigma_f: Macroscopic fission cross section (cm⁻¹).
        t_eval: Optional array of output times (s). None lets the solver choose.

    Returns:
        scipy OdeResult with .t (shape n,), .y (shape 2×n), .success, .message.
    """
    def _rhs(t: float, y: np.ndarray) -> np.ndarray:
        phi = phi_fn(t, y)
        return xenon_iodine_rhs(t, y, phi, Sigma_f)

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
        logger.error("Xenon/Iodine ODE integration failed: %s", result.message)

    return result
