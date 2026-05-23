"""
physics/msr/drift.py — MSR precursor drift terms and delay-system history buffer.

MSR delayed-neutron precursors are dissolved in the flowing salt. Precursors
that leave the core while decaying outside contribute no useful delayed neutrons.
This module implements the three governing equations for that transport effect.

SPEC equations implemented:
    Eq. 22  dCᵢ/dt = (βᵢ/Λ)n − λᵢCᵢ − (1/τ_core)Cᵢ + Cᵢ^return(t)
            [drift terms only — shared kinetics handles the first two terms]
    Eq. 23  Cᵢ^return(t) = (1/τ_core)·Cᵢ(t − τ_loop)·e^{−λᵢ τ_loop}
    Eq. 24  β_eff,flow = Σᵢ βᵢ·[λᵢτ_core/(1+λᵢτ_core)]·[1 + e^{−λᵢτ_loop}/(1+λᵢτ_core−e^{−λᵢτ_loop})]

Delay-differential equation (DDE) handling (SPEC §4.6.3):
    Eq. 23 introduces a delay: the RHS at time t depends on Cᵢ(t − τ_loop).
    The platform resolves this with the method-of-steps approach:
      • PrecursorHistory stores accepted (t, C) pairs in a sorted list.
      • Lookups use linear interpolation between the two nearest stored points.
      • For t < t_start (before the simulation began), C_init (steady-state) is returned.
      • The engine integrates in chunks ≤ τ_loop/2 so all delay lookups reference
        already-accepted history.

Architecture rules (CLAUDE.md):
    - Pure physics: no Dash, UI, or orchestrator imports.
    - @njit on the hot ODE kernel (msr_precursor_drift_rhs).
    - compute_beta_eff_flow is Python-level (called once per scenario, not per step).
"""

from __future__ import annotations

import bisect
import logging
import math

import numba
import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Numba-compiled drift kernel — called at every ODE evaluation
# ---------------------------------------------------------------------------


@numba.njit(cache=True)
def msr_precursor_drift_rhs(
    C_arr: np.ndarray,
    C_delayed: np.ndarray,
    lambda_arr: np.ndarray,
    tau_core: float,
    tau_loop: float,
) -> np.ndarray:
    """Numba JIT kernel: MSR precursor drift contribution to dCᵢ/dt (SPEC Eq. 22 drift part).

    Returns the net drift term to be ADDED to the standard point-kinetics
    precursor ODE (which the shared kinetics core already computes):

        drift_i = −(1/τ_core)·Cᵢ  +  (1/τ_core)·Cᵢ(t−τ_loop)·e^{−λᵢ τ_loop}
                  [flow-out]            [return — Eq. 23]

    At τ_core → ∞ (zero salt velocity) both terms vanish and the MSR precursor
    ODE reduces exactly to the PWR precursor ODE (ARCHITECTURE.md §5, consistency test).

    Args:
        C_arr:     Current in-core precursor concentrations Cᵢ(t), shape (6,), float64.
        C_delayed: Past precursor concentrations Cᵢ(t − τ_loop), shape (6,), float64.
                   Supplied by PrecursorHistory.get(t − τ_loop).
        lambda_arr: Precursor decay constants λᵢ (s⁻¹), shape (6,), float64.
        tau_core:  Core transit time τ_core (s). Must be > 0.
        tau_loop:  External loop transit time τ_loop (s). May be 0.

    Returns:
        Drift contribution array, shape (6,), float64.
        Positive sign convention: add directly to dCᵢ/dt.
    """
    inv_tau = 1.0 / tau_core
    result = np.empty(6)
    for i in range(6):
        flow_out = -inv_tau * C_arr[i]                                  # Eq. 22 loss term
        return_term = inv_tau * C_delayed[i] * math.exp(-lambda_arr[i] * tau_loop)  # Eq. 23
        result[i] = flow_out + return_term
    return result


# ---------------------------------------------------------------------------
# β_eff,flow — steady-state effective delayed fraction under flowing-salt conditions
# ---------------------------------------------------------------------------


def compute_beta_eff_flow(
    beta_arr: np.ndarray,
    lambda_arr: np.ndarray,
    tau_core: float,
    tau_loop: float,
) -> float:
    """Compute β_eff,flow — effective delayed neutron fraction with precursor drift.

    Implements SPEC Eq. 24 (steady-state solution of Eq. 22 at constant power):

        β_eff,flow = Σᵢ βᵢ · [λᵢτ_core / (1 + λᵢτ_core)]
                          · [1 + e^{−λᵢτ_loop} / (1 + λᵢτ_core − e^{−λᵢτ_loop})]

    Limiting cases (validated by test_beta_eff_flow_zero_velocity):
      • τ_core → ∞ (v_salt → 0):   β_eff,flow → Σβᵢ = β_eff,static  (no drift loss)
      • τ_loop = 0 (instant return): β_eff,flow = Σβᵢ                (cancelled flow-out)
      • τ_core → 0 (v_salt → ∞):   β_eff,flow → 0                   (all precursors lost)

    Args:
        beta_arr:   Delayed-neutron group fractions βᵢ, shape (6,), float64.
        lambda_arr: Precursor decay constants λᵢ (s⁻¹), shape (6,), float64.
        tau_core:   Core transit time τ_core (s). Must be > 0; large value ≈ zero flow.
        tau_loop:   External loop transit time τ_loop (s). Non-negative.

    Returns:
        β_eff,flow (dimensionless). Always in (0, β_eff_static].
    """
    beta_flow = 0.0
    for i in range(len(beta_arr)):
        li = lambda_arr[i]
        bi = beta_arr[i]
        li_tc = li * tau_core
        exp_term = math.exp(-li * tau_loop)
        denom = 1.0 + li_tc - exp_term
        if abs(denom) < 1e-30:
            # Degenerate case: treat as no drift (shouldn't occur in physical range)
            beta_flow += bi
            continue
        factor1 = li_tc / (1.0 + li_tc)
        factor2 = 1.0 + exp_term / denom
        beta_flow += bi * factor1 * factor2
    return beta_flow


# ---------------------------------------------------------------------------
# PrecursorHistory — delay-system state buffer
# ---------------------------------------------------------------------------


class PrecursorHistory:
    """Sorted history buffer for MSR precursor concentrations.

    Stores (time, C_array) pairs from accepted ODE steps. Linear interpolation
    is used to retrieve Cᵢ(t − τ_loop) at any requested time t. For any lookup
    time before the simulation start (t < t_start), the pre-filled steady-state
    value C_init is returned — consistent with the assumption that the reactor
    ran at steady state before t = 0 (SPEC §4.6.3).

    Memory management: points older than (current_time − τ_loop − margin) are
    pruned when the buffer exceeds max_size entries, keeping memory bounded.

    Usage pattern (SPEC §4.6.3 method-of-steps):
        history = PrecursorHistory(t_start=0.0, C_init=C_ss)
        for each integration chunk [t_i, t_{i+1}]:
            result = solve_ivp(..., args=(history,))
            for t_k, C_k in zip(result.t, result.y[1:7].T):
                history.add(t_k, C_k)
    """

    def __init__(
        self,
        t_start: float,
        C_init: np.ndarray,
        max_size: int = 50_000,
    ) -> None:
        """Initialise the history buffer.

        Args:
            t_start: Simulation start time (s). Lookups at t ≤ t_start return C_init.
            C_init:  Steady-state precursor concentrations, shape (6,), float64.
                     Pre-fills the "before simulation" history (SPEC §4.6.3).
            max_size: Maximum number of stored points before pruning. Default 50 000.
        """
        self._t_start = float(t_start)
        self._C_init = C_init.astype(np.float64).copy()
        self._max_size = max_size
        self._times: list[float] = []
        self._values: list[np.ndarray] = []

    # ------------------------------------------------------------------
    def get(self, t: float) -> np.ndarray:
        """Return C at time t via linear interpolation.

        Returns C_init for any lookup before or at t_start (buffer underrun).
        Logs a debug message for underruns (SPEC §4.6.5).

        Args:
            t: Time to look up (s). Typically t_current − τ_loop.

        Returns:
            Cᵢ values at time t, shape (6,), float64.
        """
        if len(self._times) == 0 or t <= self._t_start:
            if t < self._t_start - 1e-12:
                logger.debug(
                    "PrecursorHistory underrun at t=%.6f (before t_start=%.6f); "
                    "returning steady-state initial value.",
                    t, self._t_start,
                )
            return self._C_init

        # Clamp to available range
        if t >= self._times[-1]:
            return self._values[-1]
        if t <= self._times[0]:
            return self._values[0]

        # Binary search for bracketing interval [times[idx-1], times[idx]]
        idx = bisect.bisect_right(self._times, t)
        t0 = self._times[idx - 1]
        t1 = self._times[idx]
        C0 = self._values[idx - 1]
        C1 = self._values[idx]
        alpha = (t - t0) / (t1 - t0)
        return C0 + alpha * (C1 - C0)

    # ------------------------------------------------------------------
    def add(self, t: float, C: np.ndarray) -> None:
        """Record a new (t, C) history point.

        Points must be added in non-decreasing time order for correct interpolation.

        Args:
            t: Time of the new point (s).
            C: Precursor concentrations at time t, shape (6,), float64.
        """
        self._times.append(float(t))
        self._values.append(C.astype(np.float64).copy())

        # Prune oldest entries once buffer exceeds max_size
        if len(self._times) > self._max_size:
            prune_count = self._max_size // 10
            del self._times[:prune_count]
            del self._values[:prune_count]

    # ------------------------------------------------------------------
    def prune_before(self, t_cutoff: float) -> None:
        """Remove all entries with time < t_cutoff.

        Called by the engine after each chunk to keep memory bounded while
        retaining at least one full τ_loop window of history.

        Args:
            t_cutoff: Oldest time to retain (s). Typically t_current − τ_loop − margin.
        """
        idx = bisect.bisect_left(self._times, t_cutoff)
        if idx > 1:
            del self._times[: idx - 1]
            del self._values[: idx - 1]

    # ------------------------------------------------------------------
    @property
    def size(self) -> int:
        """Number of stored history points."""
        return len(self._times)
