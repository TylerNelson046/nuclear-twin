"""
physics/pwr/thermal_hydraulics.py — Lumped-parameter fuel/coolant thermal-hydraulic model.

Implements SPEC §4.2 (Governing Physics, Conservation of Energy) and §4.5.1 equations:
    Eq. 9   m_f c_f · dT_f/dt = γ_f · P − (T_f − T_c) / R_fc
    Eq. 10  m_c c_c · dT_c/dt = (T_f − T_c) / R_fc − ṁ c_c (T_c − T_in)

State vector layout (2 states):
    y[0] = T_f   Lumped fuel temperature   (K)
    y[1] = T_c   Lumped coolant temperature (K)

Coupling injection points (received from point kinetics engine at each ODE step):
    power_w   — total fission power in Watts; P = n · P_nom (n is normalised population)
    m_dot     — primary coolant mass flow rate (kg/s); supports LOF scenarios
    T_in      — coolant inlet temperature (K)

Physical constants are derived from the nominal steady-state operating point (SPEC §4.5.4)
so that the module produces T_f = T_FUEL_NOM, T_c = T_COOL_NOM at P = P_NOM, ṁ = M_DOT_NOM.
This guarantees mathematical consistency between the steady-state formula and the ODE.

Architecture rules (CLAUDE.md):
    - Pure physics: no Dash, UI, or orchestrator imports.
    - solve_ivp called with method='Radau', rtol=1e-6, atol=1e-9.
    - @njit on the hot inner kernel (thermal_hydraulics_rhs).
    - Unphysical inputs (T < 0, ṁ ≤ 0) log a warning and return NaN-filled array
      so the calling engine detects the failure state (SR-01).
"""

from __future__ import annotations

import logging
import math
from typing import Callable

import numba
import numpy as np
from scipy.integrate import solve_ivp

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Nominal steady-state operating point (SPEC §4.5.4)
# ---------------------------------------------------------------------------

# 3000 MWth — representative of a large generic commercial PWR.
# Pydantic upper bound in parameters.py is 4000 MWth; this is the midpoint reference.
P_NOM: float = 3.0e9          # W — nominal fission power

# Fraction of fission power deposited directly in the fuel (remainder in coolant/structure).
# ~97% is standard for UO2 ceramic fuel in a light-water reactor (Glasstone & Sesonske).
GAMMA_F: float = 0.97         # dimensionless

# Reference steady-state temperatures at P_NOM, consistent with parameters.py defaults.
T_FUEL_NOM: float = 900.0    # K — nominal lumped fuel temperature
T_COOL_NOM: float = 590.0    # K — nominal lumped coolant temperature
T_IN_NOM: float = 565.0      # K — nominal coolant inlet temperature (parameters.py default)

# ---------------------------------------------------------------------------
# Material properties — source: standard nuclear engineering references
# ---------------------------------------------------------------------------

# UO2 ceramic fuel specific heat at ~900 K (Fink 2000, J. Nucl. Mater. 279, 1–18).
# Treated as constant within the operating temperature range modeled.
C_FUEL: float = 300.0        # J/(kg·K)

# Lumped fuel thermal mass: ~192 fuel assemblies × ~530 kg UO2/assembly (generic 193-FA core).
# Sets the fuel thermal time constant: τ_f = M_FUEL·C_FUEL·R_FC ≈ 3.2 s.
M_FUEL: float = 101_000.0    # kg

# Light water specific heat at ~590 K, 155 bar (IAPWS-IF97 subcooled liquid).
# 5600 J/(kg·K) is representative of PWR primary coolant at hot-full-power conditions.
C_COOL: float = 5600.0       # J/(kg·K)

# Coolant mass in the active core region: estimated from core volume (~30 m³) at
# water density ≈ 700 kg/m³ at 590 K, 155 bar → ≈ 20 000 kg.
# Sets coolant advection time constant: τ_c ≈ M_COOL / M_DOT_NOM ≈ 1 s.
M_COOL: float = 20_000.0     # kg

# ---------------------------------------------------------------------------
# Derived parameters — computed analytically from nominal steady-state conditions
# so that Eqs. 9 & 10 at dT/dt = 0 reproduce T_FUEL_NOM and T_COOL_NOM exactly.
# ---------------------------------------------------------------------------

# From Eq. 9 at steady state (dT_f/dt = 0):
#   γ_f · P = (T_f − T_c) / R_fc  →  R_fc = ΔT_fc / (γ_f · P_NOM)
# Encodes the fuel-to-coolant thermal resistance through the pellet gap and cladding.
R_FC: float = (T_FUEL_NOM - T_COOL_NOM) / (GAMMA_F * P_NOM)

# From Eq. 10 at steady state (dT_c/dt = 0):
#   (T_f − T_c) / R_fc = ṁ · c_c · (T_c − T_in)  →  ṁ_nom = γ_f·P / (c_c · ΔT_rise)
# Nominal primary mass flow rate: ~20 800 kg/s, consistent with large commercial PWR
# primary circuits (~15 000–25 000 kg/s depending on plant design and number of loops).
M_DOT_NOM: float = (GAMMA_F * P_NOM) / (C_COOL * (T_COOL_NOM - T_IN_NOM))


# ---------------------------------------------------------------------------
# ODE right-hand side — Numba JIT kernel
# ---------------------------------------------------------------------------


@numba.njit(cache=True)
def thermal_hydraulics_rhs(
    t: float,
    y: np.ndarray,
    power_w: float,
    m_dot: float,
    T_in: float,
) -> np.ndarray:
    """Numba JIT kernel: 2-state fuel/coolant thermal-hydraulic ODE right-hand side.

    SPEC Eq. 9:   m_f c_f · dT_f/dt = γ_f · P − (T_f − T_c) / R_fc
    SPEC Eq. 10:  m_c c_c · dT_c/dt = (T_f − T_c) / R_fc − ṁ c_c (T_c − T_in)

    The fuel-to-coolant heat flux q_fc = (T_f − T_c) / R_fc drives both equations:
    it is the heat removed from the fuel (negative in Eq. 9) and the heat deposited
    into the coolant (positive in Eq. 10). At steady state both derivatives are zero
    and the heat balance closes exactly.

    Args:
        t: Current time (s) — required by scipy ODE driver signature.
        y: State vector [T_f, T_c] (K), shape (2,), float64.
        power_w: Total fission power delivered to the fuel node (W). Proportional to
                 the neutron population n from the kinetics engine: P = n · P_NOM.
        m_dot: Primary coolant mass flow rate (kg/s). Dynamic input; caller reduces
               this to model a loss-of-flow transient.
        T_in: Coolant inlet temperature (K). Dynamic input; constant in most scenarios.

    Returns:
        [dT_f/dt, dT_c/dt] (K/s), shape (2,), float64.
    """
    T_f = y[0]
    T_c = y[1]

    # Conductive heat flux from fuel to coolant [W] — SPEC Eq. 9 heat-transfer term.
    q_fc = (T_f - T_c) / R_FC

    dydt = np.empty(2)

    # Eq. 9 — fuel temperature
    dydt[0] = (GAMMA_F * power_w - q_fc) / (M_FUEL * C_FUEL)

    # Eq. 10 — coolant temperature (advection removes ṁ c_c ΔT from the node)
    dydt[1] = (q_fc - m_dot * C_COOL * (T_c - T_in)) / (M_COOL * C_COOL)

    return dydt


# ---------------------------------------------------------------------------
# Steady-state temperature calculator
# ---------------------------------------------------------------------------


def steady_state_temperatures(
    power_w: float,
    m_dot: float,
    T_in: float,
) -> np.ndarray:
    """Return equilibrium [T_f, T_c] for given operating conditions.

    Derived from SPEC Eqs. 9–10 by setting dT_f/dt = dT_c/dt = 0:
        T_c,ss = T_in + γ_f · P / (ṁ · c_c)
        T_f,ss = T_c,ss + γ_f · P · R_fc

    At P = 0 both temperatures equal T_in (no heat source).
    At P = P_NOM, ṁ = M_DOT_NOM, T_in = T_IN_NOM the result is
    [T_FUEL_NOM, T_COOL_NOM] exactly (by construction of R_FC and M_DOT_NOM).

    Args:
        power_w: Total fission power (W). Non-negative.
        m_dot: Primary coolant mass flow rate (kg/s). Must be > 0; zero flow is a
               Loss-Of-Coolant condition that is out of scope (SPEC §3.3).
        T_in: Coolant inlet temperature (K). Must be ≥ 0.

    Returns:
        [T_f_ss, T_c_ss] (K), shape (2,), float64.
        Returns NaN-filled array and logs a warning for unphysical inputs.
    """
    if m_dot <= 0.0:
        logger.warning(
            "steady_state_temperatures: m_dot = %.4f kg/s ≤ 0; "
            "zero/negative flow is out of scope (SPEC §3.3). Returning NaN.",
            m_dot,
        )
        return np.full(2, math.nan)

    if power_w < 0.0:
        logger.warning(
            "steady_state_temperatures: power_w = %.4e W < 0 (unphysical). Returning NaN.",
            power_w,
        )
        return np.full(2, math.nan)

    if T_in < 0.0:
        logger.warning(
            "steady_state_temperatures: T_in = %.3f K < 0 K (unphysical). Returning NaN.",
            T_in,
        )
        return np.full(2, math.nan)

    # From Eq. 10 SS: γ_f·P = ṁ·c_c·(T_c − T_in)
    T_c_ss = T_in + GAMMA_F * power_w / (m_dot * C_COOL)

    # From Eq. 9 SS: γ_f·P = (T_f − T_c) / R_fc
    T_f_ss = T_c_ss + GAMMA_F * power_w * R_FC

    return np.array([T_f_ss, T_c_ss], dtype=np.float64)


# ---------------------------------------------------------------------------
# Standalone integrator
# ---------------------------------------------------------------------------


def integrate_thermal_hydraulics(
    y0: np.ndarray,
    t_span: tuple[float, float],
    power_fn: Callable[[float, np.ndarray], float],
    m_dot_fn: Callable[[float, np.ndarray], float],
    T_in_fn: Callable[[float, np.ndarray], float],
    t_eval: np.ndarray | None = None,
):
    """Integrate the 2-state fuel/coolant thermal-hydraulic ODE over t_span.

    Uses scipy Radau (stiff implicit Runge-Kutta) with tolerances fixed per
    CLAUDE.md §Architecture Rules (rtol=1e-6, atol=1e-9).

    Suitable for standalone thermal transient analysis or as the coupled
    thermal sub-integrator in the full 11-state PWR ODE system. In the full
    11-state system the power_fn callback extracts n from the joint state vector
    and scales by P_NOM; m_dot_fn applies the coolant_flow_fraction from the UI.

    Args:
        y0: Initial state [T_f₀, T_c₀] (K), shape (2,), float64.
        t_span: (t_start, t_end) in seconds.
        power_fn: Callable(t, y) → power (W). Evaluated each solver step.
                  For constant power: ``lambda t, y: power_value``
        m_dot_fn: Callable(t, y) → mass flow rate (kg/s). Constant nominal:
                  ``lambda t, y: M_DOT_NOM``
        T_in_fn: Callable(t, y) → inlet temperature (K). Constant nominal:
                  ``lambda t, y: T_IN_NOM``
        t_eval: Optional output time array (s). None lets the solver choose.

    Returns:
        scipy OdeResult with .t (shape n,), .y (shape 2×n), .success, .message.
    """
    def _rhs(t: float, y: np.ndarray) -> np.ndarray:
        return thermal_hydraulics_rhs(
            t, y,
            power_fn(t, y),
            m_dot_fn(t, y),
            T_in_fn(t, y),
        )

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
        logger.error("Thermal-hydraulic ODE integration failed: %s", result.message)

    return result
