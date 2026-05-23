"""
physics/msr/thermal.py — Unified salt thermal-hydraulic module for the MSR twin.

In a Molten Salt Reactor the nuclear fuel is dissolved directly in the salt that
also serves as the primary coolant. There is no separate fuel pellet and no
fuel-to-coolant thermal resistance (unlike the PWR's Eq. 9–10 two-node model).
All fission energy deposits directly into the flowing salt, so the thermal model
reduces to a single node.

SPEC reference (§4.4 MSR Decomposition):
    Conservation of energy in flowing salt (1-state, fuel-coolant unified):
        M_CORE · c_salt · dT_salt/dt = P − ṁ_salt · c_salt · (T_salt − T_inlet)

    where:
      - M_CORE  = ρ_salt · V_core   salt mass in the active core (kg)
      - P       = n · P_ref         fission power (W); n is normalised neutron population
      - ṁ_salt                      primary salt mass flow rate (kg/s)
      - T_inlet                     salt temperature returning from the heat exchanger (K)

State vector layout (1 state):
    y[0] = T_salt   lumped salt temperature (K)

Nominal operating-point constants are derived analytically so that at P = P_NOM_MSR,
ṁ_salt = M_DOT_SALT_NOM, the steady-state solution T_salt = T_SALT_NOM exactly.
This guarantees mathematical consistency between the SS formula and the ODE
(same derivation strategy used for the PWR thermal module).

Architecture rules (CLAUDE.md):
    - Pure physics: no Dash, UI, or orchestrator imports.
    - solve_ivp called with method='Radau', rtol=1e-6, atol=1e-9.
    - @njit on the hot inner kernel (salt_thermal_rhs).
"""

from __future__ import annotations

import logging
import math

import numba
import numpy as np
from scipy.integrate import solve_ivp

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Nominal operating-point constants
# Physical justification for each value is documented inline.
# ---------------------------------------------------------------------------

# 500 MWth — representative mid-scale commercial-class thermal MSR.
# Covers the default MSRParameters.initial_power_mw = 500 value.
P_NOM_MSR: float = 500.0e6          # W — nominal fission power

# Nominal salt temperature at full-power steady state (SPEC §4.5.4, default from MSRParameters).
# Fluoride salts (FLiBe, FLiNaK) typically operate in the 600–750 °C range ≈ 873–1023 K.
T_SALT_NOM: float = 900.0           # K — nominal bulk salt temperature

# Salt inlet temperature (returning from primary heat exchanger at nominal flow).
# 40 K sub-cooling below T_SALT_NOM is representative of MSRE-class designs
# (Haubenreich & Engel 1970, §3 thermal data).
T_SALT_INLET_NOM: float = 860.0     # K — nominal salt inlet temperature

# FLiBe (Li₂BeF₄) specific heat at ~900 K.
# Source: Williams et al. (2006) "Assessment of Candidate Molten Salt Coolants",
# ORNL/TM-2006/12; value ≈ 1500 J/(kg·K) at 900 K within 5%.
C_SALT: float = 1500.0              # J/(kg·K)

# FLiBe density at ~900 K (from ORNL/TM-2006/12 table).
RHO_SALT: float = 2000.0            # kg/m³

# Core transit time at nominal salt velocity (matches MSRParameters default).
TAU_CORE_NOM: float = 5.0           # s — τ_core at v_salt = 1 m/s

# Nominal salt velocity derived from v_salt_nom × τ_core_nom = L_core.
V_SALT_NOM: float = 1.0             # m/s (MSRParameters default)

# Active core length = v_salt_nom × τ_core_nom.
L_CORE: float = V_SALT_NOM * TAU_CORE_NOM   # = 5.0 m

# ---------------------------------------------------------------------------
# Derived parameters — computed from nominal operating point so that
# the ODE at dT/dt = 0 reproduces T_SALT_NOM exactly (SPEC §4.6.2).
# ---------------------------------------------------------------------------

# From SS: P_NOM = ṁ_nom · c_salt · (T_SALT_NOM − T_SALT_INLET_NOM)
M_DOT_SALT_NOM: float = P_NOM_MSR / (C_SALT * (T_SALT_NOM - T_SALT_INLET_NOM))
# = 500 × 10⁶ / (1500 × 40) ≈ 8333 kg/s — consistent with large-scale MSR primary circuit.

# Core cross-section area from continuity: ṁ_nom = ρ · A · v_nom
A_CORE: float = M_DOT_SALT_NOM / (RHO_SALT * V_SALT_NOM)
# ≈ 4.17 m² — plausible for a 500 MWth cylindrical core.

# Total salt mass in the active core region.
# τ_thermal = M_CORE / ṁ_nom = (ρ · L · A) / (ρ · A · v) = L / v = τ_core_nom ✓
M_CORE_SALT: float = RHO_SALT * L_CORE * A_CORE
# ≈ 41 667 kg — by construction τ_thermal = τ_core_nom = 5 s


# ---------------------------------------------------------------------------
# ODE right-hand side — Numba JIT kernel
# ---------------------------------------------------------------------------


@numba.njit(cache=True)
def salt_thermal_rhs(
    t: float,
    y: np.ndarray,
    power_w: float,
    m_dot_salt: float,
    T_salt_inlet: float,
) -> np.ndarray:
    """Numba JIT kernel: 1-state unified salt temperature ODE.

    SPEC §4.4 (unified fuel-coolant energy balance):
        M_CORE · c_salt · dT_salt/dt = P − ṁ_salt · c_salt · (T_salt − T_inlet)

    All fission energy deposits directly into the flowing salt (γ_salt = 1.0,
    unlike the PWR where ~3% heats the coolant/structure directly).

    Args:
        t: Current time (s) — required by scipy ODE driver signature.
        y: State vector [T_salt] (K), shape (1,), float64.
        power_w: Total fission power (W). Dynamic input from the kinetics engine.
        m_dot_salt: Salt mass flow rate (kg/s). Dynamic input; may scale with v_salt.
        T_salt_inlet: Salt inlet temperature (K). Dynamic input from heat exchanger model.

    Returns:
        [dT_salt/dt] (K/s), shape (1,), float64.
    """
    T_salt = y[0]
    heat_in = power_w                                  # all fission power → salt directly
    heat_out = m_dot_salt * C_SALT * (T_salt - T_salt_inlet)  # advective removal

    dydt = np.empty(1)
    dydt[0] = (heat_in - heat_out) / (M_CORE_SALT * C_SALT)
    return dydt


# ---------------------------------------------------------------------------
# Steady-state salt temperature calculator
# ---------------------------------------------------------------------------


def steady_state_salt_temperature(
    power_w: float,
    m_dot_salt: float,
    T_inlet: float,
) -> float:
    """Return equilibrium T_salt for given operating conditions.

    Derived from the energy ODE with dT_salt/dt = 0:
        T_salt,ss = T_inlet + P / (ṁ_salt · c_salt)

    At P = 0, T_salt = T_inlet (no heat source).
    At P = P_NOM_MSR, ṁ = M_DOT_SALT_NOM, T_inlet = T_SALT_INLET_NOM:
        T_salt,ss = T_SALT_NOM  (by construction of M_DOT_SALT_NOM).

    Args:
        power_w:    Total fission power (W). Non-negative.
        m_dot_salt: Salt mass flow rate (kg/s). Must be > 0 for heat removal.
        T_inlet:    Salt inlet temperature (K). Non-negative.

    Returns:
        Steady-state salt temperature (K).
        Returns NaN and logs a warning for unphysical inputs (SR-01).
    """
    if m_dot_salt <= 0.0:
        logger.warning(
            "steady_state_salt_temperature: m_dot_salt = %.4f kg/s ≤ 0; "
            "zero/negative flow is out of scope (SPEC §3.3). Returning NaN.",
            m_dot_salt,
        )
        return math.nan

    if power_w < 0.0:
        logger.warning(
            "steady_state_salt_temperature: power_w = %.4e W < 0 (unphysical). Returning NaN.",
            power_w,
        )
        return math.nan

    if T_inlet < 0.0:
        logger.warning(
            "steady_state_salt_temperature: T_inlet = %.3f K < 0 K (unphysical). Returning NaN.",
            T_inlet,
        )
        return math.nan

    return T_inlet + power_w / (m_dot_salt * C_SALT)


def m_dot_salt_from_tau_core(tau_core: float) -> float:
    """Return salt mass flow rate for a given core transit time.

    At fixed core geometry (L_CORE, A_CORE) the transit time scales inversely
    with salt velocity, and ṁ_salt scales directly with velocity:

        ṁ_salt = ρ · A · v = ρ · A · L_CORE / τ_core = M_DOT_SALT_NOM · τ_core_nom / τ_core

    This is the key coupling between the DDE parameters (τ_core) and the thermal
    response: faster salt → more heat removal → lower steady-state T_salt.

    Args:
        tau_core: Core transit time τ_core (s). Must be > 0.

    Returns:
        ṁ_salt (kg/s). Returns M_DOT_SALT_NOM for tau_core ≤ 0 (guard against division).
    """
    if tau_core <= 0.0:
        logger.warning(
            "m_dot_salt_from_tau_core: tau_core = %.4f s ≤ 0; returning nominal flow rate.",
            tau_core,
        )
        return M_DOT_SALT_NOM
    return M_DOT_SALT_NOM * TAU_CORE_NOM / tau_core
