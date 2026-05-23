"""Bremsstrahlung radiation and thermal confinement loss calculators.

SPEC equations implemented:
    Eq. 20 — P_Brem = C_B · Z_eff · n_e² · T^{1/2} · V
    Eq. 21 — P_cond = W / τ_E

D-He3 50/50 plasma constants (derived from quasineutrality):
    n_e   = (3/2) · n_ions         (1 e⁻ per D⁺, 2 e⁻ per He3²⁺ at equal densities)
    Z_eff = Σ(nᵢ Zᵢ²) / n_e
          = ((n/2)·1² + (n/2)·2²) / (3n/2)
          = (n/2 + 2n) / (3n/2) = 5/3

Confinement time τ_E (SPEC §4.3, Eq. 21):
    τ_E is the dominant model uncertainty for the Helion twin (ARCHITECTURE.md R-02).
    This module uses a parametric FRC scaling law:
        τ_E = C_TAU × (n / N_REF) × (B / B_REF)²
    consistent with observed density and field trends in Binderbauer et al. (2015),
    Physics of Plasmas 22, 056110. Helion's actual performance is proprietary.

    Per SPEC §7.3, results should be reported across the TAU_E_UNCERTAINTY_LOWER/
    TAU_E_UNCERTAINTY_UPPER multiplier band (Risk R-02).

Architecture: pure physics — no Dash, UI, or orchestrator imports.
"""

from __future__ import annotations

import numba
import numpy as np

# Bremsstrahlung coefficient (SPEC §4.5.4, CLAUDE.md constants)
C_BREM: float = 5.35e-37  # W · m³ / keV^{1/2}

# D-He3 50/50 quasineutrality derived constants
_DHE3_NE_MULT: float = 1.5     # n_e = 1.5 · n_ions
_DHE3_Z_EFF: float = 5.0 / 3.0  # Z_eff for perfect 50/50 D-He3

# Empirical FRC τ_E parametric scaling (calibrated at nominal Helion operating point)
# τ_E = C_TAU × (n / N_REF) × (B / B_REF)²  → ~50 μs at n=1e21 m⁻³, B=5 T
C_TAU: float = 5.0e-5     # s — scaling prefactor
_N_REF: float = 1.0e21    # m⁻³ — reference density
_B_REF: float = 5.0       # T — reference magnetic field

# τ_E uncertainty band (SPEC R-02): multiply nominal τ_E by these factors for range
TAU_E_UNCERTAINTY_LOWER: float = 0.3
TAU_E_UNCERTAINTY_UPPER: float = 3.0


@numba.njit(cache=True)
def bremsstrahlung_power(T_kev: float, n_ions_m3: float, V_m3: float) -> float:
    """Bremsstrahlung radiation loss power (W). SPEC Eq. 20.

    P_Brem = C_B · Z_eff · n_e² · T^{1/2} · V
    For D-He3 50/50: n_e = 1.5 n_ions, Z_eff = 5/3.

    Args:
        T_kev:       Plasma temperature (keV). Should be > 0.
        n_ions_m3:   Total ion number density (m⁻³).
        V_m3:        Plasma volume (m³).

    Returns:
        P_Brem (W), non-negative.
    """
    n_e = 1.5 * n_ions_m3
    return 5.35e-37 * (5.0 / 3.0) * n_e * n_e * np.sqrt(T_kev) * V_m3


def confinement_time(n_ions_m3: float, B_T: float) -> float:
    """Empirical FRC energy confinement time τ_E (s).

    Parametric scaling: τ_E = C_TAU × (n / N_REF) × (B / B_REF)²
    Calibrated to give τ_E ≈ 50 μs at n=1e21 m⁻³, B=5 T.

    Helion's actual τ_E performance is proprietary. Report results across the
    TAU_E_UNCERTAINTY_LOWER–TAU_E_UNCERTAINTY_UPPER multiplier band (Risk R-02).

    Args:
        n_ions_m3: Total ion number density (m⁻³). Must be > 0.
        B_T:       Magnetic field (T). Must be > 0.

    Returns:
        τ_E (s).
    """
    return C_TAU * (n_ions_m3 / _N_REF) * (B_T / _B_REF) ** 2


def conduction_loss(W_J: float, n_ions_m3: float, B_T: float) -> float:
    """Thermal conduction loss through the magnetic field (W). SPEC Eq. 21.

    P_cond = W / τ_E

    Args:
        W_J:         Total plasma thermal energy (J).
        n_ions_m3:   Total ion number density (m⁻³).
        B_T:         Magnetic field (T).

    Returns:
        P_cond (W).
    """
    tau_e = confinement_time(n_ions_m3, B_T)
    return W_J / tau_e
