"""Adiabatic compression heating calculator for the Helion FRC twin.

SPEC Eq. 19 — Adiabatic compression of a monatomic ideal plasma (γ = 5/3):

    T_final = T_initial · Rᶜ^{2/3}     (temperature scales as volume^{−(γ−1)})
    n_final = n_initial · Rᶜ             (number density scales as 1/volume)

where Rᶜ = V_initial / V_final is the compression ratio.

Post-compression plasma thermal energy W₀ is derived from T_final, n_final, V_final
using the ideal-plasma energy relation. For D-He3 50/50 by ion number density:
    n_e = (3/2) · n_ions       (quasineutrality: Z_D=1 gives n/2 e⁻, Z_He=2 gives n e⁻)
    n_total = n_ions + n_e = (5/2) · n_ions
    W = (3/2) · n_total · (k_B T) · V = (15/4) · n_ions · (T_keV × J_per_keV) · V

The helper `plasma_temperature_kev` inverts this relation: W → T (used at every ODE step).

Architecture: pure physics — no Dash, UI, or orchestrator imports.
"""

from __future__ import annotations

# Joules per keV — SI conversion for plasma energy calculations
_J_PER_KEV: float = 1.60218e-16

# For D-He3 50/50: n_total_particles / n_ions = (n_ions + n_e) / n_ions = 5/2
# Derived from quasineutrality with Z_D=1, Z_He3=2 at equal ion densities.
_DHE3_TOTAL_PARTICLE_MULT: float = 2.5


def apply_adiabatic_compression(
    T_initial_kev: float,
    n_initial_m3: float,
    compression_ratio: float,
    volume_initial_m3: float,
) -> tuple[float, float, float, float]:
    """Apply adiabatic compression and return post-compression plasma state.

    Implements SPEC Eq. 19. Plasma is assumed to compress faster than the energy
    loss timescale so the process is isentropic (adiabatic), with γ = 5/3.

    Args:
        T_initial_kev:    Pre-compression ion temperature (keV).
        n_initial_m3:     Pre-compression total ion number density (m⁻³).
        compression_ratio: Rᶜ = V_initial / V_final ≥ 1.
        volume_initial_m3: Pre-compression plasma volume (m³).

    Returns:
        (T_final_kev, n_final_m3, V_final_m3, W_final_J):
            Post-compression temperature (keV), ion density (m⁻³),
            plasma volume (m³), and total thermal energy (J).
    """
    T_final = T_initial_kev * compression_ratio ** (2.0 / 3.0)  # SPEC Eq. 19
    n_final = n_initial_m3 * compression_ratio                   # SPEC Eq. 19
    V_final = volume_initial_m3 / compression_ratio

    W_final = _plasma_energy_j(T_final, n_final, V_final)
    return T_final, n_final, V_final, W_final


def plasma_temperature_kev(W_J: float, n_ions_m3: float, V_m3: float) -> float:
    """Derive plasma temperature (keV) from total thermal energy W.

    Inverse of the D-He3 ideal-plasma energy relation:
        W = (3/2) · n_total · (k_B T) · V  with n_total = (5/2) · n_ions
    → T_keV = (4/15) · W / (n_ions · V · J_per_keV)

    Args:
        W_J:         Total plasma thermal energy (J). Must be > 0.
        n_ions_m3:   Total ion number density (m⁻³).
        V_m3:        Plasma volume (m³).

    Returns:
        Ion temperature T (keV). Returns 0.0 for W ≤ 0 (floored state).
    """
    if W_J <= 0.0:
        return 0.0
    n_total = _DHE3_TOTAL_PARTICLE_MULT * n_ions_m3
    return (2.0 / 3.0) * W_J / (n_total * V_m3 * _J_PER_KEV)


def _plasma_energy_j(T_kev: float, n_ions_m3: float, V_m3: float) -> float:
    """Total plasma thermal energy W (J) from temperature and density."""
    n_total = _DHE3_TOTAL_PARTICLE_MULT * n_ions_m3
    return 1.5 * n_total * (T_kev * _J_PER_KEV) * V_m3
