"""
physics/pwr/feedback.py — Algebraic reactivity feedback calculators for the PWR twin.

Implements SPEC §4.2 (Governing Physics decomposition) and §4.5.1 equations:
    Eq. 11  ρ_D = α_D · (T_f − T_f,0)    Doppler fuel-temperature feedback
    Eq. 12  ρ_m = α_m · (T_c − T_c,0)    Moderator temperature feedback
    Eq. 13  ρ_B = ω_B · C_B              Boron reactivity worth

All three calculators are pure algebraic: scalar inputs → scalar Δρ in pcm.
Called by the PWR engine at every solver timestep; outputs sum into ρ_total.

Reactivity balance (ARCHITECTURE.md §4.1):
    ρ_total = ρ_rod + ρ_D(T_f) + ρ_m(T_c) + ρ_B(C_B) + ρ_Xe(X)

Unit conventions (SPEC §4.6.4):
    - Temperature inputs in Kelvin (K). Since feedback uses ΔT, pcm/K ≡ pcm/°C.
    - Reactivity outputs in pcm (= Δk/k × 10⁵).
    - Boron concentration in ppm (mass, parts per million).
    - Convert to dimensionless (÷ 10⁵) before passing to the kinetics ODE solver.

Architecture rules (CLAUDE.md):
    - Pure physics only: no Dash, UI, or orchestrator imports.
    - Coefficient defaults match CLAUDE.md §Physics Constants midpoints.
    - Each default has a comment with physical justification and source range.
    - Unphysical inputs (T < 0 K, boron_ppm < 0) log a warning and return NaN
      so the caller (engine) can detect and handle the failure state (SPEC SR-01).
"""

from __future__ import annotations

import logging
import math

logger = logging.getLogger(__name__)

# Default Doppler temperature coefficient for U-235 thermal reactor (SPEC §4.5.4).
# Physical basis: rising T_f broadens U-238 neutron absorption resonances, increasing
# parasitic capture and reducing k_eff. Sign is always negative for LWRs operating
# in the thermal spectrum. Range per CLAUDE.md: −2 to −3 pcm/K; midpoint used.
ALPHA_D_DEFAULT: float = -2.5  # pcm/K

# Default moderator temperature coefficient for a generic commercial PWR.
# Physical basis: hotter, less dense H₂O moderates neutrons less effectively,
# shifting the spectrum to higher energies and reducing thermal fission probability.
# Sign must be negative (negative void coefficient requirement). Range per CLAUDE.md:
# −20 to −50 pcm/K; midpoint of −35 used for generic large PWR at full power.
ALPHA_M_DEFAULT: float = -35.0  # pcm/K

# Default differential boron worth for H₃BO₃ dissolved in PWR primary coolant.
# Physical basis: ¹⁰B absorbs thermal neutrons (σ_a ≈ 3840 barns); worth is nearly
# linear in concentration over the normal operating range. Value per CLAUDE.md: ~−10 pcm/ppm.
OMEGA_B_DEFAULT: float = -10.0  # pcm/ppm


def doppler_feedback(
    T_fuel_k: float,
    T_fuel_ref_k: float,
    alpha_D: float = ALPHA_D_DEFAULT,
) -> float:
    """Compute Doppler (fuel temperature) reactivity feedback in pcm.

    Implements SPEC Eq. 11:  ρ_D = α_D · (T_f − T_f,0)

    Rising fuel temperature broadens U-238 resonance absorption peaks (Doppler
    broadening), reducing k_eff. With α_D < 0, any power excursion that raises
    T_f produces negative ρ_D — the primary prompt inherent safety mechanism in
    all light-water reactors.

    Inputs are fed dynamically by the thermal-hydraulics module at each solver step.

    Args:
        T_fuel_k: Current lumped fuel temperature (K).
        T_fuel_ref_k: Reference (initial steady-state) fuel temperature (K).
        alpha_D: Doppler temperature coefficient (pcm/K). Must be negative for
                 physical self-regulation. Default: ALPHA_D_DEFAULT.

    Returns:
        Doppler reactivity contribution ρ_D (pcm). Negative when T_f > T_f,0.
        Returns NaN and logs a warning for unphysical inputs (T < 0 K).
    """
    if T_fuel_k < 0.0:
        logger.warning(
            "doppler_feedback: unphysical fuel temperature %.3f K (< 0 K); returning NaN.",
            T_fuel_k,
        )
        return math.nan
    if T_fuel_ref_k < 0.0:
        logger.warning(
            "doppler_feedback: unphysical reference fuel temperature %.3f K (< 0 K); returning NaN.",
            T_fuel_ref_k,
        )
        return math.nan

    return alpha_D * (T_fuel_k - T_fuel_ref_k)


def moderator_feedback(
    T_cool_k: float,
    T_cool_ref_k: float,
    alpha_m: float = ALPHA_M_DEFAULT,
) -> float:
    """Compute moderator temperature reactivity feedback in pcm.

    Implements SPEC Eq. 12:  ρ_m = α_m · (T_c − T_c,0)

    Hotter coolant is less dense, moderates neutrons less effectively, and shifts
    the neutron energy spectrum upward, reducing thermal fission probability.
    With α_m < 0 this provides a second independent inherent safety mechanism
    that acts on the coolant rather than the fuel, with a slower thermal time
    constant than the Doppler effect.

    Inputs are fed dynamically by the thermal-hydraulics module at each solver step.

    Args:
        T_cool_k: Current lumped coolant (moderator) temperature (K).
        T_cool_ref_k: Reference (initial steady-state) coolant temperature (K).
        alpha_m: Moderator temperature coefficient (pcm/K). Must be negative in all
                 LWR designs (negative void coefficient requirement). Default: ALPHA_M_DEFAULT.

    Returns:
        Moderator reactivity contribution ρ_m (pcm). Negative when T_c > T_c,0.
        Returns NaN and logs a warning for unphysical inputs (T < 0 K).
    """
    if T_cool_k < 0.0:
        logger.warning(
            "moderator_feedback: unphysical coolant temperature %.3f K (< 0 K); returning NaN.",
            T_cool_k,
        )
        return math.nan
    if T_cool_ref_k < 0.0:
        logger.warning(
            "moderator_feedback: unphysical reference coolant temperature %.3f K (< 0 K); returning NaN.",
            T_cool_ref_k,
        )
        return math.nan

    return alpha_m * (T_cool_k - T_cool_ref_k)


def boron_worth(
    boron_ppm: float,
    omega_B: float = OMEGA_B_DEFAULT,
) -> float:
    """Compute dissolved boron reactivity worth in pcm.

    Implements SPEC Eq. 13:  ρ_B = ω_B · C_B

    ¹⁰B dissolved as H₃BO₃ absorbs thermal neutrons proportional to concentration.
    This is the primary mechanism for long-term core reactivity management: startup
    excess reactivity is held down by boron, which is diluted (reduced) as the
    fuel burns up over the fuel cycle. Unlike the temperature feedbacks, this is
    an absolute contribution, not a delta from a reference.

    Args:
        boron_ppm: Dissolved boron concentration (ppm, mass). Non-negative.
        omega_B: Differential boron worth (pcm/ppm). Always negative (absorption
                 is always a neutron poison). Default: OMEGA_B_DEFAULT.

    Returns:
        Boron reactivity contribution ρ_B (pcm). Negative for positive boron_ppm.
        Returns NaN and logs a warning for unphysical inputs (boron_ppm < 0).
    """
    if boron_ppm < 0.0:
        logger.warning(
            "boron_worth: unphysical boron concentration %.3f ppm (< 0); returning NaN.",
            boron_ppm,
        )
        return math.nan

    return omega_B * boron_ppm


def reactivity_balance(
    rho_rod_pcm: float,
    rho_doppler_pcm: float,
    rho_moderator_pcm: float,
    rho_boron_pcm: float,
    rho_xenon_pcm: float,
) -> float:
    """Sum all PWR reactivity components into total reactivity (pcm).

    Implements the reactivity balance from ARCHITECTURE.md §4.1:
        ρ_total = ρ_rod + ρ_D + ρ_m + ρ_B + ρ_Xe

    The result is passed (converted to dimensionless via pcm_to_dk_k) to the
    point kinetics ODE as the ρ argument in SPEC Eq. 1.

    Args:
        rho_rod_pcm: Control rod reactivity insertion (pcm). Positive = withdrawal.
        rho_doppler_pcm: Doppler feedback contribution (pcm). Typically negative at power.
        rho_moderator_pcm: Moderator temperature feedback (pcm). Typically negative at power.
        rho_boron_pcm: Dissolved boron worth (pcm). Negative.
        rho_xenon_pcm: Xenon-135 reactivity worth (pcm). Negative at operating power.

    Returns:
        Total reactivity ρ_total (pcm).
    """
    return (
        rho_rod_pcm
        + rho_doppler_pcm
        + rho_moderator_pcm
        + rho_boron_pcm
        + rho_xenon_pcm
    )


def pcm_to_dk_k(rho_pcm: float) -> float:
    """Convert reactivity from pcm to dimensionless Δk/k.

    1 pcm = 10⁻⁵ Δk/k (SPEC §4.6.4, Unit Consistency table).
    Used at the physics engine boundary before passing ρ to the kinetics solver.

    Args:
        rho_pcm: Reactivity in pcm.

    Returns:
        Reactivity as dimensionless Δk/k.
    """
    return rho_pcm * 1.0e-5
