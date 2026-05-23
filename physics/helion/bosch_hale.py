"""D-He3 thermonuclear reactivity via the Bosch-Hale parametrization.

SPEC Eq. 18 — implemented per Bosch & Hale (1992), Nuclear Fusion 32(4), eq. (12):

    ⟨σv⟩ = C₁ · θ · √(ξ / (m_r c² T³)) · exp(−3ξ)

    θ = T / [1 − T(C₂ + T(C₄ + TC₆)) / (1 + T(C₃ + T(C₅ + TC₇)))]
    ξ = (B_G² / 4θ)^{1/3}

Note: SPEC §4.5.2 Eq. 18 writes the formula as "C₁ θ² / (ξ B_G²) · …" which is an
algebraically non-equivalent form. This module implements the published B&H Eq. (12)
directly, which has been verified numerically against published ⟨σv⟩ tables.

Valid T range: 0.5–190 keV (R-06). Out-of-range T is clamped with a warning.
Units: T in keV → ⟨σv⟩ in m³/s (B&H original result cm³/s; converted by × 10⁻⁶).

Architecture: pure physics — no Dash, UI, or orchestrator imports.
"""

from __future__ import annotations

import logging

import numba
import numpy as np

from data.bosch_hale_coeffs import DHE3_BOSCH_HALE, DHE3_TEMPERATURE_RANGE_KEV

logger = logging.getLogger(__name__)

T_MIN_KEV: float = DHE3_TEMPERATURE_RANGE_KEV[0]   # 0.5 keV — B&H lower validity bound
T_MAX_KEV: float = DHE3_TEMPERATURE_RANGE_KEV[1]   # 190 keV — B&H upper validity bound


@numba.njit(cache=True)
def sigma_v_kernel(
    T_kev: float,
    bg: float,
    mrc2: float,
    c1: float,
    c2: float,
    c3: float,
    c4: float,
    c5: float,
    c6: float,
    c7: float,
) -> float:
    """Numba JIT kernel: Bosch-Hale ⟨σv⟩ for D-He3.

    Implements B&H (1992) Eq. (12). T must already be clamped to [0.5, 190] keV
    by the caller; no bounds enforcement is performed here.

    Args:
        T_kev:   Plasma temperature (keV).
        bg:      B_G = π α_f Z₁Z₂ √(2 m_r c²) for the reaction pair.
        mrc2:    Reduced mass energy m_r c² (keV).
        c1–c7:  Bosch-Hale fit coefficients (Table IV).

    Returns:
        ⟨σv⟩ in m³/s (float64).
    """
    theta = T_kev / (
        1.0 - T_kev * (c2 + T_kev * (c4 + T_kev * c6))
        / (1.0 + T_kev * (c3 + T_kev * (c5 + T_kev * c7)))
    )
    xi = (bg * bg / (4.0 * theta)) ** (1.0 / 3.0)
    sv_cm3_s = c1 * theta * np.sqrt(xi / (mrc2 * T_kev * T_kev * T_kev)) * np.exp(-3.0 * xi)
    return sv_cm3_s * 1.0e-6  # cm³/s → m³/s


def sigma_v_dhe3(T_kev: float) -> float:
    """D-He3 thermonuclear reactivity ⟨σv⟩(T) in m³/s.

    Validates T against the Bosch-Hale validity range and clamps with a warning
    per SPEC §4.6.5 (R-06). Returns the clamped result; results outside
    [0.5, 190] keV should be treated as extrapolations.

    Args:
        T_kev: Plasma temperature (keV).

    Returns:
        ⟨σv⟩ in m³/s.
    """
    if T_kev < T_MIN_KEV or T_kev > T_MAX_KEV:
        logger.warning(
            "Bosch-Hale T = %.4g keV outside valid range [%.1f, %.1f] keV; "
            "clamping. Results in this interval should be treated as "
            "extrapolations (SPEC R-06).",
            T_kev,
            T_MIN_KEV,
            T_MAX_KEV,
        )
        T_kev = max(T_MIN_KEV, min(T_MAX_KEV, T_kev))

    bh = DHE3_BOSCH_HALE
    return sigma_v_kernel(
        T_kev,
        bh.bg, bh.mrc2,
        bh.c1, bh.c2, bh.c3, bh.c4, bh.c5, bh.c6, bh.c7,
    )


def prewarm_jit() -> None:
    """Pre-warm the Numba JIT cache for sigma_v_kernel.

    Call once at application startup to eliminate first-call compile latency
    (ARCHITECTURE.md §7 Risk R-04, CLAUDE.md §Numba Rules).
    """
    bh = DHE3_BOSCH_HALE
    sigma_v_kernel(
        50.0,
        bh.bg, bh.mrc2,
        bh.c1, bh.c2, bh.c3, bh.c4, bh.c5, bh.c6, bh.c7,
    )
