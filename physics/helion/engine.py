"""Helion FRC plasma energy balance ODE engine — Phase 2.

SPEC equations implemented:
    Eq. 16 — dW/dt = P_heat + P_fusion − P_Brem − P_cond
    Eq. 17 — P_fusion = (n²/4) · ⟨σv⟩(T) · E_DHe3 · V
    Eq. 18 — ⟨σv⟩ via Bosch-Hale (physics/helion/bosch_hale.py)
    Eq. 19 — Adiabatic compression initial conditions (physics/helion/compression.py)
    Eq. 20 — P_Brem (physics/helion/losses.py)
    Eq. 21 — P_cond = W / τ_E (physics/helion/losses.py)

State vector (1 state, ARCHITECTURE.md §4.2):
    y[0] = W   Total plasma thermal energy (J)

All other quantities — T (keV), P_fusion, P_Brem, P_cond — are algebraic outputs
derived from W and the fixed post-compression parameters at each timestep.

Flat parameter array (_P_* constants) packs all physical constants into a
float64 array compatible with scipy's args keyword and @numba.njit.

Architecture:
    - Pure physics: no Dash, UI, or orchestrator imports.
    - Pydantic HelionParameters validated before entry (parameters.py).
    - solve_ivp(method='Radau', rtol=1e-6, atol=1e-9) per CLAUDE.md rule 6.
    - @njit on plasma_energy_rhs_kernel (hot inner loop).
    - Physics worker decoupling (CLAUDE.md Pivot 3) is handled at orchestrator level.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numba
import numpy as np
from scipy.integrate import solve_ivp

from data.bosch_hale_coeffs import DHE3_BOSCH_HALE
from physics.helion.bosch_hale import T_MAX_KEV, T_MIN_KEV, sigma_v_kernel
from physics.helion.compression import apply_adiabatic_compression, plasma_temperature_kev
from physics.helion.losses import C_TAU, _B_REF, _N_REF, bremsstrahlung_power
from physics.helion.parameters import HelionParameters

logger = logging.getLogger(__name__)

# D-He3 reaction energy (SPEC §4.5.4, CLAUDE.md constants): 18.3 MeV in SI
E_DHE3: float = 2.93e-12  # J

# J per keV — SI conversion constant for temperature derivation
_J_PER_KEV: float = 1.60218e-16

# D-He3 total particle multiplier (n_total = 2.5 × n_ions, see compression.py)
_DHE3_PM: float = 2.5

# ---------------------------------------------------------------------------
# Flat parameter array layout — every index is a named constant so the
# @numba.njit kernel never uses magic integers.  All elements are float64.
# ---------------------------------------------------------------------------
_P_N_IONS    =  0  # post-compression ion density n_final (m⁻³)
_P_V         =  1  # post-compression plasma volume V_final (m³)
_P_P_HEAT    =  2  # external heating power P_heat (W), constant during pulse
_P_B         =  3  # magnetic field B (T) — taken as post-compression value
_P_BG        =  4  # Bosch-Hale B_G
_P_MRC2      =  5  # Bosch-Hale m_r c² (keV)
_P_C1        =  6  # Bosch-Hale C1
_P_C2        =  7  # Bosch-Hale C2
_P_C3        =  8  # Bosch-Hale C3
_P_C4        =  9  # Bosch-Hale C4
_P_C5        = 10  # Bosch-Hale C5
_P_C6        = 11  # Bosch-Hale C6
_P_C7        = 12  # Bosch-Hale C7
_P_E_DHE3    = 13  # D-He3 reaction energy (J)
_P_C_BREM    = 14  # Bremsstrahlung coefficient (W·m³/keV^{1/2})
_P_C_TAU     = 15  # τ_E scaling prefactor (s)
_P_N_REF     = 16  # τ_E reference density (m⁻³)
_P_B_REF     = 17  # τ_E reference field (T)
_P_DHE3_PM   = 18  # D-He3 total particle multiplier (5/2)
_P_J_PER_KEV = 19  # J per keV conversion factor
_P_T_MIN     = 20  # Bosch-Hale T lower bound (keV)
_P_T_MAX     = 21  # Bosch-Hale T upper bound (keV)
HELION_PARAMS_LEN: int = 22


@dataclass
class HelionPulseResult:
    """Results from one integrated Helion FRC pulse simulation."""

    t: np.ndarray         # time (s), shape (n_steps,)
    W: np.ndarray         # plasma thermal energy (J), shape (n_steps,)
    T_kev: np.ndarray     # derived plasma temperature (keV), shape (n_steps,)
    sigma_v: np.ndarray   # D-He3 reactivity (m³/s), shape (n_steps,)
    tau_e: np.ndarray     # empirical confinement time (s), shape (n_steps,)
    P_fusion: np.ndarray  # fusion power (W), shape (n_steps,)
    P_brem: np.ndarray    # Bremsstrahlung loss (W), shape (n_steps,)
    P_cond: np.ndarray    # thermal conduction loss (W), shape (n_steps,)
    net_power: np.ndarray  # dW/dt = P_heat + P_fusion − P_brem − P_cond (W)
    p_heat_w: float       # constant external heating during the pulse (W)
    Q: float              # energy gain factor: total fusion energy / W_initial
    W_initial: float      # post-compression plasma energy at t=0 (J)
    success: bool
    message: str


@numba.njit(cache=True)
def plasma_energy_rhs_kernel(
    t: float,
    y: np.ndarray,
    params: np.ndarray,
) -> np.ndarray:
    """Numba JIT kernel: 1-state Helion plasma energy ODE right-hand side.

    Implements SPEC Eq. 16: dW/dt = P_heat + P_fusion − P_Brem − P_cond

    Sub-physics:
        Eq. 17: P_fusion = (n²/4) · ⟨σv⟩(T) · E_DHe3 · V
        Eq. 18: ⟨σv⟩ via sigma_v_kernel (Bosch-Hale)
        Eq. 20: P_Brem = C_B · Z_eff · n_e² · T^{1/2} · V  (via bremsstrahlung_power)
        Eq. 21: P_cond = W / τ_E

    W ≤ 0 is floored to 0 with zero derivative (SPEC §4.6.5 non-negative states).
    T is clamped to Bosch-Hale valid range silently inside the kernel (logging in wrapper).

    Args:
        t:      Current time (s) — required by scipy ODE driver signature.
        y:      State vector [W], shape (1,), float64. W in Joules.
        params: Flat parameter array of shape (HELION_PARAMS_LEN,) built by
                make_params_array().

    Returns:
        dy/dt, shape (1,), float64.
    """
    dydt = np.empty(1)
    W = y[0]

    # Non-negative state floor (SPEC §4.6.5)
    if W <= 0.0:
        dydt[0] = 0.0
        return dydt

    n      = params[_P_N_IONS]
    V      = params[_P_V]
    P_heat = params[_P_P_HEAT]
    B      = params[_P_B]

    # Derive temperature from W (inverse of ideal-plasma energy relation)
    n_total = params[_P_DHE3_PM] * n
    T_kev = (2.0 / 3.0) * W / (n_total * V * params[_P_J_PER_KEV])

    # Clamp T to Bosch-Hale valid range (logging deferred to wrapper — §4.6.5 R-06)
    T_kev_clamped = min(params[_P_T_MAX], max(params[_P_T_MIN], T_kev))

    # P_fusion (Eq. 17): (n²/4) ⟨σv⟩ E_DHe3 V
    sv = sigma_v_kernel(
        T_kev_clamped,
        params[_P_BG], params[_P_MRC2],
        params[_P_C1], params[_P_C2], params[_P_C3],
        params[_P_C4], params[_P_C5], params[_P_C6], params[_P_C7],
    )
    P_fusion = (n * n / 4.0) * sv * params[_P_E_DHE3] * V

    # P_Brem (Eq. 20): bremsstrahlung_power kernel inlined for @njit compatibility
    n_e = 1.5 * n
    P_brem = params[_P_C_BREM] * (5.0 / 3.0) * n_e * n_e * (T_kev_clamped ** 0.5) * V

    # P_cond (Eq. 21): W / τ_E  with τ_E = C_TAU × (n/N_REF) × (B/B_REF)²
    tau_e = params[_P_C_TAU] * (n / params[_P_N_REF]) * (B / params[_P_B_REF]) ** 2
    P_cond = W / tau_e

    # SPEC Eq. 16 — all D-He3 charged products retained in plasma (no neutrons)
    dydt[0] = P_heat + P_fusion - P_brem - P_cond
    return dydt


def make_params_array(
    n_final_m3: float,
    V_final_m3: float,
    P_heat_w: float,
    B_T: float,
    *,
    c_tau_scale: float = 1.0,
) -> np.ndarray:
    """Pack post-compression physics constants into the flat float64 params array.

    Args:
        n_final_m3:  Post-compression ion density (m⁻³).
        V_final_m3:  Post-compression plasma volume (m³).
        P_heat_w:    Constant external heating power (W).
        B_T:         Applied magnetic field — treated as post-compression value (T).

    Returns:
        Float64 array of shape (HELION_PARAMS_LEN,) with layout per _P_* constants.
    """
    bh = DHE3_BOSCH_HALE
    params = np.empty(HELION_PARAMS_LEN, dtype=np.float64)
    params[_P_N_IONS]    = n_final_m3
    params[_P_V]         = V_final_m3
    params[_P_P_HEAT]    = P_heat_w
    params[_P_B]         = B_T
    params[_P_BG]        = bh.bg
    params[_P_MRC2]      = bh.mrc2
    params[_P_C1]        = bh.c1
    params[_P_C2]        = bh.c2
    params[_P_C3]        = bh.c3
    params[_P_C4]        = bh.c4
    params[_P_C5]        = bh.c5
    params[_P_C6]        = bh.c6
    params[_P_C7]        = bh.c7
    params[_P_E_DHE3]    = E_DHE3
    params[_P_C_BREM]    = 5.35e-37
    params[_P_C_TAU]     = C_TAU * c_tau_scale
    params[_P_N_REF]     = _N_REF
    params[_P_B_REF]     = _B_REF
    params[_P_DHE3_PM]   = _DHE3_PM
    params[_P_J_PER_KEV] = _J_PER_KEV
    params[_P_T_MIN]     = T_MIN_KEV
    params[_P_T_MAX]     = T_MAX_KEV
    return params


def _post_process(
    t_arr: np.ndarray,
    W_arr: np.ndarray,
    params: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Compute T, ⟨σv⟩, τ_E, P_fusion, P_brem, P_cond, net power from W history."""
    del t_arr  # uniform output grid; retained for API symmetry with callers
    n    = params[_P_N_IONS]
    V    = params[_P_V]
    B    = params[_P_B]
    P_heat = params[_P_P_HEAT]
    bh   = DHE3_BOSCH_HALE
    c_tau = params[_P_C_TAU]

    T_kev = np.array([plasma_temperature_kev(W, n, V) for W in W_arr])
    T_clamped = np.clip(T_kev, T_MIN_KEV, T_MAX_KEV)

    sigma_v = np.array([
        sigma_v_kernel(
            T_clamped[i],
            bh.bg, bh.mrc2,
            bh.c1, bh.c2, bh.c3, bh.c4, bh.c5, bh.c6, bh.c7,
        )
        for i in range(len(W_arr))
    ])

    P_fusion = (n * n / 4.0) * sigma_v * E_DHE3 * V

    n_e = 1.5 * n
    P_brem = 5.35e-37 * (5.0 / 3.0) * n_e * n_e * np.sqrt(T_clamped) * V

    tau_e = np.full(len(W_arr), c_tau * (n / _N_REF) * (B / _B_REF) ** 2)
    P_cond = W_arr / tau_e
    net_power = P_heat + P_fusion - P_brem - P_cond

    return T_kev, sigma_v, tau_e, P_fusion, P_brem, P_cond, net_power


def integrate_helion_pulse(
    parameters: HelionParameters,
    P_heat_w: float = 0.0,
    *,
    t_eval: np.ndarray | None = None,
    n_t_points: int = 200,
    c_tau_scale: float = 1.0,
) -> HelionPulseResult:
    """Integrate the Helion FRC plasma energy balance ODE for one pulse.

    Workflow:
        1. Apply adiabatic compression (Eq. 19) → post-compression state.
        2. Build flat params array and initial state y₀ = [W₀].
        3. Integrate dW/dt (Eq. 16) over [0, pulse_duration_s] with Radau.
        4. Post-process W(t) → T, P_fusion, P_brem, P_cond, Q.

    SR-01: Integration failure returns a defined error HelionPulseResult with
    success=False and a plain-language message — no unhandled exception.

    Args:
        parameters: Validated Pydantic HelionParameters.
        P_heat_w:   Constant external plasma heating power (W). Default 0 (adiabatic pulse).
        t_eval:     Optional array of output times (s). If None, n_t_points uniformly
                    spaced points across the pulse duration are used.
        n_t_points: Number of output points when t_eval is None. Default 200.

    Returns:
        HelionPulseResult with time series and scalar Q factor.
    """
    T_final, n_final, V_final, W_initial = apply_adiabatic_compression(
        parameters.ion_temperature_kev,
        parameters.plasma_density_m3,
        parameters.compression_ratio,
        parameters.plasma_volume_m3,
    )

    if T_final > T_MAX_KEV:
        logger.warning(
            "Post-compression T = %.1f keV exceeds Bosch-Hale upper bound %.1f keV. "
            "⟨σv⟩ will be clamped during integration (SPEC R-06).",
            T_final, T_MAX_KEV,
        )

    params = make_params_array(
        n_final, V_final, P_heat_w, parameters.magnetic_field_t,
        c_tau_scale=c_tau_scale,
    )
    y0 = np.array([W_initial], dtype=np.float64)

    t_span = (0.0, parameters.pulse_duration_s)
    if t_eval is None:
        t_eval = np.linspace(0.0, parameters.pulse_duration_s, n_t_points)

    result = solve_ivp(
        plasma_energy_rhs_kernel,
        t_span,
        y0,
        method="Radau",
        rtol=1e-6,
        atol=1e-9,
        args=(params,),
        t_eval=t_eval,
        dense_output=False,
    )

    if not result.success:
        logger.error("Helion pulse ODE integration failed: %s", result.message)
        empty = np.zeros(1)
        return HelionPulseResult(
            t=empty, W=empty, T_kev=empty,
            sigma_v=empty, tau_e=empty,
            P_fusion=empty, P_brem=empty, P_cond=empty,
            net_power=empty, p_heat_w=P_heat_w,
            Q=0.0, W_initial=W_initial,
            success=False,
            message=f"ODE solver failed: {result.message}",
        )

    W_arr = result.y[0]
    t_arr = result.t
    T_kev, sigma_v, tau_e, P_fusion, P_brem, P_cond, net_power = _post_process(
        t_arr, W_arr, params,
    )

    # Q = total fusion energy / initial plasma energy (SPEC §4.3, FR-14)
    E_fusion = float(np.trapezoid(P_fusion, t_arr))
    E_heat_input = W_initial + P_heat_w * parameters.pulse_duration_s
    Q = E_fusion / E_heat_input if E_heat_input > 0.0 else 0.0

    return HelionPulseResult(
        t=t_arr,
        W=W_arr,
        T_kev=T_kev,
        sigma_v=sigma_v,
        tau_e=tau_e,
        P_fusion=P_fusion,
        P_brem=P_brem,
        P_cond=P_cond,
        net_power=net_power,
        p_heat_w=P_heat_w,
        Q=Q,
        W_initial=W_initial,
        success=True,
        message="OK",
    )


def prewarm_jit() -> None:
    """Pre-warm the Numba JIT cache for plasma_energy_rhs_kernel.

    Calls once at application startup to eliminate first-call compile latency
    (ARCHITECTURE.md §7 Risk R-04, CLAUDE.md §Numba Rules).
    """
    from physics.helion.bosch_hale import prewarm_jit as prewarm_bh
    prewarm_bh()

    params = make_params_array(
        n_final_m3=1.0e21,
        V_final_m3=1.0,
        P_heat_w=0.0,
        B_T=5.0,
    )
    y_dummy = np.array([1.0e8], dtype=np.float64)
    plasma_energy_rhs_kernel(0.0, y_dummy, params)
