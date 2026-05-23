"""Full 11-state PWR point-kinetics and thermal-hydraulics engine.

This module is the primary Phase 1 coupling surface for the PWR twin. It keeps
the shared 6-group kinetics, lumped thermal-hydraulics, xenon, and algebraic
feedback calculators independent, then composes them into one solver RHS.

State vector layout:
    y[0]    = n       neutron population, normalised to the reference power
    y[1:7]  = C1-C6   delayed-neutron precursor concentrations
    y[7]    = T_fuel  lumped fuel temperature (K)
    y[8]    = T_cool  lumped coolant / moderator temperature (K)
    y[9]    = I       I-135 number density (atoms/cm^3)
    y[10]   = X       Xe-135 number density (atoms/cm^3)

Coupling at every ODE evaluation:
    1. P_fission = n * P_ref feeds the thermal-hydraulics heat source.
    2. T_fuel and T_cool feed Doppler and moderator feedback.
    3. Feedback deltas are summed into total reactivity.
    4. Total reactivity drives the next point-kinetics RHS evaluation.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass

import numba
import numpy as np
from scipy.integrate import solve_ivp

from data.keepin_dnp import beta_fractions, decay_constants
from physics.pwr.feedback import (
    boron_worth,
    doppler_feedback,
    moderator_feedback,
    pcm_to_dk_k,
    reactivity_balance,
)
from physics.pwr.parameters import PWRParameters
from physics.pwr.reactivity import FeedbackCoefficients, ReferenceState
from physics.pwr.thermal_hydraulics import (
    M_DOT_NOM,
    P_NOM,
    T_COOL_NOM,
    T_FUEL_NOM,
    T_IN_NOM,
    thermal_hydraulics_rhs,
)
from physics.shared.kinetics import (
    LAMBDA_PWR,
    point_kinetics_rhs,
    steady_state_precursors,
)
from physics.shared.xenon import SIGMA_AX, steady_state_xenon, xenon_iodine_rhs

logger = logging.getLogger(__name__)

# Full-power operating point used by the existing xenon integration tests.
PHI_NOM: float = 3.1e13
SIGMA_F: float = 0.30
SIGMA_A: float = 0.55


@dataclass(frozen=True)
class PWRControls:
    """Validated external controls for one PWR ODE evaluation."""

    rod_reactivity_pcm: float = 0.0
    boron_ppm: float = 0.0
    coolant_flow_fraction: float = 1.0
    inlet_temperature_k: float = T_IN_NOM


@dataclass(frozen=True)
class PWRModelConfig:
    """Model constants for the coupled PWR ODE system."""

    reference_power_w: float = P_NOM
    reference_neutron_population: float = 1.0
    reference_flux: float = PHI_NOM
    sigma_f: float = SIGMA_F
    sigma_a: float = SIGMA_A
    base_reactivity_pcm: float = 0.0
    feedback_coefficients: FeedbackCoefficients = FeedbackCoefficients()


@dataclass(frozen=True)
class PWRReactivitySnapshot:
    """Reactivity component breakdown evaluated from an 11-state PWR vector."""

    base_pcm: float
    rods_pcm: float
    doppler_pcm: float
    moderator_pcm: float
    boron_pcm: float
    xenon_pcm: float

    @property
    def total_pcm(self) -> float:
        """Return total PWR reactivity in pcm."""
        return self.base_pcm + reactivity_balance(
            self.rods_pcm,
            self.doppler_pcm,
            self.moderator_pcm,
            self.boron_pcm,
            self.xenon_pcm,
        )

    @property
    def total_dk_k(self) -> float:
        """Return total PWR reactivity as dimensionless delta-k/k."""
        return pcm_to_dk_k(self.total_pcm)


ControlsFn = Callable[[float, np.ndarray], PWRControls]


def controls_from_parameters(parameters: PWRParameters) -> PWRControls:
    """Convert validated Pydantic PWR parameters into engine controls."""
    return PWRControls(
        rod_reactivity_pcm=parameters.rod_reactivity_pcm,
        boron_ppm=parameters.boron_ppm,
        coolant_flow_fraction=parameters.coolant_flow_fraction,
        inlet_temperature_k=parameters.inlet_temperature_k,
    )


def constant_controls_fn(controls: PWRControls) -> ControlsFn:
    """Return a controls callback for scenarios with fixed external inputs."""
    def _controls_fn(_t: float, _y: np.ndarray) -> PWRControls:
        return controls

    return _controls_fn


def build_initial_state(
    *,
    n0: float = 1.0,
    fuel_temperature_k: float = T_FUEL_NOM,
    coolant_temperature_k: float = T_COOL_NOM,
    config: PWRModelConfig = PWRModelConfig(),
    beta_arr: np.ndarray | None = None,
    lambda_arr: np.ndarray | None = None,
    Lambda: float = LAMBDA_PWR,
) -> np.ndarray:
    """Construct the 11-state PWR vector from analytical steady-state blocks."""
    if beta_arr is None:
        beta_arr = np.array(beta_fractions(), dtype=np.float64)
    if lambda_arr is None:
        lambda_arr = np.array(decay_constants(), dtype=np.float64)

    precursors = steady_state_precursors(n0, beta_arr, lambda_arr, Lambda)
    phi0 = neutron_population_to_flux(n0, config)
    iodine_xenon = steady_state_xenon(phi0, config.sigma_f)

    return np.concatenate(
        (
            np.array([n0], dtype=np.float64),
            precursors,
            np.array([fuel_temperature_k, coolant_temperature_k], dtype=np.float64),
            iodine_xenon,
        )
    )


def neutron_population_to_power(n: float, config: PWRModelConfig) -> float:
    """Scale normalised neutron population into instantaneous fission power (W)."""
    return n / config.reference_neutron_population * config.reference_power_w


def neutron_population_to_flux(n: float, config: PWRModelConfig) -> float:
    """Scale normalised neutron population into thermal neutron flux."""
    return n / config.reference_neutron_population * config.reference_flux


def xenon_reactivity_pcm(xenon_density: float, sigma_a: float = SIGMA_A) -> float:
    """Return Xe-135 poisoning reactivity in pcm using one-group perturbation theory."""
    return -SIGMA_AX * xenon_density / sigma_a * 1.0e5


def calculate_reactivity_snapshot(
    y: np.ndarray,
    controls: PWRControls,
    reference_state: ReferenceState,
    config: PWRModelConfig = PWRModelConfig(),
) -> PWRReactivitySnapshot:
    """Evaluate all reactivity feedback components from the coupled state vector."""
    coefficients = config.feedback_coefficients
    doppler_pcm = doppler_feedback(
        float(y[7]),
        reference_state.fuel_temperature_ref_k,
        coefficients.alpha_doppler_pcm_per_k,
    )
    moderator_pcm = moderator_feedback(
        float(y[8]),
        reference_state.coolant_temperature_ref_k,
        coefficients.alpha_moderator_pcm_per_k,
    )
    boron_pcm = boron_worth(
        controls.boron_ppm,
        coefficients.boron_worth_pcm_per_ppm,
    )
    xenon_pcm = xenon_reactivity_pcm(float(y[10]), config.sigma_a)

    return PWRReactivitySnapshot(
        base_pcm=config.base_reactivity_pcm,
        rods_pcm=controls.rod_reactivity_pcm,
        doppler_pcm=doppler_pcm,
        moderator_pcm=moderator_pcm,
        boron_pcm=boron_pcm,
        xenon_pcm=xenon_pcm,
    )


def critical_base_reactivity_pcm(
    y0: np.ndarray,
    controls: PWRControls,
    reference_state: ReferenceState,
    config: PWRModelConfig = PWRModelConfig(),
) -> float:
    """Return the base reactivity that makes ``y0`` exactly critical."""
    zero_base_config = PWRModelConfig(
        reference_power_w=config.reference_power_w,
        reference_neutron_population=config.reference_neutron_population,
        reference_flux=config.reference_flux,
        sigma_f=config.sigma_f,
        sigma_a=config.sigma_a,
        base_reactivity_pcm=0.0,
        feedback_coefficients=config.feedback_coefficients,
    )
    snapshot = calculate_reactivity_snapshot(
        y0,
        controls,
        reference_state,
        zero_base_config,
    )
    return -snapshot.total_pcm


def pwr_coupled_rhs(
    t: float,
    y: np.ndarray,
    controls_fn: ControlsFn,
    reference_state: ReferenceState,
    config: PWRModelConfig,
    beta_arr: np.ndarray,
    lambda_arr: np.ndarray,
    Lambda: float = LAMBDA_PWR,
) -> np.ndarray:
    """Evaluate the full 11-state closed-loop PWR ODE right-hand side."""
    controls = controls_fn(t, y)
    snapshot = calculate_reactivity_snapshot(y, controls, reference_state, config)

    kinetics_rhs = point_kinetics_rhs(
        t,
        y[:7],
        snapshot.total_dk_k,
        beta_arr,
        lambda_arr,
        Lambda,
    )

    power_w = neutron_population_to_power(float(y[0]), config)
    m_dot = controls.coolant_flow_fraction * M_DOT_NOM
    thermal_rhs = thermal_hydraulics_rhs(
        t,
        y[7:9],
        power_w,
        m_dot,
        controls.inlet_temperature_k,
    )

    phi = neutron_population_to_flux(float(y[0]), config)
    xenon_rhs = xenon_iodine_rhs(t, y[9:11], phi, config.sigma_f)

    return np.concatenate((kinetics_rhs, thermal_rhs, xenon_rhs))


def integrate_pwr(
    y0: np.ndarray,
    t_span: tuple[float, float],
    controls: PWRControls | ControlsFn,
    reference_state: ReferenceState,
    config: PWRModelConfig = PWRModelConfig(),
    *,
    beta_arr: np.ndarray | None = None,
    lambda_arr: np.ndarray | None = None,
    Lambda: float = LAMBDA_PWR,
    t_eval: np.ndarray | None = None,
    max_step: float = np.inf,
):
    """Integrate the closed-loop 11-state PWR ODE system with SciPy Radau."""
    if beta_arr is None:
        beta_arr = np.array(beta_fractions(), dtype=np.float64)
    if lambda_arr is None:
        lambda_arr = np.array(decay_constants(), dtype=np.float64)

    controls_fn = controls if callable(controls) else constant_controls_fn(controls)

    def _rhs(t: float, y: np.ndarray) -> np.ndarray:
        return pwr_coupled_rhs(
            t,
            y,
            controls_fn,
            reference_state,
            config,
            beta_arr,
            lambda_arr,
            Lambda,
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
        max_step=max_step,
    )

    if not result.success:
        logger.error("Coupled PWR ODE integration failed: %s", result.message)

    return result


# ---------------------------------------------------------------------------
# pwr_11_state_system — flat-params interface for Numba JIT preparation
#
# The functions above use Python dataclass objects and closures. This section
# re-exposes the same physics through a flat float64 parameter array so the
# hot RHS function can be decorated with @numba.njit in a subsequent task
# without changing any caller code that uses make_params_array / integrate_pwr_11state.
# ---------------------------------------------------------------------------

# Flat parameter array index layout — every index is a named constant so
# callers never use magic integers.  All elements are float64.
_P_ROD_DKK    = 0   # rod reactivity (dimensionless Δk/k)
_P_BORON_PPM  = 1   # dissolved boron concentration (ppm)
_P_FLOW_FRAC  = 2   # coolant flow fraction (1.0 = nominal)
_P_T_IN       = 3   # coolant inlet temperature (K)
_P_ALPHA_D    = 4   # Doppler coefficient (pcm/K)
_P_ALPHA_M    = 5   # moderator temperature coefficient (pcm/K)
_P_OMEGA_B    = 6   # differential boron worth (pcm/ppm)
_P_T_FUEL_REF = 7   # reference fuel temperature — zero Doppler feedback point (K)
_P_T_COOL_REF = 8   # reference coolant temperature — zero moderator feedback point (K)
_P_P_REF      = 9   # fission power at n = 1 (W); P = n * P_ref
_P_PHI_REF    = 10  # thermal flux at n = 1 (n/cm²/s); φ = n * φ_ref
_P_SIGMA_F    = 11  # macroscopic fission cross section (cm⁻¹)
_P_SIGMA_A    = 12  # one-group absorption XS used for xenon worth denominator (cm⁻¹)
_P_BASE_DKK   = 13  # base reactivity that guarantees criticality at t=0 (Δk/k)
_P_LAMBDA     = 14  # prompt neutron generation time Λ (s)
_P_BETA_0     = 15  # beta_arr[0..5] occupies indices 15–20
_P_LAMBDA_0   = 21  # lambda_arr[0..5] occupies indices 21–26

PARAMS_LEN: int = 27  # total length of the flat parameter vector


def make_params_array(
    controls: PWRControls,
    reference_state: ReferenceState,
    config: PWRModelConfig = PWRModelConfig(),
    *,
    beta_arr: np.ndarray | None = None,
    lambda_arr: np.ndarray | None = None,
    Lambda: float = LAMBDA_PWR,
) -> np.ndarray:
    """Pack PWR engine objects into the flat float64 params vector.

    The returned array is the single argument passed to pwr_11_state_system via
    scipy's args keyword:  solve_ivp(pwr_11_state_system, t_span, y0, args=(params,))

    Args:
        controls:        External inputs (rod, boron, flow, inlet temperature).
        reference_state: Zero-feedback reference temperatures.
        config:          Model constants (power, flux, cross sections, base reactivity).
        beta_arr:        Optional beta fractions; defaults to IAEA U-235 values.
        lambda_arr:      Optional decay constants; defaults to IAEA U-235 values.
        Lambda:          Prompt neutron generation time (s).

    Returns:
        Float64 array of shape (PARAMS_LEN,) with layout documented by _P_* constants.
    """
    if beta_arr is None:
        beta_arr = np.array(beta_fractions(), dtype=np.float64)
    if lambda_arr is None:
        lambda_arr = np.array(decay_constants(), dtype=np.float64)

    params = np.empty(PARAMS_LEN, dtype=np.float64)
    params[_P_ROD_DKK]    = controls.rod_reactivity_pcm * 1.0e-5
    params[_P_BORON_PPM]  = controls.boron_ppm
    params[_P_FLOW_FRAC]  = controls.coolant_flow_fraction
    params[_P_T_IN]       = controls.inlet_temperature_k
    params[_P_ALPHA_D]    = config.feedback_coefficients.alpha_doppler_pcm_per_k
    params[_P_ALPHA_M]    = config.feedback_coefficients.alpha_moderator_pcm_per_k
    params[_P_OMEGA_B]    = config.feedback_coefficients.boron_worth_pcm_per_ppm
    params[_P_T_FUEL_REF] = reference_state.fuel_temperature_ref_k
    params[_P_T_COOL_REF] = reference_state.coolant_temperature_ref_k
    # Fold the reference normalisation into the per-unit power and flux so that
    # P = n * P_ref and φ = n * φ_ref without a separate division inside the RHS.
    params[_P_P_REF]      = config.reference_power_w / config.reference_neutron_population
    params[_P_PHI_REF]    = config.reference_flux / config.reference_neutron_population
    params[_P_SIGMA_F]    = config.sigma_f
    params[_P_SIGMA_A]    = config.sigma_a
    params[_P_BASE_DKK]   = config.base_reactivity_pcm * 1.0e-5
    params[_P_LAMBDA]     = Lambda
    params[_P_BETA_0:_P_BETA_0 + 6]     = beta_arr
    params[_P_LAMBDA_0:_P_LAMBDA_0 + 6] = lambda_arr
    return params


@numba.njit(cache=True)
def pwr_11_state_system(t: float, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """Numba JIT kernel: full 11-state PWR ODE right-hand side — flat-params form.

    Implements SPEC Eqs. 1, 2 (kinetics), 9, 10 (thermal-hydraulics), 11–13
    (algebraic feedback), and 14, 15 (xenon/iodine) at a single solver call.

    Designed for use with scipy.integrate.solve_ivp via the args keyword::

        result = solve_ivp(
            pwr_11_state_system,
            t_span, y0,
            method="Radau", rtol=1e-6, atol=1e-9,
            args=(params,),
        )

    The flat ``params`` layout (documented by _P_* index constants) lets this
    function operate entirely on numpy scalar/array primitives, enabling full
    @numba.njit compilation. Sub-kernels (point_kinetics_rhs,
    thermal_hydraulics_rhs, xenon_iodine_rhs) are themselves @njit and are
    inlined by Numba at compile time.

    State vector layout (ARCHITECTURE.md §4.1, SPEC §4.6.2):
        y[0]    n         neutron population (normalised; n=1 ↔ P = P_ref)
        y[1:7]  C₁–C₆    delayed-neutron precursor concentrations
        y[7]    T_fuel    lumped fuel temperature (K)
        y[8]    T_cool    lumped coolant temperature (K)
        y[9]    I         I-135 number density (atoms/cm³)
        y[10]   X         Xe-135 number density (atoms/cm³)

    Args:
        t:      Current time (s) — required by scipy ODE driver signature.
        y:      11-state vector, float64, shape (11,).
        params: Flat parameter array returned by make_params_array(), shape (PARAMS_LEN,).

    Returns:
        dy/dt, float64, shape (11,) — all 11 coupled time-derivatives.
    """
    n      = y[0]
    T_fuel = y[7]
    T_cool = y[8]
    X      = y[10]

    # Unpack scalar parameters from the flat vector
    rod_dk_k   = params[_P_ROD_DKK]
    boron_ppm  = params[_P_BORON_PPM]
    flow_frac  = params[_P_FLOW_FRAC]
    T_in       = params[_P_T_IN]
    alpha_d    = params[_P_ALPHA_D]
    alpha_m    = params[_P_ALPHA_M]
    omega_b    = params[_P_OMEGA_B]
    T_fuel_ref = params[_P_T_FUEL_REF]
    T_cool_ref = params[_P_T_COOL_REF]
    P_ref      = params[_P_P_REF]
    phi_ref    = params[_P_PHI_REF]
    sigma_f    = params[_P_SIGMA_F]
    sigma_a    = params[_P_SIGMA_A]
    base_dk_k  = params[_P_BASE_DKK]
    Lambda     = params[_P_LAMBDA]
    beta_arr   = params[_P_BETA_0:_P_BETA_0 + 6]
    lambda_arr = params[_P_LAMBDA_0:_P_LAMBDA_0 + 6]

    # --- Algebraic feedback reactivity — SPEC Eqs. 11–13 + xenon worth ---
    # Each component computed in pcm, summed into dimensionless Δk/k.
    rho_d_pcm  = alpha_d * (T_fuel - T_fuel_ref)         # Doppler     (Eq. 11)
    rho_m_pcm  = alpha_m * (T_cool - T_cool_ref)         # moderator   (Eq. 12)
    rho_b_pcm  = omega_b * boron_ppm                     # boron       (Eq. 13)
    rho_xe_pcm = -SIGMA_AX * X / sigma_a * 1.0e5         # xenon (one-group perturbation)

    rho_total = (
        base_dk_k
        + rod_dk_k
        + (rho_d_pcm + rho_m_pcm + rho_b_pcm + rho_xe_pcm) * 1.0e-5
    )

    # --- Kinetics block — SPEC Eqs. 1 & 2 (7 states: n, C₁–C₆) ---
    kin_rhs = point_kinetics_rhs(t, y[:7], rho_total, beta_arr, lambda_arr, Lambda)

    # --- Thermal-hydraulics block — SPEC Eqs. 9 & 10 (2 states: T_fuel, T_cool) ---
    P_w   = n * P_ref             # fission power (W)
    m_dot = flow_frac * M_DOT_NOM
    th_rhs = thermal_hydraulics_rhs(t, y[7:9], P_w, m_dot, T_in)

    # --- Xenon/Iodine block — SPEC Eqs. 14 & 15 (2 states: I, X) ---
    phi    = n * phi_ref          # thermal neutron flux (n/cm²/s)
    xe_rhs = xenon_iodine_rhs(t, y[9:11], phi, sigma_f)

    dydt = np.empty(11)
    dydt[0] = kin_rhs[0]
    dydt[1] = kin_rhs[1]
    dydt[2] = kin_rhs[2]
    dydt[3] = kin_rhs[3]
    dydt[4] = kin_rhs[4]
    dydt[5] = kin_rhs[5]
    dydt[6] = kin_rhs[6]
    dydt[7] = th_rhs[0]
    dydt[8] = th_rhs[1]
    dydt[9] = xe_rhs[0]
    dydt[10] = xe_rhs[1]
    return dydt


def integrate_pwr_11state(
    y0: np.ndarray,
    t_span: tuple[float, float],
    params: np.ndarray,
    *,
    t_eval: np.ndarray | None = None,
    max_step: float = np.inf,
):
    """Integrate the 11-state PWR ODE using the flat-params interface.

    Calls scipy Radau via solve_ivp's args keyword so the params array is
    passed directly to the @numba.njit-compiled pwr_11_state_system kernel
    without a Python closure overhead on each RHS evaluation.

    Args:
        y0:       Initial 11-state vector, shape (11,), float64.
        t_span:   (t_start, t_end) in seconds.
        params:   Flat parameter array returned by make_params_array().
        t_eval:   Optional output times (s). None lets the solver choose steps.
        max_step: Maximum internal solver step (s). Default np.inf lets Radau
                  choose steps freely. Set to ~300 s for xenon-timescale
                  accelerated runs; set to ~0.5 s when tracking fast transients
                  under accelerated time to bound step size explicitly.

    Returns:
        scipy OdeResult with .t, .y (shape 11 × n_steps), .success, .message.
    """
    result = solve_ivp(
        pwr_11_state_system,
        t_span,
        y0,
        method="Radau",
        rtol=1e-6,
        atol=1e-9,
        args=(params,),
        t_eval=t_eval,
        dense_output=False,
        max_step=max_step,
    )
    if not result.success:
        logger.error("11-state PWR flat-params ODE integration failed: %s", result.message)
    return result


def prewarm_jit_11state(
    beta_arr: np.ndarray | None = None,
    lambda_arr: np.ndarray | None = None,
    Lambda: float = LAMBDA_PWR,
) -> None:
    """Pre-warm the Numba JIT cache for pwr_11_state_system.

    Triggers AOT compilation of the full 11-state RHS kernel (and its sub-kernels)
    at application startup so the first UI callback sees no compile-time latency
    (ARCHITECTURE.md §7 Risk R-04, CLAUDE.md §Numba Rules).

    Args:
        beta_arr:   Optional beta fractions; defaults to IAEA U-235 values.
        lambda_arr: Optional decay constants; defaults to IAEA U-235 values.
        Lambda:     Prompt neutron generation time (s).
    """
    if beta_arr is None:
        beta_arr = np.array(beta_fractions(), dtype=np.float64)
    if lambda_arr is None:
        lambda_arr = np.array(decay_constants(), dtype=np.float64)

    from physics.pwr.reactivity import ReferenceState
    from physics.pwr.thermal_hydraulics import T_COOL_NOM, T_FUEL_NOM, T_IN_NOM

    reference_state = ReferenceState(
        fuel_temperature_ref_k=T_FUEL_NOM,
        coolant_temperature_ref_k=T_COOL_NOM,
    )
    controls = PWRControls(
        rod_reactivity_pcm=0.0,
        boron_ppm=0.0,
        coolant_flow_fraction=1.0,
        inlet_temperature_k=T_IN_NOM,
    )
    config = PWRModelConfig()
    params = make_params_array(
        controls, reference_state, config,
        beta_arr=beta_arr, lambda_arr=lambda_arr, Lambda=Lambda,
    )
    y0 = build_initial_state(
        beta_arr=beta_arr, lambda_arr=lambda_arr, Lambda=Lambda,
    )
    pwr_11_state_system(0.0, y0, params)