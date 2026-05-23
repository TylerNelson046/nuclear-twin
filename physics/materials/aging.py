"""Step functions and remaining-life projections for material aging.

These models are first-order educational implementations. They are calibrated
against published commercial-reactor design-basis values, but each is
deliberately simplified to a closed-form rate law that can be evaluated once
per simulation tick without a separate ODE integrator.

References (informal, for orientation only — see SPEC for full citations):
- NRC Reg Guide 1.99 Rev 2: RPV embrittlement under fast fluence.
- Cathcart, J.V.; Pawel, R.E. (1977): Zircaloy oxidation kinetics.
- ORNL-4812 (MSRE final report): Hastelloy-N tellurium attack rates.
- Bosch & Hale (1992): D-D branching ratios and neutron yields.
- ASTM E521: Definition of DPA for charged-particle and neutron irradiation.
"""

from __future__ import annotations

import math

from physics.materials.state import (
    PWRMaterialState,
    MSRMaterialState,
    HelionMaterialState,
)


# ============================================================
# PWR design-basis limits (illustrative, from public literature).
# ============================================================
PWR_DESIGN_LIMITS: dict[str, float] = {
    # 200 °F shift is the conventional pressurized-thermal-shock screening
    # criterion for axial welds (10 CFR 50.61).
    "rpv_drtndt_k": 111.0,
    # ~100 µm corrosion is roughly the licensing limit for typical Zircaloy-4
    # cladding before strain-based criteria are challenged.
    "cladding_oxide_thickness_um": 100.0,
    # 600 wppm hydrogen approaches the threshold for delayed hydride cracking
    # in light-water-reactor cladding.
    "cladding_hydrogen_ppm": 600.0,
    # Replacement-time benchmark for control rod absorber depletion.
    "control_rod_b10_depletion_frac": 0.30,
    # Industry rod-average burnup license limit.
    "fuel_burnup_gwd_per_mtu": 62.0,
}

# ============================================================
# MSR design-basis limits (illustrative, from MSRE/MSBR design reports).
# ============================================================
MSR_DESIGN_LIMITS: dict[str, float] = {
    # MSBR design allowed ~1000 µm wall loss over 30 years.
    "hastelloy_n_corrosion_um": 1000.0,
    # MSRE observed ~250 µm intergranular cracks before modified alloy was
    # introduced; this is a conservative design ceiling.
    "tellurium_attack_depth_um": 250.0,
    # Heat-exchanger tube minimum-wall criterion.
    "heat_exchanger_thinning_um": 500.0,
    # Tritium inventory at which secondary-loop permeation barriers become
    # safety-class. Order-of-magnitude figure.
    "tritium_inventory_g": 100.0,
}

# ============================================================
# Helion design-basis limits (illustrative, fusion-engineering studies).
# ============================================================
HELION_DESIGN_LIMITS: dict[str, float] = {
    # ITER first-wall design DPA endpoint for the test blanket modules.
    "first_wall_dpa": 3.0,
    # Low-cycle fatigue endurance for austenitic-steel first-wall design.
    "first_wall_thermal_cycles": 1.0e7,
    # Manson-Coffin index analog; >1.0 implies the design fatigue life is met.
    "coil_thermal_fatigue_index": 1.0,
    # Typical pulse-power capacitor cycle rating.
    "capacitor_charge_cycles": 1.0e7,
    # 100 MGy is a typical kapton-style insulator total-dose limit.
    "insulator_dielectric_dose_mgy": 100.0,
}


# ============================================================
# PWR aging step.
# ============================================================
# Inner-vessel-wall fast-flux scaling: at full power, a typical 4-loop PWR
# sees ~3e10 n/cm^2-s of E > 1 MeV neutrons at the beltline. This scales
# linearly with power for a fixed loading pattern.
_PWR_RPV_FAST_FLUX_FULL_POWER = 3.0e10  # n/cm^2-s, E > 1 MeV
# Chemistry factor for typical commercial weld metal: roughly 110 K worth
# of shift coefficient (a stand-in for the Cu/Ni content table in RG 1.99).
_PWR_RPV_CHEMISTRY_FACTOR_K = 110.0
# Reference fluence appearing inside the RG 1.99 fluence factor expression.
_PWR_RPV_FLUENCE_NORM = 1.0e19  # n/cm^2 reference scale

# Cladding parabolic oxidation: dδ²/dt = K, with K Arrhenius in T_cool.
# Calibrated so a typical 4-loop PWR sees ~3 µm/year average oxide growth at
# 600 K bulk coolant (Cathcart-Pawel low-temperature regime).
_PWR_CLAD_K_PRE = 0.13  # µm^2 / s pre-exponential
_PWR_CLAD_Q_OVER_R = 8500.0  # K — apparent activation in Kelvin
_PWR_HYDROGEN_PICKUP_PPM_PER_UM = 9.0

# Control rod B-10 depletion: σ_a ≈ 3837 b for B-10 thermal absorption.
# Calibrated so a ~30 yr life sees ~30% B-10 burnout at typical 10%
# rod-insertion duty.
_PWR_B10_DEPLETION_PER_FULL_HOUR = 1.0e-5

# Power-to-burnup conversion: 1 GWd_th per MTU per day at unit specific power.
# Typical PWR loaded mass ~ 90 MTU at 3000 MWth → ~33.3 MW/MTU.
_PWR_REFERENCE_MTU = 90.0


def step_pwr_aging(
    state: PWRMaterialState,
    dt_s: float,
    power_mw: float,
    t_fuel_k: float,
    t_coolant_k: float,
    rod_inserted_fraction: float,
    p_nominal_mw: float = 3000.0,
) -> PWRMaterialState:
    """Advance the PWR material state forward by ``dt_s`` seconds.

    All damage rates scale with power level and component temperature.
    The model is linear-in-dt over each step; the caller is responsible for
    picking ``dt_s`` short enough that the power/temperature signals don't
    swing wildly across the step (the runner uses ~10 s steps by default).
    """
    if dt_s <= 0.0:
        return state

    power_fraction = max(0.0, power_mw) / max(p_nominal_mw, 1.0)

    # RPV fluence accumulation.
    fast_flux = _PWR_RPV_FAST_FLUX_FULL_POWER * power_fraction
    new_fluence = state.rpv_fluence_n_per_cm2 + fast_flux * dt_s

    # RG 1.99 Rev 2 fluence factor: FF = f^(0.28 - 0.10 log10(f)),
    # f normalized to 1e19 n/cm^2.
    f_norm = new_fluence / _PWR_RPV_FLUENCE_NORM
    if f_norm > 0.0:
        log_f = math.log10(f_norm)
        ff_exponent = 0.28 - 0.10 * log_f
        fluence_factor = f_norm ** ff_exponent
    else:
        fluence_factor = 0.0
    new_drtndt = _PWR_RPV_CHEMISTRY_FACTOR_K * fluence_factor

    # Cladding parabolic oxidation: dδ²/dt = K(T) → δ = sqrt(δ₀² + K dt).
    k_cool = _PWR_CLAD_K_PRE * math.exp(-_PWR_CLAD_Q_OVER_R / max(t_coolant_k, 300.0))
    new_oxide_sq = state.cladding_oxide_thickness_um ** 2 + k_cool * dt_s
    new_oxide = math.sqrt(max(0.0, new_oxide_sq))
    new_hydrogen = new_oxide * _PWR_HYDROGEN_PICKUP_PPM_PER_UM

    # Control rod B-10 depletion: exponential decay in inserted flux exposure.
    hours = dt_s / 3600.0
    inserted_exposure = max(0.0, rod_inserted_fraction) * power_fraction * hours
    remaining_fraction = 1.0 - state.control_rod_b10_depletion_frac
    remaining_fraction *= math.exp(-_PWR_B10_DEPLETION_PER_FULL_HOUR * inserted_exposure)
    new_b10_depletion = 1.0 - remaining_fraction

    # Fuel burnup: GW-days per MTU.
    gwd = (power_mw / 1000.0) * (dt_s / 86400.0)
    new_burnup = state.fuel_burnup_gwd_per_mtu + gwd / _PWR_REFERENCE_MTU

    new_hours = state.operating_hours + dt_s / 3600.0

    return PWRMaterialState(
        rpv_fluence_n_per_cm2=new_fluence,
        rpv_drtndt_k=new_drtndt,
        cladding_oxide_thickness_um=new_oxide,
        cladding_hydrogen_ppm=new_hydrogen,
        control_rod_b10_depletion_frac=new_b10_depletion,
        fuel_burnup_gwd_per_mtu=new_burnup,
        operating_hours=new_hours,
    )


_SECONDS_PER_YEAR = 365.25 * 86400.0


def _project_to_limit_pwr(
    state: PWRMaterialState,
    component: str,
    limit: float,
    power_mw: float,
    t_fuel_k: float,
    t_coolant_k: float,
    rod_inserted_fraction: float,
    p_nominal_mw: float,
) -> float:
    """Project years to ``limit`` for one PWR component by adaptive probe.

    Several aging laws are concave in time (RG 1.99 fluence factor, parabolic
    oxidation), so a short probe over-states the average rate. We bracket the
    answer iteratively: start with a 1-year probe, refine the probe length
    toward the projected remaining life, and stop when the answer is stable.
    """
    current = getattr(state, component)
    if current >= limit:
        return 0.0

    probe_years = 1.0
    last_estimate = math.inf
    for _ in range(6):
        probe = step_pwr_aging(
            state,
            dt_s=probe_years * _SECONDS_PER_YEAR,
            power_mw=power_mw,
            t_fuel_k=t_fuel_k,
            t_coolant_k=t_coolant_k,
            rod_inserted_fraction=rod_inserted_fraction,
            p_nominal_mw=p_nominal_mw,
        )
        delta = getattr(probe, component) - current
        if delta <= 0.0:
            return math.inf
        rate_per_year = delta / probe_years
        remaining = (limit - current) / rate_per_year
        if remaining <= 0.0:
            return 0.0
        if abs(remaining - last_estimate) < 0.05 * max(remaining, 1.0):
            return remaining
        last_estimate = remaining
        probe_years = max(0.5, min(remaining, 200.0))
    return last_estimate


def pwr_remaining_life(
    state: PWRMaterialState,
    power_mw: float,
    t_fuel_k: float,
    t_coolant_k: float,
    rod_inserted_fraction: float,
    p_nominal_mw: float = 3000.0,
) -> dict[str, float]:
    """Years of operation left at the current power and temperature.

    Adaptive projection: each component's remaining life is computed by
    iteratively choosing a probe window matched to the answer scale, so the
    concavity of the RG 1.99 fluence factor and the parabolic oxidation law
    don't bias the result toward a pessimistic "instantaneous rate".
    """
    return {
        "rpv_embrittlement_years": _project_to_limit_pwr(
            state, "rpv_drtndt_k", PWR_DESIGN_LIMITS["rpv_drtndt_k"],
            power_mw, t_fuel_k, t_coolant_k, rod_inserted_fraction, p_nominal_mw,
        ),
        "cladding_oxide_years": _project_to_limit_pwr(
            state, "cladding_oxide_thickness_um",
            PWR_DESIGN_LIMITS["cladding_oxide_thickness_um"],
            power_mw, t_fuel_k, t_coolant_k, rod_inserted_fraction, p_nominal_mw,
        ),
        "cladding_hydrogen_years": _project_to_limit_pwr(
            state, "cladding_hydrogen_ppm",
            PWR_DESIGN_LIMITS["cladding_hydrogen_ppm"],
            power_mw, t_fuel_k, t_coolant_k, rod_inserted_fraction, p_nominal_mw,
        ),
        "control_rod_years": _project_to_limit_pwr(
            state, "control_rod_b10_depletion_frac",
            PWR_DESIGN_LIMITS["control_rod_b10_depletion_frac"],
            power_mw, t_fuel_k, t_coolant_k, rod_inserted_fraction, p_nominal_mw,
        ),
        "fuel_burnup_years": _project_to_limit_pwr(
            state, "fuel_burnup_gwd_per_mtu",
            PWR_DESIGN_LIMITS["fuel_burnup_gwd_per_mtu"],
            power_mw, t_fuel_k, t_coolant_k, rod_inserted_fraction, p_nominal_mw,
        ),
    }


# ============================================================
# MSR aging step.
# ============================================================
# Hastelloy-N corrosion: at MSRE conditions (650 °C salt) the chromium
# leaching rate was ~6 µm/year. Scales weakly with temperature and strongly
# with redox potential. Divide by 4 because the test calibration assumed
# nominal redox+temperature factors of ~1; in practice the multipliers
# average above 1, so the base rate is brought back to the MSRE measurement.
_MSR_BASE_CORROSION_UM_PER_S = 1.5 / (365.25 * 86400.0)
_MSR_CORROSION_T_REF_K = 923.0
_MSR_CORROSION_T_SLOPE_PER_K = 0.025  # frac/K above reference

# Tellurium intergranular cracking — same calibration, narrower depth window.
_MSR_TE_BASE_UM_PER_S = 4.0 / (365.25 * 86400.0)

# Heat-exchanger thinning is roughly half of primary corrosion at MSBR
# conditions (cooler salt, no fission products in the salt loop).
_MSR_HX_FRACTION_OF_PRIMARY = 0.5

# Salt redox drifts oxidizing at ~3 mV/MW-year of fission-product accumulation.
_MSR_REDOX_DRIFT_V_PER_MW_YEAR = 0.003

# Tritium generation: ~5e-12 g per MJ thermal at typical Li-7-rich
# (depleted Li-6) salt. Calibrated against MSBR design-basis ~88 g/yr at
# 1 GW thermal.
_MSR_TRITIUM_G_PER_MJ = 5.0e-12
# Permeation out through the secondary loop is roughly 30% of generation
# at steady state for unbarriered piping.
_MSR_TRITIUM_PERMEATION_FRACTION = 0.30


def step_msr_aging(
    state: MSRMaterialState,
    dt_s: float,
    power_mw: float,
    t_salt_k: float,
    p_nominal_mw: float = 500.0,
) -> MSRMaterialState:
    """Advance the MSR primary-loop material state by ``dt_s`` seconds."""
    if dt_s <= 0.0:
        return state

    power_fraction = max(0.0, power_mw) / max(p_nominal_mw, 1.0)

    # Corrosion rate scaled by temperature above reference and by current
    # oxidizing-shift of the redox potential.
    temperature_factor = 1.0 + _MSR_CORROSION_T_SLOPE_PER_K * (
        t_salt_k - _MSR_CORROSION_T_REF_K
    )
    temperature_factor = max(0.1, temperature_factor)

    # Redox potential is referenced negative (reducing) at -1.34 V.
    # As it drifts upward (less negative), corrosion accelerates.
    redox_factor = 1.0 + max(0.0, state.salt_redox_potential_v + 1.34) * 4.0

    corrosion_rate = (
        _MSR_BASE_CORROSION_UM_PER_S
        * temperature_factor
        * redox_factor
        * power_fraction
    )
    new_corrosion = state.hastelloy_n_corrosion_um + corrosion_rate * dt_s

    te_rate = (
        _MSR_TE_BASE_UM_PER_S * temperature_factor * redox_factor * power_fraction
    )
    new_te = state.tellurium_attack_depth_um + te_rate * dt_s

    hx_rate = corrosion_rate * _MSR_HX_FRACTION_OF_PRIMARY
    new_hx = state.heat_exchanger_thinning_um + hx_rate * dt_s

    # Redox drift toward oxidizing.
    drift = (
        _MSR_REDOX_DRIFT_V_PER_MW_YEAR
        * power_mw
        * (dt_s / (365.25 * 86400.0))
    )
    new_redox = state.salt_redox_potential_v + drift

    # Tritium balance: generation - permeation loss.
    gen_g = _MSR_TRITIUM_G_PER_MJ * (power_mw * 1.0e3) * dt_s
    perm_loss = state.tritium_inventory_g * _MSR_TRITIUM_PERMEATION_FRACTION * (
        dt_s / (365.25 * 86400.0)
    )
    new_tritium = max(0.0, state.tritium_inventory_g + gen_g - perm_loss)

    return MSRMaterialState(
        hastelloy_n_corrosion_um=new_corrosion,
        tellurium_attack_depth_um=new_te,
        heat_exchanger_thinning_um=new_hx,
        salt_redox_potential_v=new_redox,
        tritium_inventory_g=new_tritium,
        operating_hours=state.operating_hours + dt_s / 3600.0,
    )


def _project_to_limit_msr(
    state: MSRMaterialState,
    component: str,
    limit: float,
    power_mw: float,
    t_salt_k: float,
    p_nominal_mw: float,
) -> float:
    """Adaptive remaining-life projection for one MSR component."""
    current = getattr(state, component)
    if current >= limit:
        return 0.0

    probe_years = 1.0
    last_estimate = math.inf
    for _ in range(6):
        probe = step_msr_aging(
            state, probe_years * _SECONDS_PER_YEAR, power_mw, t_salt_k,
            p_nominal_mw=p_nominal_mw,
        )
        delta = getattr(probe, component) - current
        if delta <= 0.0:
            return math.inf
        rate_per_year = delta / probe_years
        remaining = (limit - current) / rate_per_year
        if remaining <= 0.0:
            return 0.0
        if abs(remaining - last_estimate) < 0.05 * max(remaining, 1.0):
            return remaining
        last_estimate = remaining
        probe_years = max(0.5, min(remaining, 200.0))
    return last_estimate


def msr_remaining_life(
    state: MSRMaterialState,
    power_mw: float,
    t_salt_k: float,
    p_nominal_mw: float = 500.0,
) -> dict[str, float]:
    """Years of operation left at the current MSR power and salt temperature.

    Adaptive projection identical in spirit to the PWR version: linearizes
    each damage law around its average rate over a probe window matched to
    the projected remaining life.
    """

    return {
        "hastelloy_corrosion_years": _project_to_limit_msr(
            state, "hastelloy_n_corrosion_um",
            MSR_DESIGN_LIMITS["hastelloy_n_corrosion_um"],
            power_mw, t_salt_k, p_nominal_mw,
        ),
        "tellurium_attack_years": _project_to_limit_msr(
            state, "tellurium_attack_depth_um",
            MSR_DESIGN_LIMITS["tellurium_attack_depth_um"],
            power_mw, t_salt_k, p_nominal_mw,
        ),
        "hx_thinning_years": _project_to_limit_msr(
            state, "heat_exchanger_thinning_um",
            MSR_DESIGN_LIMITS["heat_exchanger_thinning_um"],
            power_mw, t_salt_k, p_nominal_mw,
        ),
        "tritium_inventory_years": _project_to_limit_msr(
            state, "tritium_inventory_g",
            MSR_DESIGN_LIMITS["tritium_inventory_g"],
            power_mw, t_salt_k, p_nominal_mw,
        ),
    }


# ============================================================
# Helion pulse-by-pulse aging.
# ============================================================
# D-D side reaction rate: at a D-He3 50/50 fuel mix, the n_D^2 component is
# 1/4 of the total fuel-ion-pair count. Of those, the D+D→3He+n branch fires
# about half the time, releasing a 2.45 MeV neutron.
_HELION_DD_NEUTRON_FRACTION = 0.5 * 0.25  # of total reaction events
# Fast-neutron fluence-to-DPA conversion for stainless-steel first wall:
# ~5e-22 DPA per (n/cm^2). The first-wall coverage area for a Helion-scale
# machine is ~10 m^2.
_HELION_DPA_PER_NEUTRON_PER_CM2 = 5.0e-22
_HELION_FIRST_WALL_AREA_M2 = 10.0
# Reference per-pulse fusion energy (J) for normalizing coil and capacitor
# fatigue. Helion's published Polaris target is ~1 MJ per pulse.
_HELION_REF_PULSE_ENERGY_J = 1.0e6
# Coil temperature swing per reference pulse.
_HELION_REF_COIL_DELTA_T_K = 30.0
# Gamma dose to switch insulators per reference pulse — order-of-magnitude
# at the upper end of pulsed-power lab data.
_HELION_INSULATOR_DOSE_MGY_PER_PULSE = 5.0e-6


def step_helion_pulse_aging(
    state: HelionMaterialState,
    pulse_fusion_energy_j: float,
    coil_delta_t_k: float,
    n_dhe3_reactions: float,
) -> HelionMaterialState:
    """Advance the Helion machine state by ONE pulse.

    Each Helion pulse is treated as a discrete damage increment because the
    aging is dominated by the pulse cycle, not the dwell time between pulses.
    """
    # D-D neutron yield from this pulse's reaction inventory.
    n_neutrons = n_dhe3_reactions * _HELION_DD_NEUTRON_FRACTION
    # Convert total neutrons to per-cm^2 fluence at the first wall.
    if n_neutrons > 0.0:
        fluence_per_pulse = n_neutrons / (_HELION_FIRST_WALL_AREA_M2 * 1.0e4)
        dpa_increment = fluence_per_pulse * _HELION_DPA_PER_NEUTRON_PER_CM2
    else:
        dpa_increment = 0.0

    energy_ratio = max(0.0, pulse_fusion_energy_j) / _HELION_REF_PULSE_ENERGY_J
    coil_ratio = max(0.0, coil_delta_t_k) / _HELION_REF_COIL_DELTA_T_K

    return HelionMaterialState(
        pulses_fired=state.pulses_fired + 1.0,
        first_wall_dpa=state.first_wall_dpa + dpa_increment,
        first_wall_thermal_cycles=state.first_wall_thermal_cycles + 1.0,
        coil_thermal_fatigue_index=state.coil_thermal_fatigue_index + coil_ratio ** 2 / 1.0e7,
        capacitor_charge_cycles=state.capacitor_charge_cycles + 1.0,
        insulator_dielectric_dose_mgy=state.insulator_dielectric_dose_mgy
        + _HELION_INSULATOR_DOSE_MGY_PER_PULSE * energy_ratio,
    )


def helion_remaining_life(
    state: HelionMaterialState,
    pulse_fusion_energy_j: float,
    coil_delta_t_k: float,
    n_dhe3_reactions: float,
    pulses_per_day: float = 86400.0,
) -> dict[str, float]:
    """Days of operation left assuming the current per-pulse damage profile.

    Helion's published target cadence is ~1 Hz; we default to 86,400
    pulses/day so the "years" projections compare meaningfully against PWR
    and MSR.
    """
    probe = step_helion_pulse_aging(
        state, pulse_fusion_energy_j, coil_delta_t_k, n_dhe3_reactions
    )

    def _project(current: float, future: float, limit: float) -> float:
        per_pulse = future - current
        if per_pulse <= 0.0:
            return math.inf
        remaining = limit - current
        if remaining <= 0.0:
            return 0.0
        return (remaining / per_pulse) / pulses_per_day / 365.25

    return {
        "first_wall_dpa_years": _project(
            state.first_wall_dpa,
            probe.first_wall_dpa,
            HELION_DESIGN_LIMITS["first_wall_dpa"],
        ),
        "thermal_cycle_years": _project(
            state.first_wall_thermal_cycles,
            probe.first_wall_thermal_cycles,
            HELION_DESIGN_LIMITS["first_wall_thermal_cycles"],
        ),
        "coil_fatigue_years": _project(
            state.coil_thermal_fatigue_index,
            probe.coil_thermal_fatigue_index,
            HELION_DESIGN_LIMITS["coil_thermal_fatigue_index"],
        ),
        "capacitor_cycle_years": _project(
            state.capacitor_charge_cycles,
            probe.capacitor_charge_cycles,
            HELION_DESIGN_LIMITS["capacitor_charge_cycles"],
        ),
        "insulator_dose_years": _project(
            state.insulator_dielectric_dose_mgy,
            probe.insulator_dielectric_dose_mgy,
            HELION_DESIGN_LIMITS["insulator_dielectric_dose_mgy"],
        ),
    }
