"""Serializable material-aging state dataclasses for the three reactor twins.

Each dataclass holds the accumulated damage indicators tracked per component.
States are advanced by step functions in `aging.py`. They are deliberately
flat (scalars only) so they round-trip through dcc.Store / HDF5 without
custom serializers.

All fluences are stored in n/cm^2 (E > 1 MeV) — this is the industry-standard
convention for RPV embrittlement work and matches NRC Reg Guide 1.99 Rev 2.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, replace
from typing import Any


@dataclass(frozen=True)
class PWRMaterialState:
    """Accumulated damage indicators for the PWR pressure vessel + core.

    rpv_fluence_n_per_cm2:
        Time-integrated fast-neutron fluence (E > 1 MeV) at the inner RPV wall.
    rpv_drtndt_k:
        Embrittlement shift in the nil-ductility transition temperature
        (Reg Guide 1.99 Rev 2 simplified, chemistry factor for typical
        commercial weld metal).
    cladding_oxide_thickness_um:
        Average outer-surface oxide thickness on Zircaloy cladding.
        Cathcart-Pawel-like parabolic kinetics in the coolant-temperature
        regime.
    cladding_hydrogen_ppm:
        Hydrogen pickup in the cladding, roughly proportional to oxide
        thickness via a ~15% pickup fraction.
    control_rod_b10_depletion_frac:
        Fraction of original B-10 burned out of the control rods. Scales with
        flux exposure × insertion fraction.
    fuel_burnup_gwd_per_mtu:
        Integrated thermal energy released per unit fuel mass.
        Commercial PWR discharge ~50-62 GWd/MTU.
    operating_hours:
        Wall-clock hours the reactor has been at-power. Used for time-base
        normalization in the UI.
    """

    rpv_fluence_n_per_cm2: float
    rpv_drtndt_k: float
    cladding_oxide_thickness_um: float
    cladding_hydrogen_ppm: float
    control_rod_b10_depletion_frac: float
    fuel_burnup_gwd_per_mtu: float
    operating_hours: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "PWRMaterialState":
        return cls(**{k: float(payload[k]) for k in cls.__dataclass_fields__})

    def replace(self, **kwargs: float) -> "PWRMaterialState":
        return replace(self, **kwargs)


@dataclass(frozen=True)
class MSRMaterialState:
    """Accumulated damage indicators for an MSR primary loop.

    hastelloy_n_corrosion_um:
        Average wall-thickness loss on Hastelloy-N piping from chromium
        leaching into the fluoride salt.
    tellurium_attack_depth_um:
        Intergranular cracking depth from Te embrittlement of Hastelloy-N
        grain boundaries (MSRE was the canonical observation).
    heat_exchanger_thinning_um:
        Wall-thickness loss on the secondary-side heat exchanger tubes.
    salt_redox_potential_v:
        UF4/UF3 redox potential (volts vs. F2/F-). Drives the corrosion
        rate. Drifts oxidizing over time as fission products accumulate.
    tritium_inventory_g:
        Tritium produced from Li-6/Li-7 reactions in the salt; mass-balanced
        against permeation losses through the heat exchanger.
    operating_hours:
        Wall-clock hours the reactor has been at-power.
    """

    hastelloy_n_corrosion_um: float
    tellurium_attack_depth_um: float
    heat_exchanger_thinning_um: float
    salt_redox_potential_v: float
    tritium_inventory_g: float
    operating_hours: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "MSRMaterialState":
        return cls(**{k: float(payload[k]) for k in cls.__dataclass_fields__})

    def replace(self, **kwargs: float) -> "MSRMaterialState":
        return replace(self, **kwargs)


@dataclass(frozen=True)
class HelionMaterialState:
    """Accumulated damage indicators for a Helion-class pulsed FRC machine.

    Damage on a pulsed machine accumulates per shot, not per second. The
    Helion runner advances this state by ONE pulse at a time.

    pulses_fired:
        Total integrated count of full-energy pulses.
    first_wall_dpa:
        Displacements-per-atom in the first-wall structural alloy from
        D-D side-reaction neutrons (the He-3 side of the D-He3 reaction is
        aneutronic, but ~5% of fuel-ion collisions are D-D, which produce
        2.45 MeV neutrons half the time).
    first_wall_thermal_cycles:
        Number of thermal cycles experienced by the first wall (1 per pulse).
        Drives low-cycle fatigue cracking.
    coil_thermal_fatigue_index:
        Accumulated thermal-mechanical fatigue on the compression coils.
        Each pulse adds (delta_T / delta_T_ref)^2 — Manson-Coffin style.
    capacitor_charge_cycles:
        Charge-discharge cycles on the energy-storage capacitor bank.
        Each cap typically rated 1e6-1e7 cycles.
    insulator_dielectric_dose_mgy:
        Integrated dose to switch-insulators from gamma + neutron radiation.
        Drives breakdown-voltage degradation.
    """

    pulses_fired: float
    first_wall_dpa: float
    first_wall_thermal_cycles: float
    coil_thermal_fatigue_index: float
    capacitor_charge_cycles: float
    insulator_dielectric_dose_mgy: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "HelionMaterialState":
        return cls(**{k: float(payload[k]) for k in cls.__dataclass_fields__})

    def replace(self, **kwargs: float) -> "HelionMaterialState":
        return replace(self, **kwargs)


def fresh_pwr_material_state(operating_hours: float = 0.0) -> PWRMaterialState:
    """Brand-new PWR vessel and core; zero accumulated damage."""
    return PWRMaterialState(
        rpv_fluence_n_per_cm2=0.0,
        rpv_drtndt_k=0.0,
        cladding_oxide_thickness_um=0.0,
        cladding_hydrogen_ppm=0.0,
        control_rod_b10_depletion_frac=0.0,
        fuel_burnup_gwd_per_mtu=0.0,
        operating_hours=operating_hours,
    )


def fresh_msr_material_state(operating_hours: float = 0.0) -> MSRMaterialState:
    """Brand-new MSR primary loop; nominal salt redox, zero corrosion."""
    return MSRMaterialState(
        hastelloy_n_corrosion_um=0.0,
        tellurium_attack_depth_um=0.0,
        heat_exchanger_thinning_um=0.0,
        salt_redox_potential_v=-1.34,
        tritium_inventory_g=0.0,
        operating_hours=operating_hours,
    )


def fresh_helion_material_state() -> HelionMaterialState:
    """Brand-new Helion machine; zero pulses fired."""
    return HelionMaterialState(
        pulses_fired=0.0,
        first_wall_dpa=0.0,
        first_wall_thermal_cycles=0.0,
        coil_thermal_fatigue_index=0.0,
        capacitor_charge_cycles=0.0,
        insulator_dielectric_dose_mgy=0.0,
    )
