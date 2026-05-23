"""Bounds tests for reactor input parameter schemas."""

import pytest
from pydantic import ValidationError

from physics.helion.parameters import HelionParameters
from physics.msr.parameters import MSRParameters
from physics.pwr.parameters import PWRParameters


@pytest.mark.parametrize(
    ("schema", "minimums", "maximums", "below_min", "above_max"),
    [
        (
            PWRParameters,
            {
                "rod_reactivity_pcm": -1000.0,
                "boron_ppm": 0.0,
                "coolant_flow_fraction": 0.2,
                "inlet_temperature_k": 540.0,
                "initial_power_mw": 1.0,
                "fuel_temperature_k": 550.0,
                "coolant_temperature_k": 540.0,
            },
            {
                "rod_reactivity_pcm": 1000.0,
                "boron_ppm": 2500.0,
                "coolant_flow_fraction": 1.2,
                "inlet_temperature_k": 610.0,
                "initial_power_mw": 4000.0,
                "fuel_temperature_k": 1500.0,
                "coolant_temperature_k": 650.0,
            },
            ("boron_ppm", -1.0),
            ("initial_power_mw", 4000.1),
        ),
        (
            HelionParameters,
            {
                "ion_temperature_kev": 0.5,
                "plasma_density_m3": 1.0e19,
                "compression_ratio": 1.0,
                "magnetic_field_t": 0.1,
                "plasma_volume_m3": 0.01,
                "pulse_duration_s": 1.0e-7,
            },
            {
                "ion_temperature_kev": 190.0,
                "plasma_density_m3": 1.0e23,
                "compression_ratio": 1000.0,
                "magnetic_field_t": 100.0,
                "plasma_volume_m3": 100.0,
                "pulse_duration_s": 1.0e-2,
            },
            ("ion_temperature_kev", 0.49),
            ("ion_temperature_kev", 190.1),
        ),
        (
            MSRParameters,
            {
                "external_reactivity_pcm": -1000.0,
                "initial_power_mw": 1.0,
                "salt_temperature_k": 800.0,
                "salt_velocity_m_s": 0.0,
                "core_transit_time_s": 0.1,
                "loop_transit_time_s": 0.0,
                "salt_temp_coeff_pcm_per_k": -20.0,
            },
            {
                "external_reactivity_pcm": 1000.0,
                "initial_power_mw": 3000.0,
                "salt_temperature_k": 1100.0,
                "salt_velocity_m_s": 10.0,
                "core_transit_time_s": 100.0,
                "loop_transit_time_s": 1000.0,
                "salt_temp_coeff_pcm_per_k": -0.1,
            },
            ("salt_velocity_m_s", -0.1),
            ("salt_temperature_k", 1100.1),
        ),
    ],
)
def test_schema_bounds_accept_minimum_and_maximum_values(
    schema: type[PWRParameters | HelionParameters | MSRParameters],
    minimums: dict[str, float],
    maximums: dict[str, float],
    below_min: tuple[str, float],
    above_max: tuple[str, float],
) -> None:
    assert schema(**minimums)
    assert schema(**maximums)

    with pytest.raises(ValidationError):
        schema(**{**minimums, below_min[0]: below_min[1]})

    with pytest.raises(ValidationError):
        schema(**{**maximums, above_max[0]: above_max[1]})


def test_parameter_schemas_reject_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        PWRParameters(unknown_control=1.0)
