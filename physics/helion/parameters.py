"""Pydantic schema for Helion FRC input parameters and physical bounds."""

from pydantic import BaseModel, ConfigDict, Field

from data.bosch_hale_coeffs import DHE3_TEMPERATURE_RANGE_KEV


class HelionParameters(BaseModel):
    """Validated controls and initial conditions for the Phase 2 Helion twin."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    # Bosch-Hale D-He3 reactivity fit is only valid over 0.5-190 keV.
    ion_temperature_kev: float = Field(
        default=20.0,
        ge=DHE3_TEMPERATURE_RANGE_KEV[0],
        le=DHE3_TEMPERATURE_RANGE_KEV[1],
    )
    # FRC density range spans low demonstration plasma through aggressive compact-fusion assumptions.
    plasma_density_m3: float = Field(default=1.0e21, ge=1.0e19, le=1.0e23)
    # Compression ratio is positive and capped before spatial/magnet hardware modeling would be required.
    compression_ratio: float = Field(default=10.0, ge=1.0, le=1000.0)
    # Magnetic field range covers laboratory FRC fields without entering unsupported magnet-engineering detail.
    magnetic_field_t: float = Field(default=5.0, ge=0.1, le=100.0)
    # Plasma volume remains macroscopic but small enough for compact pulsed FRC assumptions.
    plasma_volume_m3: float = Field(default=1.0, ge=0.01, le=100.0)
    # Pulse duration is bounded to pulsed FRC timescales, not steady-state fusion plant operation.
    pulse_duration_s: float = Field(default=1.0e-5, ge=1.0e-7, le=1.0e-2)