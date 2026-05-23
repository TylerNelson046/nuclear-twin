"""Pydantic schema for PWR input parameters and physical bounds."""

from pydantic import BaseModel, ConfigDict, Field


class PWRParameters(BaseModel):
    """Validated controls and initial conditions for the Phase 1 PWR twin."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    # Rod worth in the simplified transient UI is limited to small reactivity insertions, not accident-scale analysis.
    rod_reactivity_pcm: float = Field(default=0.0, ge=-1000.0, le=1000.0)
    # Commercial PWR soluble boron concentrations are nonnegative and normally below a few thousand ppm.
    boron_ppm: float = Field(default=1000.0, ge=0.0, le=2500.0)
    # Flow fraction keeps the lumped coolant model near nominal forced-circulation operation.
    coolant_flow_fraction: float = Field(default=1.0, ge=0.2, le=1.2)
    # Inlet temperature is bounded below cold shutdown and below PWR saturation-limited operating temperatures.
    inlet_temperature_k: float = Field(default=565.0, ge=540.0, le=610.0)
    # Educational full-power range covers startup through generic large commercial PWR thermal power.
    initial_power_mw: float = Field(default=3000.0, ge=1.0, le=4000.0)
    # Fuel temperature remains below simplified ceramic fuel safety margins and above hot operating coolant.
    fuel_temperature_k: float = Field(default=900.0, ge=550.0, le=1500.0)
    # Coolant temperature is bounded around single-phase pressurized-water operating conditions.
    coolant_temperature_k: float = Field(default=590.0, ge=540.0, le=650.0)