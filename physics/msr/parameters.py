"""Pydantic schema for MSR input parameters and physical bounds."""

from pydantic import BaseModel, ConfigDict, Field


class MSRParameters(BaseModel):
    """Validated controls and initial conditions for the Phase 3 MSR twin."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    # Reactivity insertion is limited to educational transients, not accident progression.
    external_reactivity_pcm: float = Field(default=0.0, ge=-1000.0, le=1000.0)
    # Generic thermal MSR power range covers experimental scale through commercial-scale concepts.
    initial_power_mw: float = Field(default=500.0, ge=1.0, le=3000.0)
    # Salt temperature remains above typical fluoride melting points and below structural material limits.
    salt_temperature_k: float = Field(default=900.0, ge=800.0, le=1100.0)
    # Salt velocity can reach zero for the no-flow comparison and is capped before detailed pump modeling is needed.
    salt_velocity_m_s: float = Field(default=1.0, ge=0.0, le=10.0)
    # Core transit time is positive and bounded to compact circulating-fuel core geometries.
    core_transit_time_s: float = Field(default=5.0, ge=0.1, le=100.0)
    # External loop transit captures precursor decay outside the core without long-duration chemistry modeling.
    loop_transit_time_s: float = Field(default=20.0, ge=0.0, le=1000.0)
    # Salt feedback coefficient is negative for the self-regulating generic graphite-moderated MSR model.
    salt_temp_coeff_pcm_per_k: float = Field(default=-5.0, ge=-20.0, le=-0.1)