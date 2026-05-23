"""Integration tests for the closed-loop PWR kinetics/thermal-hydraulics engine."""

from __future__ import annotations

import numpy as np
import pytest

from data.keepin_dnp import beta_fractions, decay_constants
from physics.pwr.engine import (
    PWRControls,
    PWRModelConfig,
    build_initial_state,
    calculate_reactivity_snapshot,
    critical_base_reactivity_pcm,
    integrate_pwr,
    pwr_coupled_rhs,
)
from physics.pwr.reactivity import ReferenceState
from physics.pwr.thermal_hydraulics import T_COOL_NOM, T_FUEL_NOM


def _critical_initial_setup() -> tuple[
    np.ndarray,
    PWRControls,
    ReferenceState,
    PWRModelConfig,
]:
    """Build a full-power critical state with xenon held down by base reactivity."""
    reference_state = ReferenceState(
        fuel_temperature_ref_k=T_FUEL_NOM,
        coolant_temperature_ref_k=T_COOL_NOM,
    )
    controls = PWRControls(
        rod_reactivity_pcm=0.0,
        boron_ppm=0.0,
        coolant_flow_fraction=1.0,
    )
    y0 = build_initial_state()
    base_reactivity_pcm = critical_base_reactivity_pcm(
        y0,
        controls,
        reference_state,
    )
    config = PWRModelConfig(base_reactivity_pcm=base_reactivity_pcm)
    return y0, controls, reference_state, config


def test_coupled_rhs_is_zero_at_full_power_critical_state() -> None:
    """At the constructed operating point all coupled derivatives are zero."""
    y0, controls, reference_state, config = _critical_initial_setup()

    dydt = pwr_coupled_rhs(
        0.0,
        y0,
        lambda _t, _y: controls,
        reference_state,
        config,
        beta_arr=np.array(beta_fractions(), dtype=np.float64),
        lambda_arr=np.array(decay_constants(), dtype=np.float64),
    )

    assert dydt.shape == (11,)
    np.testing.assert_allclose(dydt[:9], 0.0, atol=1e-8)
    np.testing.assert_allclose(dydt[9:], 0.0, atol=1e-3)


def test_small_reactivity_insertion_is_suppressed_by_temperature_feedback() -> None:
    """A +50 pcm insertion produces a limited peak then settles under feedback."""
    y0, _controls, reference_state, base_config = _critical_initial_setup()
    transient_controls = PWRControls(
        rod_reactivity_pcm=50.0,
        boron_ppm=0.0,
        coolant_flow_fraction=1.0,
    )

    result = integrate_pwr(
        y0,
        (0.0, 200.0),
        transient_controls,
        reference_state,
        base_config,
        t_eval=np.linspace(0.0, 200.0, 401),
    )

    assert result.success, result.message

    power = result.y[0]
    fuel_temperature = result.y[7]
    coolant_temperature = result.y[8]
    final_state = result.y[:, -1]
    final_snapshot = calculate_reactivity_snapshot(
        final_state,
        transient_controls,
        reference_state,
        base_config,
    )

    assert power.max() > 1.01
    assert power[-1] < power.max() * 0.99
    assert power[-1] < 1.05

    assert fuel_temperature[-1] > T_FUEL_NOM
    assert coolant_temperature[-1] > T_COOL_NOM
    assert final_snapshot.doppler_pcm < 0.0
    assert final_snapshot.moderator_pcm < 0.0
    assert final_snapshot.total_pcm == pytest.approx(0.0, abs=5.0)
