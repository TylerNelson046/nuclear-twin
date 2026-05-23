"""Integration tests for PWR reactivity balance feeding point kinetics."""

import numpy as np
import pytest

from data.keepin_dnp import beta_fractions, decay_constants
from physics.pwr.reactivity import (
    FeedbackCoefficients,
    FeedbackState,
    ReferenceState,
    build_reactivity_fn,
    calculate_reactivity_components,
    pcm_to_dk_k,
)
from physics.shared.kinetics import (
    LAMBDA_PWR,
    build_initial_state,
    integrate_kinetics,
    point_kinetics_rhs,
)


def test_reactivity_balance_feeds_total_reactivity_into_point_kinetics() -> None:
    """Mock feedback values are summed and drive the 6-group kinetics RHS."""
    beta_arr = np.array(beta_fractions(), dtype=np.float64)
    lambda_arr = np.array(decay_constants(), dtype=np.float64)
    y0 = build_initial_state(1.0, beta_arr, lambda_arr, LAMBDA_PWR)

    feedback_state = FeedbackState(
        fuel_temperature_k=901.0,
        coolant_temperature_k=591.0,
        boron_ppm=1.0,
    )
    reference_state = ReferenceState(
        fuel_temperature_ref_k=900.0,
        coolant_temperature_ref_k=590.0,
    )
    coefficients = FeedbackCoefficients(
        alpha_doppler_pcm_per_k=-2.0,
        alpha_moderator_pcm_per_k=-30.0,
        boron_worth_pcm_per_ppm=-10.0,
    )

    components = calculate_reactivity_components(
        feedback_state,
        reference_state,
        base_reactivity_pcm=120.0,
        rod_reactivity_pcm=30.0,
        coefficients=coefficients,
    )
    assert components.doppler_pcm == pytest.approx(-2.0)
    assert components.moderator_pcm == pytest.approx(-30.0)
    assert components.boron_pcm == pytest.approx(-10.0)
    assert components.total_pcm == pytest.approx(108.0)

    rho_fn = build_reactivity_fn(
        lambda _t, _y: feedback_state,
        reference_state,
        base_reactivity_pcm=120.0,
        rod_reactivity_pcm=30.0,
        coefficients=coefficients,
    )

    rho_total = rho_fn(0.0, y0)
    assert rho_total == pytest.approx(pcm_to_dk_k(108.0))

    dydt = point_kinetics_rhs(0.0, y0, rho_total, beta_arr, lambda_arr, LAMBDA_PWR)
    assert dydt[0] == pytest.approx(rho_total / LAMBDA_PWR)
    assert dydt[0] > 0.0

    result = integrate_kinetics(
        y0,
        t_span=(0.0, 1.0),
        rho_fn=rho_fn,
        beta_arr=beta_arr,
        lambda_arr=lambda_arr,
        Lambda=LAMBDA_PWR,
    )

    assert result.success, result.message
    assert result.y[0, -1] > y0[0]
