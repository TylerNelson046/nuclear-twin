"""PWR reactivity balance integration layer.

This module composes the algebraic PWR feedback calculators into the dynamic
reactivity callable consumed by the shared 6-group point kinetics solver.

SPEC equations connected here:
    Eq. 11  Doppler feedback
    Eq. 12  Moderator feedback
    Eq. 13  Boron worth
    Eq. 1   Point kinetics driving reactivity input
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from physics.pwr.feedback import (
    ALPHA_D_DEFAULT,
    ALPHA_M_DEFAULT,
    OMEGA_B_DEFAULT,
    boron_worth,
    doppler_feedback,
    moderator_feedback,
)

PCM_TO_DK_K = 1.0e-5


@dataclass(frozen=True)
class FeedbackState:
    """Current PWR feedback variables supplied by controls and thermal hydraulics."""

    fuel_temperature_k: float
    coolant_temperature_k: float
    boron_ppm: float


@dataclass(frozen=True)
class ReferenceState:
    """Reference temperatures for zero feedback at the initial steady state."""

    fuel_temperature_ref_k: float
    coolant_temperature_ref_k: float


@dataclass(frozen=True)
class FeedbackCoefficients:
    """Linearized PWR reactivity feedback coefficients."""

    alpha_doppler_pcm_per_k: float = ALPHA_D_DEFAULT
    alpha_moderator_pcm_per_k: float = ALPHA_M_DEFAULT
    boron_worth_pcm_per_ppm: float = OMEGA_B_DEFAULT


@dataclass(frozen=True)
class ReactivityComponents:
    """PWR reactivity component breakdown in pcm."""

    base_pcm: float
    rods_pcm: float
    doppler_pcm: float
    moderator_pcm: float
    boron_pcm: float

    @property
    def total_pcm(self) -> float:
        """Return total core reactivity in pcm."""
        return (
            self.base_pcm
            + self.rods_pcm
            + self.doppler_pcm
            + self.moderator_pcm
            + self.boron_pcm
        )

    @property
    def total_dk_k(self) -> float:
        """Return total core reactivity as dimensionless delta-k/k."""
        return pcm_to_dk_k(self.total_pcm)


def pcm_to_dk_k(rho_pcm: float) -> float:
    """Convert reactivity from pcm to dimensionless delta-k/k."""
    return rho_pcm * PCM_TO_DK_K


def calculate_reactivity_components(
    feedback_state: FeedbackState,
    reference_state: ReferenceState,
    *,
    base_reactivity_pcm: float = 0.0,
    rod_reactivity_pcm: float = 0.0,
    coefficients: FeedbackCoefficients = FeedbackCoefficients(),
) -> ReactivityComponents:
    """Calculate the PWR reactivity component breakdown in pcm.

    Thermal-hydraulics can update ``feedback_state.fuel_temperature_k`` and
    ``feedback_state.coolant_temperature_k`` at each ODE call in Week 3 without
    changing the shared kinetics solver interface.
    """
    doppler_pcm = doppler_feedback(
        feedback_state.fuel_temperature_k,
        reference_state.fuel_temperature_ref_k,
        coefficients.alpha_doppler_pcm_per_k,
    )
    moderator_pcm = moderator_feedback(
        feedback_state.coolant_temperature_k,
        reference_state.coolant_temperature_ref_k,
        coefficients.alpha_moderator_pcm_per_k,
    )
    boron_pcm = boron_worth(
        feedback_state.boron_ppm,
        coefficients.boron_worth_pcm_per_ppm,
    )

    return ReactivityComponents(
        base_pcm=base_reactivity_pcm,
        rods_pcm=rod_reactivity_pcm,
        doppler_pcm=doppler_pcm,
        moderator_pcm=moderator_pcm,
        boron_pcm=boron_pcm,
    )


def calculate_total_reactivity(
    feedback_state: FeedbackState,
    reference_state: ReferenceState,
    *,
    base_reactivity_pcm: float = 0.0,
    rod_reactivity_pcm: float = 0.0,
    coefficients: FeedbackCoefficients = FeedbackCoefficients(),
) -> float:
    """Return total PWR reactivity as dimensionless delta-k/k."""
    return calculate_reactivity_components(
        feedback_state,
        reference_state,
        base_reactivity_pcm=base_reactivity_pcm,
        rod_reactivity_pcm=rod_reactivity_pcm,
        coefficients=coefficients,
    ).total_dk_k


def build_reactivity_fn(
    feedback_state_fn: Callable[[float, np.ndarray], FeedbackState],
    reference_state: ReferenceState,
    *,
    base_reactivity_pcm: float = 0.0,
    rod_reactivity_pcm: float = 0.0,
    coefficients: FeedbackCoefficients = FeedbackCoefficients(),
) -> Callable[[float, np.ndarray], float]:
    """Build the dynamic ``rho_fn(t, y)`` required by ``integrate_kinetics``.

    ``feedback_state_fn`` is the hook point for Week 3 coupling: it can read
    temperatures from the future 11-state PWR vector while the shared 7-state
    kinetics core remains unchanged.
    """

    def rho_fn(t: float, y: np.ndarray) -> float:
        feedback_state = feedback_state_fn(t, y)
        return calculate_total_reactivity(
            feedback_state,
            reference_state,
            base_reactivity_pcm=base_reactivity_pcm,
            rod_reactivity_pcm=rod_reactivity_pcm,
            coefficients=coefficients,
        )

    return rho_fn
