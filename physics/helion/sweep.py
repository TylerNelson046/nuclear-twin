"""2D ignition boundary sweep for the Helion FRC twin.

Week 7 — Helion Scenarios + Parameter Sweep.

Sweeps a 2D grid of compression ratio × plasma density, integrating one
Helion pulse at each grid point and storing the Q factor. The resulting
Q_map is the primary data source for the ignition boundary heatmap (Week 8).

Architecture: pure physics — no Dash, UI, or orchestrator imports.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from pydantic import ValidationError

from physics.helion.engine import integrate_helion_pulse
from physics.helion.parameters import HelionParameters

logger = logging.getLogger(__name__)


@dataclass
class IgnitionBoundaryResult:
    """Results of a 2D ignition boundary sweep over compression ratio × density."""

    compression_ratios: np.ndarray   # shape (N_C,), input axis (pre-compression Rᶜ)
    densities_m3: np.ndarray         # shape (N_D,), input axis (pre-compression n, m⁻³)
    Q_map: np.ndarray                # shape (N_C, N_D); Q_map[i, j] for (RC[i], n[j])
    ignition_mask: np.ndarray        # bool, shape (N_C, N_D); True where Q > 1
    n_failed: int                    # grid points where ODE failed or Pydantic rejected


def ignition_boundary_sweep(
    compression_ratios: np.ndarray,
    densities_m3: np.ndarray,
    base_params: HelionParameters,
    P_heat_w: float = 0.0,
    n_t_points: int = 100,
    *,
    c_tau_scale: float = 1.0,
) -> IgnitionBoundaryResult:
    """Run a 2D ignition boundary sweep across compression ratio × plasma density.

    For each (Rᶜ, n) grid point a new HelionParameters is constructed by
    overriding `compression_ratio` and `plasma_density_m3` on base_params while
    keeping all other fields fixed (temperature, magnetic field, volume, duration).
    Grid points that fall outside Pydantic bounds are skipped with Q = 0.

    Args:
        compression_ratios: 1D array of Rᶜ values. Each must be in [1, 1000].
        densities_m3:        1D array of pre-compression ion densities (m⁻³).
                             Each must be in [1e19, 1e23].
        base_params:         HelionParameters providing fixed inputs. Its own
                             compression_ratio and plasma_density_m3 are ignored
                             and replaced by the sweep axes.
        P_heat_w:            Constant external heating power (W). Default 0.
        n_t_points:          ODE output resolution per pulse. Use lower values
                             (50–100) for fast sweeps; higher (200+) for accuracy.

    Returns:
        IgnitionBoundaryResult with Q_map[i, j] for grid point
        (compression_ratios[i], densities_m3[j]).
    """
    N_C = len(compression_ratios)
    N_D = len(densities_m3)
    Q_map = np.zeros((N_C, N_D), dtype=np.float64)
    n_failed = 0

    for i, rc in enumerate(compression_ratios):
        for j, n_m3 in enumerate(densities_m3):
            try:
                params = HelionParameters(
                    ion_temperature_kev=base_params.ion_temperature_kev,
                    plasma_density_m3=float(n_m3),
                    compression_ratio=float(rc),
                    magnetic_field_t=base_params.magnetic_field_t,
                    plasma_volume_m3=base_params.plasma_volume_m3,
                    pulse_duration_s=base_params.pulse_duration_s,
                )
            except ValidationError:
                # Grid point outside Pydantic physical bounds — skip silently
                n_failed += 1
                continue

            result = integrate_helion_pulse(
                params, P_heat_w=P_heat_w, n_t_points=n_t_points, c_tau_scale=c_tau_scale,
            )
            if result.success:
                Q_map[i, j] = result.Q
            else:
                logger.warning(
                    "Sweep point (Rᶜ=%.2g, n=%.2e m⁻³) ODE failed: %s",
                    rc, n_m3, result.message,
                )
                n_failed += 1

    return IgnitionBoundaryResult(
        compression_ratios=np.asarray(compression_ratios, dtype=np.float64),
        densities_m3=np.asarray(densities_m3, dtype=np.float64),
        Q_map=Q_map,
        ignition_mask=Q_map > 1.0,
        n_failed=n_failed,
    )
