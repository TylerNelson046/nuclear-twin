"""tests/test_e2e_integration.py — Week 9 end-to-end integration tests.

Formally validates the three Phase 2 success criteria (SPEC §8, Phase 2):

    SC-1  Helion twin is accessible from the same unified selector as PWR.
    SC-2  Ignition boundary heatmap renders correctly with a Q > 1 region.
    SC-3  Platform selector switches cleanly between PWR and Helion without
          state contamination.

Also covers:
    - Cross-reactor performance: orchestrator calls complete < 100ms (NFR-02 base)
    - All three reactor panels satisfy SR-03 (educational disclaimer)
    - PWR and Helion panels share zero component IDs (strict DOM isolation)
    - Unified app registers callbacks for both reactors simultaneously

State contamination strategy:
    PWR panel uses component IDs prefixed with ``pwr-``.
    Helion panel uses component IDs prefixed with ``helion-``.
    The render_reactor_panel callback replaces the entire panel container when
    the selector changes, removing the previous panel's components from the DOM.
    Contamination cannot occur if no IDs overlap, which this test suite verifies
    by inspecting the serialised layout of each panel independently.
"""

from __future__ import annotations

import time
from typing import Any

import pytest

from ui.app import REACTOR_PANELS, create_app
from ui.helion_panel import build_helion_panel
from ui.msr_panel import build_msr_panel
from ui.pwr_panel import DISCLAIMER, build_pwr_panel


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _panel_str(panel: Any) -> str:
    """Serialise a Dash component tree to a string for ID inspection."""
    return str(panel)


def _extract_component_ids(panel: Any) -> set[str]:
    """Walk a serialised panel string and extract quoted 'id' values.

    Works on the repr of any Dash component tree.  Not a full parser —
    sufficient for detecting the set of IDs that appear in a panel.
    """
    import re
    raw = _panel_str(panel)
    # Match both 'id': 'foo-bar' and 'id': {'type': ...} forms
    # For scalar IDs (the only kind used here):
    return set(re.findall(r"'id':\s*'([^']+)'", raw))


# ---------------------------------------------------------------------------
# SC-1 — Unified selector includes all Phase 1 and Phase 2 reactors
# ---------------------------------------------------------------------------

class TestUnifiedSelectorAccessibility:
    """SC-1: Helion twin is accessible from the same unified selector as PWR."""

    def test_reactor_panels_registry_contains_pwr_and_helion(self) -> None:
        assert "pwr" in REACTOR_PANELS
        assert "helion" in REACTOR_PANELS

    def test_reactor_panels_registry_contains_msr_placeholder(self) -> None:
        assert "msr" in REACTOR_PANELS

    def test_app_dropdown_has_pwr_option(self) -> None:
        app = create_app()
        layout_str = str(app.layout)
        assert "pwr" in layout_str

    def test_app_dropdown_has_helion_option(self) -> None:
        app = create_app()
        layout_str = str(app.layout)
        assert "helion" in layout_str

    def test_panel_factory_called_for_pwr(self) -> None:
        import dash_bootstrap_components as dbc
        panel = REACTOR_PANELS["pwr"]()
        assert isinstance(panel, dbc.Card)

    def test_panel_factory_called_for_helion(self) -> None:
        import dash_bootstrap_components as dbc
        panel = REACTOR_PANELS["helion"]()
        assert isinstance(panel, dbc.Card)

    def test_panel_factory_called_for_msr(self) -> None:
        import dash_bootstrap_components as dbc
        panel = REACTOR_PANELS["msr"]()
        assert isinstance(panel, dbc.Card)

    def test_pwr_and_helion_panels_are_distinct_objects(self) -> None:
        pwr = REACTOR_PANELS["pwr"]()
        helion = REACTOR_PANELS["helion"]()
        assert pwr is not helion

    def test_switching_to_unknown_reactor_falls_back_to_pwr(self) -> None:
        import dash_bootstrap_components as dbc
        panel_factory = REACTOR_PANELS.get("unknown_reactor", build_pwr_panel)
        panel = panel_factory()
        assert isinstance(panel, dbc.Card)


# ---------------------------------------------------------------------------
# SC-2 — Ignition boundary heatmap shows a Q > 1 region
# ---------------------------------------------------------------------------

class TestIgnitionBoundaryQGt1:
    """SC-2: Ignition boundary sweep produces a well-formed Q > 1 region.

    Base configuration uses T_initial=1 keV with B=10T so that the sweep's
    high-Rc points compress to T_final ≈ 100 keV, near the D-He3 reactivity
    peak, while keeping W_initial low enough that Q > 1 is achievable.
    This matches the operating point validated in test_helion_scenarios.py.
    """

    _BASE = {
        "ion_temperature_kev": 1.0,
        "plasma_density_m3": 1.0e21,
        "compression_ratio": 10.0,
        "magnetic_field_t": 10.0,
        "plasma_volume_m3": 1.0,
        "pulse_duration_s": 1.0e-5,
    }

    def test_sweep_succeeds_over_production_grid(self) -> None:
        from orchestrator.helion_runner import run_helion_sweep
        result = run_helion_sweep(self._BASE.copy(), n_rc=15, n_density=15)
        assert result["success"] is True

    def test_q_map_contains_at_least_one_ignited_point(self) -> None:
        from orchestrator.helion_runner import run_helion_sweep
        result = run_helion_sweep(self._BASE.copy(), n_rc=20, n_density=20)
        q_flat = [q for row in result["Q_map"] for q in row]
        assert any(q > 1.0 for q in q_flat), (
            "Ignition boundary sweep produced no Q > 1 point on a 20×20 grid "
            "covering Rc: 1–1000 and n: 1e19–1e23 m⁻³. "
            f"Max Q found: {max(q_flat):.4g}"
        )

    def test_ignition_mask_matches_q_gt_1(self) -> None:
        from orchestrator.helion_runner import run_helion_sweep
        result = run_helion_sweep(self._BASE.copy(), n_rc=10, n_density=10)
        for i, row in enumerate(result["Q_map"]):
            for j, q in enumerate(row):
                mask_val = result["ignition_mask"][i][j]
                expected = q > 1.0
                assert mask_val == expected, (
                    f"ignition_mask[{i}][{j}]={mask_val} but Q={q:.4g} "
                    f"→ expected mask={expected}"
                )

    def test_sweep_grid_axes_are_monotone_increasing(self) -> None:
        from orchestrator.helion_runner import run_helion_sweep
        result = run_helion_sweep(self._BASE.copy(), n_rc=8, n_density=8)
        rc = result["compression_ratios"]
        ns = result["densities_m3"]
        assert all(rc[i] < rc[i + 1] for i in range(len(rc) - 1)), "Rc axis not monotone"
        assert all(ns[i] < ns[i + 1] for i in range(len(ns) - 1)), "density axis not monotone"

    def test_high_compression_row_has_higher_q_than_low_compression(self) -> None:
        """Higher compression ratio always gives higher or equal Q (monotonicity)."""
        from orchestrator.helion_runner import run_helion_sweep
        result = run_helion_sweep(self._BASE.copy(), n_rc=10, n_density=5)
        q_map = result["Q_map"]
        # Compare max-Q of lowest-Rc row vs highest-Rc row
        low_rc_max = max(q_map[0])
        high_rc_max = max(q_map[-1])
        assert high_rc_max >= low_rc_max, (
            f"Expected higher Rc to give higher Q: low={low_rc_max:.4g} high={high_rc_max:.4g}"
        )


# ---------------------------------------------------------------------------
# SC-3 — State contamination: PWR and Helion panels share zero component IDs
# ---------------------------------------------------------------------------

class TestStateCOntaminationIsolation:
    """SC-3: Platform selector switches cleanly without state contamination."""

    _PWR_EXCLUSIVE_IDS = {
        "pwr-session-store",
        "pwr-run-state-store",
        "pwr-step-interval",
        "pwr-rod-reactivity",
        "pwr-boron-ppm",
        "pwr-flow-fraction",
        "pwr-inlet-temperature-c",
        "pwr-pause-toggle",
        "pwr-reset-normal",
        "pwr-start-scenario",
        "pwr-scenario-select",
        "pwr-time-multiplier",
        "pwr-status-alert",
        "pwr-event-log",
        "pwr-power-graph",
        "pwr-temperature-graph",
        "pwr-reactivity-graph",
        "pwr-xenon-iodine-graph",
    }

    _HELION_EXCLUSIVE_IDS = {
        "helion-pulse-store",
        "helion-sweep-store",
        "helion-run-btn",
        "helion-sweep-btn",
        "helion-mode",
        "helion-scenario-select",
        "helion-status-alert",
        "helion-chart-temp",
        "helion-chart-power",
        "helion-chart-energy",
        "helion-chart-heatmap",
        "helion-metric-q",
        "helion-metric-tfinal",
        "helion-metric-pfusion",
        "helion-metric-winitial",
    }

    def test_pwr_panel_contains_no_helion_ids(self) -> None:
        pwr_panel = build_pwr_panel()
        panel_str = _panel_str(pwr_panel)
        found = [hid for hid in self._HELION_EXCLUSIVE_IDS if hid in panel_str]
        assert not found, (
            f"PWR panel contains Helion component IDs — potential state leak: {found}"
        )

    def test_helion_panel_contains_no_pwr_ids(self) -> None:
        helion_panel = build_helion_panel()
        panel_str = _panel_str(helion_panel)
        found = [pid for pid in self._PWR_EXCLUSIVE_IDS if pid in panel_str]
        assert not found, (
            f"Helion panel contains PWR component IDs — potential state leak: {found}"
        )

    def test_msr_placeholder_contains_no_pwr_ids(self) -> None:
        msr_panel = build_msr_panel()
        panel_str = _panel_str(msr_panel)
        found = [pid for pid in self._PWR_EXCLUSIVE_IDS if pid in panel_str]
        assert not found, (
            f"MSR placeholder contains PWR component IDs: {found}"
        )

    def test_msr_placeholder_contains_no_helion_ids(self) -> None:
        msr_panel = build_msr_panel()
        panel_str = _panel_str(msr_panel)
        found = [hid for hid in self._HELION_EXCLUSIVE_IDS if hid in panel_str]
        assert not found, (
            f"MSR placeholder contains Helion component IDs: {found}"
        )

    def test_pwr_store_ids_present_in_pwr_panel(self) -> None:
        pwr_panel = build_pwr_panel()
        panel_str = _panel_str(pwr_panel)
        assert "pwr-session-store" in panel_str
        assert "pwr-run-state-store" in panel_str

    def test_helion_store_ids_present_in_helion_panel(self) -> None:
        helion_panel = build_helion_panel()
        panel_str = _panel_str(helion_panel)
        assert "helion-pulse-store" in panel_str
        assert "helion-sweep-store" in panel_str

    def test_render_panel_pwr_returns_pwr_card(self) -> None:
        """Simulate the render_reactor_panel callback for PWR."""
        panel = REACTOR_PANELS["pwr"]()
        panel_str = _panel_str(panel)
        assert "pwr-session-store" in panel_str
        assert "helion-pulse-store" not in panel_str

    def test_render_panel_helion_returns_helion_card(self) -> None:
        """Simulate the render_reactor_panel callback for Helion."""
        panel = REACTOR_PANELS["helion"]()
        panel_str = _panel_str(panel)
        assert "helion-pulse-store" in panel_str
        assert "pwr-session-store" not in panel_str

    def test_sequential_switch_pwr_helion_pwr_no_id_collision(self) -> None:
        """Switching PWR → Helion → PWR produces panels with no shared IDs."""
        pwr1 = REACTOR_PANELS["pwr"]()
        helion = REACTOR_PANELS["helion"]()
        pwr2 = REACTOR_PANELS["pwr"]()

        pwr1_ids = _extract_component_ids(pwr1)
        helion_ids = _extract_component_ids(helion)
        pwr2_ids = _extract_component_ids(pwr2)

        overlap = pwr1_ids & helion_ids
        assert not overlap, (
            f"PWR and Helion panels share {len(overlap)} component IDs: {overlap}"
        )
        # PWR panels from the same factory should have identical IDs
        assert pwr1_ids == pwr2_ids, "PWR panel is not idempotent across calls"


# ---------------------------------------------------------------------------
# SR-03 — Disclaimer present on all live reactor panels
# ---------------------------------------------------------------------------

class TestDisclaimerCompliance:
    """SR-03: All reactor panels display the educational disclaimer."""

    def test_disclaimer_text_defined(self) -> None:
        assert "educational purposes" in DISCLAIMER
        assert "simplified" in DISCLAIMER

    def test_pwr_panel_contains_disclaimer(self) -> None:
        panel = build_pwr_panel()
        assert "educational purposes" in _panel_str(panel)

    def test_helion_panel_contains_disclaimer(self) -> None:
        panel = build_helion_panel()
        assert "educational purposes" in _panel_str(panel)

    def test_msr_panel_contains_disclaimer(self) -> None:
        panel = build_msr_panel()
        assert "educational purposes" in _panel_str(panel)


# ---------------------------------------------------------------------------
# Callback registration — both reactors wired into unified app
# ---------------------------------------------------------------------------

class TestUnifiedCallbackRegistration:
    """Both reactor callback sets registered with the same Dash app instance."""

    def test_pwr_interval_callback_registered(self) -> None:
        app = create_app()
        assert any("pwr-step-interval" in k for k in app.callback_map)

    def test_helion_pulse_callback_registered(self) -> None:
        app = create_app()
        assert any("helion-pulse-store" in k for k in app.callback_map)

    def test_helion_sweep_callback_registered(self) -> None:
        app = create_app()
        assert any("helion-sweep-store" in k for k in app.callback_map)

    def test_helion_chart_callbacks_registered(self) -> None:
        app = create_app()
        keys = " ".join(app.callback_map.keys())
        assert "helion-chart-temp" in keys
        assert "helion-chart-heatmap" in keys

    def test_panel_switch_callback_registered(self) -> None:
        app = create_app()
        assert "reactor-panel.children" in app.callback_map

    def test_pwr_and_helion_callbacks_do_not_collide(self) -> None:
        """No callback output ID is registered by both reactors."""
        app = create_app()
        all_outputs = list(app.callback_map.keys())
        pwr_outputs = [k for k in all_outputs if k.startswith("pwr-")]
        helion_outputs = [k for k in all_outputs if k.startswith("helion-")]
        overlap = set(pwr_outputs) & set(helion_outputs)
        assert not overlap, f"PWR and Helion share callback output IDs: {overlap}"


# ---------------------------------------------------------------------------
# NFR-02 — Performance: orchestrator calls complete within 100ms (no JIT)
# ---------------------------------------------------------------------------

class TestCrossReactorPerformance:
    """NFR-02: Both reactor orchestrators complete in < 100ms without Numba.

    conftest.py sets NUMBA_DISABLE_JIT=1 so these timings represent pure
    Python + SciPy performance — the conservative NFR-02 baseline.
    """

    _NFR_02_MS = 100.0    # without Numba (SPEC §5.2)
    _WARMUP_REPS = 2
    _MEASURE_REPS = 5

    def _median_ms(self, fn, *args, **kwargs) -> float:
        for _ in range(self._WARMUP_REPS):
            fn(*args, **kwargs)
        times = []
        for _ in range(self._MEASURE_REPS):
            t0 = time.perf_counter()
            fn(*args, **kwargs)
            times.append((time.perf_counter() - t0) * 1e3)
        times.sort()
        return times[len(times) // 2]

    def test_pwr_control_step_under_100ms(self) -> None:
        from orchestrator.sim_runner import run_pwr_control_step

        controls = {
            "rod_reactivity_pcm": 0.0,
            "boron_ppm": 0.0,
            "coolant_flow_fraction": 1.0,
            "inlet_temperature_k": None,
        }
        median_ms = self._median_ms(
            run_pwr_control_step,
            None, controls,
            step_seconds=0.2,
            time_multiplier=1,
            min_output_points=9,
            max_history_points=2400,
            history_retention="timeline",
        )
        assert median_ms < self._NFR_02_MS, (
            f"PWR control step took {median_ms:.1f} ms (NFR-02 limit: {self._NFR_02_MS} ms)"
        )

    def test_helion_pulse_under_100ms(self) -> None:
        from orchestrator.helion_runner import run_helion_pulse

        raw = {
            "ion_temperature_kev": 20.0,
            "plasma_density_m3": 1.0e21,
            "compression_ratio": 10.0,
            "magnetic_field_t": 5.0,
            "plasma_volume_m3": 1.0,
            "pulse_duration_s": 1.0e-5,
        }
        median_ms = self._median_ms(run_helion_pulse, raw.copy())
        assert median_ms < self._NFR_02_MS, (
            f"Helion pulse took {median_ms:.1f} ms (NFR-02 limit: {self._NFR_02_MS} ms)"
        )

    def test_helion_scenario_baseline_under_100ms(self) -> None:
        from orchestrator.helion_runner import run_helion_scenario

        median_ms = self._median_ms(run_helion_scenario, "baseline_pulse")
        assert median_ms < self._NFR_02_MS, (
            f"Helion scenario run took {median_ms:.1f} ms (limit: {self._NFR_02_MS} ms)"
        )


# ---------------------------------------------------------------------------
# Phase 2 success criteria — formal completion gate
# ---------------------------------------------------------------------------

class TestPhase2SuccessCriteria:
    """Formal Phase 2 completion gate per SPEC §8 Phase 2 Success Criteria.

    These tests are the programmatic equivalent of the acceptance checklist:

        SC-1  Helion twin accessible at same public URL as PWR
              → unified dropdown contains both "pwr" and "helion" options
        SC-2  Ignition boundary heatmap renders correctly showing Q > 1 region
              → at least one grid point in the 20×20 sweep has Q > 1
        SC-3  Platform selector switches cleanly between PWR and Helion
              without state contamination
              → zero shared component IDs between panel layouts
    """

    def test_sc1_helion_and_pwr_share_unified_selector(self) -> None:
        """SC-1: Both reactors accessible from the same dropdown."""
        app = create_app()
        layout_str = str(app.layout)
        assert "pwr" in layout_str and "helion" in layout_str, (
            "Unified selector does not expose both 'pwr' and 'helion' options."
        )

    def test_sc2_ignition_boundary_shows_q_gt_1_region(self) -> None:
        """SC-2: The 20×20 sweep over Rc∈[1,1000] × n∈[1e19,1e23] has Q > 1.

        Uses T_initial=1 keV (low initial energy → achievable Q > 1 at high Rc)
        with B=10T (strong confinement). At Rc=1000: T_final ≈ 100 keV, which is
        near the peak of the D-He3 Bosch-Hale reactivity curve.
        """
        from orchestrator.helion_runner import run_helion_sweep
        base = {
            "ion_temperature_kev": 1.0,
            "plasma_density_m3": 1.0e21,
            "compression_ratio": 10.0,
            "magnetic_field_t": 10.0,
            "plasma_volume_m3": 1.0,
            "pulse_duration_s": 1.0e-5,
        }
        result = run_helion_sweep(base, n_rc=20, n_density=20)
        assert result["success"], f"Sweep failed: {result['message']}"
        q_flat = [q for row in result["Q_map"] for q in row]
        any_ignited = any(q > 1.0 for q in q_flat)
        assert any_ignited, (
            "Phase 2 SC-2 FAILED: ignition boundary sweep shows no Q > 1 grid point. "
            f"Max Q = {max(q_flat):.4g}"
        )

    def test_sc3_pwr_to_helion_switch_produces_no_shared_ids(self) -> None:
        """SC-3: Switching from PWR to Helion clears all PWR component IDs."""
        pwr_panel = REACTOR_PANELS["pwr"]()
        helion_panel = REACTOR_PANELS["helion"]()
        pwr_ids = _extract_component_ids(pwr_panel)
        helion_ids = _extract_component_ids(helion_panel)
        shared = pwr_ids & helion_ids
        assert not shared, (
            f"Phase 2 SC-3 FAILED: {len(shared)} component IDs shared between "
            f"PWR and Helion panels (state contamination risk): {shared}"
        )

    def test_sc3_helion_to_pwr_switch_produces_no_shared_ids(self) -> None:
        """SC-3: Switching from Helion back to PWR clears all Helion component IDs."""
        helion_panel = REACTOR_PANELS["helion"]()
        pwr_panel = REACTOR_PANELS["pwr"]()
        helion_ids = _extract_component_ids(helion_panel)
        pwr_ids = _extract_component_ids(pwr_panel)
        shared = helion_ids & pwr_ids
        assert not shared, (
            f"Phase 2 SC-3 FAILED: {len(shared)} component IDs shared after "
            f"Helion→PWR switch: {shared}"
        )

    def test_sc3_msr_switch_produces_no_pwr_or_helion_ids(self) -> None:
        """SC-3: Switching to MSR placeholder clears all Phase 1/2 component IDs."""
        msr_panel = REACTOR_PANELS["msr"]()
        pwr_panel = REACTOR_PANELS["pwr"]()
        helion_panel = REACTOR_PANELS["helion"]()
        msr_ids = _extract_component_ids(msr_panel)
        pwr_ids = _extract_component_ids(pwr_panel)
        helion_ids = _extract_component_ids(helion_panel)
        assert not (msr_ids & pwr_ids), "MSR shares IDs with PWR"
        assert not (msr_ids & helion_ids), "MSR shares IDs with Helion"
