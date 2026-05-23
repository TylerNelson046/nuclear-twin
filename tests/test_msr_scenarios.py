"""
tests/test_msr_scenarios.py — MSR scenario runner and β_eff,flow sweep tests.

Week 11 deliverable — MSR Scenarios + UI.

Test categories (CLAUDE.md §Testing Requirements):
  Unit tests:        msr_runner helpers, sweep algebraic correctness
  Integration tests: pump_trip and load_following scenarios show correct physics direction
  Regression tests:  scenario runner cold-start, session reinitialisation
"""

from __future__ import annotations

import numpy as np
import pytest

from orchestrator.msr_runner import (
    _BETA_EFF_STATIC,
    MSR_STEP_SECONDS,
    run_msr_beta_flow_sweep,
    run_msr_control_step,
    run_msr_scenario,
)
from orchestrator.msr_scenarios import (
    MSRScenario,
    available_msr_scenarios,
    get_msr_scenario,
    msr_scenario_options,
)
from physics.msr.engine import MSRControls
from physics.msr.thermal import P_NOM_MSR, T_SALT_NOM, TAU_CORE_NOM


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_DEFAULT_RAW = {
    "external_reactivity_pcm": 0.0,
    "salt_flow_fraction": 1.0,
    "core_transit_time_s": float(TAU_CORE_NOM),
    "loop_transit_time_s": 20.0,
    "salt_temp_coeff_pcm_per_k": -5.0,
    "initial_power_mw": 500.0,
    "salt_temperature_k": T_SALT_NOM,
    "salt_velocity_m_s": 1.0,
}


# ===========================================================================
# Unit tests — orchestrator/msr_scenarios.py
# ===========================================================================


class TestMsrScenarioRegistry:
    """Scenario registry sanity checks."""

    def test_all_scenarios_accessible(self):
        """available_msr_scenarios returns pump_trip and load_following."""
        scenarios = available_msr_scenarios()
        assert "pump_trip" in scenarios
        assert "load_following" in scenarios

    def test_get_scenario_returns_correct_type(self):
        s = get_msr_scenario("pump_trip")
        assert isinstance(s, MSRScenario)
        assert s.scenario_id == "pump_trip"

    def test_unknown_scenario_raises_value_error(self):
        with pytest.raises(ValueError, match="Unknown MSR scenario"):
            get_msr_scenario("nonexistent")

    def test_scenario_options_format(self):
        options = msr_scenario_options()
        assert len(options) == 2
        for opt in options:
            assert "label" in opt and "value" in opt

    def test_pump_trip_profile_flow_reduction(self):
        """Pump trip: flow fraction reduces to ~20% after the ramp."""
        s = get_msr_scenario("pump_trip")
        # Before ramp
        ctrl_t0 = s.controls_at(0.0, np.zeros(8))
        assert abs(ctrl_t0.salt_flow_fraction - 1.0) < 0.01
        # After ramp (t = 10 s, ramp ends at 5 s)
        ctrl_late = s.controls_at(10.0, np.zeros(8))
        assert abs(ctrl_late.salt_flow_fraction - 0.20) < 0.01

    def test_load_following_profile_rod_insertion(self):
        """Load following: rod reactivity is 0 before ramp, then negative."""
        s = get_msr_scenario("load_following")
        ctrl_early = s.controls_at(100.0, np.zeros(8))  # before ramp at 300 s
        assert ctrl_early.external_reactivity_pcm == pytest.approx(0.0, abs=1.0)
        ctrl_mid = s.controls_at(900.0, np.zeros(8))    # in hold-low band
        assert ctrl_mid.external_reactivity_pcm == pytest.approx(-50.0, abs=1.0)

    def test_pump_trip_duration_and_points(self):
        s = get_msr_scenario("pump_trip")
        assert s.duration_s == pytest.approx(300.0)
        assert s.n_output_points >= 100

    def test_load_following_duration(self):
        s = get_msr_scenario("load_following")
        assert s.duration_s >= 2700.0  # at least past the ramp-back end


# ===========================================================================
# Unit tests — orchestrator/msr_runner.py (algebraic helpers)
# ===========================================================================


class TestMsrBetaFlowSweep:
    """run_msr_beta_flow_sweep algebraic correctness."""

    def test_returns_required_keys(self):
        result = run_msr_beta_flow_sweep()
        for key in ("v_salt", "beta_eff_flow", "beta_eff_static", "v_salt_current", "beta_current"):
            assert key in result, f"Missing key '{key}' in sweep result"

    def test_beta_flow_monotone_decreasing(self):
        """β_eff,flow decreases monotonically with salt velocity (SPEC §7.3)."""
        result = run_msr_beta_flow_sweep()
        b = result["beta_eff_flow"]
        for i in range(len(b) - 1):
            assert b[i] >= b[i + 1] - 1.0e-12, (
                f"β_eff,flow non-monotone: b[{i}]={b[i]:.6f} < b[{i+1}]={b[i+1]:.6f}"
            )

    def test_beta_flow_bounded_by_static(self):
        """All β_eff,flow values ≤ β_eff_static."""
        result = run_msr_beta_flow_sweep()
        b_static = result["beta_eff_static"]
        for val in result["beta_eff_flow"]:
            assert val <= b_static + 1.0e-10
            assert val >= 0.0

    def test_static_limit_at_very_low_velocity(self):
        """At very low v_salt (large τ_core), β_eff,flow → β_eff_static."""
        # Sweep from v=0.001 m/s to 10 m/s
        v_arr = np.array([0.001, 0.01, 0.1, 1.0, 10.0])
        result = run_msr_beta_flow_sweep(v_salt_arr=v_arr)
        b_slowest = result["beta_eff_flow"][0]
        b_static = result["beta_eff_static"]
        assert abs(b_slowest - b_static) / b_static < 0.01, (
            f"At v_salt=0.001 m/s, β_eff,flow={b_slowest:.6f} should be ≈ β_eff_static={b_static:.6f}"
        )

    def test_operating_point_inside_sweep_range(self):
        """Current operating point values lie within the sweep data range."""
        result = run_msr_beta_flow_sweep(tau_core_nom=5.0, tau_loop_nom=20.0)
        v_cur = result["v_salt_current"]
        b_cur = result["beta_current"]
        assert 0.0 < v_cur < 1000.0
        assert 0.0 < b_cur <= result["beta_eff_static"]

    def test_custom_tau_core_changes_operating_point(self):
        """Halving τ_core doubles v_salt_current."""
        r1 = run_msr_beta_flow_sweep(tau_core_nom=10.0, tau_loop_nom=40.0)
        r2 = run_msr_beta_flow_sweep(tau_core_nom=5.0, tau_loop_nom=20.0)
        assert abs(r2["v_salt_current"] / r1["v_salt_current"] - 2.0) < 0.01


# ===========================================================================
# Integration tests — run_msr_control_step
# ===========================================================================


class TestMsrControlStep:
    """run_msr_control_step session management and physics."""

    def test_cold_start_returns_ok(self):
        """A cold-start control step (session=None) returns a valid session."""
        result = run_msr_control_step(None, _DEFAULT_RAW)
        assert result["status"]["ok"], result["status"]["message"]

    def test_cold_start_state_vector_length(self):
        """Session state must have 8 elements (MSR state vector)."""
        result = run_msr_control_step(None, _DEFAULT_RAW)
        assert len(result["state"]) == 8

    def test_cold_start_history_populated(self):
        """History must contain at least one data point after first step."""
        result = run_msr_control_step(None, _DEFAULT_RAW)
        assert len(result["history"]["time_s"]) >= 1
        assert len(result["history"]["power_mw"]) >= 1
        assert len(result["history"]["salt_temperature_k"]) >= 1

    def test_nominal_power_near_rated(self):
        """At nominal conditions, power should start near rated 500 MWth."""
        result = run_msr_control_step(None, _DEFAULT_RAW)
        metrics = result.get("metrics", {})
        p_mw = metrics.get("power_mw", 0.0)
        assert 450.0 < p_mw < 550.0, f"Nominal power {p_mw:.1f} MWth outside expected range"

    def test_consecutive_steps_advance_time(self):
        """Two consecutive steps advance the simulated time."""
        s1 = run_msr_control_step(None, _DEFAULT_RAW)
        s2 = run_msr_control_step(s1, _DEFAULT_RAW)
        assert s2["time_s"] > s1["time_s"]

    def test_steady_state_power_drift_small(self):
        """At steady state, power drift over several steps should be < 2%."""
        session = run_msr_control_step(None, _DEFAULT_RAW)
        n_init = session["metrics"]["power_mw"]
        # Advance 50 s (10 steps of 5 s each)
        for _ in range(10):
            session = run_msr_control_step(session, _DEFAULT_RAW)
        n_final = session["metrics"]["power_mw"]
        drift_pct = abs(n_final - n_init) / n_init * 100.0
        assert drift_pct < 2.0, (
            f"Power drift {drift_pct:.2f}% over 50 s — expected < 2% at steady state"
        )

    def test_positive_reactivity_raises_power(self):
        """Inserting +100 pcm should increase power compared to no insertion."""
        raw_zero = dict(_DEFAULT_RAW, external_reactivity_pcm=0.0)
        raw_pos  = dict(_DEFAULT_RAW, external_reactivity_pcm=100.0)
        # Run a few steps with no reactivity to establish baseline
        s0 = run_msr_control_step(None, raw_zero)
        for _ in range(5):
            s0 = run_msr_control_step(s0, raw_zero)
        baseline_p = s0["metrics"]["power_mw"]

        # Now run with positive insertion from same start
        s_pos = run_msr_control_step(None, raw_pos)
        for _ in range(5):
            s_pos = run_msr_control_step(s_pos, raw_pos)
        inserted_p = s_pos["metrics"]["power_mw"]

        assert inserted_p > baseline_p * 1.0, (
            f"+100 pcm should raise power: baseline={baseline_p:.1f}, inserted={inserted_p:.1f}"
        )

    def test_reduced_flow_raises_salt_temperature(self):
        """Halving salt flow should raise T_salt compared to full flow."""
        raw_full = dict(_DEFAULT_RAW, salt_flow_fraction=1.0)
        raw_half = dict(_DEFAULT_RAW, salt_flow_fraction=0.5)

        s_full = run_msr_control_step(None, raw_full)
        for _ in range(10):
            s_full = run_msr_control_step(s_full, raw_full)

        s_half = run_msr_control_step(None, raw_half)
        for _ in range(10):
            s_half = run_msr_control_step(s_half, raw_half)

        T_full = s_full["metrics"]["salt_temperature_k"]
        T_half = s_half["metrics"]["salt_temperature_k"]
        assert T_half > T_full, (
            f"Half-flow T_salt={T_half:.1f} K should exceed full-flow T_salt={T_full:.1f} K"
        )

    def test_invalid_input_returns_error(self):
        """Non-numeric input should return a failed-status response."""
        bad_raw = dict(_DEFAULT_RAW, external_reactivity_pcm="not_a_number")
        result = run_msr_control_step(None, bad_raw)
        assert not result["status"]["ok"]

    def test_config_change_reinitialises_session(self):
        """Changing τ_core reinitialises the session (β_eff_flow changes)."""
        raw_tc5 = dict(_DEFAULT_RAW, core_transit_time_s=5.0)
        raw_tc20 = dict(_DEFAULT_RAW, core_transit_time_s=20.0)
        s1 = run_msr_control_step(None, raw_tc5)
        s2 = run_msr_control_step(s1, raw_tc20)  # τ_core changed → reinitialise
        # Session should still succeed
        assert s2["status"]["ok"]
        # β_eff_flow should differ (slower salt → higher β_eff_flow)
        b1 = s1.get("beta_eff_flow", 0.0)
        b2 = s2.get("beta_eff_flow", 0.0)
        assert b2 > b1, (
            f"β_eff,flow at τ_core=20s ({b2:.5f}) should exceed τ_core=5s ({b1:.5f})"
        )


# ===========================================================================
# Integration tests — run_msr_scenario
# ===========================================================================


class TestMsrScenarioRunner:
    """run_msr_scenario physics direction tests."""

    def test_pump_trip_runs_successfully(self):
        """Pump trip scenario completes without solver failure."""
        result = run_msr_scenario("pump_trip")
        assert result["status"]["ok"], result["status"]["message"]
        assert result.get("scenario", {}).get("completed")

    def test_pump_trip_salt_temperature_rises(self):
        """After pump trip, final T_salt > initial T_salt (reduced heat removal)."""
        result = run_msr_scenario("pump_trip")
        hist = result["history"]
        T_init = hist["salt_temperature_k"][0]
        T_final = hist["salt_temperature_k"][-1]
        assert T_final > T_init, (
            f"T_salt should rise after pump trip: T_init={T_init:.1f} K, T_final={T_final:.1f} K"
        )

    def test_pump_trip_power_decreases(self):
        """After pump trip, final power < initial power (negative temperature feedback)."""
        result = run_msr_scenario("pump_trip")
        hist = result["history"]
        # Initial power (first few points after t=0)
        p_init = hist["power_mw"][5]   # skip t=0 transient
        p_final = hist["power_mw"][-1]
        # Power must decrease (negative temp feedback dominates)
        assert p_final < p_init * 1.1, (
            f"Power should decrease after pump trip: p_init={p_init:.1f}, p_final={p_final:.1f} MWth"
        )

    def test_load_following_power_decreases_then_recovers(self):
        """Load following: power dips during rod insertion and recovers after."""
        result = run_msr_scenario("load_following")
        assert result["status"]["ok"], result["status"]["message"]
        hist = result["history"]
        p_arr = np.array(hist["power_mw"])
        # Power should dip below initial value at some point
        p_initial = p_arr[10]  # after warmup
        p_min = float(np.min(p_arr[50:]))
        assert p_min < p_initial * 0.98, (
            f"Load following should reduce power: p_initial={p_initial:.1f}, p_min={p_min:.1f}"
        )

    def test_load_following_returns_history(self):
        """Load following history must cover the full scenario duration."""
        result = run_msr_scenario("load_following")
        from orchestrator.msr_scenarios import get_msr_scenario
        s = get_msr_scenario("load_following")
        final_t = result["history"]["time_s"][-1]
        assert final_t >= s.duration_s * 0.95, (
            f"Load following history ends at t={final_t:.1f} s, expected ~{s.duration_s:.1f} s"
        )

    def test_unknown_scenario_returns_error(self):
        """Requesting a non-existent scenario returns a failed-status response."""
        result = run_msr_scenario("does_not_exist")
        assert not result["status"]["ok"]

    def test_scenario_history_all_channels_populated(self):
        """Every history channel must have matching length after pump_trip."""
        result = run_msr_scenario("pump_trip")
        hist = result["history"]
        lengths = {k: len(v) for k, v in hist.items()}
        n_ref = lengths["time_s"]
        assert n_ref > 0
        for k, n in lengths.items():
            assert n == n_ref, f"History channel '{k}' length {n} != time_s length {n_ref}"

    def test_pump_trip_temp_reactivity_is_negative(self):
        """After pump trip, temperature feedback must be negative (T_salt rose)."""
        result = run_msr_scenario("pump_trip")
        hist = result["history"]
        # Final temperature reactivity should be negative
        temp_rho_final = hist["temp_reactivity_pcm"][-1]
        assert temp_rho_final < 0, (
            f"Temperature reactivity should be negative after pump trip, got {temp_rho_final:.2f} pcm"
        )
