"""Tests for PWR vs MSR comparison runner."""

from orchestrator.comparison_runner import run_pwr_msr_comparison


def test_run_pwr_msr_comparison_load_following() -> None:
    result = run_pwr_msr_comparison("load_following", "load_following")
    assert result.get("ok") is True
    assert result.get("pwr", {}).get("history", {}).get("time_s")
    assert result.get("msr", {}).get("history", {}).get("time_s")
    assert result.get("beta_sweep", {}).get("v_salt")
