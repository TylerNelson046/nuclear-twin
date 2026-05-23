"""Tests for the illustrative materials aging module + runner integration."""

from __future__ import annotations

import math

import pytest

from physics.materials import (
    HELION_DESIGN_LIMITS,
    MSR_DESIGN_LIMITS,
    PWR_DESIGN_LIMITS,
    HelionMaterialState,
    MSRMaterialState,
    PWRMaterialState,
    fresh_helion_material_state,
    fresh_msr_material_state,
    fresh_pwr_material_state,
    helion_remaining_life,
    msr_remaining_life,
    pwr_remaining_life,
    step_helion_pulse_aging,
    step_msr_aging,
    step_pwr_aging,
)


# ---------------------------------------------------------------------------
# Dataclass round-trip
# ---------------------------------------------------------------------------


class TestStateRoundtrip:
    def test_pwr_state_to_from_dict_lossless(self) -> None:
        original = fresh_pwr_material_state(operating_hours=123.4)
        payload = original.to_dict()
        rebuilt = PWRMaterialState.from_dict(payload)
        assert rebuilt == original

    def test_msr_state_to_from_dict_lossless(self) -> None:
        original = fresh_msr_material_state(operating_hours=99.0)
        rebuilt = MSRMaterialState.from_dict(original.to_dict())
        assert rebuilt == original

    def test_helion_state_to_from_dict_lossless(self) -> None:
        original = fresh_helion_material_state()
        rebuilt = HelionMaterialState.from_dict(original.to_dict())
        assert rebuilt == original

    def test_pwr_state_immutable(self) -> None:
        state = fresh_pwr_material_state()
        # frozen dataclass — direct attribute mutation must fail.
        with pytest.raises(Exception):
            state.rpv_drtndt_k = 50.0  # type: ignore[misc]

    def test_pwr_state_replace_returns_new_instance(self) -> None:
        state = fresh_pwr_material_state()
        new_state = state.replace(rpv_drtndt_k=42.0)
        assert state.rpv_drtndt_k == 0.0
        assert new_state.rpv_drtndt_k == 42.0
        assert new_state is not state


# ---------------------------------------------------------------------------
# PWR aging behaviour
# ---------------------------------------------------------------------------


class TestPWRAging:
    def test_zero_power_advances_nothing(self) -> None:
        state = fresh_pwr_material_state()
        new = step_pwr_aging(state, 86400.0, 0.0, 900.0, 600.0, 0.0, 3000.0)
        # Operating hours still ticks even at zero power (calendar life).
        assert new.operating_hours == pytest.approx(24.0)
        # Fluence-driven damage stays at zero.
        assert new.rpv_fluence_n_per_cm2 == 0.0
        assert new.rpv_drtndt_k == 0.0
        assert new.control_rod_b10_depletion_frac == 0.0

    def test_zero_dt_is_noop(self) -> None:
        state = fresh_pwr_material_state(operating_hours=10.0)
        new = step_pwr_aging(state, 0.0, 3000.0, 900.0, 600.0, 0.5)
        assert new == state

    def test_negative_dt_is_noop(self) -> None:
        state = fresh_pwr_material_state()
        new = step_pwr_aging(state, -100.0, 3000.0, 900.0, 600.0, 0.5)
        assert new == state

    def test_damage_monotonic_in_power(self) -> None:
        s0 = fresh_pwr_material_state()
        low = step_pwr_aging(s0, 86400.0, 500.0, 900.0, 600.0, 0.1, 3000.0)
        high = step_pwr_aging(s0, 86400.0, 3000.0, 900.0, 600.0, 0.1, 3000.0)
        assert high.rpv_fluence_n_per_cm2 > low.rpv_fluence_n_per_cm2
        assert high.fuel_burnup_gwd_per_mtu > low.fuel_burnup_gwd_per_mtu

    def test_oxide_grows_faster_at_higher_temperature(self) -> None:
        s0 = fresh_pwr_material_state()
        cool = step_pwr_aging(s0, 86400.0, 3000.0, 900.0, 560.0, 0.1)
        hot = step_pwr_aging(s0, 86400.0, 3000.0, 900.0, 620.0, 0.1)
        assert hot.cladding_oxide_thickness_um > cool.cladding_oxide_thickness_um

    def test_one_year_at_full_power_calibration(self) -> None:
        """Hit the design-basis annual numbers within a factor of ~2.

        These are sanity checks against published commercial PWR values;
        the model is illustrative so a generous tolerance is appropriate.
        """
        state = fresh_pwr_material_state()
        for _ in range(365):
            state = step_pwr_aging(state, 86400.0, 3000.0, 900.0, 600.0, 0.1)

        # Fast fluence ~1e18 n/cm^2/yr at beltline
        assert 5e17 < state.rpv_fluence_n_per_cm2 < 2e18
        # ΔRT_NDT mid-tens of K after 1 year for typical chemistry
        assert 20.0 < state.rpv_drtndt_k < 80.0
        # Cladding oxide ~1-5 µm/yr at PWR coolant temperatures
        assert 0.5 < state.cladding_oxide_thickness_um < 8.0
        # Burnup ~12 GWd/MTU/yr at full power
        assert 8.0 < state.fuel_burnup_gwd_per_mtu < 16.0

    def test_hydrogen_pickup_proportional_to_oxide(self) -> None:
        state = fresh_pwr_material_state()
        state = step_pwr_aging(state, 86400.0 * 30, 3000.0, 900.0, 600.0, 0.1)
        ratio = state.cladding_hydrogen_ppm / state.cladding_oxide_thickness_um
        # Coefficient is the constant in aging.py — 9 wppm/µm.
        assert 8.0 < ratio < 10.0


class TestPWRRemainingLife:
    def test_zero_power_returns_inf_for_fluence_components(self) -> None:
        state = fresh_pwr_material_state()
        life = pwr_remaining_life(state, 0.0, 900.0, 600.0, 0.0)
        # No power means no fluence, no burnup, no rod exposure.
        assert math.isinf(life["rpv_embrittlement_years"])
        assert math.isinf(life["fuel_burnup_years"])
        assert math.isinf(life["control_rod_years"])

    def test_full_power_gives_finite_projection(self) -> None:
        state = fresh_pwr_material_state()
        life = pwr_remaining_life(state, 3000.0, 900.0, 600.0, 0.2)
        # Burnup is the binding limit for a fresh core at full power.
        assert math.isfinite(life["fuel_burnup_years"])
        # And it's roughly 5 years (62 GWd/MTU / 12 GWd/MTU per yr).
        assert 3.0 < life["fuel_burnup_years"] < 7.0

    def test_projection_zero_when_already_past_limit(self) -> None:
        # Synthesize a state already past the RPV limit.
        state = fresh_pwr_material_state().replace(
            rpv_drtndt_k=PWR_DESIGN_LIMITS["rpv_drtndt_k"] + 10.0
        )
        life = pwr_remaining_life(state, 3000.0, 900.0, 600.0, 0.1)
        assert life["rpv_embrittlement_years"] == 0.0


# ---------------------------------------------------------------------------
# MSR aging behaviour
# ---------------------------------------------------------------------------


class TestMSRAging:
    def test_zero_power_keeps_corrosion_zero(self) -> None:
        s0 = fresh_msr_material_state()
        new = step_msr_aging(s0, 86400.0, 0.0, 923.0)
        assert new.hastelloy_n_corrosion_um == 0.0
        assert new.tellurium_attack_depth_um == 0.0

    def test_corrosion_calibrated_to_six_um_per_year(self) -> None:
        state = fresh_msr_material_state()
        for _ in range(365):
            state = step_msr_aging(state, 86400.0, 500.0, 923.0)
        assert 3.0 < state.hastelloy_n_corrosion_um < 12.0

    def test_tritium_inventory_grows_at_reactor_power(self) -> None:
        state = fresh_msr_material_state()
        for _ in range(30):
            state = step_msr_aging(state, 86400.0, 500.0, 923.0)
        # ~5-10 g after a month at 500 MW
        assert state.tritium_inventory_g > 0.5
        assert state.tritium_inventory_g < 50.0

    def test_msr_remaining_life_tellurium_binds_in_a_decade(self) -> None:
        state = fresh_msr_material_state()
        # One year of operation establishes a damage rate.
        for _ in range(365):
            state = step_msr_aging(state, 86400.0, 500.0, 923.0)
        life = msr_remaining_life(state, 500.0, 923.0)
        # Te attack: MSRE saw 250 µm at ~20 yr, we calibrated 8.4 yr ish.
        assert 3.0 < life["tellurium_attack_years"] < 25.0


# ---------------------------------------------------------------------------
# Helion pulse aging
# ---------------------------------------------------------------------------


class TestHelionAging:
    def test_one_pulse_increments_count(self) -> None:
        state = fresh_helion_material_state()
        new = step_helion_pulse_aging(state, 1e6, 30.0, 3e17)
        assert new.pulses_fired == 1.0
        assert new.first_wall_thermal_cycles == 1.0
        assert new.capacitor_charge_cycles == 1.0

    def test_more_reactions_yields_more_dpa(self) -> None:
        s0 = fresh_helion_material_state()
        low = step_helion_pulse_aging(s0, 1e6, 30.0, 1e17)
        high = step_helion_pulse_aging(s0, 1e6, 30.0, 1e19)
        assert high.first_wall_dpa > low.first_wall_dpa

    def test_coil_fatigue_quadratic_in_delta_t(self) -> None:
        s0 = fresh_helion_material_state()
        a = step_helion_pulse_aging(s0, 1e6, 30.0, 1e17)
        b = step_helion_pulse_aging(s0, 1e6, 60.0, 1e17)
        # delta_T doubled ⇒ fatigue index should ~4x (within the same pulse).
        assert b.coil_thermal_fatigue_index > 3.5 * a.coil_thermal_fatigue_index
        assert b.coil_thermal_fatigue_index < 4.5 * a.coil_thermal_fatigue_index

    def test_helion_remaining_life_finite(self) -> None:
        s = fresh_helion_material_state()
        for _ in range(100):
            s = step_helion_pulse_aging(s, 1e6, 30.0, 3e17)
        life = helion_remaining_life(s, 1e6, 30.0, 3e17)
        # Capacitor cycle count is the typical binding limit for pulsed systems.
        for v in life.values():
            assert v > 0.0


# ---------------------------------------------------------------------------
# Runner integration
# ---------------------------------------------------------------------------


def test_pwr_runner_attaches_materials_payload() -> None:
    from orchestrator.sim_runner import run_pwr_control_step

    session = None
    for _ in range(3):
        session = run_pwr_control_step(
            session,
            {
                "rod_reactivity_pcm": -50.0,
                "boron_ppm": 1000.0,
                "coolant_flow_fraction": 1.0,
                "inlet_temperature_k": 565.0,
            },
        )
    materials = session["materials"]
    assert "state" in materials
    assert "remaining_life_years" in materials
    assert materials["state"]["operating_hours"] > 0.0
    # Fresh core projection: burnup is the binding limit ~5 yr
    assert 0.0 < materials["remaining_life_years"]["fuel_burnup_years"] < 15.0


def test_msr_runner_attaches_materials_payload() -> None:
    from orchestrator.msr_runner import run_msr_control_step

    session = None
    for _ in range(2):
        session = run_msr_control_step(
            session,
            {
                "external_reactivity_pcm": 0.0,
                "salt_flow_fraction": 1.0,
                "core_transit_time_s": 5.0,
                "loop_transit_time_s": 20.0,
                "salt_temp_coeff_pcm_per_k": -2.0,
            },
        )
    materials = session["materials"]
    assert "state" in materials
    assert materials["state"]["operating_hours"] >= 0.0
    # Hastelloy corrosion ticks up immediately
    assert materials["state"]["hastelloy_n_corrosion_um"] >= 0.0


def test_helion_runner_returns_materials_when_passed_in() -> None:
    from orchestrator.helion_runner import run_helion_pulse
    from orchestrator.materials_integration import initial_helion_materials_payload

    mats = initial_helion_materials_payload()
    result = run_helion_pulse(
        {
            "ion_temperature_kev": 5.0,
            "plasma_density_m3": 1e21,
            "compression_ratio": 100.0,
            "magnetic_field_t": 10.0,
            "plasma_volume_m3": 1.0,
            "pulse_duration_s": 1e-5,
        },
        materials_in=mats,
    )
    assert result["success"]
    assert "materials" in result
    assert result["materials"]["state"]["pulses_fired"] == 1.0


def test_helion_runner_omits_materials_when_not_passed() -> None:
    from orchestrator.helion_runner import run_helion_pulse

    result = run_helion_pulse(
        {
            "ion_temperature_kev": 5.0,
            "plasma_density_m3": 1e21,
            "compression_ratio": 100.0,
            "magnetic_field_t": 10.0,
            "plasma_volume_m3": 1.0,
            "pulse_duration_s": 1e-5,
        },
    )
    assert result["success"]
    assert "materials" not in result


# ---------------------------------------------------------------------------
# UI smoke tests for the operator view
# ---------------------------------------------------------------------------


def test_pwr_operator_view_handles_empty_session() -> None:
    from ui.pwr_panel import _build_pwr_operator_view

    diagram, fig, bars, cards, summary, tiles = _build_pwr_operator_view(None)
    assert diagram is not None
    assert fig is not None
    assert bars == []
    assert cards == []
    assert "Awaiting" in summary
    assert tiles == []


def test_pwr_operator_view_populated_session() -> None:
    from ui.pwr_panel import _build_pwr_operator_view

    session = {
        "metrics": {
            "power_mw": 2950.0,
            "fuel_temperature_k": 920.0,
            "coolant_temperature_k": 595.0,
        },
        "controls": {"inlet_temperature_k": 565.0, "rod_reactivity_pcm": -50.0},
        "materials": {
            "state": fresh_pwr_material_state(operating_hours=100.0).to_dict(),
            "remaining_life_years": {
                "rpv_embrittlement_years": 15.0,
                "cladding_oxide_years": 50.0,
                "cladding_hydrogen_years": 80.0,
                "control_rod_years": 30.0,
                "fuel_burnup_years": 5.0,
            },
        },
    }
    out = _build_pwr_operator_view(session)
    assert len(out) == 6
    diagram, fig, bars, cards, summary, tiles = out
    assert len(bars) == 5
    assert len(cards) == 5
    assert len(tiles) == 4
    assert "year" in summary.lower() or "month" in summary.lower()


def test_msr_operator_view_handles_empty_session() -> None:
    from ui.msr_panel import _build_msr_operator_view

    out = _build_msr_operator_view(None)
    assert len(out) == 6


def test_helion_operator_view_empty_state() -> None:
    from ui.helion_panel import _build_helion_operator_view

    diagram, fig, label, bars, cards, summary, tiles = _build_helion_operator_view(
        None, None, None
    )
    assert "Run a pulse" in label
    assert len(bars) == 5
    assert len(cards) == 5


def test_helion_operator_view_with_pulse() -> None:
    from ui.helion_panel import _build_helion_operator_view
    from orchestrator.helion_runner import run_helion_pulse
    from orchestrator.materials_integration import initial_helion_materials_payload

    mats = initial_helion_materials_payload()
    pulse = run_helion_pulse(
        {
            "ion_temperature_kev": 5.0,
            "plasma_density_m3": 1e21,
            "compression_ratio": 100.0,
            "magnetic_field_t": 10.0,
            "plasma_volume_m3": 1.0,
            "pulse_duration_s": 1e-5,
        },
        materials_in=mats,
    )
    out = _build_helion_operator_view(pulse, pulse["materials"], None)
    assert len(out) == 7
    diagram, fig, label, bars, cards, summary, tiles = out
    assert "Phase:" in label


# ---------------------------------------------------------------------------
# Diagram primitives smoke tests
# ---------------------------------------------------------------------------


def test_diagram_render_helpers_return_strings() -> None:
    from ui.diagrams import render_pwr_diagram, render_msr_diagram, render_helion_diagram

    s_pwr = render_pwr_diagram(0.5, 0.5, 0.3, 0.2, 1.0)
    assert isinstance(s_pwr, str)
    assert "<svg" in s_pwr and "</svg>" in s_pwr

    s_msr = render_msr_diagram(950.0, 850.0, 0.3, 0.2)
    assert "<svg" in s_msr

    s_helion = render_helion_diagram(5e-6, 1e-5, 0.2, 0.1, 50.0)
    assert "<svg" in s_helion
    # Plasma elements should be present in markup.
    assert "helion-plasma-merged" in s_helion


def test_damage_panel_renders_finite_and_infinite_lives() -> None:
    from ui.diagrams import update_damage_panel_children

    bars, cards, summary = update_damage_panel_children(
        damage_bars=[
            ("RPV", 30.0, 100.0, "K"),
            ("Cladding", 20.0, 100.0, "µm"),
        ],
        remaining_life_years={"rpv": 12.5, "cladding": math.inf},
    )
    assert len(bars) == 2
    assert len(cards) == 2
    assert "12.5" in summary
