"""Plain-language simulation event narration utilities.

This module is intentionally UI-agnostic: it accepts raw simulation state and
returns JSON-serializable event records that Dash can store in ``dcc.Store``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

Severity = Literal["info", "warning"]

MAX_EVENT_LOG_ENTRIES = 80
CONTROL_EPSILON = 1.0e-6
FUEL_WARNING_C = 700.0
COOLANT_WARNING_C = 340.0
POWER_CHANGE_THRESHOLD = 0.05
POWER_LIMIT_WARNING = 1.20
REACTIVITY_LIMIT_PCM = 650.0


@dataclass(frozen=True)
class EventRecord:
    """One plain-language event emitted by the simulation runner."""

    time_s: float
    message: str
    severity: Severity = "info"
    code: str = "event"

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable event payload."""
        return {
            "time_s": self.time_s,
            "message": self.message,
            "severity": self.severity,
            "code": self.code,
        }


class EventLogBuffer:
    """Bounded in-memory queue for recent simulation events."""

    def __init__(self, entries: list[dict[str, Any]] | None = None, max_entries: int = MAX_EVENT_LOG_ENTRIES):
        self._entries = list(entries or [])[-max_entries:]
        self._max_entries = max_entries

    def append(self, event: EventRecord) -> None:
        """Append one event and trim the buffer to its configured size."""
        self._entries.append(event.as_dict())
        self._entries = self._entries[-self._max_entries:]

    def extend(self, events: list[EventRecord]) -> None:
        """Append multiple events in order."""
        for event in events:
            self.append(event)

    def to_list(self) -> list[dict[str, Any]]:
        """Return the current event buffer as JSON-serializable records."""
        return list(self._entries)


def initial_event_log(time_s: float = 0.0) -> list[dict[str, Any]]:
    """Return a new event log seeded with a startup narration."""
    return [
        EventRecord(
            time_s=time_s,
            message="PWR twin initialized at a critical full-power operating point.",
            code="startup",
        ).as_dict()
    ]


def generate_pwr_events(
    *,
    time_s: float,
    previous_controls: dict[str, Any] | None,
    current_controls: dict[str, Any],
    previous_metrics: dict[str, Any] | None,
    current_metrics: dict[str, Any],
    previous_flags: dict[str, bool] | None,
    history: dict[str, list[float]],
) -> tuple[list[EventRecord], dict[str, bool]]:
    """Translate PWR state and control changes into plain-language events."""
    flags = dict(previous_flags or {})
    events: list[EventRecord] = []

    events.extend(_control_change_events(time_s, previous_controls, current_controls))
    events.extend(_thermal_events(time_s, current_metrics, flags))
    events.extend(_power_events(time_s, previous_metrics, current_metrics))
    events.extend(_xenon_events(time_s, history, flags))

    return events, flags


def generate_pwr_scenario_events(
    *,
    scenario_id: str,
    scenario_label: str,
    scenario_description: str,
    history: dict[str, list[float]],
) -> list[EventRecord]:
    """Summarize a completed predefined PWR transient in plain language."""
    events = [
        EventRecord(
            time_s=0.0,
            message=f"Scenario started: {scenario_label}. {scenario_description}",
            code="scenario_started",
        )
    ]
    times = history.get("time_s", [])
    power = history.get("power_normalized", [])
    fuel_temperature = history.get("fuel_temperature_k", [])
    total_reactivity = history.get("total_reactivity_pcm", [])
    xenon = history.get("xenon_concentration", [])

    if times and power:
        peak_idx = _max_index(power)
        peak_power = power[peak_idx]
        events.append(
            EventRecord(
                time_s=times[peak_idx],
                message=f"Core power peaked at {peak_power * 100.0:.1f}% of the initial operating point.",
                severity="warning" if peak_power >= POWER_LIMIT_WARNING else "info",
                code="scenario_power_peak",
            )
        )

    if times and total_reactivity:
        peak_rho_idx = _max_abs_index(total_reactivity)
        peak_rho = total_reactivity[peak_rho_idx]
        if abs(peak_rho) >= REACTIVITY_LIMIT_PCM:
            events.append(
                EventRecord(
                    time_s=times[peak_rho_idx],
                    message=(
                        f"WARNING: Net reactivity reached {peak_rho:.0f} pcm, "
                        "near the delayed-neutron control margin."
                    ),
                    severity="warning",
                    code="scenario_reactivity_limit",
                )
            )

    if times and fuel_temperature:
        peak_temp_idx = _max_index(fuel_temperature)
        peak_temp_c = _kelvin_to_celsius(fuel_temperature[peak_temp_idx])
        if peak_temp_c >= FUEL_WARNING_C:
            events.append(
                EventRecord(
                    time_s=times[peak_temp_idx],
                    message=(
                        f"WARNING: Fuel temperature peaked at {peak_temp_c:.0f} deg C; "
                        "Doppler feedback is opposing the transient."
                    ),
                    severity="warning",
                    code="scenario_fuel_temperature_limit",
                )
            )

    if scenario_id == "xenon_peak_post_shutdown" and times and xenon:
        xenon_idx = _max_index(xenon)
        events.append(
            EventRecord(
                time_s=times[xenon_idx],
                message="Xenon-135 reached its post-shutdown peak as iodine decay overtook neutron burnout.",
                code="scenario_xenon_peak",
            )
        )

    if times and power:
        events.append(_scenario_stability_event(times, power))

    return events


def _control_change_events(
    time_s: float,
    previous_controls: dict[str, Any] | None,
    current_controls: dict[str, Any],
) -> list[EventRecord]:
    """Narrate operator input changes."""
    if previous_controls is None:
        return [
            EventRecord(
                time_s=time_s,
                message=(
                    "Control rods moved to "
                    f"{_rod_position_percent(current_controls):.1f}%. "
                    f"{_as_float(current_controls.get('rod_reactivity_pcm')):.1f} pcm of reactivity applied."
                ),
                code="rod_control",
            ),
            EventRecord(
                time_s=time_s,
                message=f"Soluble boron adjusted to {_as_float(current_controls.get('boron_ppm')):.0f} ppm.",
                code="boron_control",
            ),
            EventRecord(
                time_s=time_s,
                message=(
                    "Primary coolant flow set to "
                    f"{_as_float(current_controls.get('coolant_flow_fraction')) * 100.0:.0f}% of nominal."
                ),
                code="flow_control",
            ),
            EventRecord(
                time_s=time_s,
                message=(
                    "Core inlet temperature boundary set to "
                    f"{_kelvin_to_celsius(_as_float(current_controls.get('inlet_temperature_k'))):.1f} deg C."
                ),
                code="inlet_temperature",
            ),
        ]

    events: list[EventRecord] = []
    previous_rod = _as_float(previous_controls.get("rod_reactivity_pcm"))
    current_rod = _as_float(current_controls.get("rod_reactivity_pcm"))
    if abs(current_rod - previous_rod) > CONTROL_EPSILON:
        events.append(
            EventRecord(
                time_s=time_s,
                message=(
                    "Control rods moved to "
                    f"{_rod_position_percent(current_controls):.1f}%. "
                    f"{current_rod:.1f} pcm of reactivity applied."
                ),
                code="rod_control",
            )
        )

    previous_boron = _as_float(previous_controls.get("boron_ppm"))
    current_boron = _as_float(current_controls.get("boron_ppm"))
    if abs(current_boron - previous_boron) > CONTROL_EPSILON:
        events.append(
            EventRecord(
                time_s=time_s,
                message=f"Soluble boron adjusted to {current_boron:.0f} ppm.",
                code="boron_control",
            )
        )

    previous_flow = _as_float(previous_controls.get("coolant_flow_fraction"))
    current_flow = _as_float(current_controls.get("coolant_flow_fraction"))
    if abs(current_flow - previous_flow) > CONTROL_EPSILON:
        events.append(
            EventRecord(
                time_s=time_s,
                message=f"Primary coolant flow set to {current_flow * 100.0:.0f}% of nominal.",
                code="flow_control",
            )
        )

    previous_inlet = _as_float(previous_controls.get("inlet_temperature_k"))
    current_inlet = _as_float(current_controls.get("inlet_temperature_k"))
    if abs(current_inlet - previous_inlet) > CONTROL_EPSILON:
        events.append(
            EventRecord(
                time_s=time_s,
                message=f"Core inlet temperature boundary set to {_kelvin_to_celsius(current_inlet):.1f} deg C.",
                code="inlet_temperature",
            )
        )

    return events


def _thermal_events(
    time_s: float,
    current_metrics: dict[str, Any],
    flags: dict[str, bool],
) -> list[EventRecord]:
    """Narrate thermal limit and feedback events."""
    events: list[EventRecord] = []
    fuel_c = _kelvin_to_celsius(_as_float(current_metrics.get("fuel_temperature_k")))
    coolant_c = _kelvin_to_celsius(_as_float(current_metrics.get("coolant_temperature_k")))

    if fuel_c >= FUEL_WARNING_C and not flags.get("fuel_temperature_warning"):
        flags["fuel_temperature_warning"] = True
        events.append(
            EventRecord(
                time_s=time_s,
                message=(
                    f"WARNING: Fuel temperature exceeded {FUEL_WARNING_C:.0f} deg C. "
                    "Inherent Doppler feedback engaging."
                ),
                severity="warning",
                code="fuel_temperature_warning",
            )
        )
    elif fuel_c < FUEL_WARNING_C - 25.0:
        flags["fuel_temperature_warning"] = False

    if coolant_c >= COOLANT_WARNING_C and not flags.get("coolant_temperature_warning"):
        flags["coolant_temperature_warning"] = True
        events.append(
            EventRecord(
                time_s=time_s,
                message=(
                    f"WARNING: Coolant temperature exceeded {COOLANT_WARNING_C:.0f} deg C. "
                    "Moderator feedback is reducing reactivity."
                ),
                severity="warning",
                code="coolant_temperature_warning",
            )
        )
    elif coolant_c < COOLANT_WARNING_C - 10.0:
        flags["coolant_temperature_warning"] = False

    return events


def _power_events(
    time_s: float,
    previous_metrics: dict[str, Any] | None,
    current_metrics: dict[str, Any],
) -> list[EventRecord]:
    """Narrate large normalized power shifts."""
    if previous_metrics is None:
        return []

    previous_power = _as_float(previous_metrics.get("power_normalized"))
    current_power = _as_float(current_metrics.get("power_normalized"))
    delta = current_power - previous_power
    if abs(delta) < POWER_CHANGE_THRESHOLD:
        return []

    direction = "increased" if delta > 0.0 else "decreased"
    return [
        EventRecord(
            time_s=time_s,
            message=f"Core power {direction} to {current_power * 100.0:.1f}% of the initial operating point.",
            code="power_shift",
        )
    ]


def _xenon_events(
    time_s: float,
    history: dict[str, list[float]],
    flags: dict[str, bool],
) -> list[EventRecord]:
    """Narrate xenon transients once a local peak is observed after low power."""
    xenon = history.get("xenon_concentration", [])
    power = history.get("power_normalized", [])
    if len(xenon) < 3 or len(power) < 3 or flags.get("xenon_peak_reported"):
        return []

    previous_slope = xenon[-2] - xenon[-3]
    latest_slope = xenon[-1] - xenon[-2]
    low_power_recently = min(power[-3:]) < 0.10
    if low_power_recently and previous_slope > 0.0 and latest_slope <= 0.0:
        flags["xenon_peak_reported"] = True
        return [
            EventRecord(
                time_s=time_s,
                message="Xenon-135 concentration has reached peak post-shutdown equilibrium.",
                code="xenon_peak",
            )
        ]

    return []


def _scenario_stability_event(times: list[float], power: list[float]) -> EventRecord:
    """Describe whether the final part of a scenario appears settled."""
    window = max(3, min(len(power), len(power) // 10))
    recent_power = power[-window:]
    mean_power = sum(recent_power) / len(recent_power)
    span = max(recent_power) - min(recent_power)
    if span / max(abs(mean_power), 1.0e-9) < 0.02:
        message = "Transient stabilized over the final scenario window."
        code = "scenario_stabilized"
    else:
        message = "Scenario window complete; the transient is still evolving."
        code = "scenario_in_progress"
    return EventRecord(time_s=times[-1], message=message, code=code)


def _max_index(values: list[float]) -> int:
    """Return the index of the largest numeric value in a non-empty list."""
    return max(range(len(values)), key=lambda idx: values[idx])


def _max_abs_index(values: list[float]) -> int:
    """Return the index of the largest absolute numeric value in a non-empty list."""
    return max(range(len(values)), key=lambda idx: abs(values[idx]))


def _rod_position_percent(controls: dict[str, Any]) -> float:
    """Map the UI rod reactivity range (-1000 to +1000 pcm) onto 0-100%."""
    rod_pcm = _as_float(controls.get("rod_reactivity_pcm"))
    return max(0.0, min(100.0, (rod_pcm + 1000.0) / 2000.0 * 100.0))


def _as_float(value: Any) -> float:
    """Best-effort float conversion for already-validated runner values."""
    if value is None or value == "":
        return 0.0
    return float(value)


def _kelvin_to_celsius(value: float) -> float:
    """Convert Kelvin into Celsius for plain-language operator messages."""
    return value - 273.15


# ---------------------------------------------------------------------------
# MSR event narration (parity with PWR event log)
# ---------------------------------------------------------------------------

MSR_BETA_DROP_THRESHOLD = 0.0005  # dimensionless — flag meaningful β_eff,flow change
MSR_SALT_TEMP_WARNING_K = 950.0


def initial_msr_event_log(time_s: float = 0.0) -> list[dict[str, Any]]:
    """Return a new MSR event log seeded with a startup narration."""
    return [
        EventRecord(
            time_s=time_s,
            message=(
                "MSR twin initialized at flowing steady state with precursor drift enabled."
            ),
            code="msr_startup",
        ).as_dict()
    ]


def generate_msr_events(
    *,
    time_s: float,
    previous_controls: dict[str, Any] | None,
    current_controls: dict[str, Any],
    previous_metrics: dict[str, Any] | None,
    current_metrics: dict[str, Any],
    previous_flags: dict[str, bool] | None,
    config: dict[str, Any],
) -> tuple[list[EventRecord], dict[str, bool]]:
    """Translate MSR state and control changes into plain-language events."""
    flags = dict(previous_flags or {})
    events: list[EventRecord] = []

    events.extend(_msr_control_change_events(time_s, previous_controls, current_controls))
    events.extend(_msr_beta_events(time_s, previous_metrics, current_metrics, flags))
    events.extend(_msr_thermal_events(time_s, current_metrics, flags))
    events.extend(_msr_power_events(time_s, previous_metrics, current_metrics))

    tau_core = _as_float(config.get("tau_core", 0.0))
    if tau_core > 0.0 and not flags.get("tau_core_announced"):
        events.append(
            EventRecord(
                time_s=time_s,
                message=(
                    f"Core transit time τ_core = {tau_core:.2f} s; "
                    f"loop transit τ_loop = {_as_float(config.get('tau_loop', 0.0)):.1f} s."
                ),
                code="msr_transit_times",
            )
        )
        flags["tau_core_announced"] = True

    return events, flags


def generate_msr_scenario_events(
    *,
    scenario_id: str,
    scenario_label: str,
    scenario_description: str,
    history: dict[str, list[float]],
) -> list[EventRecord]:
    """Summarize a completed MSR scenario in plain language."""
    events = [
        EventRecord(
            time_s=0.0,
            message=f"MSR scenario started: {scenario_label}. {scenario_description}",
            code="msr_scenario_started",
        )
    ]
    times = history.get("time_s", [])
    power = history.get("power_mw", [])
    beta = history.get("beta_eff_flow", [])
    flow = history.get("salt_flow_fraction", [])

    if times and power:
        peak_idx = _max_index(power)
        events.append(
            EventRecord(
                time_s=times[peak_idx],
                message=f"Thermal power peaked at {power[peak_idx]:.1f} MWth.",
                code="msr_scenario_power_peak",
            )
        )

    if times and beta:
        min_beta_idx = min(range(len(beta)), key=lambda i: beta[i])
        events.append(
            EventRecord(
                time_s=times[min_beta_idx],
                message=(
                    f"β_eff,flow reached {beta[min_beta_idx] * 100:.2f}% of static β "
                    "as salt velocity increased precursor loss."
                ),
                code="msr_scenario_beta_minimum",
            )
        )

    if scenario_id == "pump_trip" and times and flow:
        trip_idx = min(range(len(flow)), key=lambda i: flow[i])
        events.append(
            EventRecord(
                time_s=times[trip_idx],
                message=(
                    f"Salt flow dropped to {flow[trip_idx] * 100:.0f}% of nominal — "
                    "delayed neutrons return faster, stiffening kinetics."
                ),
                severity="warning",
                code="msr_pump_trip",
            )
        )

    return events


def _msr_control_change_events(
    time_s: float,
    previous_controls: dict[str, Any] | None,
    current_controls: dict[str, Any],
) -> list[EventRecord]:
    """Narrate MSR operator input changes."""
    if previous_controls is None:
        return [
            EventRecord(
                time_s=time_s,
                message=(
                    f"External reactivity set to "
                    f"{_as_float(current_controls.get('external_reactivity_pcm')):.0f} pcm."
                ),
                code="msr_rod_control",
            ),
            EventRecord(
                time_s=time_s,
                message=(
                    "Salt flow at "
                    f"{_as_float(current_controls.get('salt_flow_fraction')) * 100.0:.0f}% of nominal."
                ),
                code="msr_flow_control",
            ),
        ]

    events: list[EventRecord] = []
    prev_rod = _as_float(previous_controls.get("external_reactivity_pcm"))
    curr_rod = _as_float(current_controls.get("external_reactivity_pcm"))
    if abs(curr_rod - prev_rod) > CONTROL_EPSILON:
        events.append(
            EventRecord(
                time_s=time_s,
                message=f"External reactivity changed to {curr_rod:.0f} pcm.",
                code="msr_rod_control",
            )
        )

    prev_flow = _as_float(previous_controls.get("salt_flow_fraction"))
    curr_flow = _as_float(current_controls.get("salt_flow_fraction"))
    if abs(curr_flow - prev_flow) > CONTROL_EPSILON:
        events.append(
            EventRecord(
                time_s=time_s,
                message=f"Salt flow fraction set to {curr_flow * 100.0:.0f}% of nominal.",
                code="msr_flow_control",
            )
        )
    return events


def _msr_beta_events(
    time_s: float,
    previous_metrics: dict[str, Any] | None,
    current_metrics: dict[str, Any],
    flags: dict[str, bool],
) -> list[EventRecord]:
    """Flag significant changes in flowing effective delayed fraction."""
    if previous_metrics is None:
        return []

    prev_beta = _as_float(previous_metrics.get("beta_eff_flow"))
    curr_beta = _as_float(current_metrics.get("beta_eff_flow"))
    if abs(curr_beta - prev_beta) < MSR_BETA_DROP_THRESHOLD:
        return []

    if curr_beta < prev_beta and not flags.get("beta_drop_warned"):
        flags["beta_drop_warned"] = True
        return [
            EventRecord(
                time_s=time_s,
                message=(
                    f"β_eff,flow decreased to {curr_beta:.5f} — MSR responds faster than a "
                    "static PWR at the same salt velocity."
                ),
                severity="warning",
                code="msr_beta_drop",
            )
        ]
    return []


def _msr_thermal_events(
    time_s: float,
    current_metrics: dict[str, Any],
    flags: dict[str, bool],
) -> list[EventRecord]:
    """Warn when salt temperature exceeds a teaching threshold."""
    T_k = _as_float(current_metrics.get("salt_temperature_k"))
    if T_k >= MSR_SALT_TEMP_WARNING_K and not flags.get("salt_temp_warned"):
        flags["salt_temp_warned"] = True
        return [
            EventRecord(
                time_s=time_s,
                message=(
                    f"Salt temperature reached {_kelvin_to_celsius(T_k):.0f} °C; "
                    "negative salt temperature feedback is opposing the transient."
                ),
                severity="warning",
                code="msr_salt_temperature",
            )
        ]
    return []


def _msr_power_events(
    time_s: float,
    previous_metrics: dict[str, Any] | None,
    current_metrics: dict[str, Any],
) -> list[EventRecord]:
    """Narrate large MSR power swings."""
    if previous_metrics is None:
        return []

    prev_p = _as_float(previous_metrics.get("power_mw"))
    curr_p = _as_float(current_metrics.get("power_mw"))
    if prev_p <= 0.0:
        return []

    rel_change = abs(curr_p - prev_p) / prev_p
    if rel_change < POWER_CHANGE_THRESHOLD:
        return []

    direction = "rose" if curr_p > prev_p else "fell"
    return [
        EventRecord(
            time_s=time_s,
            message=f"Thermal power {direction} to {curr_p:.1f} MWth.",
            code="msr_power_change",
        )
    ]
