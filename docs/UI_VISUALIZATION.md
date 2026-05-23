# UI Visualization Reference

Maps **[CURRENT]** (shipped in `ui/`) vs **[ADDITIONAL]** displays per reactor twin. Orchestrator history keys are the data contract for time-series charts.

## Global

| Item | Status | Location |
|------|--------|----------|
| Reactor selector (PWR / Helion / MSR) | CURRENT | `ui/app.py` |
| Educational disclaimer (SR-03) | CURRENT | All panels |
| Scenario launchers | CURRENT | PWR, MSR, Helion |
| Status alert | CURRENT | All panels |
| HDF5 save / load (PWR) | ADDITIONAL | `ui/pwr_panel.py` + `utils/data_handler.py` |
| CSV export (PWR, MSR, Helion) | ADDITIONAL | `utils/history_export.py` |
| Timeline scrubber (scenario replay) | ADDITIONAL | `ui/scenario_scrubber.py`, PWR/MSR/Helion panels |
| Global units toggle (K/°C, s/min/h) | ADDITIONAL | `ui/display_units_bar.py`, `utils/display_units.py` |
| Unified comparison view (PWR vs MSR) | ADDITIONAL | `ui/comparison_panel.py` |

## PWR vs MSR Comparison (`ui/comparison_panel.py`)

Select **PWR vs MSR Comparison** in the reactor dropdown. Pick a PWR and MSR scenario, then **Run Comparison** to overlay:

- Normalized power (% of each twin's nominal rating)
- Total reactivity (pcm) vs time
- β_eff,flow vs salt velocity with PWR static β_eff reference

Backend: `orchestrator/comparison_runner.py` runs both scenarios without requiring live twin sessions.

## PWR (`ui/pwr_panel.py`)

### Charts — CURRENT

- Core thermal power (`power_mw` vs `time_s`) with normalized power overlay
- Fuel & coolant temperature (°C)
- Reactivity breakdown (6 traces, pcm)
- Iodine & xenon (dual y-axis)

### Charts — ADDITIONAL

- Delayed precursor groups C₁–C₆
- Fuel–coolant ΔT
- Control schedule (boron, flow, ṁ, rod pcm, T_in °C) — 2×2 panel
- Xenon worth vs concentration (phase scatter)
- Neutron population n & thermal flux φ
- Power vs total reactivity (phase-plane)
- State derivatives (dn/dt, dT_f/dt, dT_c/dt, dI/dt, dX/dt)
- ρ_base reference line + typical Xe equilibrium band on reactivity chart
- Scenario metadata banner above metrics

### Metrics — CURRENT + ADDITIONAL

- Thermal power, fuel T, coolant T, total ρ, sim time, power %, xenon worth (pcm)

### Other — CURRENT

- Event log, time multiplier, scenario mode, HDF5 + CSV download

## MSR (`ui/msr_panel.py`)

### Charts — CURRENT

- Power, salt temperature, reactivity breakdown, β_eff,flow vs v_salt (algebraic sweep)

### Charts — ADDITIONAL

- Salt flow fraction vs time with ṁ (kg/s) and τ_core / τ_loop annotation
- β_eff,flow vs time with PWR static β reference
- Precursor groups C₁–C₆
- Operator control schedule (flow + ρ_ext)
- Precursor drift flow-out vs return (all 6 groups, 2×3 panel)
- ρ_base reference on reactivity chart
- β sweep: PWR static β_eff overlay for validation
- Scenario metadata banner

### Metrics — ADDITIONAL

- τ_core / τ_loop badges from session config

### Other — ADDITIONAL

- Event log (parity with PWR), CSV download

## Helion (`ui/helion_panel.py`)

### Charts — CURRENT

- Plasma T, power balance, plasma energy W, ignition heatmap

### Charts — ADDITIONAL

- Fusion reactivity ⟨σv⟩ and τ_E vs time
- Q(t) and Brem/fusion ratio
- Net power dW/dt (`net_power_w` from orchestrator)
- Electron density n_e vs time
- Pre- vs post-compression grouped bar chart (T, n, V)
- P_heat trace on power balance when non-zero
- τ_E uncertainty contours on sweep (0.3× / 3.0× C_TAU)
- Ignition mask overlay (green tint, Q>1 cells)
- Post-run pulse parameter summary (pre/post compression)

### Metrics — ADDITIONAL

- Integrated E_fusion, E_brem, E_cond

### Other — ADDITIONAL

- CSV download of last pulse time series
