# Nuclear Reactor Digital Twin Platform — Architecture

**Version:** 0.2  
**Stack:** Python 3.11 · Dash · SciPy · Numba · Pydantic · HDF5  
**Phases:** PWR (Phase 1) → Helion FRC (Phase 2) → MSR (Phase 3)

---

## 1. Project Overview

A multi-reactor nuclear digital twin platform implementing simplified analytical models of three distinct reactor architectures:

| Reactor | Type | Physics Core |
|---|---|---|
| Pressurized Water Reactor (PWR) | Solid-fuel fission | 6-group point kinetics + thermal-hydraulics |
| Helion FRC | D-He3 fusion | Plasma energy balance + Bosch-Hale reactivity |
| Molten Salt Reactor (MSR) | Liquid-fuel fission | Shared PWR kinetics core + precursor drift |

Every software subsystem is derived directly from its governing physical law. Every simplification is named, justified, and documented in code comments. The reduced-order (zero-dimensional, single-node) modeling approach is a deliberate scoping decision, not a limitation.

---

## 2. Layer Separation

The platform is organized into three strict layers. Data flows in one direction: UI → Orchestrator → Physics. The physics engine never imports from UI or orchestrator layers.

```
┌─────────────────────────────────────┐
│              UI Layer               │  Dash + Plotly + DBC
│  pwr_panel · helion_panel · msr_panel│  Real-time controls, plots, event log
└────────────────┬────────────────────┘
                 │ parameter dicts (validated)
┌────────────────▼────────────────────┐
│          Orchestrator Layer         │  sim_runner.py
│  Drives solve_ivp · manages state   │  Routes between reactor engines
│  Handles HDF5 save/load             │  Generates event log narration
└────────────────┬────────────────────┘
                 │ ODE state vectors
┌────────────────▼────────────────────┐
│           Physics Layer             │  NumPy · SciPy · Numba
│  shared/ · pwr/ · helion/ · msr/    │  Pure functions, no Dash imports
└─────────────────────────────────────┘
```

**Rule:** Physics modules have zero knowledge of Dash, callbacks, or UI state. They accept numpy arrays and return numpy arrays. This makes them independently testable and reusable.

---

## 3. File Structure

```
nuclear-twin/
│
├── app.py                          # Dash app entry point; reactor selector
├── pyproject.toml                  # Poetry dependency manifest
├── ARCHITECTURE.md                 # This file
├── CLAUDE.md                       # Claude Code agent instructions
├── .cursorrules                    # Cursor agent instructions
├── README.md                       # Project overview + live demo link
│
├── docs/
│   └── SPEC.md                     # Full engineering specification (source of truth)
│
├── physics/
│   ├── shared/
│   │   ├── kinetics.py             # 6-group point kinetics ODE RHS (PWR + MSR shared)
│   │   └── xenon.py                # Xe-135 / I-135 ODE system
│   │
│   ├── pwr/
│   │   ├── engine.py               # Full 11-state PWR ODE system
│   │   ├── feedback.py             # Doppler, moderator, boron calculators (algebraic)
│   │   └── parameters.py           # Pydantic schema for PWR inputs + bounds
│   │
│   ├── helion/
│   │   ├── engine.py               # Plasma energy balance ODE
│   │   ├── bosch_hale.py           # D-He3 reactivity <σv> parametrization
│   │   ├── compression.py          # Adiabatic compression heating calculator
│   │   ├── losses.py               # Bremsstrahlung + confinement loss calculators
│   │   └── parameters.py           # Pydantic schema for Helion inputs + bounds
│   │
│   └── msr/
│       ├── engine.py               # Modified point kinetics + precursor drift
│       ├── drift.py                # Precursor return delay (Eq. 22-24)
│       ├── thermal.py              # Unified salt thermal-hydraulic module
│       └── parameters.py           # Pydantic schema for MSR inputs + bounds
│
├── orchestrator/
│   └── sim_runner.py               # Drives solve_ivp, manages time stepping, HDF5 I/O
│
├── ui/
│   ├── pwr_panel.py                # PWR control panel + Plotly time-series layouts
│   ├── helion_panel.py             # Helion control panel + ignition heatmap layout
│   └── msr_panel.py                # MSR control panel + β_eff,flow plots
│
├── data/
│   ├── keepin_dnp.py               # Keepin 6-group delayed neutron parameters (U-235)
│   └── bosch_hale_coeffs.py        # Bosch-Hale C1-C7 constants (D-He3, Table IV)
│
└── tests/
    ├── test_kinetics.py            # Unit + integration tests: steady-state, Doppler, prompt crit
    ├── test_xenon.py               # Xenon equilibrium, peak post-shutdown, burnout
    ├── test_feedback.py            # Algebraic feedback calculators
    ├── test_helion.py              # Energy conservation, Q boundary, Bosch-Hale scaling
    ├── test_msr.py                 # β_eff,flow reduction, shared kinetics consistency
    └── test_pydantic_bounds.py     # All input bounds at min, max, and just outside limits
```

---

## 4. ODE State Vectors

### 4.1 PWR — 11-State System

| Index | Variable | Description | Units |
|---|---|---|---|
| 0 | n | Neutron population (proportional to power) | — |
| 1–6 | C₁–C₆ | Delayed neutron precursor group concentrations | neutrons/cm³ |
| 7 | T_fuel | Lumped fuel temperature | K |
| 8 | T_cool | Lumped coolant temperature | K |
| 9 | I | I-135 number density | atoms/cm³ |
| 10 | X | Xe-135 number density | atoms/cm³ |

Reactivity components (algebraic, computed at each timestep, not ODE states):
`ρ_total = ρ_rod + ρ_Doppler(T_fuel) + ρ_moderator(T_cool) + ρ_boron(C_B) + ρ_xenon(X)`

### 4.2 Helion FRC — 1-State System

| Index | Variable | Description | Units |
|---|---|---|---|
| 0 | W | Total plasma thermal energy | J |

Derived quantities at each timestep: T (from W and n), P_fusion (Bosch-Hale), P_Brem, P_cond, Q.  
Post-compression initial conditions set by adiabatic heating: `T_final = T_initial · Rᶜ^(2/3)`, `n_final = n_initial · Rᶜ`.

### 4.3 MSR — 11-State System (extends PWR)

Same state vector as PWR. Two additional terms injected into the precursor ODEs (Eq. 22):
- **Flow-out term:** `−(1/τ_core) · Cᵢ`
- **Return term:** `+(1/τ_core) · Cᵢ(t − τ_loop) · e^(−λᵢ τ_loop)`

The return term introduces history dependence requiring full precursor history storage. This is the source of MSR numerical stiffness at high `v_salt` (Risk R-03 in spec).

---

## 5. Shared Kinetics Core Design Decision

The PWR and MSR share a single point kinetics implementation in `physics/shared/kinetics.py`.

**Design:** The MSR engine does not reimplement kinetics. It wraps the shared core and injects drift modification terms as additional ODE contributions via a callback/injection pattern.

**Why:** Ensures kinetics mathematics is implemented and tested once. MSR-specific modifications are isolated and clearly identified. Enables the key validation test: at zero salt velocity, MSR results must be numerically identical to PWR results given identical inputs.

**Switch logic lives in `orchestrator/sim_runner.py`**, which routes between PWR, Helion, and MSR engines without state contamination between reactor sessions.

---

## 6. Data Flow

### Real-Time Control Mode
```
User adjusts parameter
  → Pydantic validates input bounds (parameters.py)
  → Orchestrator updates reactor state (sim_runner.py)
  → Physics engine integrates one timestep (engine.py via solve_ivp Radau)
  → State vector returned to orchestrator
  → Plotly traces updated (ui/*.py)
  → Event log narration generated
  → Dash UI renders (< 500ms target, NFR-01)
```

### Scenario Mode
```
User selects scenario
  → Orchestrator loads scenario config from HDF5
  → Physics engine integrates full scenario duration
  → Results stored to HDF5
  → UI renders complete time-series
  → User scrubs through timeline
```

---

## 7. Performance Architecture

| Concern | Solution |
|---|---|
| ODE stiffness | SciPy `solve_ivp` with `method='Radau'`, `rtol=1e-6`, `atol=1e-9` |
| Hot loop performance | Numba `@njit` on kinetics RHS and Bosch-Hale lookup |
| UI responsiveness | `dcc.Interval` + `dcc.Store` for state; never block the callback thread |
| MSR history dependence | Ring buffer for precursor history at high `v_salt` |
| Deployment constraints | Free-tier Render/Railway; no GPU, standard memory |

**Benchmark targets (NFR-02):**
- Without Numba: full ODE timestep < 100ms
- With Numba (hot): full ODE timestep < 20ms

---

## 8. Input Validation Rules

All reactor inputs are validated by Pydantic schemas before reaching the physics engine (NFR-04, SR-01, SR-02).

- Every field bound is derived from a physical constraint, not an arbitrary software limit
- The physical justification for each bound is documented in a code comment on the field
- Inputs outside bounds return a defined error state with a plain-language explanation — no unhandled exceptions exposed to users
- All bounds are tested at minimum, maximum, and just-outside-maximum values

---

## 9. Testing & Validation Requirements

Target: **> 80% test coverage on all physics engine modules** (NFR-06).

### Physics Validation Checkpoints

**PWR:**
- Steady-state: power drift < 0.1% over 1000 simulated seconds with no inputs
- Doppler self-regulation: +50 pcm rod insertion stabilizes at new equilibrium (not runaway)
- Xenon equilibrium: after 50 hours at full power, xenon worth within −2500 to −3000 pcm
- Xenon peak: peaks 6–10 hours post-shutdown, returns to near-zero by 40–50 hours
- Prompt criticality: +1$ insertion produces divergent (not controlled) transient

**Helion:**
- Energy conservation: fusion output + losses = input + yield within 1%
- Q boundary: sub-ignition conditions yield Q < 1; ignition conditions yield Q > 1
- Bremsstrahlung dominance: at T < ~30 keV, radiation losses exceed fusion power

**MSR:**
- β_eff,flow = static β_eff at zero salt flow; decreases monotonically with velocity
- Shared kinetics consistency: at v_salt = 0, MSR results identical to PWR results

---

## 10. Key Technical Risks

| Risk | Description | Mitigation |
|---|---|---|
| R-02 | τ_E confinement scaling uncertainty dominates Helion model accuracy | Report Q as a range across uncertainty band, not a single value |
| R-03 | MSR precursor return history dependence causes stiffness at high v_salt | Ring buffer + Radau solver; document valid v_salt range |
| R-04 | Numba JIT compile time on first call adds latency | Pre-warm JIT during app startup with dummy call |

---

## 11. Out of Scope

The following are explicitly excluded from all three reactor models:

- Spatial neutron flux distributions or multi-group diffusion
- Two-phase flow or departure from nucleate boiling (DNB)
- Fuel burnup, isotopic depletion, or long-term reactivity management
- Control system automation or PID controller modeling
- Accident progression beyond early transient phase
- Salt chemistry or fission product solubility (MSR)
- Neutron transport (Monte Carlo or deterministic)
- Regulatory compliance or safety system modeling

---

## 12. Phase Completion Criteria

| Phase | Completion Gate |
|---|---|
| Pre-Phase | `pytest` passes on data loading tests; Dash app live at public URL |
| Phase 1 (PWR) | PWR twin at public URL; all 5 physics validation benchmarks pass; xenon peak and rod ejection scenarios demonstrable |
| Phase 2 (Helion) | Helion twin at same URL as PWR; ignition heatmap renders; Q boundary correct; platform selector switches without state contamination |
| Phase 3 (MSR) | All three twins in unified dashboard; β_eff,flow decreases with salt velocity; shared kinetics consistency test passes; full regression suite green |

---

*Update this document at the end of each phase to reflect architectural decisions made during implementation.*
