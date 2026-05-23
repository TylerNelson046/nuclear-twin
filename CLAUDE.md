# CLAUDE.md — Instructions for Claude Code

This file is read automatically by Claude Code at session start. Follow all instructions here throughout the session.

---

## Project Identity

**Nuclear Reactor Digital Twin Platform** — a Python/Dash scientific simulation app modeling three reactor architectures: PWR (fission, solid fuel), Helion FRC (D-He3 fusion), and MSR (fission, liquid fuel). The full engineering specification is in `docs/SPEC.md`. The architecture is in `ARCHITECTURE.md`. Read both before making any significant changes.

---

## Stack

- **Language:** Python 3.11+
- **UI:** Dash + Dash Bootstrap Components (DBC) + Plotly
- **Physics:** NumPy + SciPy (`solve_ivp`, `method='Radau'`)
- **Performance:** Numba (`@njit` on hot ODE loops)
- **Validation:** Pydantic v2 (all reactor input schemas)
- **Storage:** HDF5 via h5py
- **Testing:** pytest + pytest-benchmark
- **Linting:** Ruff
- **Dependency management:** Poetry (`pyproject.toml`)

---

## Architecture Rules — Non-Negotiable

1. **Strict layer separation.** Physics modules (`physics/`) never import from `ui/` or `orchestrator/`. They are pure functions: numpy arrays in, numpy arrays out. This is what makes them independently testable.

2. **Shared kinetics core.** PWR and MSR share `physics/shared/kinetics.py`. The MSR engine wraps the shared core and injects drift terms — it does not reimplement kinetics. Never duplicate the kinetics math.

3. **Pydantic first.** All reactor inputs are validated by Pydantic schemas in `physics/<reactor>/parameters.py` before reaching any physics function. No raw dicts enter the physics layer.

4. **No unhandled exceptions.** All physics functions must handle unphysical states gracefully and return a defined error state with a plain-language message. Never let a solver failure surface as a Python traceback to the user.

5. **Bounds are physical, not arbitrary.** Every Pydantic field bound must have a code comment explaining its physical justification (e.g., `# Coolant cannot exceed saturation temperature at PWR operating pressure`).

6. **ODE solver config is fixed.** Always use `solve_ivp(method='Radau', rtol=1e-6, atol=1e-9)`. Do not change solver or tolerances without explicit user instruction.

---

## Physics Constants — Source of Truth

**Keepin 6-group parameters for U-235** are in `data/keepin_dnp.py`. Do not hardcode these values inline in engine files — always import from the data module.

**Bosch-Hale D-He3 coefficients (C1–C7)** are in `data/bosch_hale_coeffs.py`. Source: Bosch & Hale (1992), *Nuclear Fusion* 32(4), Table IV. Valid temperature range: 0.5–190 keV. Enforce bounds.

Key constants:
- `beta_eff` (U-235 thermal): ~0.0065
- `Lambda` (PWR prompt neutron lifetime): ~1e-5 s
- `alpha_D` (Doppler coefficient): −2 to −3 pcm/°C
- `alpha_m` (moderator temperature coefficient): −20 to −50 pcm/°C
- `omega_B` (differential boron worth): ~−10 pcm/ppm
- `E_DHe3` (D-He3 reaction energy): 18.3 MeV = 2.93e-12 J
- `C_Brem` (Bremsstrahlung coefficient): 5.35e-37 W·m³/keV^(1/2)

---

## Numba Rules

- Apply `@njit` to the kinetics RHS function (`point_kinetics_rhs`) and the Bosch-Hale lookup
- Do **not** apply `@njit` to Pydantic model methods or anything involving Python objects
- Pre-warm JIT at app startup with a dummy call to avoid first-call latency on user interaction
- Benchmark with `pytest-benchmark` before and after Numba addition to confirm improvement

---

## Testing Requirements

- **> 80% coverage** on all `physics/` modules — this is a hard requirement (NFR-06)
- Write tests alongside implementation, not after
- Every equation implemented as a function gets a unit test against a hand-calculated value
- Key integration tests that must always pass:
  - `test_steady_state_stability`: power drift < 0.1% over 1000 simulated seconds
  - `test_doppler_self_regulation`: +50 pcm insertion stabilizes (does not diverge)
  - `test_xenon_equilibrium`: xenon worth within −2500 to −3000 pcm after 50 hours at full power
  - `test_shared_kinetics_consistency`: MSR at v_salt=0 produces identical results to PWR
- Run `pytest` before committing any physics change

---

## Dash / UI Rules

- Use `dcc.Store` for all simulation state shared between callbacks — never use global Python variables for state
- Use `dcc.Interval` for real-time updates
- Never perform ODE integration inside a Dash callback directly — always call through `orchestrator/sim_runner.py`
- All Plotly figures use `plotly.graph_objects`, not `plotly.express`, for fine-grained control
- Target UI response time: < 500ms from parameter change to rendered update (NFR-01)
- The disclaimer "Models are simplified analytical representations for educational purposes and do not represent licensed reactor safety analysis" must appear on every reactor panel (SR-03)

---

## HDF5 File Format

Simulation states saved to HDF5 must:
- Use descriptive dataset names (e.g., `/pwr/time`, `/pwr/power_MW`, `/pwr/T_fuel_K`)
- Include `units` as a dataset attribute on every dataset (NFR-05)
- Include a `/metadata` group with: `reactor_type`, `timestamp`, `parameters` (JSON string), `spec_version`

---

## What Is Out of Scope

Do not implement, suggest, or scaffold any of the following — they are explicitly excluded from the spec:
- Spatial neutron flux or multi-group diffusion
- Two-phase flow or DNB calculations
- Fuel burnup or isotopic depletion
- PID controllers or automated control systems
- Accident progression beyond early transient
- Salt chemistry (MSR)
- Monte Carlo neutron transport
- Regulatory safety analysis

If a user asks about any of these, explain they are out of scope per the spec and suggest how the existing models bound the relevant behavior.

---

## Development Phases

The project builds in three phases. Do not implement Phase 2 or 3 components while working in Phase 1 — keep scope clean.

| Phase | Reactor | Key Output |
|---|---|---|
| 1 | PWR | Shared kinetics core + full PWR twin + UI |
| 2 | Helion FRC | Plasma energy balance + ignition heatmap |
| 3 | MSR | Shared kinetics extension + precursor drift |

---

## When Scaffolding Files

When asked to scaffold the project, create all directories and files from the structure in `ARCHITECTURE.md` Section 3. Each file should contain:
1. A module-level docstring explaining what the module contains and which spec equations it implements
2. Import stubs
3. Placeholder function signatures with docstrings
4. No implementation yet (unless explicitly asked)

This gives Cursor meaningful autocomplete context without polluting files with unreviewed physics code.


## Phase 1 → Phase 2 Architectural Pivots — MANDATORY READ BEFORE WRITING PHASE 2 CODE

Phase 1 (PWR lumped-parameter twin) is complete. Before writing any Phase 2 code, every session must internalize the following five irreversible architectural pivot decisions. Full analysis in `docs/phase_1_retrospective.md`.

### Pivot 1 — Node-Indexed State Tensor (NOT a flat 1D array)
Phase 1 uses a flat 11-element state vector `y[0..10]`. Phase 2 introduces N spatial nodes. The internal physics representation becomes `y_2d[node_idx, state_idx]` — shape `(N_NODES, 11)`. SciPy's `solve_ivp` still receives a 1D flat vector; a `flatten_state` / `unflatten_state` adapter sits at the solve_ivp boundary. The state index constants (`_S_N=0`, `_S_C1=1`, …, `_S_XENON=10`) remain valid as the **column** index of the 2D tensor. Do not eliminate them — repurpose them.

### Pivot 2 — Replace `make_params_array` / `PARAMS_LEN=27` with Structured Per-Node Params
The 27-element flat params array was designed for scalar (single-node) Numba JIT. It cannot represent per-node cross sections, thermal masses, or inter-node diffusion coupling. Phase 2 adopts a `MultiNodeParams` dataclass that holds scalar globals (Λ, β_arr, λ_arr) alongside per-node numpy arrays (`sigma_f[N]`, `m_fuel[N]`, `neutron_coupling[N,N]`). The `_P_*` index constants are retired when the multi-node path is adopted. At N=1, `MultiNodeParams` must produce identical results to the Phase 1 scalar params (regression test).

### Pivot 3 — Decouple Physics from the Dash Callback Thread
Phase 1 calls `integrate_pwr_11state` synchronously inside the Dash callback. Phase 2 multi-node integrations are O(N) slower and will freeze the UI if run on the callback thread. Architecture: physics runs in a **separate OS process** (`multiprocessing`), writes results to a shared ring buffer, and the Dash callback reads display-resolution snapshots without calling solve_ivp. The `sim_runner.run_pwr_control_step()` API becomes a control-change message to the physics worker. Implement `PhysicsWorker` spike **before** writing Phase 2 RHS code.

### Pivot 4 — Pre-Allocated Ring Buffers Replace `list.append()` History
Phase 1's `_empty_history()` dict-of-lists pattern copies ~3,900 floats per callback — acceptable for 11 states. For N=50 nodes × 10 channels × 300 history points = 150,000 floats, Python list copies dominate latency. Phase 2 uses pre-allocated `np.ndarray` ring buffers (`utils/ring_buffer.py`). The physics process writes new points in-place via modular indexing; the Dash callback reads a contiguous display-slice. Only the display slice (~300 × display_channels floats) flows to `dcc.Store`.

### Pivot 5 — Numba Parallel Array Dispatch for Multi-Node Loops
Phase 1 `@njit` kernels are scalar-per-node (sequential). Phase 2 adds a parallel dispatch wrapper using `@njit(parallel=True)` with `numba.prange(N_NODES)` so each node's kinetics computation runs concurrently. The existing sub-kernels (`point_kinetics_rhs`, `thermal_hydraulics_rhs`, `xenon_iodine_rhs`) remain unchanged — only the dispatch wrapper is new. Pre-warm must be updated to `prewarm_jit_multi_node(N_NODES)` at startup.

### Non-Negotiable Invariants That Carry Forward
- Strict layer separation: `physics/` still never imports `ui/` or `orchestrator/`
- Pydantic validation at the orchestrator boundary — now includes node count and mesh parameters
- `solve_ivp(method='Radau', rtol=1e-6, atol=1e-9)` — do not change solver or tolerances
- All hot ODE kernels operate on `float64` numpy arrays only — no Python objects inside `@njit`
- `dcc.Store` for UI session state; no process-global mutable variables

---

8. Milestone & Iterative Development Plan
Pre-Phase: Environment Setup
Duration: Week 1 Deliverables:
•	Poetry environment configured with all dependencies installed and locked
•	GitHub repository initialized with branch protection and README
•	ARCHITECTURE.md created in repository root covering layer separation, engine boundaries, shared kinetics core design decision, and 0DE state vectors. Updated at the end of each phase.
•	Dash boilerplate application running locally with placeholder reactor selector
•	IAEA delayed neutron database downloaded and parsed into project data directory
•	Keepin 6-group parameters loaded and unit-tested
•	Bosch-Hale D-He3 coefficients implemented and verified against published ⟨σv⟩ tables
•	Pydantic base schemas defined for all three reactor parameter sets
•	Render or Railway deployment pipeline connected to GitHub — deploy the boilerplate immediately so deployment is never a last-minute problem
Success Criteria: poetry run pytest passes on data loading tests. Dash app accessible at public URL showing placeholder UI.
 
Phase 1: PWR Twin
Duration: Weeks 2–5 This phase builds the platform, not just the PWR. Every architectural decision made here persists through Phases 2 and 3.
Week 2 — Kinetics Engine:
•	Implement 6-group point kinetics ODE system
•	Implement Xe-135/I-135 kinetics ODE system
•	Implement Doppler and moderator feedback calculators
•	Implement boron worth calculator
•	Implement reactivity balance summation
•	Unit test all components individually
•	Integration test: steady-state stability
Week 3 — Thermal Hydraulics + Full PWR System:
•	Implement lumped fuel and coolant temperature ODEs
•	Couple thermal-hydraulic module to kinetics engine via feedback loop
•	Implement full 11-state PWR ODE system
•	Apply Numba JIT to hot loops
•	Benchmark performance against NFR-02 target
•	Validate Doppler self-regulation and xenon equilibrium
Week 4 — PWR UI:
•	Build PWR control panel (Dash + DBC): rod, boron, flow rate, temperature inputs
•	Build real-time Plotly time-series plots: power, temperatures, reactivity breakdown, Xe/I concentrations
•	Implement accelerated time mode (time multiplier slider)
•	Implement plain-language event log
•	Implement HDF5 save/load for simulation states
Week 5 — PWR Scenarios + Validation:
•	Implement scenario mode with predefined PWR scenarios: rod ejection, boron dilution, xenon peak post-shutdown, load following
•	Complete physics validation checklist (Section 9.3)
•	Polish UI, fix performance issues
•	Deploy Phase 1 to production URL
•	Write Phase 1 retrospective: what architectural decisions need to change before Phase 2
Phase 1 Success Criteria: PWR twin accessible at public URL. All physics validation benchmarks passed. Xenon peak transient demonstrable. Rod ejection scenario shows Doppler self-regulation.
 
Phase 2: Helion FRC Twin
Duration: Weeks 6–9 This phase validates platform modularity. Helion's physics is fundamentally different — if the platform architecture accommodates it cleanly, Phase 3 is low risk.
Week 6 — Helion Physics Engine:
•	Implement plasma energy balance ODE
•	Implement Bosch-Hale reactivity lookup (already unit-tested from pre-phase)
•	Implement adiabatic compression heating calculator
•	Implement Bremsstrahlung radiation loss
•	Implement empirical confinement time scaling with documented uncertainty range
•	Unit and integration tests
Week 7 — Helion Scenarios + Parameter Sweep:
•	Implement pulse simulation (microsecond timescale, no time acceleration needed)
•	Implement 2D ignition boundary sweep across compression ratio × plasma density
•	Implement Q factor calculation
•	Validate energy conservation and temperature scaling
Week 8 — Helion UI:
•	Build Helion control panel: compression ratio, plasma density, temperature, pulse duration inputs
•	Build time-series plots: temperature, fusion power, loss channels
•	Build ignition boundary heatmap (most visually striking output in the platform)
•	Integrate into unified reactor selector with PWR
Week 9 — Integration + Polish:
•	End-to-end testing of PWR + Helion together in unified dashboard
•	Performance validation on deployed instance
•	Deploy Phase 2 to production
Phase 2 Success Criteria: Helion twin accessible at same public URL as PWR. Ignition boundary heatmap renders correctly showing Q > 1 region. Platform selector switches cleanly between PWR and Helion without state contamination.
 
Phase 3: MSR Twin
Duration: Weeks 10–12 This phase demonstrates the shared kinetics infrastructure was worth building correctly. MSR reuses the PWR kinetics core with drift modifications.
Week 10 — MSR Physics Engine:
•	Subclass/extend shared kinetics core with precursor drift terms
•	Implement precursor return delay with exponential decay
•	Implement β_eff,flow calculation
•	Implement unified salt thermal-hydraulic module
•	Implement temperature feedback (single salt coefficient)
•	Unit test drift modification against zero-flow equality with PWR results
Week 11 — MSR Scenarios + UI:
•	Implement MSR scenarios: pump trip, load following, comparison of β_eff,flow at different salt velocities
•	Build MSR control panel and plots
•	Integrate into unified dashboard
Week 12 — Final Integration + Portfolio Polish:
•	Three-reactor platform fully integrated and deployed
•	Complete physics validation for MSR (Section 9.3)
•	Full regression test suite passing across all three reactors
•	README written for GitHub: project purpose, physics summary, live demo link, tech stack
•	Engineering spec document finalized and linked from README
•	Final performance audit on deployed instance
Phase 3 Success Criteria: All three reactor twins accessible from unified dashboard at public URL. MSR β_eff,flow demonstrably decreases with salt velocity. Shared kinetics consistency test passes. Full physics validation checklist complete across all three reactors.