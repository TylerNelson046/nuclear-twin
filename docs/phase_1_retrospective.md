# Phase 1 Retrospective and Phase 2 Readiness Report

**Author:** Principal Software Architect / Senior Nuclear Engineer review  
**Date:** 2026-05-17  
**Phase 1 Status:** Complete — all NFR-02 benchmarks passed, 374 tests green  
**Scope of this review:** Identify structural obligations before writing a single line of Phase 2 code

---

## 1. Technical Achievements

Phase 1 successfully delivered a coupled, numerically verified digital twin of a pressurized water reactor. The following were built and validated:

### 1.1 Physics Engine

| Module | What was verified |
|---|---|
| `physics/shared/kinetics.py` | 6-group point kinetics (SPEC Eqs. 1–2); steady-state precursor equilibrium; supercritical warning guard |
| `physics/shared/xenon.py` | I-135/Xe-135 coupled ODE (SPEC Eqs. 14–15); equilibrium at constant flux; peak-post-shutdown iodine pit |
| `physics/pwr/feedback.py` | Algebraic Doppler (Eq. 11), moderator (Eq. 12), boron worth (Eq. 13); NaN guard on unphysical inputs |
| `physics/pwr/thermal_hydraulics.py` | Lumped fuel/coolant ODEs (Eqs. 9–10); analytic steady-state consistency at P_NOM |
| `physics/pwr/engine.py` | Full 11-state coupled system: two solver interfaces (`pwr_coupled_rhs` + `pwr_11_state_system`); flat-params Numba JIT interface; JIT pre-warm pattern |

### 1.2 Numerical Performance

NFR-02 benchmark results (2026-05-17, `scripts/benchmark_pwr_system.py`):

| Metric | Result | Target |
|---|---|---|
| Full 100 s transient wall time | 63.9 ms | — |
| Real-time factor | 1,564× | — |
| Per-0.5 s callback median (Numba) | 0.50 ms | < 20 ms |
| Per-0.5 s callback p95 (Numba) | 1.30 ms | < 20 ms |
| No-JIT 0.5 s baseline | 1.76 ms | < 100 ms |
| UI render (dash.Patch delta) | 1.71 ms median | < 500 ms |
| Windows passing NFR-02 | 20/20 | 20/20 |
| Peak heap delta | 316.6 KiB | — |

CPU bottleneck: `radau.py:_step_impl → lu_solve` (implicit Jacobian factorization inside SciPy Radau) dominates cost — not the physics RHS. Numba's 3.5× speedup is real but modest because the RHS is already cheap relative to the LU-solve overhead. **This ratio will invert completely when Phase 2 adds spatial nodes.**

### 1.3 Orchestrator and UI

- `orchestrator/sim_runner.py`: clean JSON-serializable session contract; Pydantic validation boundary; adaptive `max_step` selection based on dominant physics timescale; scenario mode with 4 predefined PWR transients
- `ui/pwr_panel.py`: `dash.Patch` delta-update pattern for 4 charts; 40× render speedup vs full figure rebuild; 500 ms interval (4 Hz live refresh)
- `utils/data_handler.py`: HDF5 persistence with descriptive dataset names, units metadata, and warm-restart state vector

### 1.4 Correctness

- Doppler self-regulation confirmed: +50 pcm rod insertion stabilizes at new equilibrium
- Xenon equilibrium within −2500 to −3000 pcm after 50 hours at full power
- Xenon iodine pit: peak 6–10 hours post-shutdown
- Prompt criticality produces divergent transient
- MSR at zero salt velocity produces numerically identical results to PWR (shared kinetics consistency)

---

## 2. Architectural Bottlenecks and Tech Debt

This section identifies **every structural decision in Phase 1 that creates friction or failure risk when Phase 2 introduces spatially-discretized multi-node physics, 3D core meshing, and parallel execution.**

### 2.1 The State Vector is a Hardcoded 1D Array of Length 11

**Where:** `physics/pwr/engine.py:503`, `physics/shared/kinetics.py:83`, `physics/pwr/thermal_hydraulics.py:137`, `physics/shared/xenon.py:78`

Every Numba kernel terminates with a hardcoded allocation:
```python
dydt = np.empty(11)   # engine.py — full system
dydt = np.empty(7)    # kinetics.py
dydt = np.empty(2)    # thermal_hydraulics.py, xenon.py
```

And every kernel accesses state by raw integer index:
```python
n      = y[0]
T_fuel = y[7]
T_cool = y[8]
X      = y[10]
```

**Phase 2 impact:** Spatial physics requires `N_NODES` copies of each state variable. The natural internal representation becomes a 2D tensor `y[node_idx, state_idx]`, but SciPy's `solve_ivp` requires a flat 1D state vector. The bridge — reshape-on-entry / ravel-on-exit — must be built explicitly. The current architecture has no hook for this translation; adding it requires modifying every kernel signature.

**Debt classification:** Structural. Must be resolved before writing Phase 2 RHS kernels.

---

### 2.2 The Flat Params Array Is Sized for Scalar Physics Only

**Where:** `physics/pwr/engine.py:342–360`, `make_params_array()`

```python
_P_SIGMA_F    = 11  # one scalar: cm⁻¹
_P_SIGMA_A    = 12  # one scalar: cm⁻¹
PARAMS_LEN: int = 27
```

The flat 27-element params array was architected correctly for a zero-dimensional model — it enabled Numba JIT without closures. But it encodes the assumption that cross sections, thermal masses, and flow rates are global scalars. For N-node spatial physics:

- Each node has its own `sigma_f[i]`, `sigma_a[i]`, enrichment, thermal mass, and coolant path
- Inter-node neutron coupling requires a diffusion coupling matrix `M[i, j]`
- The flat-array indexing scheme (`_P_SIGMA_F = 11`) cannot represent per-node parameters without a complete redesign of the params layout

**Debt classification:** Moderate-to-severe. The `make_params_array` interface must be replaced with a structured parameter object (dataclass or dict of numpy arrays) for Phase 2. The `_P_*` index constants become meaningless once parameters become node-indexed.

---

### 2.3 Reactivity Feedback Is Computed Algebraically at Scalars

**Where:** `physics/pwr/feedback.py`, `physics/pwr/engine.py:480–489`

```python
rho_d_pcm  = alpha_d * (T_fuel - T_fuel_ref)   # scalar × scalar
rho_m_pcm  = alpha_m * (T_cool - T_cool_ref)   # scalar × scalar
rho_total  = base_dk_k + rod_dk_k + ...
```

In a multi-node model, every node generates its own Doppler and moderator feedback contribution, and these are **weighted by local neutron importance** (adjoint flux weighting) before being summed into a net core reactivity. The current scalar-summation pattern cannot express spatial reactivity distributions, power-peaking coefficients, or per-assembly feedback.

**Debt classification:** Physics model limitation (out of scope for lumped model), but the code pattern must change: the Phase 2 feedback calculator receives arrays and returns arrays, not scalars.

---

### 2.4 The `_append_pwr_result` Post-Processing Loop Is O(n\_points) Python

**Where:** `orchestrator/sim_runner.py:429–447`

```python
for idx in range(start_idx, len(times)):
    y = states[:, idx]
    snapshot = calculate_reactivity_snapshot(y, ...)  # Python call
    power_mw = neutron_population_to_power(float(y[0]), config)
    history["time_s"].append(float(times[idx]))
    ...
```

For Phase 1 (21 points per step), this loop costs ~0.5 ms. For a multi-node simulation producing dense output (e.g., 100 nodes × 200 output points per callback), the equivalent loop becomes 20,000 Python iterations per callback — roughly 10–20 ms of pure Python overhead, consuming the entire NFR-02 budget before any chart rendering begins.

**Debt classification:** Performance debt. The post-processing pipeline must be vectorized with NumPy before Phase 2 is built. The `calculate_reactivity_snapshot` calls inside the loop are particularly expensive in the spatial case.

---

### 2.5 Session History Is Stored as Python Lists in `dcc.Store` JSON

**Where:** `orchestrator/sim_runner.py:510–526`, `ui/pwr_panel.py:579–585`

The session state is a Python dict of Python lists (13 channels × 300 points = 3,900 floats) serialized to JSON on every callback and deserialized from the browser. This is acceptable for Phase 1 because the payload is ~31 KiB per tick.

For Phase 2 with N spatial nodes:
- Power distribution: N values per timestep
- Temperature distribution: N values per timestep  
- Reactivity distribution: N values per timestep

At N=50 nodes × 10 channels × 300 history points = 150,000 floats → ~1.2 MB per callback round-trip. At 4 Hz, that is 4.8 MB/s flowing through browser JSON serialization. The `dash.Patch` optimization only saves chart rendering time — the `dcc.Store` round-trip still serializes the full session on every write.

**Debt classification:** Architecture constraint. The store-per-callback pattern must be decomposed: physics state belongs in a server-side ring buffer; only display-resolution snapshots go to `dcc.Store`.

---

### 2.6 The `dcc.Interval` Drives Physics Synchronously on the Dash Callback Thread

**Where:** `ui/pwr_panel.py:143–218`, `orchestrator/sim_runner.py:90–170`

```python
@app.callback(...)
def update_pwr_panel(...):
    updated = run_pwr_control_step(session, controls)   # <-- physics here
    ...
    return updated, ..., power_fig, ...
```

The Dash callback thread calls `run_pwr_control_step` → `integrate_pwr_11state` → `solve_ivp` — all synchronously. For Phase 1 (0.5 ms physics cost), this is harmless. For Phase 2:

- Multi-node integrations are slower by O(N) relative to the scalar case
- The Radau solver for a stiff N-node system requires more LU-solve iterations (the Jacobian is now an `(11N × 11N)` matrix)
- Python's GIL means even if the solver releases the GIL inside Numba/NumPy, the Python orchestration overhead (session rebuild, snapshot generation, JSON serialization) still blocks the callback thread

For a 50-node spatial model, a single 0.5 s callback could take 50–200 ms. At 4 Hz, the Dash process would be CPU-bound and the UI would appear frozen between updates.

**Debt classification:** Concurrency architecture debt. Phase 2 requires physics to run in a background process (not thread — due to the GIL on Python orchestration code), communicating results to the UI via a queue or shared memory snapshot.

---

### 2.7 Thermal-Hydraulic Constants Are Module-Level Scalars

**Where:** `physics/pwr/thermal_hydraulics.py:47–93`

```python
M_FUEL: float = 101_000.0    # kg — total core fuel mass
M_COOL: float = 20_000.0     # kg — total core coolant mass
R_FC: float = (T_FUEL_NOM - T_COOL_NOM) / (GAMMA_F * P_NOM)   # computed at import
M_DOT_NOM: float = (GAMMA_F * P_NOM) / (C_COOL * (T_COOL_NOM - T_IN_NOM))
```

These are correct for a zero-dimensional lumped model — there is one fuel lump and one coolant lump for the entire core. For N axial or radial nodes, each node has its own `m_fuel[i]`, `m_cool[i]`, `r_fc[i]`, and `m_dot[i]`. The module-level constants cannot be vectorized; every node's `thermal_hydraulics_rhs` call would need its own parameter set.

The current `@njit` kernel uses these constants as Numba compile-time globals (they're module-level floats referenced inside the `@njit` function). This works for scalar physics but prevents per-node parameterization without changing the kernel signature.

**Debt classification:** Moderate. The constants can be promoted to per-call parameters without changing the Numba kernel pattern, but it requires kernel signature changes and updated callers.

---

### 2.8 Numba JIT Kernels Are Not Prepared for Parallel Array Loops

**Where:** All four `@njit(cache=True)` kernels

The current `point_kinetics_rhs` uses a Python-style `for i in range(6)` loop that Numba compiles to sequential scalar code. For a multi-node spatial model, the equivalent computation would be:

```python
for node in range(N_NODES):
    dydt[node, :] = point_kinetics_rhs_for_node(y[node, :], rho[node], ...)
```

This `node` loop is **embarrassingly parallel**: each node's kinetics are independent given a known reactivity field. Numba's `@njit(parallel=True)` with `prange(N_NODES)` can parallelize this across CPU cores without any algorithmic changes. However:

1. The current kernel signatures take 1D arrays and return 1D arrays — they cannot be directly called inside a `prange` loop without a wrapper that unpacks the 2D tensor
2. There is no `prewarm_jit` pattern established for parallel kernels; the existing pre-warm calls a serial scalar computation

**Debt classification:** Forward-looking preparation gap. Kernels must be refactored to accept node-index or full-array arguments before parallel acceleration can be added.

---

### 2.9 No Separation Between "Physics State" and "Display State"

**Where:** `orchestrator/sim_runner.py:420–471`

The session dict serves three purposes simultaneously:
1. **Physics checkpoint** — the 11-state ODE vector for warm-restarts (`session["state"]`)
2. **Display buffer** — the rolling 300-point history for Plotly traces (`session["history"]`)
3. **UI metadata** — metrics, events, scenario information

These concerns are conflated in one JSON blob. For Phase 2, the display buffer must be separated from the physics checkpoint:

- Physics state: `float64[N_NODES, 11]` — should live in process memory, not a browser store
- Display state: a downsampled snapshot — the only thing that should flow to `dcc.Store`
- Warm-restart checkpoint: a compact serialized form of the final ODE state

**Debt classification:** Design debt. Not a blocking issue for Phase 1 but it becomes the primary bottleneck for Phase 2 UI scalability.

---

## 3. Mandatory Pivot Decisions for Phase 2

These are architectural decisions that **must be made and implemented before the first Phase 2 physics function is written.** Each decision is an irreversible structural choice that downstream code will depend on.

---

### Pivot 1: Adopt a Node-Indexed State Representation

**Decision:** Define a canonical `NodeState` data structure that separates spatial indexing from variable indexing.

**Implementation contract:**

The ODE solver receives a 1D flat vector (SciPy's requirement). The physics kernels operate on a 2D view. A thin reshaping adapter sits at the boundary:

```python
# Conceptual contract — not implementation prescription
STATES_PER_NODE = 11    # same 11 variables as Phase 1
N_NODES         = ?     # determined by core mesh configuration

def flatten_state(y_nd: np.ndarray) -> np.ndarray:
    """(N_NODES, STATES_PER_NODE) → (N_NODES * STATES_PER_NODE,) for solve_ivp."""
    return y_nd.ravel()

def unflatten_state(y_flat: np.ndarray, n_nodes: int) -> np.ndarray:
    """(N_NODES * STATES_PER_NODE,) → (N_NODES, STATES_PER_NODE) for physics kernels."""
    return y_flat.reshape(n_nodes, STATES_PER_NODE)
```

The state indices `_S_N=0, _S_C1=1, ..., _S_XENON=10` remain valid as the column index of the 2D tensor. All existing physics sub-kernels remain valid if called per-node. No kinetics math needs to change — only the dispatch layer changes.

**What does NOT change:** The Keepin 6-group parameters, the SPEC equations, the Radau solver tolerance, the xenon ODEs.

**What DOES change:** The `pwr_11_state_system` RHS function becomes a loop over nodes; `build_initial_state` returns a 2D array; `_append_pwr_result` vectorizes over nodes.

---

### Pivot 2: Replace the Flat Params Scalar Array with a Structured Node-Parameter Object

**Decision:** The `make_params_array()` / `PARAMS_LEN=27` interface was purpose-built for scalar Numba JIT. For spatial physics, adopt a structured parameter container that can hold both scalar (global) and array (per-node) parameters:

```python
# Conceptual contract — not implementation prescription
@dataclass
class MultiNodeParams:
    # Global scalars (same as Phase 1)
    Lambda: float
    beta_arr: np.ndarray   # shape (6,)
    lambda_arr: np.ndarray # shape (6,)
    base_dk_k: float
    rod_dk_k: float        # or rod_dk_k_per_node: np.ndarray shape (N,)

    # Per-node arrays — new in Phase 2
    sigma_f: np.ndarray    # shape (N_NODES,)   cm⁻¹, per-node fission XS
    sigma_a: np.ndarray    # shape (N_NODES,)   cm⁻¹, per-node absorption XS
    phi_ref: np.ndarray    # shape (N_NODES,)   reference flux per node
    m_fuel: np.ndarray     # shape (N_NODES,)   fuel thermal mass per node (kg)
    m_cool: np.ndarray     # shape (N_NODES,)   coolant mass per node (kg)
    r_fc: np.ndarray       # shape (N_NODES,)   fuel-coolant resistance per node
    m_dot: np.ndarray      # shape (N_NODES,)   coolant flow per node (kg/s)

    # Inter-node coupling — new in Phase 2
    neutron_coupling: np.ndarray  # shape (N_NODES, N_NODES) diffusion coupling matrix
```

The Numba JIT wrapper for the multi-node RHS unpacks this into local numpy arrays for the `prange` loop. The `_P_*` index constants are retired when the multi-node path is adopted.

---

### Pivot 3: Decouple Physics Execution from the Dash Callback Thread

**Decision:** Before Phase 2 physics code is written, establish the concurrency boundary that separates the physics engine from the UI render cycle.

**The concurrency problem:** Dash's callback model is single-threaded by default. Even with `--processes N` gunicorn, each worker process is GIL-bound for Python orchestration. A multi-node spatial integration that takes 50–200 ms per step cannot be called synchronously inside a 500 ms callback without causing visible UI stuttering.

**Required architecture for Phase 2:**

```
┌────────────────────────────────────────────────────┐
│  Physics Process (separate OS process)             │
│  - Runs integrate_multi_node() continuously        │
│  - Writes results to shared ring buffer            │
│  - GIL-irrelevant: physics runs at full speed      │
└─────────────────────┬──────────────────────────────┘
                      │ multiprocessing.Queue  OR
                      │ shared_memory.SharedMemory
┌─────────────────────▼──────────────────────────────┐
│  Dash Callback Thread (main process)               │
│  - Reads a display-resolution snapshot from buffer │
│  - Never calls solve_ivp                           │
│  - dcc.Store payload stays small (display only)    │
│  - dcc.Interval drives render, not physics advance │
└────────────────────────────────────────────────────┘
```

The `sim_runner.py` orchestrator must grow a `start_physics_worker()` / `stop_physics_worker()` interface before Phase 2 engine code is written. The current synchronous `run_pwr_control_step()` API becomes an adapter that posts a control-change message to the physics worker rather than computing inline.

---

### Pivot 4: Adopt Pre-Allocated Ring Buffers for High-Frequency Logging

**Decision:** Replace the current `list.append()` + `_trim_history()` pattern with pre-allocated circular buffers for all time-series channels.

**The scaling problem:** The current pattern allocates a new Python list slice on every trim:
```python
return {key: values[-max_points:] for key, values in history.items()}
```
For 13 channels × 300 points, this is 3,900 floats copied per callback — acceptable. For 50 nodes × 10 channels × 300 points, this is 150,000 floats copied per callback at 4 Hz → 600,000 Python float allocations per second.

**Required pattern:**
- Pre-allocate `np.ndarray` buffers of shape `(N_CHANNELS, RING_CAPACITY)` with `dtype=float64`
- Maintain a write pointer and filled-count
- Physics writes `new_points[:, 0:n_new]` into the ring buffer using modular indexing (zero copy in the physics process)
- UI reads a contiguous snapshot view: `buffer.get_display_slice(n_points=300)` returns a numpy array, not a Python list
- Only the display slice is serialized to JSON for `dcc.Store` — the ring buffer lives in the physics process

This eliminates all list-copy overhead and makes the display latency independent of history depth.

---

### Pivot 5: Prepare Numba Kernels for Parallel Multi-Node Dispatch

**Decision:** Before Phase 2 RHS functions are written, establish the pattern for Numba parallel array loops so that per-node physics is automatically parallelized.

**Required kernel pattern (illustrative):**

```python
@numba.njit(parallel=True, cache=True)
def multi_node_kinetics_rhs(
    y_2d: np.ndarray,        # shape (N_NODES, 7)
    rho_arr: np.ndarray,     # shape (N_NODES,)
    beta_arr: np.ndarray,    # shape (6,) — shared across nodes
    lambda_arr: np.ndarray,  # shape (6,) — shared across nodes
    Lambda: float,           # scalar — shared
) -> np.ndarray:
    n_nodes = y_2d.shape[0]
    out = np.empty_like(y_2d)
    for i in numba.prange(n_nodes):   # parallel loop over nodes
        out[i, :] = point_kinetics_rhs(0.0, y_2d[i, :], rho_arr[i], beta_arr, lambda_arr, Lambda)
    return out
```

This pattern:
1. Keeps the existing `point_kinetics_rhs` sub-kernel unchanged
2. Adds a parallel dispatch wrapper that is O(N_NODES / N_CORES) instead of O(N_NODES)
3. Is compatible with the reshape pattern from Pivot 1

The pre-warm strategy must be updated: `prewarm_jit_multi_node(N_NODES)` must be called at startup with the actual node count to trigger AOT compilation of the correct array shapes.

---

## 4. What Phase 1 Got Right (Preserve These)

These structural decisions were correct and should carry forward unchanged:

| Decision | Why it matters for Phase 2 |
|---|---|
| Strict layer separation: `physics/` never imports `ui/` or `orchestrator/` | Multi-node physics modules remain independently testable; UI changes never touch physics |
| `dcc.Store` for session state (no process-global mutable variables) | Multi-worker deployment (gunicorn) works without session contamination; still correct |
| Pydantic validation boundary before physics | Spatial inputs (node count, mesh parameters) get the same validated treatment as scalar inputs |
| Radau solver with `rtol=1e-6, atol=1e-9` | Multi-node stiff ODE system is correctly handled by implicit Runge-Kutta; do not change solver or tolerances |
| Flat `float64` state arrays (no Python objects inside ODE kernel) | Prerequisite for Numba JIT; remains mandatory for Phase 2 kernels |
| `@njit(cache=True)` on all hot inner loops | Cache invalidates automatically when the function signature changes; safe to keep |
| Pre-warm JIT at startup | Becomes more important in Phase 2 where compile time for larger array types will be longer |
| HDF5 with units metadata | Extend naturally to per-node datasets: `/pwr/node_power[N]`, `/pwr/node_temp[N]` |
| Scenarios in `orchestrator/scenarios.py`, not in `ui/` | Multi-node scenarios (rod ejection in a specific assembly) belong in the same layer |

---

## 5. Phase 2 Entry Checklist

Do not write Phase 2 physics code until all items below are resolved:

- [ ] **Define the node-indexed state tensor contract** (`N_NODES`, state layout, flatten/unflatten pair) in a new `physics/spatial/core_mesh.py` module
- [ ] **Retire or extend `make_params_array`** — decide whether to keep it as a scalar shortcut for single-node operation or replace it with `MultiNodeParams` for all callers
- [ ] **Establish the ring buffer module** (`utils/ring_buffer.py`) with pre-allocated numpy backing and a thread-safe snapshot method
- [ ] **Spike the concurrency boundary** — write a minimal `PhysicsWorker` that runs a background process and exposes a `put_control_change` / `get_display_snapshot` interface; verify it works with Dash before wiring in Phase 2 physics
- [ ] **Write the parallel Numba dispatch wrapper** (`multi_node_kinetics_rhs` + pre-warm) and benchmark it against the serial Phase 1 equivalent at N=1 (must match), N=10, N=50
- [ ] **Update `ARCHITECTURE.md`** with the new spatial layer diagram and node-indexed state conventions before any Phase 2 PR is reviewed
