# Nuclear Reactor Digital Twin Platform

A multi-reactor nuclear digital twin platform implementing reduced-order analytical models of three reactor architectures: Pressurized Water Reactor (PWR), Helion-style Field-Reversed Configuration (FRC) fusion reactor, and Molten Salt Reactor (MSR). Built as a scientific computing and interactive visualization project demonstrating applied software engineering at the intersection of reactor physics, numerical simulation, and real-time UI.

**Live demo:** [nuclear-twin.onrender.com](https://nuclear-twin.onrender.com) *(free-tier host — allow ~30 s cold start)*

---

## What it models

### Pressurized Water Reactor (PWR)
Six-group point kinetics (Keepin U-235 parameters) coupled to two-node lumped thermal-hydraulics with Doppler and moderator temperature feedback, Xe-135/I-135 dynamics, and dissolved boron reactivity control. The 11-state ODE system is integrated with SciPy Radau. Demonstrates inherent self-regulation, xenon transients, and the prompt-delayed neutron distinction.

**Physics:** SPEC Eqs. 1–2 (kinetics), 9–10 (thermal-hydraulics), 11–13 (feedback), 14–15 (xenon/iodine)

**Validated against:**
- Steady-state power drift < 0.1% over 1000 simulated seconds
- +50 pcm rod insertion stabilises via Doppler self-regulation; final power in analytically predicted range
- Equilibrium xenon worth −2500 to −3000 pcm after 50 simulated hours at full power

### Helion FRC Fusion Reactor
Plasma energy balance ODE integrating fusion self-heating (D-He3 Bosch-Hale reactivity), Bremsstrahlung radiation loss, and empirical magnetic confinement loss. Adiabatic compression heating sets post-compression initial conditions. The ignition boundary heatmap sweeps a 20×20 grid across compression ratio × plasma density to map Q > 1 regions.

**Physics:** SPEC Eqs. 16–21 (Bosch-Hale: Bosch & Hale 1992, *Nuclear Fusion* 32(4), Table IV)

**Validated against:**
- Energy conservation within 1% across all simulated pulses
- Bremsstrahlung losses exceed fusion power at T < ~30 keV (D-He3 physics)
- Q > 1 region visible in the ignition boundary heatmap at high compression

### Molten Salt Reactor (MSR)
Extends the shared PWR kinetics core with precursor drift terms: precursors carried out of the core by flowing salt decay outside and cannot sustain the chain reaction, reducing the effective delayed neutron fraction β_eff,flow. The delay-differential return term is solved using a circular precursor history buffer with linear interpolation. Demonstrates how salt velocity fundamentally changes reactor transient response.

**Physics:** SPEC Eqs. 22–24 (precursor drift, return delay, β_eff,flow)

**Validated against:**
- β_eff,flow = static β_eff at zero salt flow; decreases monotonically with increasing salt velocity
- Pump trip: salt temperature rises, power decreases via negative temperature feedback
- Shared kinetics consistency: at zero salt velocity, MSR kinetics is numerically identical to PWR kinetics

---

## Tech stack

| Layer | Tool | Purpose |
|---|---|---|
| Language | Python 3.13 | Primary implementation |
| UI | Dash 4.1 + Dash Bootstrap Components 2.0 | React-compiled interactive dashboard |
| Charting | Plotly 6.7 (graph_objects) | Time-series, heatmaps, parameter sweeps |
| Physics | NumPy 2.4 + SciPy 1.17 (`solve_ivp`, Radau) | ODE integration, numerical simulation |
| Performance | Numba 0.65 (`@njit`) | JIT-compiled kinetics and Bosch-Hale hot loops |
| Validation | Pydantic v2 | Reactor parameter schemas with physical bounds |
| Storage | h5py 3.16 | HDF5 save/load for simulation runs |
| Testing | pytest 9.0 + pytest-benchmark | 604 tests, all passing |
| Linting | Ruff | Fast Python linting |
| Deployment | Render (free tier) | Gunicorn WSGI, `render.yaml` |
| Dep management | Poetry | Locked reproducible environment |

---

## Architecture

Strict three-layer separation — physics never imports from UI:

```
UI (Dash + Plotly)
    ↓ validated parameter dicts
Orchestrator (sim_runner, msr_runner, helion_runner)
    ↓ ODE state vectors
Physics (physics/shared, physics/pwr, physics/helion, physics/msr)
```

PWR and MSR share a single point kinetics implementation (`physics/shared/kinetics.py`). The MSR engine wraps the shared core and injects drift modification terms — no kinetics math is duplicated. Full details in `ARCHITECTURE.md`.

---

## Run locally

```bash
# From the nuclear-twin/ directory
poetry install
poetry run python app.py
# Open http://127.0.0.1:8050
```

If the Poetry virtualenv path requires explicit activation on your system:

```bash
PATH="/path/to/nuclear-sim/venv/bin:$PATH" poetry install --no-root
PATH="/path/to/nuclear-sim/venv/bin:$PATH" poetry run python app.py
```

---

## Test

```bash
poetry run pytest
# 604 tests, all passing
```

Test categories across all three reactor engines:
- Unit tests: each physics equation tested against hand-calculated values
- Integration tests: full ODE system steady-state and transient validation
- Boundary tests: Pydantic bounds at min, max, and just-outside limits
- Scenario tests: pump trip, load following, rod ejection, xenon peak, boron dilution
- E2E tests: unified dashboard state isolation, disclaimer compliance, callback registration

---

## Documents

- [`docs/SPEC.md`](docs/SPEC.md) — Full engineering specification: physics decomposition, governing equations, requirements, validation strategy, and phased delivery plan
- [`ARCHITECTURE.md`](ARCHITECTURE.md) — Layer separation, file structure, ODE state vectors, shared kinetics core design decision, and performance architecture

---

## Deployment

The repository includes production entry points for Render, Heroku-style Procfile hosts, and Docker:

```bash
gunicorn app:server --config gunicorn.conf.py
```

`app.py` exposes `server = app.server` for WSGI hosting. `gunicorn.conf.py` binds to `0.0.0.0:$PORT` with a local fallback of `8050`. `render.yaml` configures automatic deployment from the main branch.

```bash
docker build -t nuclear-twin .
docker run --rm -p 8050:8050 -e PORT=8050 nuclear-twin
```

---

## Performance

All orchestrators run well within the 500 ms UI response target (NFR-01) and the 100 ms no-JIT baseline (NFR-02):

| Reactor | Median step (no JIT) | NFR-02 limit |
|---|---|---|
| PWR | 0.1 ms | 100 ms |
| MSR | 2.2 ms | 100 ms |
| Helion | 1.0 ms | 100 ms |

---

## Disclaimer

Models are simplified analytical representations for educational purposes and do not represent licensed reactor safety analysis.
