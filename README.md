# Nuclear Reactor Digital Twin Platform

Educational digital twin platform for simplified analytical models of three reactor architectures:

- Pressurized Water Reactor (PWR): 6-group point kinetics, thermal feedback, boron worth, and xenon dynamics.
- Helion-style Field-Reversed Configuration (FRC): D-He3 Bosch-Hale fusion reactivity, compression heating, and loss channels.
- Molten Salt Reactor (MSR): shared fission kinetics with delayed-neutron precursor drift in circulating fuel salt.

The project intentionally uses reduced-order, zero-dimensional and lumped-parameter models. It is for learning, visualization, and software engineering demonstration only. It is not licensed reactor safety analysis.

## Current Status

Pre-Phase environment setup is underway. The repository currently includes:

- Poetry dependency manifest and lockfile.
- Dash boilerplate app with placeholder reactor selector.
- Parsed IAEA U-235 thermal-spectrum delayed-neutron data.
- Keepin-style six-group parameter loader.
- Bosch-Hale D-He3 coefficient table.
- Pydantic base parameter schemas for PWR, Helion FRC, and MSR inputs.
- Pytest smoke and data validation tests.

## Tech Stack

- Python 3.13
- Dash, Dash Bootstrap Components, Plotly
- NumPy, SciPy, Numba
- Pydantic v2
- HDF5 via h5py
- pytest, pytest-benchmark
- Ruff

## Run Locally

From this directory:

```bash
PATH="/Users/kingkahuna/nuclear-sim/venv/bin:$PATH" poetry install --no-root
PATH="/Users/kingkahuna/nuclear-sim/venv/bin:$PATH" poetry run python app.py
```

Then open:

```text
http://127.0.0.1:8050
```

If Poetry is available on your normal shell path, the shorter form also works:

```bash
poetry install --no-root
poetry run python app.py
```

## Test

```bash
PATH="/Users/kingkahuna/nuclear-sim/venv/bin:$PATH" poetry run pytest
```

## Project Documents

- `docs/SPEC.md`: engineering specification and phased delivery plan.
- `ARCHITECTURE.md`: layer separation, file structure, state vectors, and engine boundaries.

## Deployment

The repository includes production entry points for Render, Heroku-style Procfile hosts, and Docker:

```bash
gunicorn app:server --config gunicorn.conf.py
```

`app.py` exposes `server = app.server` for WSGI hosting. `gunicorn.conf.py` binds to `0.0.0.0:$PORT` with a local fallback of `8050`, which matches Render, Heroku, Railway, and container platform expectations.

Render can deploy directly from `render.yaml`. The Docker image can be built and smoke-tested locally with:

```bash
docker build -t nuclear-twin .
docker run --rm -p 8050:8050 -e PORT=8050 nuclear-twin
```
