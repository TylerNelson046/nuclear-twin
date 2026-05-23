# Pre-Phase Deployment Checklist

The local deployment scaffold is complete:

- `app.py` exposes `server = app.server` for WSGI hosting.
- `gunicorn` is included as a production dependency.
- `render.yaml` defines a free Render web service.
- `.github/workflows/ci.yml` runs tests on pushes to `main` and on pull requests.

External steps that must be completed in GitHub/Render:

1. Push this repository to GitHub.
2. In GitHub, enable branch protection for `main`.
3. Require the `CI / test` check before merging.
4. In Render, create a new Blueprint from the GitHub repository.
5. Deploy the `nuclear-twin` service and add the public URL to `README.md`.

The production start command is:

```bash
poetry run gunicorn app:server --bind 0.0.0.0:$PORT
```
