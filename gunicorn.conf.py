"""Gunicorn production settings for the Dash WSGI server."""

import multiprocessing
import os


bind = f"0.0.0.0:{os.environ.get('PORT', '8050')}"
workers = int(os.environ.get("WEB_CONCURRENCY", "1"))
threads = int(os.environ.get("GUNICORN_THREADS", "2"))
timeout = int(os.environ.get("GUNICORN_TIMEOUT", "120"))
graceful_timeout = int(os.environ.get("GUNICORN_GRACEFUL_TIMEOUT", "30"))
keepalive = int(os.environ.get("GUNICORN_KEEPALIVE", "5"))

worker_class = "sync"
worker_tmp_dir = "/dev/shm" if os.path.isdir("/dev/shm") else None
max_requests = int(os.environ.get("GUNICORN_MAX_REQUESTS", "1000"))
max_requests_jitter = int(os.environ.get("GUNICORN_MAX_REQUESTS_JITTER", "50"))

accesslog = "-"
errorlog = "-"
loglevel = os.environ.get("GUNICORN_LOG_LEVEL", "info")

# Keep the default conservative for free-tier hosts, but allow larger instances
# to scale without editing deployment files.
default_worker_count = max(1, multiprocessing.cpu_count() * 2 + 1)
if "WEB_CONCURRENCY" not in os.environ and os.environ.get("GUNICORN_AUTO_WORKERS") == "1":
    workers = default_worker_count
