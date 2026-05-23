"""Compatibility entry point for launching the Dash boilerplate app.

Run with:
    poetry run python main.py
"""

from app import app


if __name__ == "__main__":
    app.run(debug=True, port=8050)
