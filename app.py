"""Dash entry point for the nuclear reactor digital twin platform."""

from ui.app import create_app

app = create_app()
server = app.server


if __name__ == "__main__":
    app.run(debug=True, port=8050)
