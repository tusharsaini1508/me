import os
import webbrowser
from pathlib import Path
from threading import Timer

from streamlit.web import bootstrap


def main() -> None:
    app_path = Path(__file__).resolve().parent / "app.py"
    if not app_path.exists():
        raise SystemExit(f"Could not find app.py at {app_path}")

    os.chdir(app_path.parent)

    Timer(2.0, lambda: webbrowser.open("http://127.0.0.1:8501")).start()

    bootstrap.run(
        str(app_path),
        "streamlit run app.py",
        [],
        {
            "server.headless": True,
            "server.port": 8501,
            "server.address": "127.0.0.1",
            "global.disableWatchdogWarning": True,
        },
    )


if __name__ == "__main__":
    main()
