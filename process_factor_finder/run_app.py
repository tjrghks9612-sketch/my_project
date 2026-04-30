"""Executable launcher for Process Factor Finder.

This file is intentionally small. PyInstaller uses it as the entry point,
then Streamlit runs the bundled app.py from the temporary extracted folder.
"""

from __future__ import annotations

import os
import socket
import sys
import threading
import time
import webbrowser
from pathlib import Path


def _bundle_dir() -> Path:
    """Return the folder that contains bundled app files."""
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        return Path(sys._MEIPASS)  # type: ignore[attr-defined]
    return Path(__file__).resolve().parent


def _open_browser_later(url: str, delay_seconds: float = 2.5) -> None:
    """Open the local Streamlit URL after the server has had time to start."""
    def _open() -> None:
        time.sleep(delay_seconds)
        try:
            webbrowser.open(url, new=2)
        except Exception:
            print(f"브라우저가 자동으로 열리지 않으면 직접 접속하세요: {url}")

    threading.Thread(target=_open, daemon=True).start()


def _is_port_available(port: int) -> bool:
    """Return True when localhost can bind the requested port."""
    addresses = [("0.0.0.0", socket.AF_INET), ("127.0.0.1", socket.AF_INET)]
    for host, family in addresses:
        with socket.socket(family, socket.SOCK_STREAM) as sock:
            try:
                sock.bind((host, port))
            except OSError:
                return False
    try:
        with socket.socket(socket.AF_INET6, socket.SOCK_STREAM) as sock:
            sock.bind(("::1", port))
    except OSError:
        return False
    return True


def _find_available_port(preferred_port: int = 8501, max_port: int = 8599) -> int:
    """Find a usable Streamlit port, preferring 8501."""
    if _is_port_available(preferred_port):
        return preferred_port
    for port in range(preferred_port + 1, max_port + 1):
        if _is_port_available(port):
            print(f"Port {preferred_port} is already in use. Using port {port} instead.")
            return port
    raise RuntimeError(f"{preferred_port}-{max_port} 범위에서 사용 가능한 포트를 찾지 못했습니다.")


def main() -> int:
    """Start the Streamlit dashboard."""
    base_dir = _bundle_dir()
    app_path = base_dir / "app.py"
    if not app_path.exists():
        raise FileNotFoundError(f"app.py를 찾을 수 없습니다: {app_path}")

    os.environ.setdefault("STREAMLIT_BROWSER_GATHER_USAGE_STATS", "false")
    os.environ.setdefault("STREAMLIT_SERVER_HEADLESS", "true")
    os.environ.setdefault("STREAMLIT_GLOBAL_DEVELOPMENT_MODE", "false")
    runtime_dir = Path(sys.executable).resolve().parent if getattr(sys, "frozen", False) else base_dir
    os.environ.setdefault("PFF_RUNTIME_DIR", str(runtime_dir))

    sys.path.insert(0, str(base_dir))
    preferred_port = int(os.environ.get("PFF_PORT", "8501"))
    port = _find_available_port(preferred_port)
    url = f"http://localhost:{port}"
    print(f"Process Factor Finder URL: {url}")
    _open_browser_later(url)

    sys.argv = [
        "streamlit",
        "run",
        str(app_path),
        "--server.headless=true",
        f"--server.port={port}",
        "--browser.serverAddress=localhost",
        "--global.developmentMode=false",
    ]

    from streamlit.web import cli as streamlit_cli

    return int(streamlit_cli.main() or 0)


if __name__ == "__main__":
    raise SystemExit(main())
