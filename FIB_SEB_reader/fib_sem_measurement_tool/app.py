from __future__ import annotations

from fib_sem_measurement_tool.ui.main_window import MainWindow


def run_app() -> None:
    app = MainWindow()
    app.mainloop()

