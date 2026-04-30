from __future__ import annotations

import customtkinter as ctk


class ResultTable(ctk.CTkFrame):
    """Compact metadata strip used below the main viewer."""

    def __init__(self, master, **kwargs):
        super().__init__(master, fg_color="#0d1721", border_color="#223242", border_width=1, corner_radius=6, **kwargs)
        self.values = []
        self._build()

    def _build(self) -> None:
        self.labels = []
        for title, value in [
            ("가속 전압", "-"),
            ("프로브 전류", "-"),
            ("작업 거리", "-"),
            ("이미지 크기", "-"),
            ("스캔 라인", "-"),
            ("픽셀 스케일", "-"),
        ]:
            cell = ctk.CTkFrame(self, fg_color="transparent")
            cell.pack(side="left", fill="x", expand=True, padx=8, pady=10)
            ctk.CTkLabel(cell, text=title, text_color="#8191a2", font=ctk.CTkFont(size=11)).pack(anchor="w")
            label = ctk.CTkLabel(cell, text=value, text_color="#f0f4f8", font=ctk.CTkFont(size=13))
            label.pack(anchor="w", pady=(4, 0))
            self.labels.append(label)

    def update_info(self, image_size, scan_line_count: int, px_to_real: float, unit: str) -> None:
        width, height = image_size if image_size else ("-", "-")
        values = ["-", "-", "-", f"{width} x {height}", str(scan_line_count), f"{px_to_real:.4g} {unit}/px"]
        for label, value in zip(self.labels, values):
            label.configure(text=value)

