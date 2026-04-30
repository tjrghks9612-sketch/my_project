from __future__ import annotations

import tkinter as tk
from typing import Optional, Tuple

import cv2
import customtkinter as ctk
import numpy as np


class ProfileGraph(ctk.CTkFrame):
    def __init__(self, master, **kwargs):
        super().__init__(master, fg_color="#0d1721", border_color="#223242", border_width=1, corner_radius=6, **kwargs)
        self.mode_var = tk.StringVar(value="Both")
        self.coord_var = tk.StringVar(value="Hover image to inspect grayscale profile")
        self.gray: Optional[np.ndarray] = None
        self.cursor: Optional[Tuple[int, int]] = None
        self._image_id = None
        self._pending_cursor: Optional[Tuple[Optional[int], Optional[int]]] = None
        self._draw_after_id = None
        self._last_drawn_cursor: Optional[Tuple[int, int]] = None
        self._build()

    def _build(self) -> None:
        header = ctk.CTkFrame(self, fg_color="transparent")
        header.pack(fill="x", padx=10, pady=(8, 4))
        ctk.CTkLabel(header, text="Grayscale Profile", font=ctk.CTkFont(size=13, weight="bold")).pack(side="left")
        ctk.CTkLabel(header, textvariable=self.coord_var, text_color="#90a2b4").pack(side="left", padx=12)
        self.mode = ctk.CTkSegmentedButton(
            header,
            values=["Both", "Horizontal", "Vertical"],
            variable=self.mode_var,
            command=lambda _value: self._draw_graph(),
            width=220,
        )
        self.mode.pack(side="right")

        self.canvas = tk.Canvas(self, bg="#071018", height=210, highlightthickness=0)
        self.canvas.pack(fill="both", expand=True, padx=10, pady=(0, 8))
        self.canvas.bind("<Configure>", lambda _event: self._draw_graph())

    def set_image(self, image_bgr) -> None:
        if image_bgr is None:
            self.gray = None
            self.cursor = None
            self._image_id = None
            self._last_drawn_cursor = None
            self.coord_var.set("No image")
            self._draw_graph()
            return
        image_id = (id(image_bgr), image_bgr.shape)
        if image_id == self._image_id:
            return
        if image_bgr.ndim == 2:
            self.gray = image_bgr.copy()
        else:
            self.gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        self._image_id = image_id
        self.cursor = None
        self._last_drawn_cursor = None
        self.coord_var.set("Hover image to inspect grayscale profile")
        self._draw_graph()

    def clear_cursor(self) -> None:
        if self.cursor is None and self._pending_cursor == (None, None):
            return
        self.cursor = None
        self._last_drawn_cursor = None
        self.coord_var.set("Hover image to inspect grayscale profile")
        self._draw_graph()

    def update_cursor(self, x: Optional[int], y: Optional[int]) -> None:
        if self.gray is None or x is None or y is None:
            self.clear_cursor()
            return
        height, width = self.gray.shape[:2]
        x = max(0, min(width - 1, int(x)))
        y = max(0, min(height - 1, int(y)))
        if self._last_drawn_cursor == (x, y):
            return
        self._pending_cursor = (x, y)
        if self._draw_after_id is None:
            self._draw_after_id = self.after(35, self._flush_cursor_update)

    def _flush_cursor_update(self) -> None:
        self._draw_after_id = None
        if self._pending_cursor is None:
            return
        x, y = self._pending_cursor
        self._pending_cursor = None
        if self.gray is None or x is None or y is None:
            self.clear_cursor()
            return
        self.cursor = (x, y)
        self._last_drawn_cursor = self.cursor
        self.coord_var.set(f"x={x}, y={y}, gray={int(self.gray[y, x])}")
        self._draw_graph()

    def _profile_points(self, profile: np.ndarray, width: int, height: int):
        if profile.size <= 1:
            return []
        xs = np.linspace(0, width - 1, profile.size)
        ys = height - 1 - (profile.astype(np.float32) / 255.0) * (height - 16) - 8
        return [(float(x), float(y)) for x, y in zip(xs, ys)]

    def _draw_polyline(self, points, color: str, step: int = 1) -> None:
        if len(points) < 2:
            return
        if step > 1:
            points = points[::step]
        flat = []
        for x, y in points:
            flat.extend([x, y])
        self.canvas.create_line(*flat, fill=color, width=2, smooth=True)

    def _draw_graph(self) -> None:
        self.canvas.delete("all")
        width = max(1, self.canvas.winfo_width())
        height = max(1, self.canvas.winfo_height())
        self.canvas.create_rectangle(0, 0, width, height, fill="#071018", outline="")
        for value in (64, 128, 192):
            y = height - 1 - (value / 255.0) * (height - 16) - 8
            self.canvas.create_line(0, y, width, y, fill="#142536")
            self.canvas.create_text(4, y - 2, text=str(value), fill="#647487", anchor="sw", font=("Arial", 8))

        if self.gray is None:
            self.canvas.create_text(width / 2, height / 2, text="Load an image", fill="#8293a6")
            return
        if self.cursor is None:
            self.canvas.create_text(width / 2, height / 2, text="Move mouse over main image", fill="#8293a6")
            return

        x, y = self.cursor
        mode = self.mode_var.get()
        if mode in ("Both", "Horizontal"):
            horizontal = self.gray[y, :]
            step = max(1, int(len(horizontal) / max(width, 1)))
            self._draw_polyline(self._profile_points(horizontal, width, height), "#38a8ff", step)
            self.canvas.create_text(width - 8, 14, text="H", fill="#38a8ff", anchor="ne", font=("Arial", 10, "bold"))
        if mode in ("Both", "Vertical"):
            vertical = self.gray[:, x]
            step = max(1, int(len(vertical) / max(width, 1)))
            self._draw_polyline(self._profile_points(vertical, width, height), "#61f27a", step)
            self.canvas.create_text(width - 8, 30, text="V", fill="#61f27a", anchor="ne", font=("Arial", 10, "bold"))
