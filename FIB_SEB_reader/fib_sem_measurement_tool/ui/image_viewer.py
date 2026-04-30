from __future__ import annotations

import math
import tkinter as tk
from typing import Callable, Optional, Sequence, Tuple

import cv2
import customtkinter as ctk
from PIL import Image, ImageTk


class ImageViewer(ctk.CTkFrame):
    def __init__(
        self,
        master,
        on_roi_changed: Callable[[Tuple[int, int, int, int]], None],
        on_calibration_line: Callable[[Tuple[int, int, int, int], float], None],
        on_overlay_toggled: Callable[[bool], None],
        on_hover_profile: Optional[Callable[[Optional[int], Optional[int]], None]] = None,
        **kwargs,
    ):
        super().__init__(master, fg_color="#0d1721", border_color="#223242", border_width=1, corner_radius=6, **kwargs)
        self.on_roi_changed = on_roi_changed
        self.on_calibration_line = on_calibration_line
        self.on_overlay_toggled = on_overlay_toggled
        self.on_hover_profile = on_hover_profile

        self.title_var = tk.StringVar(value="메인 이미지 미리보기")
        self.meta_var = tk.StringVar(value="이미지를 불러오세요")
        self.status_var = tk.StringVar(value="")
        self.zoom_var = tk.StringVar(value="Fit")
        self.mode_var = tk.StringVar(value="roi")
        self.overlay_enabled = True

        self.image_bgr = None
        self.render_bgr = None
        self.tk_image: Optional[ImageTk.PhotoImage] = None
        self.fit_mode = True
        self.zoom = 1.0
        self.scale = 1.0
        self.offset_x = 0
        self.offset_y = 0
        self.draw_w = 0
        self.draw_h = 0
        self.drag_start_canvas: Optional[Tuple[int, int]] = None
        self.drag_item: Optional[int] = None
        self._content_image_id = None

        self._build()

    def _build(self) -> None:
        header = ctk.CTkFrame(self, fg_color="transparent")
        header.pack(fill="x", padx=12, pady=(10, 6))
        ctk.CTkLabel(header, textvariable=self.title_var, font=ctk.CTkFont(size=15, weight="bold")).pack(side="left")
        ctk.CTkLabel(header, textvariable=self.status_var, text_color="#52f36d").pack(side="right")

        ctk.CTkLabel(self, textvariable=self.meta_var, anchor="w", text_color="#aebccb").pack(fill="x", padx=12, pady=(0, 6))

        self.canvas = tk.Canvas(self, bg="#070d13", highlightthickness=0, cursor="crosshair")
        self.canvas.pack(fill="both", expand=True, padx=12, pady=(0, 8))
        self.canvas.bind("<Configure>", lambda _event: self._render())
        self.canvas.bind("<ButtonPress-1>", self._on_press)
        self.canvas.bind("<B1-Motion>", self._on_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_release)
        self.canvas.bind("<MouseWheel>", self._on_mouse_wheel)
        self.canvas.bind("<Motion>", self._on_motion)
        self.canvas.bind("<Leave>", self._on_leave)

        controls = ctk.CTkFrame(self, fg_color="#0a121b")
        controls.pack(fill="x", padx=12, pady=(0, 10))
        ctk.CTkButton(controls, text="ROI", width=54, command=lambda: self.set_mode("roi")).pack(side="left", padx=(8, 4), pady=8)
        ctk.CTkButton(controls, text="Cal Line", width=78, fg_color="#203246", command=lambda: self.set_mode("calibration")).pack(side="left", padx=4, pady=8)
        ctk.CTkButton(controls, text="-", width=38, fg_color="#142234", command=self.zoom_out).pack(side="left", padx=4, pady=8)
        ctk.CTkButton(controls, text="+", width=38, fg_color="#142234", command=self.zoom_in).pack(side="left", padx=4, pady=8)
        ctk.CTkButton(controls, text="Fit", width=54, fg_color="#142234", command=self.fit).pack(side="left", padx=4, pady=8)
        ctk.CTkLabel(controls, textvariable=self.zoom_var, width=70).pack(side="left", padx=(8, 4))
        self.overlay_switch = ctk.CTkSwitch(controls, text="Overlay", command=self._toggle_overlay)
        self.overlay_switch.select()
        self.overlay_switch.pack(side="right", padx=10, pady=8)

    def set_mode(self, mode: str) -> None:
        self.mode_var.set(mode)
        self.canvas.configure(cursor="tcross" if mode == "calibration" else "crosshair")
        self.status_var.set("수동 캘리브레이션 선 모드" if mode == "calibration" else self.status_var.get())

    def set_content(self, image_bgr, render_bgr, title: str, meta: str, status: str) -> None:
        image_id = (id(image_bgr), image_bgr.shape if image_bgr is not None else None)
        self.image_bgr = image_bgr
        self.render_bgr = render_bgr
        self.title_var.set(title)
        self.meta_var.set(meta)
        self.status_var.set(status)
        if image_id != self._content_image_id:
            self.fit_mode = True
            self._content_image_id = image_id
        self._render()

    def clear(self) -> None:
        self.image_bgr = None
        self.render_bgr = None
        self.canvas.delete("all")
        self.title_var.set("메인 이미지 미리보기")
        self.meta_var.set("이미지를 불러오세요")
        self.status_var.set("")

    def _toggle_overlay(self) -> None:
        self.overlay_enabled = bool(self.overlay_switch.get())
        self.on_overlay_toggled(self.overlay_enabled)

    def zoom_in(self) -> None:
        self.fit_mode = False
        self.zoom = min(8.0, self.zoom * 1.25)
        self._render()

    def zoom_out(self) -> None:
        self.fit_mode = False
        self.zoom = max(0.05, self.zoom / 1.25)
        self._render()

    def fit(self) -> None:
        self.fit_mode = True
        self._render()

    def _on_mouse_wheel(self, event) -> None:
        if event.delta > 0:
            self.zoom_in()
        else:
            self.zoom_out()

    def _render(self) -> None:
        if self.render_bgr is None:
            return
        canvas_w = max(1, self.canvas.winfo_width())
        canvas_h = max(1, self.canvas.winfo_height())
        h, w = self.render_bgr.shape[:2]
        if self.fit_mode:
            self.scale = min(canvas_w / w, canvas_h / h)
            self.zoom = self.scale
        else:
            self.scale = self.zoom
        self.draw_w = max(1, int(w * self.scale))
        self.draw_h = max(1, int(h * self.scale))
        self.offset_x = int((canvas_w - self.draw_w) / 2)
        self.offset_y = int((canvas_h - self.draw_h) / 2)
        rgb = cv2.cvtColor(self.render_bgr, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb).resize((self.draw_w, self.draw_h), Image.BILINEAR)
        self.tk_image = ImageTk.PhotoImage(pil)
        self.canvas.delete("all")
        self.canvas.create_image(self.offset_x, self.offset_y, image=self.tk_image, anchor="nw")
        if self.fit_mode:
            self.zoom_var.set("Fit")
        else:
            self.zoom_var.set(f"{self.scale * 100:.0f}%")

    def _is_canvas_inside_image(self, x: int, y: int) -> bool:
        return (
            self.image_bgr is not None
            and self.offset_x <= x < self.offset_x + self.draw_w
            and self.offset_y <= y < self.offset_y + self.draw_h
        )

    def _canvas_to_image(self, x: int, y: int) -> Tuple[int, int]:
        if self.image_bgr is None:
            return 0, 0
        h, w = self.image_bgr.shape[:2]
        ix = int(round((x - self.offset_x) / max(self.scale, 1e-6)))
        iy = int(round((y - self.offset_y) / max(self.scale, 1e-6)))
        return max(0, min(w - 1, ix)), max(0, min(h - 1, iy))

    def _on_motion(self, event) -> None:
        if self.on_hover_profile is None:
            return
        if not self._is_canvas_inside_image(event.x, event.y):
            self.on_hover_profile(None, None)
            return
        self.on_hover_profile(*self._canvas_to_image(event.x, event.y))

    def _on_leave(self, _event) -> None:
        if self.on_hover_profile is not None:
            self.on_hover_profile(None, None)

    def _on_press(self, event) -> None:
        if self.image_bgr is None:
            return
        self.drag_start_canvas = (event.x, event.y)
        if self.drag_item:
            self.canvas.delete(self.drag_item)
            self.drag_item = None

    def _on_drag(self, event) -> None:
        if self.drag_start_canvas is None:
            return
        if self.drag_item:
            self.canvas.delete(self.drag_item)
        x0, y0 = self.drag_start_canvas
        if self.mode_var.get() == "calibration":
            self.drag_item = self.canvas.create_line(x0, y0, event.x, event.y, fill="#48d7ff", width=2)
        else:
            self.drag_item = self.canvas.create_rectangle(x0, y0, event.x, event.y, outline="#ffd23c", dash=(4, 3), width=2)

    def _on_release(self, event) -> None:
        if self.drag_start_canvas is None:
            return
        start = self._canvas_to_image(*self.drag_start_canvas)
        end = self._canvas_to_image(event.x, event.y)
        self.drag_start_canvas = None
        if self.drag_item:
            self.canvas.delete(self.drag_item)
            self.drag_item = None

        if self.mode_var.get() == "calibration":
            x1, y1 = start
            x2, y2 = end
            length = math.hypot(x2 - x1, y2 - y1)
            if length >= 3:
                self.on_calibration_line((x1, y1, x2, y2), length)
            return

        x1, y1 = start
        x2, y2 = end
        if abs(x2 - x1) >= 8 and abs(y2 - y1) >= 8:
            self.on_roi_changed((min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)))
