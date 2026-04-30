from __future__ import annotations

import tkinter as tk
from typing import Callable, Dict, Optional

import customtkinter as ctk

from fib_sem_measurement_tool.models.settings import (
    DISTANCE_METHODS,
    DISTANCE_METHOD_BY_LABEL,
    EDGE_REFERENCES,
    EDGE_REFERENCE_BY_LABEL,
    MEASUREMENT_TYPES,
    MEASUREMENT_TYPE_BY_LABEL,
    NOISE_LEVELS,
    NOISE_LEVEL_BY_LABEL,
    ROI_APPLY_MODES,
    ROI_APPLY_MODE_BY_LABEL,
    SCOPES,
    SCOPE_BY_LABEL,
    MeasurementSettings,
)


class OptionPanel(ctk.CTkFrame):
    def __init__(
        self,
        master,
        on_option_changed: Callable[[bool], None],
        on_apply_settings: Callable[[], None],
        on_apply_roi: Callable[[], None],
        on_apply_calibration: Callable[[], None],
        on_detect_scale_bar: Callable[[], None],
        on_draw_calibration_line: Callable[[], None],
        on_measure_scope: Callable[[], None],
        on_create_group: Callable[[], None],
        on_ungroup: Callable[[], None],
        on_reset_current: Callable[[str], None],
        **kwargs,
    ):
        super().__init__(master, fg_color="#0d1721", border_color="#223242", border_width=1, corner_radius=6, **kwargs)
        self.on_option_changed = on_option_changed
        self.on_apply_settings = on_apply_settings
        self.on_apply_roi = on_apply_roi
        self.on_apply_calibration = on_apply_calibration
        self.on_detect_scale_bar = on_detect_scale_bar
        self.on_draw_calibration_line = on_draw_calibration_line
        self.on_measure_scope = on_measure_scope
        self.on_create_group = on_create_group
        self.on_ungroup = on_ungroup
        self.on_reset_current = on_reset_current

        self._loading = False
        self.advanced_visible = False

        self.measurement_type_var = tk.StringVar(value=MEASUREMENT_TYPES["distance_both"])
        self.taper_side_var = tk.StringVar(value="left")
        self.distance_method_var = tk.StringVar(value=DISTANCE_METHODS["mean"])
        self.edge_reference_var = tk.StringVar(value=EDGE_REFERENCES["inner"])
        self.noise_level_var = tk.StringVar(value=NOISE_LEVELS["medium"])
        self.scope_var = tk.StringVar(value=SCOPES["current"])
        self.roi_apply_mode_var = tk.StringVar(value=ROI_APPLY_MODES["relative_copy"])
        self.calibration_mode_var = tk.StringVar(value="manual")
        self.detected_px_var = tk.StringVar(value="")
        self.manual_px_var = tk.StringVar(value="")
        self.actual_length_var = tk.StringVar(value="")
        self.unit_var = tk.StringVar(value="um")

        self.advanced_vars: Dict[str, tk.StringVar] = {}
        self.advanced_value_vars: Dict[str, tk.StringVar] = {}
        self.advanced_sliders: Dict[str, ctk.CTkSlider] = {}
        self.advanced_specs = {
            "blur_kernel": (0, 11, 1, "odd_int"),
            "median_filter_size": (0, 7, 1, "odd_int"),
            "background_correction_strength": (0.0, 2.0, 0.05, "float"),
            "sensitivity": (0.1, 2.0, 0.05, "float"),
            "peak_prominence": (0.02, 0.50, 0.01, "float"),
            "scan_line_count": (5, 61, 2, "int"),
            "minimum_valid_line_count": (3, 40, 1, "int"),
            "min_valid_line_ratio": (0.10, 0.90, 0.05, "float"),
            "outlier_rejection_strength": (0.5, 3.0, 0.05, "float"),
            "fit_error_threshold": (1.0, 20.0, 0.5, "float"),
            "confidence_threshold": (40.0, 95.0, 1.0, "float"),
        }
        self.advanced_descriptions = {
            "blur_kernel": "이미지를 살짝 흐리게 해서 작은 노이즈를 줄입니다.",
            "median_filter_size": "점 잡음이나 밝은 먼지 같은 튀는 픽셀을 줄입니다.",
            "background_correction_strength": "전체 배경 밝기 기울기를 보정합니다.",
            "sensitivity": "값이 클수록 더 강한 경계만 edge로 인정합니다.",
            "peak_prominence": "약한 변화와 진짜 경계를 구분하는 기준입니다.",
            "scan_line_count": "ROI 안에서 반복 측정할 줄 수입니다.",
            "minimum_valid_line_count": "성공으로 인정할 최소 유효 측정 줄 수입니다.",
            "min_valid_line_ratio": "전체 scan line 중 성공해야 하는 최소 비율입니다.",
            "outlier_rejection_strength": "튀는 측정값을 얼마나 엄격하게 버릴지 정합니다.",
            "fit_error_threshold": "Taper 직선 맞춤에서 허용할 흔들림 기준입니다.",
            "confidence_threshold": "OK로 볼 신뢰도 기준입니다.",
        }

        self._build()

    def _build(self) -> None:
        header = ctk.CTkFrame(self, fg_color="transparent")
        header.pack(fill="x", padx=12, pady=(10, 4))
        ctk.CTkLabel(header, text="옵션 설정", font=ctk.CTkFont(size=15, weight="bold")).pack(side="left")
        ctk.CTkButton(header, text="기본값", width=72, fg_color="#142234", command=lambda: self.on_reset_current("global")).pack(side="right")

        self.body = ctk.CTkScrollableFrame(self, fg_color="transparent")
        self.body.pack(fill="both", expand=True, padx=8, pady=(0, 8))

        self._section_label("기본 옵션")
        self._combo_row("측정 타입", self.measurement_type_var, list(MEASUREMENT_TYPES.values()), lambda _v: self._changed(False))
        self._radio_row("Taper 측면", self.taper_side_var, [("Left", "left"), ("Right", "right")])
        self._combo_row("대표값 계산", self.distance_method_var, list(DISTANCE_METHODS.values()), lambda _v: self._changed(False))
        self._combo_row("경계 기준", self.edge_reference_var, list(EDGE_REFERENCES.values()), lambda _v: self._changed(False))
        self._combo_row("노이즈 프리셋", self.noise_level_var, list(NOISE_LEVELS.values()), lambda _v: self._noise_changed())

        self._section_label("캘리브레이션")
        self._radio_row("모드", self.calibration_mode_var, [("자동", "auto"), ("수동", "manual")])
        button_row = ctk.CTkFrame(self.body, fg_color="transparent")
        button_row.pack(fill="x", pady=(4, 6))
        ctk.CTkButton(button_row, text="스케일바 자동 검출", command=self.on_detect_scale_bar).pack(side="left", fill="x", expand=True, padx=(0, 4))
        ctk.CTkButton(button_row, text="수동 선", fg_color="#203246", command=self.on_draw_calibration_line).pack(side="left", fill="x", expand=True, padx=(4, 0))
        self._entry_row("감지 px", self.detected_px_var)
        self._entry_row("수동 px", self.manual_px_var)
        self._entry_row("실제 길이", self.actual_length_var)
        self._combo_row("단위", self.unit_var, ["nm", "um", "µm", "mm"], lambda _v: self._changed(False))
        ctk.CTkButton(self.body, text="캘리브레이션 적용", command=self.on_apply_calibration).pack(fill="x", pady=(6, 10))

        self._section_label("적용 범위")
        self._combo_row("범위", self.scope_var, list(SCOPES.values()), lambda _v: None)
        self._combo_row("ROI 적용", self.roi_apply_mode_var, list(ROI_APPLY_MODES.values()), lambda _v: self._changed(False))
        ctk.CTkButton(self.body, text="설정을 범위에 적용", command=self.on_apply_settings).pack(fill="x", pady=(6, 4))
        ctk.CTkButton(self.body, text="ROI를 범위에 적용", command=self.on_apply_roi).pack(fill="x", pady=4)
        ctk.CTkButton(self.body, text="측정 실행", fg_color="#0f5eb8", command=self.on_measure_scope).pack(fill="x", pady=(4, 10))

        self._section_label("그룹")
        group_row = ctk.CTkFrame(self.body, fg_color="transparent")
        group_row.pack(fill="x", pady=(0, 8))
        ctk.CTkButton(group_row, text="선택 이미지 그룹 생성", command=self.on_create_group).pack(side="left", fill="x", expand=True, padx=(0, 4))
        ctk.CTkButton(group_row, text="그룹 해제", fg_color="#203246", command=self.on_ungroup).pack(side="left", fill="x", expand=True, padx=(4, 0))
        reset_row = ctk.CTkFrame(self.body, fg_color="transparent")
        reset_row.pack(fill="x", pady=(0, 8))
        ctk.CTkButton(reset_row, text="그룹 설정으로", fg_color="#142234", command=lambda: self.on_reset_current("group")).pack(side="left", fill="x", expand=True, padx=(0, 4))
        ctk.CTkButton(reset_row, text="개별 설정 초기화", fg_color="#142234", command=lambda: self.on_reset_current("clear")).pack(side="left", fill="x", expand=True, padx=(4, 0))

        self.advanced_button = ctk.CTkButton(self.body, text="고급 옵션 보기", fg_color="#142234", command=self.toggle_advanced)
        self.advanced_button.pack(fill="x", pady=(4, 6))
        self.advanced_frame = ctk.CTkFrame(self.body, fg_color="#0a121b", border_color="#223242", border_width=1, corner_radius=6)
        self._build_advanced()

    def _section_label(self, text: str) -> None:
        ctk.CTkLabel(self.body, text=text, font=ctk.CTkFont(size=13, weight="bold"), anchor="w").pack(fill="x", pady=(10, 4))

    def _row(self, label: str) -> ctk.CTkFrame:
        row = ctk.CTkFrame(self.body, fg_color="transparent")
        row.pack(fill="x", pady=4)
        row.grid_columnconfigure(0, minsize=112)
        row.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(row, text=label, anchor="w", text_color="#c7d2df").grid(row=0, column=0, sticky="w", padx=(0, 8))
        return row

    def _combo_row(self, label: str, variable: tk.StringVar, values, command) -> ctk.CTkOptionMenu:
        row = self._row(label)
        menu = ctk.CTkOptionMenu(row, variable=variable, values=values, command=command, height=32)
        menu.grid(row=0, column=1, sticky="ew")
        return menu

    def _entry_row(self, label: str, variable: tk.StringVar) -> ctk.CTkEntry:
        row = self._row(label)
        entry = ctk.CTkEntry(row, textvariable=variable, height=32)
        entry.grid(row=0, column=1, sticky="ew")
        entry.bind("<Return>", lambda _event: self._changed(False))
        entry.bind("<FocusOut>", lambda _event: self._changed(False))
        return entry

    def _radio_row(self, label: str, variable: tk.StringVar, values) -> None:
        row = self._row(label)
        box = ctk.CTkFrame(row, fg_color="transparent")
        box.grid(row=0, column=1, sticky="ew")
        for text, value in values:
            ctk.CTkRadioButton(box, text=text, variable=variable, value=value, command=lambda: self._changed(False)).pack(side="left", padx=(0, 12))

    def _build_advanced(self) -> None:
        fields = [
            ("blur_kernel", "blur kernel"),
            ("median_filter_size", "median filter"),
            ("background_correction_strength", "background"),
            ("sensitivity", "sensitivity"),
            ("peak_prominence", "peak prominence"),
            ("scan_line_count", "scan lines"),
            ("minimum_valid_line_count", "min valid"),
            ("min_valid_line_ratio", "valid ratio"),
            ("outlier_rejection_strength", "outlier strength"),
            ("fit_error_threshold", "fit error"),
            ("confidence_threshold", "confidence"),
        ]
        for key, label in fields:
            self._slider_row(key, label)

        self.overlay_save_var = tk.BooleanVar(value=False)
        self.profile_graph_var = tk.BooleanVar(value=False)
        ctk.CTkCheckBox(self.advanced_frame, text="overlay image 저장", variable=self.overlay_save_var, command=lambda: self._changed(True)).pack(anchor="w", padx=8, pady=(6, 2))
        ctk.CTkCheckBox(self.advanced_frame, text="profile graph 표시", variable=self.profile_graph_var, command=lambda: self._changed(True)).pack(anchor="w", padx=8, pady=(2, 8))

    def _slider_row(self, key: str, label: str) -> None:
        low, high, _step, _kind = self.advanced_specs[key]
        value_var = tk.StringVar(value="")
        display_var = tk.StringVar(value="")
        self.advanced_vars[key] = value_var
        self.advanced_value_vars[key] = display_var

        row = ctk.CTkFrame(self.advanced_frame, fg_color="transparent")
        row.pack(fill="x", padx=8, pady=5)
        row.grid_columnconfigure(0, minsize=142)
        row.grid_columnconfigure(1, weight=1)
        row.grid_columnconfigure(2, minsize=54)
        ctk.CTkLabel(row, text=label, anchor="w", text_color="#aebccb").grid(row=0, column=0, sticky="w", padx=(0, 8))
        slider = ctk.CTkSlider(row, from_=low, to=high, command=lambda value, slider_key=key: self._slider_changed(slider_key, value))
        slider.grid(row=0, column=1, sticky="ew", padx=(0, 8))
        self.advanced_sliders[key] = slider
        ctk.CTkLabel(row, textvariable=display_var, width=52, anchor="e", text_color="#dbe7f2").grid(row=0, column=2, sticky="e")
        ctk.CTkLabel(
            row,
            text=self.advanced_descriptions.get(key, ""),
            anchor="w",
            justify="left",
            text_color="#7f91a5",
            font=ctk.CTkFont(size=11),
            wraplength=360,
        ).grid(row=1, column=0, columnspan=3, sticky="ew", pady=(2, 0))

    def _quantize_slider_value(self, key: str, value: float):
        low, high, step, kind = self.advanced_specs[key]
        value = max(float(low), min(float(high), float(value)))
        if step:
            value = round((value - float(low)) / float(step)) * float(step) + float(low)
        if kind == "odd_int":
            ivalue = int(round(value))
            if ivalue > 0 and ivalue % 2 == 0:
                ivalue += 1
            return max(int(low), min(int(high), ivalue))
        if kind == "int":
            return int(round(value))
        return round(float(value), 3)

    def _format_slider_value(self, value) -> str:
        if isinstance(value, int):
            return str(value)
        return f"{float(value):.2f}".rstrip("0").rstrip(".")

    def _slider_changed(self, key: str, raw_value: float) -> None:
        value = self._quantize_slider_value(key, raw_value)
        self.advanced_vars[key].set(str(value))
        self.advanced_value_vars[key].set(self._format_slider_value(value))
        if not self._loading:
            self._changed(True)

    def toggle_advanced(self) -> None:
        self.advanced_visible = not self.advanced_visible
        if self.advanced_visible:
            self.advanced_frame.pack(fill="x", pady=(0, 8))
            self.advanced_button.configure(text="고급 옵션 숨기기")
        else:
            self.advanced_frame.pack_forget()
            self.advanced_button.configure(text="고급 옵션 보기")

    def _changed(self, advanced: bool) -> None:
        if not self._loading:
            self.on_option_changed(advanced)

    def _noise_changed(self) -> None:
        if self._loading:
            return
        settings = self.get_settings(MeasurementSettings())
        settings.custom_option = False
        settings.apply_noise_preset(force=True)
        self._set_advanced_values(settings)
        self.on_option_changed(False)

    def _float(self, text: str, default: float) -> float:
        try:
            return float(text)
        except (TypeError, ValueError):
            return default

    def _int(self, text: str, default: int) -> int:
        try:
            return int(float(text))
        except (TypeError, ValueError):
            return default

    def get_settings(self, base: Optional[MeasurementSettings] = None, mark_custom: bool = False) -> MeasurementSettings:
        settings = base.clone() if base is not None else MeasurementSettings()
        settings.measurement_type = MEASUREMENT_TYPE_BY_LABEL.get(self.measurement_type_var.get(), "distance_both")
        settings.taper_side = self.taper_side_var.get()
        settings.distance_method = DISTANCE_METHOD_BY_LABEL.get(self.distance_method_var.get(), "mean")
        settings.edge_reference = EDGE_REFERENCE_BY_LABEL.get(self.edge_reference_var.get(), "inner")
        settings.noise_level = NOISE_LEVEL_BY_LABEL.get(self.noise_level_var.get(), "medium")
        settings.roi_apply_mode = ROI_APPLY_MODE_BY_LABEL.get(self.roi_apply_mode_var.get(), "relative_copy")
        settings.calibration.mode = self.calibration_mode_var.get()
        settings.calibration.unit = self.unit_var.get()

        adv = settings.advanced
        adv.blur_kernel = self._int(self.advanced_vars["blur_kernel"].get(), adv.blur_kernel)
        adv.median_filter_size = self._int(self.advanced_vars["median_filter_size"].get(), adv.median_filter_size)
        adv.background_correction_strength = self._float(self.advanced_vars["background_correction_strength"].get(), adv.background_correction_strength)
        adv.sensitivity = self._float(self.advanced_vars["sensitivity"].get(), adv.sensitivity)
        adv.peak_prominence = self._float(self.advanced_vars["peak_prominence"].get(), adv.peak_prominence)
        adv.scan_line_count = self._int(self.advanced_vars["scan_line_count"].get(), adv.scan_line_count)
        adv.minimum_valid_line_count = self._int(self.advanced_vars["minimum_valid_line_count"].get(), adv.minimum_valid_line_count)
        adv.min_valid_line_ratio = self._float(self.advanced_vars["min_valid_line_ratio"].get(), adv.min_valid_line_ratio)
        adv.outlier_rejection_strength = self._float(self.advanced_vars["outlier_rejection_strength"].get(), adv.outlier_rejection_strength)
        adv.fit_error_threshold = self._float(self.advanced_vars["fit_error_threshold"].get(), adv.fit_error_threshold)
        adv.confidence_threshold = self._float(self.advanced_vars["confidence_threshold"].get(), adv.confidence_threshold)
        adv.overlay_save_enabled = bool(self.overlay_save_var.get())
        adv.profile_graph_enabled = bool(self.profile_graph_var.get())
        if mark_custom:
            settings.custom_option = True
        return settings

    def set_settings(self, settings: MeasurementSettings) -> None:
        self._loading = True
        self.measurement_type_var.set(MEASUREMENT_TYPES.get(settings.measurement_type, MEASUREMENT_TYPES["distance_both"]))
        self.taper_side_var.set(settings.taper_side)
        self.distance_method_var.set(DISTANCE_METHODS.get(settings.distance_method, DISTANCE_METHODS["mean"]))
        self.edge_reference_var.set(EDGE_REFERENCES.get(settings.edge_reference, EDGE_REFERENCES["inner"]))
        self.noise_level_var.set(NOISE_LEVELS.get(settings.noise_level, NOISE_LEVELS["medium"]))
        self.roi_apply_mode_var.set(ROI_APPLY_MODES.get(settings.roi_apply_mode, ROI_APPLY_MODES["relative_copy"]))
        self.calibration_mode_var.set(settings.calibration.mode)
        self.unit_var.set(settings.calibration.unit if settings.calibration.unit != "px" else "um")
        self.detected_px_var.set("" if settings.calibration.detected_scale_bar_px is None else f"{settings.calibration.detected_scale_bar_px:.3f}")
        self.manual_px_var.set("" if settings.calibration.manual_pixel_length is None else f"{settings.calibration.manual_pixel_length:.3f}")
        self.actual_length_var.set("" if settings.calibration.actual_scale_bar_length is None else f"{settings.calibration.actual_scale_bar_length:.6g}")
        self._set_advanced_values(settings)
        self._loading = False

    def _set_advanced_values(self, settings: MeasurementSettings) -> None:
        adv = settings.advanced
        values = {
            "blur_kernel": adv.blur_kernel,
            "median_filter_size": adv.median_filter_size,
            "background_correction_strength": adv.background_correction_strength,
            "sensitivity": adv.sensitivity,
            "peak_prominence": adv.peak_prominence,
            "scan_line_count": adv.scan_line_count,
            "minimum_valid_line_count": adv.minimum_valid_line_count,
            "min_valid_line_ratio": adv.min_valid_line_ratio,
            "outlier_rejection_strength": adv.outlier_rejection_strength,
            "fit_error_threshold": adv.fit_error_threshold,
            "confidence_threshold": adv.confidence_threshold,
        }
        for key, value in values.items():
            value = self._quantize_slider_value(key, value)
            self.advanced_vars[key].set(str(value))
            self.advanced_value_vars[key].set(self._format_slider_value(value))
            if key in self.advanced_sliders:
                self.advanced_sliders[key].set(value)
        self.overlay_save_var.set(bool(adv.overlay_save_enabled))
        self.profile_graph_var.set(bool(adv.profile_graph_enabled))

    def get_scope(self) -> str:
        return SCOPE_BY_LABEL.get(self.scope_var.get(), "current")

    def get_roi_apply_mode(self) -> str:
        return ROI_APPLY_MODE_BY_LABEL.get(self.roi_apply_mode_var.get(), "relative_copy")

    def get_calibration_inputs(self):
        mode = self.calibration_mode_var.get()
        pixel_text = self.detected_px_var.get() if mode == "auto" else self.manual_px_var.get()
        try:
            pixel_length = float(pixel_text)
        except (TypeError, ValueError):
            pixel_length = 0.0
        try:
            actual_length = float(self.actual_length_var.get())
        except (TypeError, ValueError):
            actual_length = 0.0
        return mode, pixel_length, actual_length, self.unit_var.get()

    def set_detected_scale_bar(self, pixel_length: Optional[float]) -> None:
        self.detected_px_var.set("" if pixel_length is None else f"{pixel_length:.3f}")
        if pixel_length:
            self.calibration_mode_var.set("auto")

    def set_manual_pixel_length(self, pixel_length: float) -> None:
        self.manual_px_var.set(f"{pixel_length:.3f}")
        self.calibration_mode_var.set("manual")
