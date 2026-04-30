from __future__ import annotations

import math
from collections import OrderedDict
from copy import deepcopy
from pathlib import Path
from tkinter import filedialog, messagebox, simpledialog
from typing import Dict, List, Optional

import customtkinter as ctk

from fib_sem_measurement_tool.core.calibration import apply_calibration, detect_scale_bar
from fib_sem_measurement_tool.core.image_io import (
    filter_image_paths,
    list_image_files,
    load_image_unicode,
    save_image_unicode,
    read_image_metadata,
)
from fib_sem_measurement_tool.core.measurement_runner import run_measurement
from fib_sem_measurement_tool.core.overlay import draw_overlay
from fib_sem_measurement_tool.core.roi_utils import apply_roi_to_image, normalize_roi
from fib_sem_measurement_tool.export.csv_exporter import export_results_to_csv
from fib_sem_measurement_tool.models.group_item import GroupItem
from fib_sem_measurement_tool.models.image_item import ImageItem
from fib_sem_measurement_tool.models.settings import (
    MEASUREMENT_TYPES,
    MeasurementSettings,
    default_global_settings,
    resolve_effective_settings,
)
from fib_sem_measurement_tool.ui.image_viewer import ImageViewer
from fib_sem_measurement_tool.ui.option_panel import OptionPanel
from fib_sem_measurement_tool.ui.profile_graph import ProfileGraph
from fib_sem_measurement_tool.ui.thumbnail_panel import ThumbnailPanel


class MainWindow(ctk.CTk):
    def __init__(self):
        super().__init__()
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")
        self.title("FIB-SEM 측정기")
        self.geometry("1680x940")
        self.minsize(1280, 760)

        self.global_settings = default_global_settings()
        self.image_items: List[ImageItem] = []
        self.group_settings: Dict[str, GroupItem] = {}
        self.current_index = -1
        self.current_image = None
        self.current_image_path = ""
        self.overlay_enabled = True
        self.scale_bar_bboxes: Dict[str, tuple] = {}
        self.calibration_lines: Dict[str, tuple] = {}
        self._auto_measure_after_id = None
        self.auto_measure_delay_ms = 250
        self._thumbnail_refresh_after_id = None
        self._profile_image_path = ""
        self._last_option_signature = None
        self.image_cache = OrderedDict()
        self.image_cache_limit = 8
        self.render_cache = OrderedDict()
        self.render_cache_limit = 4
        self.status_var = ctk.StringVar(value="이미지를 불러오면 ROI를 드래그해서 측정을 시작할 수 있습니다.")

        self._build()

    def _build(self) -> None:
        self.configure(fg_color="#08111a")
        toolbar = ctk.CTkFrame(self, fg_color="#08111a")
        toolbar.pack(fill="x", padx=14, pady=(10, 6))
        ctk.CTkLabel(toolbar, text="FIB-SEM 측정기", font=ctk.CTkFont(size=20, weight="bold")).pack(side="left", padx=(4, 18))
        ctk.CTkButton(toolbar, text="이미지 불러오기", width=140, command=self.load_images_dialog).pack(side="left", padx=4)
        ctk.CTkButton(toolbar, text="폴더 불러오기", width=140, command=self.load_folder_dialog).pack(side="left", padx=4)
        ctk.CTkButton(toolbar, text="이전", width=90, fg_color="#142234", command=self.previous_image).pack(side="left", padx=(18, 4))
        ctk.CTkButton(toolbar, text="다음", width=90, fg_color="#142234", command=self.next_image).pack(side="left", padx=4)
        ctk.CTkButton(toolbar, text="현재 이미지 측정", width=140, command=lambda: self.measure_scope("current")).pack(side="left", padx=(18, 4))
        ctk.CTkButton(toolbar, text="일괄 측정", width=120, fg_color="#203246", command=lambda: self.measure_scope("all")).pack(side="left", padx=4)
        ctk.CTkButton(toolbar, text="CSV 저장", width=110, command=self.export_csv).pack(side="right", padx=4)

        main = ctk.CTkFrame(self, fg_color="#08111a")
        main.pack(fill="both", expand=True, padx=14, pady=(0, 8))
        main.grid_columnconfigure(0, weight=7, uniform="main")
        main.grid_columnconfigure(1, weight=3, uniform="main")
        main.grid_columnconfigure(2, weight=4, uniform="main")
        main.grid_rowconfigure(0, weight=1)

        left = ctk.CTkFrame(main, fg_color="transparent")
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        left.grid_rowconfigure(0, weight=5)
        left.grid_rowconfigure(1, weight=1)
        left.grid_columnconfigure(0, weight=1)
        self.viewer = ImageViewer(
            left,
            on_roi_changed=self.on_roi_changed,
            on_calibration_line=self.on_calibration_line,
            on_overlay_toggled=self.on_overlay_toggled,
            on_hover_profile=self.on_profile_hover,
        )
        self.viewer.grid(row=0, column=0, sticky="nsew")
        self.profile_graph = ProfileGraph(left)
        self.profile_graph.grid(row=1, column=0, sticky="nsew", pady=(8, 0))

        self.thumbnail_panel = ThumbnailPanel(
            main,
            on_select_image=self.select_image,
            on_selection_changed=self.refresh_thumbnail_panel,
        )
        self.thumbnail_panel.grid(row=0, column=1, sticky="nsew", padx=4)

        self.option_panel = OptionPanel(
            main,
            on_option_changed=self.on_option_changed,
            on_apply_settings=self.apply_settings_to_scope,
            on_apply_roi=self.apply_roi_to_scope,
            on_apply_calibration=self.apply_calibration_to_scope,
            on_detect_scale_bar=self.detect_current_scale_bar,
            on_draw_calibration_line=self.activate_calibration_line_mode,
            on_measure_scope=lambda: self.measure_scope(self.option_panel.get_scope()),
            on_create_group=self.create_group_from_selection,
            on_ungroup=self.ungroup_selection,
            on_reset_current=self.reset_current_settings,
        )
        self.option_panel.grid(row=0, column=2, sticky="nsew", padx=(8, 0))
        self.option_panel.set_settings(self.global_settings)

        status = ctk.CTkFrame(self, fg_color="#08111a")
        status.pack(fill="x", padx=18, pady=(0, 8))
        ctk.CTkLabel(status, textvariable=self.status_var, anchor="w", text_color="#9dacbc").pack(side="left", fill="x", expand=True)

    def set_status(self, message: str) -> None:
        self.status_var.set(message)
        self.update_idletasks()

    def load_images_dialog(self) -> None:
        paths = filedialog.askopenfilenames(
            title="FIB-SEM 이미지 선택",
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff"), ("All files", "*.*")],
        )
        self.add_image_paths(paths)

    def load_folder_dialog(self) -> None:
        folder = filedialog.askdirectory(title="이미지 폴더 선택")
        if folder:
            self.add_image_paths(list_image_files(folder))

    def add_image_paths(self, paths) -> None:
        image_paths = filter_image_paths(paths)
        if not image_paths:
            return
        existing = {item.image_path for item in self.image_items}
        added = 0
        errors = []
        for path in image_paths:
            if path in existing:
                continue
            try:
                image_size, thumbnail = read_image_metadata(path)
                self.image_items.append(ImageItem.from_path(path, image_size, thumbnail))
                added += 1
            except Exception as exc:
                errors.append(f"{Path(path).name}: {exc}")
        if self.current_index < 0 and self.image_items:
            self.current_index = 0
            self.load_current_image()
        self.refresh_all()
        self.set_status(f"{added}개 이미지를 불러왔습니다." + (f" 실패 {len(errors)}개" if errors else ""))
        if errors:
            messagebox.showwarning("일부 이미지 로드 실패", "\n".join(errors[:8]))

    def resolve_settings_for_item(self, item: ImageItem) -> MeasurementSettings:
        return resolve_effective_settings(item, self.group_settings, self.global_settings)

    def current_item(self) -> Optional[ImageItem]:
        if 0 <= self.current_index < len(self.image_items):
            return self.image_items[self.current_index]
        return None

    def current_settings(self) -> MeasurementSettings:
        item = self.current_item()
        if item is None:
            return self.global_settings.clone()
        return self.resolve_settings_for_item(item)

    def select_image(self, index: int) -> None:
        if not (0 <= index < len(self.image_items)):
            return
        if index == self.current_index and self.current_image is not None:
            return
        self._cancel_auto_measure()
        self.current_index = index
        self.load_current_image()
        self.schedule_thumbnail_panel_refresh()

    def previous_image(self) -> None:
        if not self.image_items:
            return
        self.select_image(max(0, self.current_index - 1))

    def next_image(self) -> None:
        if not self.image_items:
            return
        self.select_image(min(len(self.image_items) - 1, self.current_index + 1))

    def _cancel_auto_measure(self) -> None:
        if self._auto_measure_after_id is not None:
            try:
                self.after_cancel(self._auto_measure_after_id)
            except ValueError:
                pass
            self._auto_measure_after_id = None

    def load_image_cached(self, path: str):
        cached = self.image_cache.get(path)
        if cached is not None:
            self.image_cache.move_to_end(path)
            return cached
        image = load_image_unicode(path)
        self.image_cache[path] = image
        self.image_cache.move_to_end(path)
        while len(self.image_cache) > self.image_cache_limit:
            self.image_cache.popitem(last=False)
        return image

    def load_current_image(self) -> None:
        item = self.current_item()
        if item is None:
            self.current_image = None
            self.current_image_path = ""
            self.viewer.clear()
            self.profile_graph.set_image(None)
            self._profile_image_path = ""
            self._last_option_signature = None
            return
        if self.current_image_path != item.image_path:
            self.current_image = self.load_image_cached(item.image_path)
            self.current_image_path = item.image_path
            self._last_option_signature = None
        self.render_current_image()

    def render_current_image(self) -> None:
        item = self.current_item()
        if item is None or self.current_image is None:
            self.viewer.clear()
            return
        settings = self.resolve_settings_for_item(item)
        rendered = self.get_rendered_preview(item, settings)
        group_label = f"{item.group_id or '-'} / {item.group_name or '-'}"
        measurement_label = MEASUREMENT_TYPES.get(settings.measurement_type, settings.measurement_type)
        status = item.result.status + f" {item.result.overall_confidence:.0f}%" if item.result else settings.settings_source
        title = f"{item.file_name}"
        meta = f"{group_label} | {measurement_label} | {settings.settings_source}"
        self.viewer.set_content(self.current_image, rendered, title, meta, status)
        if self._profile_image_path != item.image_path:
            self.profile_graph.set_image(self.current_image)
            self._profile_image_path = item.image_path
        self.option_panel.set_settings(settings)

    def _render_cache_key(self, item: ImageItem, settings: MeasurementSettings):
        calibration_line = self.calibration_lines.get(item.image_path)
        scale_bar_bbox = self.scale_bar_bboxes.get(item.image_path)
        calibration = settings.calibration
        return (
            item.image_path,
            id(item.result),
            settings.roi,
            settings.measurement_type,
            settings.edge_reference,
            settings.distance_method,
            settings.taper_side,
            round(float(calibration.px_to_real), 8),
            calibration.unit,
            self.overlay_enabled,
            calibration_line,
            scale_bar_bbox,
        )

    def get_rendered_preview(self, item: ImageItem, settings: MeasurementSettings):
        key = self._render_cache_key(item, settings)
        cached = self.render_cache.get(key)
        if cached is not None:
            self.render_cache.move_to_end(key)
            return cached
        rendered = draw_overlay(
            self.current_image,
            settings.roi,
            item.result,
            settings,
            show_overlay=self.overlay_enabled,
            calibration_line=self.calibration_lines.get(item.image_path),
            scale_bar_bbox=self.scale_bar_bboxes.get(item.image_path),
        )
        self.render_cache[key] = rendered
        self.render_cache.move_to_end(key)
        while len(self.render_cache) > self.render_cache_limit:
            self.render_cache.popitem(last=False)
        return rendered

    def on_profile_hover(self, x: Optional[int], y: Optional[int]) -> None:
        self.profile_graph.update_cursor(x, y)

    def refresh_thumbnail_panel(self) -> None:
        self.thumbnail_panel.refresh(self.image_items, self.current_index, self.resolve_settings_for_item)

    def schedule_thumbnail_panel_refresh(self) -> None:
        if self._thumbnail_refresh_after_id is not None:
            try:
                self.after_cancel(self._thumbnail_refresh_after_id)
            except ValueError:
                pass
        self._thumbnail_refresh_after_id = self.after(320, self._flush_thumbnail_panel_refresh)

    def _flush_thumbnail_panel_refresh(self) -> None:
        self._thumbnail_refresh_after_id = None
        self.refresh_thumbnail_panel()

    def refresh_all(self) -> None:
        self.render_current_image()
        self.refresh_thumbnail_panel()

    def _ensure_item_settings(self, item: ImageItem, source: str = "image_specific") -> MeasurementSettings:
        settings = self.resolve_settings_for_item(item)
        settings.settings_source = source
        item.settings = settings
        return settings

    def _settings_signature(self, settings: MeasurementSettings):
        adv = settings.advanced
        cal = settings.calibration
        return (
            settings.roi,
            settings.measurement_type,
            settings.taper_side,
            settings.distance_method,
            settings.edge_reference,
            settings.noise_level,
            adv.blur_kernel,
            adv.median_filter_size,
            round(float(adv.background_correction_strength), 4),
            round(float(adv.sensitivity), 4),
            round(float(adv.peak_prominence), 4),
            adv.scan_line_count,
            adv.minimum_valid_line_count,
            round(float(adv.min_valid_line_ratio), 4),
            round(float(adv.outlier_rejection_strength), 4),
            round(float(adv.fit_error_threshold), 4),
            round(float(adv.confidence_threshold), 4),
            bool(adv.overlay_save_enabled),
            bool(adv.profile_graph_enabled),
            round(float(cal.px_to_real), 8),
            cal.unit,
        )

    def on_option_changed(self, advanced: bool) -> None:
        item = self.current_item()
        if item is None:
            self.global_settings = self.option_panel.get_settings(self.global_settings, mark_custom=advanced)
            self.global_settings.settings_source = "global_default"
            return
        base = self.resolve_settings_for_item(item)
        settings = self.option_panel.get_settings(base, mark_custom=advanced)
        settings.settings_source = "image_specific"
        signature = self._settings_signature(settings)
        if signature == self._last_option_signature:
            return
        self._last_option_signature = signature
        item.settings = settings
        if settings.roi is not None and self.current_image is not None:
            item.result = None
            self.render_current_image()
            self._schedule_current_auto_measure()
        else:
            self.render_current_image()
            self.schedule_thumbnail_panel_refresh()

    def _schedule_current_auto_measure(self) -> None:
        if self._auto_measure_after_id is not None:
            try:
                self.after_cancel(self._auto_measure_after_id)
            except ValueError:
                pass
        self._auto_measure_after_id = self.after(self.auto_measure_delay_ms, self._auto_measure_current)

    def _auto_measure_current(self) -> None:
        self._auto_measure_after_id = None
        item = self.current_item()
        if item is None or self.current_image is None:
            return
        settings = self.resolve_settings_for_item(item)
        if settings.roi is None:
            self.render_current_image()
            self.refresh_thumbnail_panel()
            return
        item.result = run_measurement(self.current_image, settings)
        self.render_current_image()
        self.schedule_thumbnail_panel_refresh()
        self.set_status(f"옵션 변경 자동 적용: {item.file_name} / {item.result.status} {item.result.overall_confidence:.0f}%")

    def on_roi_changed(self, roi) -> None:
        item = self.current_item()
        if item is None:
            return
        clean_roi = normalize_roi(roi, item.image_size)
        if clean_roi is None:
            self.set_status("ROI가 너무 작습니다.")
            return

        mode = self.option_panel.get_roi_apply_mode()
        selected_targets = [image for image in self.image_items if image.selected]
        targets = selected_targets or [item]
        if item not in targets:
            targets = [item] + targets

        applied = 0
        for target_item in targets:
            target_roi = apply_roi_to_image(clean_roi, item.image_size, target_item.image_size, mode)
            if target_roi is None:
                continue
            source = "image_specific" if target_item is item else "copied_from_previous"
            settings = self._ensure_item_settings(target_item, source)
            settings.roi = target_roi
            settings.roi_apply_mode = mode
            settings.roi_source_image = item.file_name
            target_item.result = None
            applied += 1

        self.set_status(f"ROI 적용 완료: {applied}/{len(targets)}개 이미지 ({mode})")
        self.refresh_all()
        if self.current_image is not None:
            self._schedule_current_auto_measure()

    def on_calibration_line(self, line, length: float) -> None:
        item = self.current_item()
        if item is None:
            return
        self.calibration_lines[item.image_path] = line
        self.option_panel.set_manual_pixel_length(length)
        self.set_status(f"수동 캘리브레이션 선 길이: {length:.2f} px")
        self.render_current_image()

    def on_overlay_toggled(self, enabled: bool) -> None:
        self.overlay_enabled = enabled
        self.render_current_image()

    def activate_calibration_line_mode(self) -> None:
        self.viewer.set_mode("calibration")
        self.set_status("메인 이미지 위에서 캘리브레이션 기준선을 드래그하세요.")

    def detect_current_scale_bar(self) -> None:
        item = self.current_item()
        if item is None or self.current_image is None:
            return
        result = detect_scale_bar(self.current_image)
        if result.get("status") == "detected":
            pixel_length = result.get("pixel_length")
            self.option_panel.set_detected_scale_bar(float(pixel_length))
            self.scale_bar_bboxes[item.image_path] = result.get("bbox")
            self.set_status(f"스케일바 후보 감지: {pixel_length:.2f} px")
        else:
            self.option_panel.set_detected_scale_bar(None)
            self.set_status(str(result.get("message", "스케일바 감지 실패")))
            messagebox.showinfo("스케일바 감지", str(result.get("message", "스케일바 감지 실패")))
        self.render_current_image()

    def _targets_for_scope(self, scope: str) -> List[ImageItem]:
        item = self.current_item()
        if scope == "all":
            return list(self.image_items)
        if scope == "selected":
            selected = [image for image in self.image_items if image.selected]
            return selected or ([item] if item else [])
        if scope == "group" and item and item.group_id:
            return [image for image in self.image_items if image.group_id == item.group_id]
        return [item] if item else []

    def _copy_measurement_options(self, source: MeasurementSettings, target: MeasurementSettings) -> MeasurementSettings:
        target.measurement_type = source.measurement_type
        target.taper_side = source.taper_side
        target.distance_method = source.distance_method
        target.edge_reference = source.edge_reference
        target.noise_level = source.noise_level
        target.roi_apply_mode = source.roi_apply_mode
        target.advanced = deepcopy(source.advanced)
        target.custom_option = source.custom_option
        return target

    def apply_settings_to_scope(self) -> None:
        if not self.image_items:
            self.global_settings = self.option_panel.get_settings(self.global_settings)
            return
        scope = self.option_panel.get_scope()
        source = self.option_panel.get_settings(self.current_settings())
        item = self.current_item()
        if scope == "group" and item and item.group_id and item.group_id in self.group_settings:
            group = self.group_settings[item.group_id]
            shared = self._copy_measurement_options(source, group.shared_settings.clone())
            shared.settings_source = "group_shared"
            group.shared_settings = shared
            self.set_status(f"{group.group_name}: 그룹 공통 설정 적용")
        else:
            targets = self._targets_for_scope(scope)
            for target_item in targets:
                target = self.resolve_settings_for_item(target_item)
                updated = self._copy_measurement_options(source, target)
                updated.settings_source = "image_specific"
                target_item.settings = updated
            self.set_status(f"{len(targets)}개 이미지에 설정 적용")
        self.refresh_all()

    def apply_roi_to_scope(self) -> None:
        source_item = self.current_item()
        if source_item is None:
            return
        source_settings = self.current_settings()
        if source_settings.roi is None:
            messagebox.showwarning("ROI 없음", "현재 이미지에 ROI를 먼저 지정하세요.")
            return

        scope = self.option_panel.get_scope()
        mode = self.option_panel.get_roi_apply_mode()
        targets = self._targets_for_scope(scope)
        applied = 0

        if scope == "group" and source_item.group_id and source_item.group_id in self.group_settings:
            group = self.group_settings[source_item.group_id]
            group.shared_settings.roi = source_settings.roi
            group.shared_settings.roi_apply_mode = mode
            group.shared_settings.roi_source_image = source_item.file_name
            group.shared_settings.settings_source = "group_shared"

        if scope == "all":
            self.global_settings.roi = source_settings.roi
            self.global_settings.roi_apply_mode = mode
            self.global_settings.roi_source_image = source_item.file_name

        for target_item in targets:
            target_roi = apply_roi_to_image(source_settings.roi, source_item.image_size, target_item.image_size, mode)
            if target_roi is None:
                continue
            settings_source = "group_shared" if scope == "group" and target_item.group_id == source_item.group_id else "copied_from_previous"
            target_settings = self._ensure_item_settings(target_item, settings_source)
            target_settings.roi = target_roi
            target_settings.roi_apply_mode = mode
            target_settings.roi_source_image = source_item.file_name
            applied += 1

        self.set_status(f"ROI 적용 완료: {applied}/{len(targets)}개 이미지 ({mode})")
        self.refresh_all()

    def apply_calibration_to_scope(self) -> None:
        mode, pixel_length, actual_length, unit = self.option_panel.get_calibration_inputs()
        calibration = apply_calibration(pixel_length, actual_length, unit, mode=mode)
        if calibration.status != "calibrated":
            messagebox.showwarning("캘리브레이션 실패", "pixel length와 실제 길이를 올바르게 입력하세요.")
            return
        scope = self.option_panel.get_scope()
        item = self.current_item()
        if scope == "group" and item and item.group_id and item.group_id in self.group_settings:
            self.group_settings[item.group_id].shared_settings.calibration = calibration
            self.group_settings[item.group_id].shared_settings.settings_source = "group_shared"
            self.set_status(f"{self.group_settings[item.group_id].group_name}: 그룹 캘리브레이션 적용")
        else:
            targets = self._targets_for_scope(scope)
            for target_item in targets:
                settings = self._ensure_item_settings(target_item, "image_specific")
                settings.calibration = deepcopy(calibration)
            self.set_status(f"{len(targets)}개 이미지에 캘리브레이션 적용 ({calibration.px_to_real:.6g} {unit}/px)")
        self.refresh_all()

    def create_group_from_selection(self) -> None:
        if not self.image_items:
            return
        targets = [item for item in self.image_items if item.selected]
        if not targets and self.current_item():
            targets = [self.current_item()]
        group_id = f"group_{len(self.group_settings) + 1:02d}"
        default_name = f"set_{len(self.group_settings) + 1:02d}"
        group_name = simpledialog.askstring("그룹 생성", "그룹 이름", initialvalue=default_name, parent=self) or default_name
        source_item = self.current_item() or targets[0]
        shared = self.option_panel.get_settings(self.current_settings())
        shared.settings_source = "group_shared"
        shared.roi_source_image = source_item.file_name
        self.group_settings[group_id] = GroupItem(group_id, group_name, source_item.file_name, shared)
        for item in targets:
            item.group_id = group_id
            item.group_name = group_name
        self.set_status(f"{len(targets)}개 이미지를 {group_id} / {group_name} 그룹으로 묶었습니다.")
        self.refresh_all()

    def ungroup_selection(self) -> None:
        targets = [item for item in self.image_items if item.selected]
        if not targets and self.current_item():
            targets = [self.current_item()]
        for item in targets:
            item.group_id = ""
            item.group_name = ""
        used = {item.group_id for item in self.image_items if item.group_id}
        for group_id in list(self.group_settings.keys()):
            if group_id not in used:
                del self.group_settings[group_id]
        self.set_status(f"{len(targets)}개 이미지의 그룹을 해제했습니다.")
        self.refresh_all()

    def reset_current_settings(self, mode: str) -> None:
        item = self.current_item()
        if item is None:
            self.global_settings = default_global_settings()
            self.option_panel.set_settings(self.global_settings)
            return
        if mode == "clear" or mode == "group":
            item.settings = None
            self.set_status(f"{item.file_name}: 개별 설정을 제거했습니다.")
        elif mode == "global":
            settings = default_global_settings()
            settings.settings_source = "global_default"
            item.settings = settings
            self.set_status(f"{item.file_name}: 전역 기본값으로 되돌렸습니다.")
        self.refresh_all()

    def measure_scope(self, force_scope: Optional[str] = None) -> None:
        if not self.image_items:
            return
        scope = force_scope or self.option_panel.get_scope()
        targets = self._targets_for_scope(scope)
        if not targets:
            return
        failures = 0
        for idx, item in enumerate(targets, start=1):
            self.set_status(f"측정 중 {idx}/{len(targets)}: {item.file_name}")
            image = self.load_image_cached(item.image_path)
            settings = self.resolve_settings_for_item(item)
            item.result = run_measurement(image, settings)
            if item.result.status == "Fail":
                failures += 1
            if settings.advanced.overlay_save_enabled:
                rendered = draw_overlay(
                    image,
                    settings.roi,
                    item.result,
                    settings,
                    show_overlay=True,
                    calibration_line=self.calibration_lines.get(item.image_path),
                    scale_bar_bbox=self.scale_bar_bboxes.get(item.image_path),
                )
                out_path = Path(item.image_path).with_name(f"{Path(item.image_path).stem}_overlay.png")
                save_image_unicode(str(out_path), rendered)
        self.load_current_image()
        self.refresh_thumbnail_panel()
        self.set_status(f"{len(targets)}개 이미지 측정 완료" + (f" / Fail {failures}개" if failures else ""))

    def export_csv(self) -> None:
        if not self.image_items:
            return
        path = filedialog.asksaveasfilename(
            title="CSV 저장",
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv")],
        )
        if not path:
            return
        export_results_to_csv(path, self.image_items, self.group_settings, self.global_settings)
        self.set_status(f"CSV 저장 완료: {path}")
        messagebox.showinfo("CSV 저장", "결과 CSV를 저장했습니다.")
