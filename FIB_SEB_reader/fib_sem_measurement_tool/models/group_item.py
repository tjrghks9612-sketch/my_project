from __future__ import annotations

from dataclasses import dataclass

from .settings import MeasurementSettings


@dataclass
class GroupItem:
    group_id: str
    group_name: str
    source_image: str
    shared_settings: MeasurementSettings

