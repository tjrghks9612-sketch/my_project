from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

from PIL import Image

from .result import MeasurementResult
from .settings import MeasurementSettings


@dataclass
class ImageItem:
    image_path: str
    file_name: str
    image_size: Tuple[int, int]
    thumbnail: Optional[Image.Image] = None
    group_id: str = ""
    group_name: str = ""
    selected: bool = False
    settings: Optional[MeasurementSettings] = None
    result: Optional[MeasurementResult] = None
    last_error: str = ""

    @classmethod
    def from_path(cls, path: str, image_size: Tuple[int, int], thumbnail: Optional[Image.Image]) -> "ImageItem":
        return cls(
            image_path=str(path),
            file_name=Path(path).name,
            image_size=image_size,
            thumbnail=thumbnail,
        )

