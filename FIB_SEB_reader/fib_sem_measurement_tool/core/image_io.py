from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def list_image_files(folder: str) -> List[str]:
    root = Path(folder)
    paths: List[str] = []
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            paths.append(str(path))
    return sorted(paths)


def filter_image_paths(paths: Iterable[str]) -> List[str]:
    return [str(path) for path in paths if Path(path).suffix.lower() in IMAGE_EXTENSIONS]


def load_image_unicode(path: str) -> np.ndarray:
    data = np.fromfile(path, dtype=np.uint8)
    if data.size == 0:
        raise ValueError(f"이미지를 읽을 수 없습니다: {path}")
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"지원하지 않거나 손상된 이미지입니다: {path}")
    return image


def save_image_unicode(path: str, image: np.ndarray) -> None:
    ext = Path(path).suffix or ".png"
    ok, encoded = cv2.imencode(ext, image)
    if not ok:
        raise ValueError(f"이미지 저장 인코딩 실패: {path}")
    encoded.tofile(path)


def to_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def create_thumbnail(image: np.ndarray, size: Tuple[int, int] = (148, 96)) -> Image.Image:
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    pil.thumbnail(size, Image.LANCZOS)
    thumb = Image.new("RGB", size, (14, 24, 33))
    x = (size[0] - pil.width) // 2
    y = (size[1] - pil.height) // 2
    thumb.paste(pil, (x, y))
    return thumb


def read_image_metadata(path: str) -> Tuple[Tuple[int, int], Optional[Image.Image]]:
    image = load_image_unicode(path)
    height, width = image.shape[:2]
    return (width, height), create_thumbnail(image)

