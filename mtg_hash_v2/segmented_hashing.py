from __future__ import annotations

from typing import Any

import cv2
import imagehash
from PIL import Image

from .config import SegmentedHashConfig


def normalize_card(image_bgr, size=(744, 1040)):
    width, height = size
    return cv2.resize(image_bgr, (width, height), interpolation=cv2.INTER_AREA)


def crop_top_half(image_bgr, top_fraction=0.55):
    h, w = image_bgr.shape[:2]
    y2 = max(1, min(h, int(round(h * top_fraction))))
    return image_bgr[:y2, :w]


def split_into_grid(image_bgr, rows, cols):
    h, w = image_bgr.shape[:2]
    regions = []
    for r in range(rows):
        y1 = int(round(r * h / rows))
        y2 = int(round((r + 1) * h / rows))
        for c in range(cols):
            x1 = int(round(c * w / cols))
            x2 = int(round((c + 1) * w / cols))
            region = image_bgr[y1:y2, x1:x2]
            regions.append({
                "index": r * cols + c,
                "row": r,
                "col": c,
                "bbox": [x1, y1, x2, y2],
                "image": region,
            })
    return regions


def _compute_hashes(region_bgr):
    rgb = cv2.cvtColor(region_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)
    ph = str(imagehash.phash(pil_img))
    dh = str(imagehash.dhash(pil_img))
    return ph, dh


def compute_segmented_hashes(image_bgr, cfg: SegmentedHashConfig):
    normalized = normalize_card(image_bgr, cfg.normalized_size)
    top_half = crop_top_half(normalized, cfg.top_half_fraction)
    regions = split_into_grid(top_half, cfg.grid_rows, cfg.grid_cols)

    hash_regions: list[dict[str, Any]] = []
    for reg in regions:
        ph, dh = _compute_hashes(reg["image"])
        hash_regions.append(
            {
                "index": reg["index"],
                "row": reg["row"],
                "col": reg["col"],
                "bbox": reg["bbox"],
                "phash": ph,
                "dhash": dh,
            }
        )

    return {
        "normalized_size": list(cfg.normalized_size),
        "top_half_fraction": cfg.top_half_fraction,
        "rows": cfg.grid_rows,
        "cols": cfg.grid_cols,
        "regions": hash_regions,
    }


def hamming_distance(hex_a: str, hex_b: str) -> int:
    return imagehash.hex_to_hash(hex_a) - imagehash.hex_to_hash(hex_b)
