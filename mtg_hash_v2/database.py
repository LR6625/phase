from __future__ import annotations

import json
from pathlib import Path

import cv2

from .config import SegmentedHashConfig
from .segmented_hashing import compute_segmented_hashes


def make_card_key(name: str | None, set_code: str | None, scryfall_uri: str | None) -> str:
    if scryfall_uri:
        return f"uri::{scryfall_uri.strip().lower()}"
    name_norm = (name or "unknown").strip().lower()
    set_norm = (set_code or "unknown").strip().lower()
    return f"name_set::{name_norm}::{set_norm}"


def make_card_key_from_info(card_info: dict) -> str:
    return make_card_key(card_info.get("name"), card_info.get("set"), card_info.get("scryfall_uri"))


def save_segmented_db(records: list[dict], output_path: str | Path):
    output = Path(output_path)
    with output.open("w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False)


def load_segmented_db(path: str | Path) -> list[dict]:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def build_segmented_db_from_manifest(manifest_path: str | Path, output_path: str | Path, cfg: SegmentedHashConfig):
    """
    Manifest format:
    [
      {
        "name": "Shriekgeist",
        "set": "isd",
        "scryfall_uri": "https://scryfall.com/card/...",
        "image_path": "Image_storage/shriekgeist.jpg"
      }
    ]
    """
    manifest_path = Path(manifest_path)
    with manifest_path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)

    records: list[dict] = []
    root_dir = manifest_path.parent
    for entry in manifest:
        image_path = Path(entry["image_path"])
        if not image_path.is_absolute():
            image_path = (root_dir / image_path).resolve()

        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")

        key = make_card_key(entry.get("name"), entry.get("set"), entry.get("scryfall_uri"))
        segmented = compute_segmented_hashes(image, cfg)
        record = {
            "card_key": key,
            "name": entry.get("name"),
            "set": entry.get("set"),
            "scryfall_uri": entry.get("scryfall_uri"),
            "image_path": str(image_path),
            "segmented": segmented,
        }
        records.append(record)

    save_segmented_db(records, output_path)
    return records
