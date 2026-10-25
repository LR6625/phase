"""
Convert YOLO bbox labels (cx cy w h) to OBB labels (x1y1 x2y2 x3y3 x4y4)
for val and test splits, where boxes are axis-aligned (angle=0).
OBB format per line: class x1 y1 x2 y2 x3 y3 x4 y4
"""
import os
from pathlib import Path

SPLITS = ["val", "test"]
SRC_ROOT = Path("yolo_dataset/labels")
DST_ROOT = Path("yolo_dataset/labels_obb")


def bbox_to_obb(cx: float, cy: float, w: float, h: float) -> str:
    """Return OBB string for axis-aligned box: top-left → top-right → bottom-right → bottom-left."""
    half_w, half_h = w / 2, h / 2
    x1, y1 = cx - half_w, cy - half_h  # top-left
    x2, y2 = cx + half_w, cy - half_h  # top-right
    x3, y3 = cx + half_w, cy + half_h  # bottom-right
    x4, y4 = cx - half_w, cy + half_h  # bottom-left
    return f"{x1:.6f} {y1:.6f} {x2:.6f} {y2:.6f} {x3:.6f} {y3:.6f} {x4:.6f} {y4:.6f}"


def convert_split(split: str) -> None:
    src_dir = SRC_ROOT / split
    dst_dir = DST_ROOT / split
    dst_dir.mkdir(parents=True, exist_ok=True)

    files = list(src_dir.glob("*.txt"))
    converted = 0
    for f in files:
        lines_out = []
        for line in f.read_text().strip().splitlines():
            parts = line.split()
            if len(parts) != 5:
                # Pass through any already-OBB lines or empty lines
                lines_out.append(line)
                continue
            cls = parts[0]
            cx, cy, w, h = map(float, parts[1:])
            obb = bbox_to_obb(cx, cy, w, h)
            lines_out.append(f"{cls} {obb}")
        (dst_dir / f.name).write_text("\n".join(lines_out) + "\n")
        converted += 1

    print(f"  [{split}] converted {converted} files → {dst_dir}")


if __name__ == "__main__":
    for split in SPLITS:
        convert_split(split)
    print("Done.")
