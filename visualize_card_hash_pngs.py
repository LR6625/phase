from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import imagehash
import matplotlib
import numpy as np
from PIL import Image

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mtg_hash_v2.config import SegmentedHashConfig
from mtg_hash_v2.segmented_hashing import crop_top_half, normalize_card, split_into_grid


def _find_default_example_image(workspace_root: Path) -> Path | None:
    search_roots = [
        workspace_root / "Identified_Cards",
        workspace_root / "buggingme",
        workspace_root / "Debugging images",
        workspace_root / "Image_storage",
    ]
    preferred_patterns = [
        "**/top_crop.png",
        "**/rotated_cropped_image.png",
        "**/locator_09_final_crop.png",
        "**/card_*.jpg",
        "**/card_*.png",
        "**/*.jpg",
        "**/*.png",
    ]

    for root in search_roots:
        if not root.exists():
            continue
        for pattern in preferred_patterns:
            matches = sorted(root.glob(pattern))
            if matches:
                return matches[0]
    return None


def _load_bgr(image_path: Path) -> np.ndarray:
    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        raise ValueError(f"Could not load image: {image_path}")
    return image_bgr


def _phash_intermediates(image_bgr: np.ndarray) -> dict[str, object]:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    gray32 = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA)
    f32 = np.float32(gray32)
    dct = cv2.dct(f32)
    low8 = dct[:8, :8]
    median = float(np.median(low8[1:, 1:]))
    bits8 = (low8 > median).astype(np.uint8)

    pil_img = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
    phash_obj = imagehash.phash(pil_img)
    return {
        "phash": phash_obj,
        "gray32": gray32,
        "dct32": dct,
        "low8": low8,
        "bits8": bits8,
        "median": median,
    }


def _build_bit_grid_image(bits8: np.ndarray, title: str) -> np.ndarray:
    cell = 40
    pad = 18
    h = (8 * cell) + (2 * pad)
    w = (8 * cell) + (2 * pad)
    canvas = np.full((h, w, 3), (255, 255, 255), dtype=np.uint8)

    for r in range(8):
        for c in range(8):
            y1 = pad + (r * cell)
            x1 = pad + (c * cell)
            y2 = y1 + cell - 2
            x2 = x1 + cell - 2
            bit = int(bits8[r, c])
            fill = (0, 0, 0) if bit == 1 else (255, 255, 255)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), fill, -1)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (90, 90, 90), 2)
            cv2.putText(
                canvas,
                str(bit),
                (x1 + 11, y1 + 27),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 255) if bit == 1 else (210, 80, 80),
                1,
                cv2.LINE_AA,
            )

    cv2.putText(canvas, title, (16, h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (20, 90, 170), 2, cv2.LINE_AA)
    return canvas


def _build_segment_mosaic(segment_bits: list[np.ndarray], region_hashes: list[str], cfg: SegmentedHashConfig) -> np.ndarray:
    rows = int(cfg.grid_rows)
    cols = int(cfg.grid_cols)
    bit_cell = 6
    bit_pad = 6
    grid_size = (8 * bit_cell) + (2 * bit_pad)
    tile_w = grid_size + 24
    tile_h = grid_size + 38
    margin = 18
    gap = 12

    canvas_w = (cols * tile_w) + ((cols - 1) * gap) + (2 * margin)
    canvas_h = (rows * tile_h) + ((rows - 1) * gap) + (2 * margin) + 42
    canvas = np.full((canvas_h, canvas_w, 3), (255, 255, 255), dtype=np.uint8)

    for idx in range(rows * cols):
        r = idx // cols
        c = idx % cols
        x0 = margin + c * (tile_w + gap)
        y0 = margin + r * (tile_h + gap)
        bits8 = segment_bits[idx] if idx < len(segment_bits) else np.zeros((8, 8), dtype=np.uint8)
        hash_text = region_hashes[idx] if idx < len(region_hashes) else ""

        cv2.rectangle(canvas, (x0, y0), (x0 + tile_w - 1, y0 + tile_h - 1), (0, 120, 220), 2)
        cv2.putText(
            canvas,
            f"R{idx + 1}",
            (x0 + 6, y0 + 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 120, 220),
            2,
            cv2.LINE_AA,
        )
        if hash_text:
            cv2.putText(
                canvas,
                hash_text[:10],
                (x0 + 6, y0 + tile_h - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.44,
                (70, 70, 70),
                1,
                cv2.LINE_AA,
            )

        gx = x0 + 9
        gy = y0 + 23
        for rr in range(8):
            for cc in range(8):
                bx1 = gx + bit_pad + (cc * bit_cell)
                by1 = gy + bit_pad + (rr * bit_cell)
                bx2 = bx1 + bit_cell - 1
                by2 = by1 + bit_cell - 1
                bit_val = int(bits8[rr, cc])
                fill = (0, 0, 0) if bit_val == 1 else (255, 255, 255)
                cv2.rectangle(canvas, (bx1, by1), (bx2, by2), fill, -1)
                cv2.rectangle(canvas, (bx1, by1), (bx2, by2), (140, 140, 140), 1)

    cv2.putText(
        canvas,
        "Segmented top-half hash tiles (each tile is an 8x8 perceptual hash)",
        (14, canvas_h - 14),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.74,
        (20, 90, 170),
        2,
        cv2.LINE_AA,
    )
    return canvas


def _build_phash_figure(image_bgr: np.ndarray, source_name: str) -> plt.Figure:
    details = _phash_intermediates(image_bgr)
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    low8 = details["low8"]
    bits8 = details["bits8"]

    fig = plt.figure(figsize=(16, 11), dpi=180, facecolor="#FFFFFF")
    gs = fig.add_gridspec(2, 2, height_ratios=[1.05, 1.0], wspace=0.16, hspace=0.18)

    ax_original = fig.add_subplot(gs[0, 0])
    ax_gray32 = fig.add_subplot(gs[0, 1])
    ax_low8 = fig.add_subplot(gs[1, 0])
    ax_bits = fig.add_subplot(gs[1, 1])

    ax_original.imshow(rgb)
    ax_original.set_title("Original identified card", fontsize=15, fontweight="bold")
    ax_original.axis("off")

    ax_gray32.imshow(details["gray32"], cmap="gray")
    ax_gray32.set_title("32x32 grayscale input", fontsize=15, fontweight="bold")
    ax_gray32.axis("off")

    ax_low8.imshow(low8, cmap="gray_r", interpolation="nearest")
    ax_low8.set_title("8x8 low-frequency DCT block", fontsize=15, fontweight="bold")
    ax_low8.set_xticks(range(8))
    ax_low8.set_yticks(range(8))
    ax_low8.grid(True, alpha=0.28)
    for i in range(8):
        for j in range(8):
            ax_low8.text(j, i, f"{low8[i, j]:.0f}", ha="center", va="center", fontsize=7, color="#7A1C1C")

    ax_bits.imshow(bits8, cmap="gray_r", interpolation="nearest")
    ax_bits.set_title("Final 8x8 hash bits", fontsize=15, fontweight="bold")
    ax_bits.set_xticks(range(8))
    ax_bits.set_yticks(range(8))
    ax_bits.grid(True, alpha=0.28)
    for i in range(8):
        for j in range(8):
            ax_bits.text(j, i, "1" if bits8[i, j] else "0", ha="center", va="center", fontsize=8, color="#C62828")

    hash_hex = str(details["phash"])
    binary = bin(int(hash_hex, 16))[2:].zfill(64)
    summary = (
        f"Source: {source_name}\n"
        f"Hash hex: {hash_hex}\n"
        f"Binary: {binary}\n"
        f"Mean threshold from low8 (excluding DC cell): {details['median']:.2f}"
    )
    fig.text(
        0.5,
        0.02,
        summary,
        ha="center",
        va="bottom",
        fontsize=12,
        color="#1F2E3A",
        family="monospace",
        bbox=dict(boxstyle="round,pad=0.45", facecolor="#FFF6E8", edgecolor="#F0C36D", linewidth=1.8),
    )
    fig.suptitle("Perceptual hash visualization", fontsize=22, fontweight="bold", y=0.98)
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    return fig


def _build_segmented_figure(image_bgr: np.ndarray, source_name: str, cfg: SegmentedHashConfig) -> plt.Figure:
    normalized = normalize_card(image_bgr, cfg.normalized_size)
    top_half = crop_top_half(normalized, cfg.top_half_fraction)
    segments = split_into_grid(top_half, cfg.grid_rows, cfg.grid_cols)

    segment_bits: list[np.ndarray] = []
    region_hashes: list[str] = []
    for segment in segments:
        seg_img = segment["image"]
        if seg_img is None or not isinstance(seg_img, np.ndarray) or seg_img.size == 0:
            seg_img = np.zeros((16, 16, 3), dtype=np.uint8)
        details = _phash_intermediates(seg_img)
        segment_bits.append(details["bits8"])
        region_hashes.append(str(details["phash"]))

    overlay = normalized.copy()
    top_h = int(round(overlay.shape[0] * float(cfg.top_half_fraction)))
    top_h = max(1, min(overlay.shape[0], top_h))
    rows = max(1, int(cfg.grid_rows))
    cols = max(1, int(cfg.grid_cols))
    cv2.rectangle(overlay, (0, 0), (overlay.shape[1] - 1, top_h - 1), (255, 200, 0), 2)
    for r in range(1, rows):
        y = int(round((top_h * r) / rows))
        cv2.line(overlay, (0, y), (overlay.shape[1] - 1, y), (255, 200, 0), 1, cv2.LINE_AA)
    for c in range(1, cols):
        x = int(round((overlay.shape[1] * c) / cols))
        cv2.line(overlay, (x, 0), (x, top_h - 1), (255, 200, 0), 1, cv2.LINE_AA)

    mosaic = _build_segment_mosaic(segment_bits, region_hashes, cfg)

    fig = plt.figure(figsize=(18, 11), dpi=180, facecolor="#FFFFFF")
    gs = fig.add_gridspec(2, 2, height_ratios=[1.1, 0.26], width_ratios=[1.05, 1.15], wspace=0.13, hspace=0.14)

    ax_overlay = fig.add_subplot(gs[0, 0])
    ax_mosaic = fig.add_subplot(gs[0, 1])
    ax_info = fig.add_subplot(gs[1, :])

    ax_overlay.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    ax_overlay.set_title("Normalized top half with 3x3 segmentation", fontsize=15, fontweight="bold")
    ax_overlay.axis("off")

    ax_mosaic.imshow(cv2.cvtColor(mosaic, cv2.COLOR_BGR2RGB))
    ax_mosaic.set_title("Segmented hash tiles", fontsize=15, fontweight="bold")
    ax_mosaic.axis("off")

    ax_info.axis("off")
    lines = [
        f"Source: {source_name}",
        f"Config: normalized_size={cfg.normalized_size}, top_half_fraction={cfg.top_half_fraction}, grid={cfg.grid_rows}x{cfg.grid_cols}",
        f"Regions: {len(region_hashes)} tiles, each tile produces its own 8x8 perceptual hash",
    ]
    for idx, region_hash in enumerate(region_hashes, start=1):
        lines.append(f"R{idx}: {region_hash}")
    ax_info.text(
        0.01,
        0.5,
        "\n".join(lines),
        ha="left",
        va="center",
        fontsize=10,
        family="monospace",
        color="#1F2E3A",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#FFF7E9", edgecolor="#F0C36D", linewidth=1.6),
    )

    fig.suptitle("Segmented hash visualization", fontsize=22, fontweight="bold", y=0.98)
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    return fig


def _resolve_input_path(args: argparse.Namespace, workspace_root: Path) -> Path:
    if args.input:
        return Path(args.input)

    example = _find_default_example_image(workspace_root)
    if example is None:
        raise FileNotFoundError(
            "No default card image found. Pass --input with a card crop or identified card image."
        )
    return example


def main() -> int:
    parser = argparse.ArgumentParser(description="Export PNGs for a card's 8x8 hash and segmented hash visualizations.")
    parser.add_argument("--input", help="Path to an identified card image or crop.")
    parser.add_argument(
        "--output-dir",
        default="hash_visualizations",
        help="Directory where PNGs will be written.",
    )
    parser.add_argument(
        "--mode",
        choices=("both", "phash", "segmented"),
        default="both",
        help="Which visualization(s) to export.",
    )
    args = parser.parse_args()

    workspace_root = Path(__file__).resolve().parent
    image_path = _resolve_input_path(args, workspace_root)
    image_bgr = _load_bgr(image_path)

    output_dir = workspace_root / args.output_dir / image_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)

    outputs: list[Path] = []

    if args.mode in ("both", "phash"):
        fig = _build_phash_figure(image_bgr, image_path.name)
        phash_path = output_dir / f"{image_path.stem}_phash_8x8.png"
        fig.savefig(phash_path, facecolor=fig.get_facecolor())
        plt.close(fig)
        outputs.append(phash_path)

    if args.mode in ("both", "segmented"):
        cfg = SegmentedHashConfig()
        fig = _build_segmented_figure(image_bgr, image_path.name, cfg)
        segmented_path = output_dir / f"{image_path.stem}_segmented_hash.png"
        fig.savefig(segmented_path, facecolor=fig.get_facecolor())
        plt.close(fig)
        outputs.append(segmented_path)

    for output_path in outputs:
        print(f"Saved {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())