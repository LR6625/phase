import argparse
import time
from pathlib import Path

import cv2

from mtg_hash_v2.config import SegmentedHashConfig
from mtg_hash_v2.matcher import HybridTopHalfMatcher


WORKSPACE_DIR = Path(__file__).resolve().parent


def iter_images(image_dir: Path):
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"):
        yield from image_dir.glob(ext)


def main():
    parser = argparse.ArgumentParser(description="Benchmark segmented matcher v2 on a folder of warped card images.")
    parser.add_argument("--image-dir", required=True, help="Folder containing warped card images.")
    parser.add_argument("--segmented-db", default="scryfall_segmented_tophalf_db_v2.json")
    parser.add_argument("--source-module", default="Expo_P.H.A.S.E.py")
    parser.add_argument("--full-db", default="scryfall_phash_db_sample.json")
    args = parser.parse_args()

    matcher = HybridTopHalfMatcher(
        source_module_path=WORKSPACE_DIR / args.source_module,
        full_card_db_path=WORKSPACE_DIR / args.full_db,
        segmented_db_path=WORKSPACE_DIR / args.segmented_db,
        cfg=SegmentedHashConfig(),
    )

    image_dir = Path(args.image_dir)
    image_paths = sorted(iter_images(image_dir))
    if not image_paths:
        raise FileNotFoundError(f"No images found under: {image_dir}")

    durations = []
    total = len(image_paths)
    t_start = time.perf_counter()
    for i, path in enumerate(image_paths):
        img = cv2.imread(str(path))
        if img is None:
            print(f"[{i+1}/{total}] SKIP (unreadable): {path.name}")
            continue

        t0 = time.perf_counter()
        result = matcher.identify(img)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        durations.append(elapsed_ms)

        elapsed_total = time.perf_counter() - t_start
        done = i + 1
        rate = done / elapsed_total if elapsed_total > 0 else 0
        eta_s = (total - done) / rate if rate > 0 else 0
        eta_str = f"{int(eta_s//60)}m{int(eta_s%60):02d}s" if eta_s >= 60 else f"{eta_s:.0f}s"
        pct = done / total * 100

        best = result.get("best")
        card_str = f"{best['name']}" if best else "reject"
        print(f"[{done}/{total} {pct:.0f}% ETA {eta_str}] {path.name}: {card_str} | {result['status']} | {elapsed_ms:.1f} ms")

    if not durations:
        print("No images processed.")
        return

    mean = sum(durations) / len(durations)
    p95 = sorted(durations)[max(0, int(len(durations) * 0.95) - 1)]
    total_elapsed = time.perf_counter() - t_start
    print(f"\nSamples={len(durations)} | Mean={mean:.1f} ms | P95={p95:.1f} ms | Total={total_elapsed:.1f}s")


if __name__ == "__main__":
    main()
