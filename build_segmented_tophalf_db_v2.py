import argparse

from mtg_hash_v2.config import SegmentedHashConfig
from mtg_hash_v2.database import build_segmented_db_from_manifest


def main():
    parser = argparse.ArgumentParser(description="Build segmented top-half hash database v2.")
    parser.add_argument("--manifest", required=True, help="Path to manifest JSON with card metadata and image paths.")
    parser.add_argument("--output", default="scryfall_segmented_tophalf_db_v2.json", help="Output JSON path.")
    parser.add_argument("--rows", type=int, default=3)
    parser.add_argument("--cols", type=int, default=3)
    parser.add_argument("--top-half", type=float, default=0.55)
    args = parser.parse_args()

    cfg = SegmentedHashConfig(grid_rows=args.rows, grid_cols=args.cols, top_half_fraction=args.top_half)
    records = build_segmented_db_from_manifest(args.manifest, args.output, cfg)
    print(f"Built segmented DB with {len(records)} records -> {args.output}", flush=True)


if __name__ == "__main__":
    main()
