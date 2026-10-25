import argparse
import json
import time
from pathlib import Path

import cv2
import requests

from mtg_hash_v2.config import SegmentedHashConfig
from mtg_hash_v2.database import make_card_key
from mtg_hash_v2.segmented_hashing import compute_segmented_hashes

USER_AGENT = "mtg-sorter/1.0 (+local-dev)"


class RateLimitedFetcher:
    def __init__(self, min_interval_seconds: float = 0.12, timeout_seconds: float = 30.0):
        import time

        self.time = time
        self.min_interval_seconds = max(min_interval_seconds, 0.11)
        self.timeout_seconds = timeout_seconds
        self.last_request = 0.0
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": USER_AGENT})

    def get_bytes(self, url: str, max_retries: int = 5) -> bytes:
        for _ in range(max_retries):
            elapsed = self.time.time() - self.last_request
            if elapsed < self.min_interval_seconds:
                self.time.sleep(self.min_interval_seconds - elapsed)

            response = self.session.get(url, timeout=self.timeout_seconds)
            self.last_request = self.time.time()

            if response.status_code == 429:
                retry_after = float(response.headers.get("Retry-After", "1"))
                self.time.sleep(max(retry_after, 1.0))
                continue

            response.raise_for_status()
            return response.content

        raise RuntimeError(f"Failed to fetch image after retries: {url}")


def image_url_from_scryfall_uri(scryfall_uri: str) -> str | None:
    # https://scryfall.com/card/{set}/{collector}/{slug}?utm_source=api
    # -> https://api.scryfall.com/cards/{set}/{collector} to read image_uris
    if "/card/" not in scryfall_uri:
        return None
    cleaned = scryfall_uri.split("?")[0].strip()
    marker = "/card/"
    idx = cleaned.find(marker)
    if idx < 0:
        return None
    tail = cleaned[idx + len(marker) :].strip("/")
    parts = tail.split("/")
    if len(parts) < 2:
        return None

    set_code = parts[0]
    collector_number = parts[1]
    return f"https://api.scryfall.com/cards/{set_code}/{collector_number}"


def fetch_normal_image_url(fetcher: RateLimitedFetcher, scryfall_uri: str) -> str | None:
    api_card_url = image_url_from_scryfall_uri(scryfall_uri)
    if not api_card_url:
        return None

    response_bytes = fetcher.get_bytes(api_card_url)
    payload = json.loads(response_bytes.decode("utf-8"))

    image_uris = payload.get("image_uris") or {}
    if image_uris.get("normal"):
        return image_uris["normal"]

    for face in payload.get("card_faces") or []:
        face_uris = face.get("image_uris") or {}
        if face_uris.get("normal"):
            return face_uris["normal"]

    return None


def decode_image(image_bytes: bytes):
    import numpy as np

    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def write_progress(progress_path: Path, payload: dict):
    progress_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Build segmented top-half DB from existing pHash DB card list.")
    parser.add_argument("--phash-db", default="scryfall_phash_db_sample.updated.json")
    parser.add_argument("--output", default="scryfall_segmented_tophalf_db_v2.json")
    parser.add_argument("--rows", type=int, default=3)
    parser.add_argument("--cols", type=int, default=3)
    parser.add_argument("--top-half", type=float, default=0.55)
    parser.add_argument("--min-interval", type=float, default=0.12)
    parser.add_argument("--max-cards", type=int, default=0, help="0 means all")
    parser.add_argument("--set", dest="set_code", default=None)
    parser.add_argument("--resume", action="store_true", help="Resume from existing output file if present")
    parser.add_argument("--save-every", type=int, default=500, help="Write output checkpoint every N built cards")
    parser.add_argument("--progress-every", type=int, default=25, help="Update progress file every N scanned cards")
    parser.add_argument("--progress-file", default="", help="Optional progress JSON path")
    args = parser.parse_args()

    cfg = SegmentedHashConfig(grid_rows=args.rows, grid_cols=args.cols, top_half_fraction=args.top_half)

    with Path(args.phash_db).open("r", encoding="utf-8") as f:
        source_cards = json.load(f)

    output_path = Path(args.output)
    progress_path = Path(args.progress_file) if args.progress_file else Path(f"{args.output}.progress.json")

    filtered_entries = []
    for entry in source_cards:
        if args.set_code and (entry.get("set", "").lower() != args.set_code.lower()):
            continue
        if not entry.get("scryfall_uri"):
            continue
        filtered_entries.append(entry)

    if args.max_cards > 0:
        filtered_entries = filtered_entries[: args.max_cards]

    fetcher = RateLimitedFetcher(min_interval_seconds=args.min_interval)
    output_records = []
    seen_keys = set()

    if args.resume and output_path.exists():
        try:
            with output_path.open("r", encoding="utf-8") as f:
                output_records = json.load(f)
            seen_keys = {record.get("card_key") for record in output_records if record.get("card_key")}
            print(f"Resuming with {len(output_records)} existing segmented records.", flush=True)
        except Exception:
            output_records = []
            seen_keys = set()

    total_candidates = len(filtered_entries)
    start_time = time.time()
    scanned = 0
    built = len(output_records)
    skipped = 0
    errors = 0

    write_progress(
        progress_path,
        {
            "status": "running",
            "started_at": start_time,
            "updated_at": time.time(),
            "total_candidates": total_candidates,
            "scanned": scanned,
            "built": built,
            "skipped": skipped,
            "errors": errors,
            "percent": 0.0,
            "eta_seconds": None,
            "output": str(output_path),
        },
    )

    def maybe_write_progress():
        if scanned % args.progress_every != 0 and scanned != total_candidates:
            return
        elapsed = max(time.time() - start_time, 1e-6)
        scan_rate = scanned / elapsed
        remaining = max(total_candidates - scanned, 0)
        eta_seconds = remaining / max(scan_rate, 1e-6)
        percent = (scanned / total_candidates * 100.0) if total_candidates > 0 else 100.0

        write_progress(
            progress_path,
            {
                "status": "running",
                "started_at": start_time,
                "updated_at": time.time(),
                "total_candidates": total_candidates,
                "scanned": scanned,
                "built": built,
                "skipped": skipped,
                "errors": errors,
                "percent": percent,
                "eta_seconds": eta_seconds,
                "output": str(output_path),
            },
        )

    for entry in filtered_entries:
        scanned += 1
        scryfall_uri = entry.get("scryfall_uri")
        card_key = make_card_key(entry.get("name"), entry.get("set"), scryfall_uri)
        should_process = True

        if card_key in seen_keys:
            skipped += 1
            should_process = False

        if should_process:
            try:
                normal_image_url = fetch_normal_image_url(fetcher, scryfall_uri)
                if not normal_image_url:
                    skipped += 1
                else:
                    image_bytes = fetcher.get_bytes(normal_image_url)
                    image = decode_image(image_bytes)
                    if image is None:
                        skipped += 1
                    else:
                        segmented = compute_segmented_hashes(image, cfg)
                        output_records.append(
                            {
                                "card_key": card_key,
                                "name": entry.get("name"),
                                "set": entry.get("set"),
                                "scryfall_uri": scryfall_uri,
                                "segmented": segmented,
                            }
                        )
                        seen_keys.add(card_key)
                        built += 1

                        if built % args.save_every == 0:
                            with output_path.open("w", encoding="utf-8") as f:
                                json.dump(output_records, f, ensure_ascii=False)
                            print(f"Checkpoint saved: built={built} scanned={scanned}/{total_candidates}", flush=True)

            except Exception:
                errors += 1

        maybe_write_progress()

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(output_records, f, ensure_ascii=False)

    elapsed = time.time() - start_time
    write_progress(
        progress_path,
        {
            "status": "done",
            "started_at": start_time,
            "updated_at": time.time(),
            "total_candidates": total_candidates,
            "scanned": scanned,
            "built": built,
            "skipped": skipped,
            "errors": errors,
            "percent": 100.0,
            "eta_seconds": 0.0,
            "elapsed_seconds": elapsed,
            "output": str(output_path),
        },
    )

    print(
        f"Done. Segmented records: {len(output_records)} -> {args.output} | "
        f"scanned={scanned}/{total_candidates} skipped={skipped} errors={errors} elapsed={elapsed/3600:.2f}h",
        flush=True,
    )


if __name__ == "__main__":
    main()
