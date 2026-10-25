import argparse
import json
import time
from pathlib import Path
from typing import Any

import cv2
import imagehash
import requests
from PIL import Image

SCRYFALL_BULK_DATA_URL = "https://api.scryfall.com/bulk-data"
USER_AGENT = "mtg-sorter/1.0 (+local-dev)"


class RateLimitedSession:
    def __init__(self, min_interval_seconds: float = 0.12, timeout_seconds: float = 30.0):
        self.min_interval_seconds = max(min_interval_seconds, 0.11)
        self.timeout_seconds = timeout_seconds
        self._last_request_time = 0.0
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": USER_AGENT,
                "Accept": "application/json",
            }
        )

    def _wait_if_needed(self):
        elapsed = time.time() - self._last_request_time
        if elapsed < self.min_interval_seconds:
            time.sleep(self.min_interval_seconds - elapsed)

    def get_json(self, url: str, max_retries: int = 5) -> dict[str, Any]:
        for attempt in range(max_retries):
            self._wait_if_needed()
            response = self.session.get(url, timeout=self.timeout_seconds)
            self._last_request_time = time.time()

            if response.status_code == 429:
                retry_after = float(response.headers.get("Retry-After", "1"))
                time.sleep(max(retry_after, 1.0))
                continue
            response.raise_for_status()
            return response.json()

        raise RuntimeError(f"Failed to fetch JSON after retries: {url}")

    def get_bytes(self, url: str, max_retries: int = 5) -> bytes:
        for attempt in range(max_retries):
            self._wait_if_needed()
            response = self.session.get(url, timeout=self.timeout_seconds)
            self._last_request_time = time.time()

            if response.status_code == 429:
                retry_after = float(response.headers.get("Retry-After", "1"))
                time.sleep(max(retry_after, 1.0))
                continue
            response.raise_for_status()
            return response.content

        raise RuntimeError(f"Failed to fetch bytes after retries: {url}")


def compute_phash_from_bytes(image_bytes: bytes) -> str:
    import numpy as np

    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    image_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError("Could not decode image bytes")

    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)
    return str(imagehash.phash(pil_img))


def load_existing_db(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_db(path: Path, records: list[dict[str, Any]]):
    with path.open("w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False)


def extract_image_url(card: dict[str, Any]) -> str | None:
    image_uris = card.get("image_uris") or {}
    if image_uris.get("normal"):
        return image_uris["normal"]

    # Double-faced cards
    for face in card.get("card_faces") or []:
        face_uris = face.get("image_uris") or {}
        if face_uris.get("normal"):
            return face_uris["normal"]

    return None


def should_include_card(card: dict[str, Any], set_filter: str | None) -> bool:
    if card.get("layout") in {"token", "art_series", "reversible_card"}:
        return False
    if card.get("set_type") in {"token", "memorabilia"}:
        return False
    if card.get("games") and "paper" not in card.get("games", []):
        return False
    if set_filter and card.get("set", "").lower() != set_filter.lower():
        return False
    return True


def get_oracle_bulk_download_uri(http: RateLimitedSession) -> str:
    payload = http.get_json(SCRYFALL_BULK_DATA_URL)
    data = payload.get("data", [])
    for entry in data:
        if entry.get("type") == "oracle_cards":
            return entry["download_uri"]
    raise RuntimeError("Could not locate oracle_cards bulk data URI")


def build_existing_index(records: list[dict[str, Any]]) -> set[str]:
    keys = set()
    for rec in records:
        uri = rec.get("scryfall_uri")
        if uri:
            keys.add(uri.strip().lower())
    return keys


def main():
    parser = argparse.ArgumentParser(description="Safely update scryfall_phash_db_sample.json with new cards.")
    parser.add_argument("--db", default="scryfall_phash_db_sample.json", help="Path to existing pHash DB JSON")
    parser.add_argument("--out", default="scryfall_phash_db_sample.updated.json", help="Output updated DB path")
    parser.add_argument("--set", dest="set_code", default=None, help="Optional set code filter (e.g. dsk)")
    parser.add_argument("--max-new", type=int, default=0, help="Optional cap of newly added cards (0 = no cap)")
    parser.add_argument("--min-interval", type=float, default=0.12, help="Minimum delay between API/image requests")
    args = parser.parse_args()

    db_path = Path(args.db)
    out_path = Path(args.out)

    existing = load_existing_db(db_path)
    seen_uris = build_existing_index(existing)

    http = RateLimitedSession(min_interval_seconds=args.min_interval)
    download_uri = get_oracle_bulk_download_uri(http)
    oracle_cards = http.get_json(download_uri)

    updated = list(existing)
    added = 0
    skipped_no_image = 0

    for card in oracle_cards:
        if not should_include_card(card, args.set_code):
            continue

        scryfall_uri = (card.get("scryfall_uri") or "").strip()
        if not scryfall_uri:
            continue
        key = scryfall_uri.lower()
        if key in seen_uris:
            continue

        image_url = extract_image_url(card)
        if not image_url:
            skipped_no_image += 1
            continue

        try:
            image_bytes = http.get_bytes(image_url)
            phash = compute_phash_from_bytes(image_bytes)
        except Exception:
            continue

        updated.append(
            {
                "phash": phash,
                "scryfall_uri": scryfall_uri,
                "name": card.get("name", ""),
                "set": card.get("set", ""),
                "mana_cost": card.get("mana_cost", ""),
            }
        )
        seen_uris.add(key)
        added += 1

        if args.max_new > 0 and added >= args.max_new:
            break

        if added % 100 == 0:
            print(f"Added {added} new cards...", flush=True)

    save_db(out_path, updated)
    print(
        f"Done. Existing={len(existing)} | Added={added} | SkippedNoImage={skipped_no_image} | Total={len(updated)}\n"
        f"Output: {out_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
