import json
import sys
import time
import imagehash
import numpy as np
from annoy import AnnoyIndex

# --- Configuration ---
HASH_DB_PATH = "scryfall_phash_db_sample.json"
ANNOY_INDEX_PATH = "phash_annoy_index.ann"
IDX_TO_CARD_PATH = "phash_idx_to_card.json"
PHASH_TO_IDX_PATH = "phash_to_idx.json"
ANNOY_DIM = 64


def phash_to_vec(phash_str):
    """Convert a hex perceptual hash string to a 64-dim float vector (0/1)."""
    h = imagehash.hex_to_hash(phash_str)
    return h.hash.flatten().astype(np.float32)


def _bar(done, total, width=40):
    pct = done / total if total else 0
    filled = int(width * pct)
    bar = "#" * filled + "-" * (width - filled)
    return f"[{bar}] {done}/{total} ({pct*100:.1f}%)"


# Load hash database
print(f"Loading {HASH_DB_PATH} ...")
with open(HASH_DB_PATH, "r", encoding="utf-8") as f:
    card_hash_db = json.load(f)
print(f"  Loaded {len(card_hash_db):,} records.")

annoy_index = AnnoyIndex(ANNOY_DIM, 'hamming')
phash_to_idx = {}
idx_to_card = {}

total = len(card_hash_db)
t0 = time.perf_counter()
PROGRESS_EVERY = max(1, total // 200)  # ~200 updates

print("Adding items to Annoy index ...")
for idx, entry in enumerate(card_hash_db):
    phash = entry.get('phash')
    if phash is None:
        continue
    vec = phash_to_vec(phash)
    annoy_index.add_item(idx, vec)
    phash_to_idx[phash] = idx
    idx_to_card[idx] = entry

    if (idx + 1) % PROGRESS_EVERY == 0 or idx + 1 == total:
        elapsed = time.perf_counter() - t0
        rate = (idx + 1) / elapsed if elapsed > 0 else 0
        eta = (total - idx - 1) / rate if rate > 0 else 0
        sys.stdout.write(f"\r  {_bar(idx + 1, total)}  ETA {eta:.0f}s   ")
        sys.stdout.flush()

elapsed_add = time.perf_counter() - t0
print(f"\n  Done adding {len(idx_to_card):,} items in {elapsed_add:.1f}s.")

print(f"Building Annoy index (10 trees) — this may take 1-3 minutes ...")
t_build = time.perf_counter()
annoy_index.build(10)
elapsed_build = time.perf_counter() - t_build
print(f"  Built in {elapsed_build:.1f}s.")

annoy_index.save(ANNOY_INDEX_PATH)
print(f"  Saved to {ANNOY_INDEX_PATH}")

print("Saving JSON mappings ...")
with open(IDX_TO_CARD_PATH, "w", encoding="utf-8") as f:
    json.dump(idx_to_card, f)
with open(PHASH_TO_IDX_PATH, "w", encoding="utf-8") as f:
    json.dump(phash_to_idx, f)
print(f"  {IDX_TO_CARD_PATH}")
print(f"  {PHASH_TO_IDX_PATH}")
print("All done.")
