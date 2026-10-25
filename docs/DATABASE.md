# Card hash database

The sorter identifies cards by comparing perceptual hashes of the camera image
against hashes of every card image on [Scryfall](https://scryfall.com). The
database files are about 180 MB in total, so they are not in the git history.
Download them from the **Releases** page and drop them in the repo root.

| File | Size | What it is |
|---|---|---|
| `scryfall_phash_db_sample.json` | ~18 MB | One 64-bit pHash per card, plus name, set, mana cost and Scryfall URL |
| `phash_annoy_index.ann` | ~17 MB | [Annoy](https://github.com/spotify/annoy) nearest-neighbour index over those hashes |
| `phash_idx_to_card.json` | ~19 MB | Maps Annoy index positions back to card records |
| `phash_to_idx.json` | ~2 MB | Maps a pHash string to its index position |
| `scryfall_segmented_tophalf_db_v2.json` | ~139 MB | About 103k cards. For each: a 3x3 grid over the top 55% of the card, with a pHash and dHash per cell |

The databases contain hashes and card metadata only, not images. Card data
comes from Scryfall's API and bulk data; card text and artwork belong to
Wizards of the Coast.

## Rebuilding

Needs internet access. Scryfall asks clients to stay under ~10 requests/second;
the scripts sleep between requests (`--min-interval`).

```bash
# 1. Add new cards to the full-card pHash DB (downloads each image and hashes it)
python update_scryfall_phash_db_safe.py --db scryfall_phash_db_sample.json --out scryfall_phash_db_sample.updated.json

# 2. Rebuild the Annoy index + lookup tables
#    (reads scryfall_phash_db_sample.json; rename the .updated.json first)
python build_phash_annoy_index.py

# 3. Build the segmented top-half DB (resumable)
python build_segmented_tophalf_db_v2_from_phash_db.py --phash-db scryfall_phash_db_sample.json --resume
```

A full rebuild downloads one image per card, so it takes many hours.
`segmented_manifest_template_v2.json` shows the manifest format for the
offline variant, `build_segmented_tophalf_db_v2.py`.
