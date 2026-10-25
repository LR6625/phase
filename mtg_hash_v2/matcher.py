from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from pathlib import Path
import sys

import cv2

try:
    from .config import SegmentedHashConfig
    from .database import load_segmented_db, make_card_key_from_info
    from .segmented_hashing import compute_segmented_hashes, hamming_distance
except ImportError:
    # Allow running this module directly (python mtg_hash_v2/matcher.py).
    workspace_root = str(Path(__file__).resolve().parent.parent)
    if workspace_root not in sys.path:
        sys.path.insert(0, workspace_root)
    from mtg_hash_v2.config import SegmentedHashConfig
    from mtg_hash_v2.database import load_segmented_db, make_card_key_from_info
    from mtg_hash_v2.segmented_hashing import compute_segmented_hashes, hamming_distance


@dataclass
class CandidateScore:
    card_key: str
    name: str
    set_code: str | None
    orientation: str
    stage1_distance: float
    weighted_matches: float
    matched_regions: int
    total_regions: int
    avg_region_distance: float
    final_score: float


class HybridTopHalfMatcher:
    def __init__(
        self,
        source_module_path: str | Path,
        full_card_db_path: str | Path,
        segmented_db_path: str | Path,
        cfg: SegmentedHashConfig | None = None,
    ):
        self.cfg = cfg or SegmentedHashConfig()
        self.source_module = self._load_source_module(source_module_path)

        with Path(full_card_db_path).open("r", encoding="utf-8") as f:
            full_card_db = __import__("json").load(f)
        self.annoy_index, self.idx_to_card = self.source_module.load_or_build_annoy_index(full_card_db)

        segmented_records = load_segmented_db(segmented_db_path)
        self.segmented_by_key = {record["card_key"]: record for record in segmented_records}

    @staticmethod
    def _load_source_module(module_path: str | Path):
        module_path = Path(module_path)
        spec = importlib.util.spec_from_file_location("exfcc_copy_module_v2", module_path)
        module = importlib.util.module_from_spec(spec)
        if spec.loader is None:
            raise RuntimeError(f"Could not load source module: {module_path}")
        spec.loader.exec_module(module)
        return module

    def _stage1_shortlist(self, image_bgr):
        raw = self.source_module.annoy_nearest_phash(
            image_bgr, self.annoy_index, self.idx_to_card, n=self.cfg.stage1_k
        )
        if not raw:
            return []

        best_distance = min(float(dist) for _, dist in raw)
        max_distance = best_distance + self.cfg.stage1_distance_window

        unique = {}
        for card_info, dist in raw:
            dist = float(dist)
            if dist > max_distance:
                continue
            key = make_card_key_from_info(card_info)
            if key not in unique or dist < unique[key]["distance"]:
                unique[key] = {"card_info": card_info, "distance": dist}

        candidates = sorted(unique.values(), key=lambda x: x["distance"])
        return candidates[: self.cfg.stage1_max_candidates]

    def _score_candidate(self, scan_bundle: dict, candidate_record: dict, stage1_distance: float, orientation: str):
        total_regions = self.cfg.expected_regions()
        candidate_regions = {r["index"]: r for r in candidate_record["segmented"]["regions"]}

        weighted_matches = 0.0
        matched_regions = 0
        dist_sum = 0.0
        dist_count = 0

        for scan_region in scan_bundle["regions"]:
            idx = scan_region["index"]
            cand_region = candidate_regions.get(idx)
            if cand_region is None:
                continue

            ph_dist = hamming_distance(scan_region["phash"], cand_region["phash"])
            dh_dist = hamming_distance(scan_region["dhash"], cand_region["dhash"])
            dist_sum += (ph_dist + dh_dist) * 0.5
            dist_count += 1

            if ph_dist <= self.cfg.phash_threshold and dh_dist <= self.cfg.dhash_threshold:
                matched_regions += 1
                weighted_matches += self.cfg.get_weight(idx)

        avg_region_distance = dist_sum / max(dist_count, 1)
        final_score = weighted_matches - (0.08 * avg_region_distance) - (0.03 * stage1_distance)

        return CandidateScore(
            card_key=candidate_record["card_key"],
            name=candidate_record.get("name") or "Unknown",
            set_code=candidate_record.get("set"),
            orientation=orientation,
            stage1_distance=stage1_distance,
            weighted_matches=weighted_matches,
            matched_regions=matched_regions,
            total_regions=total_regions,
            avg_region_distance=avg_region_distance,
            final_score=final_score,
        )

    def _evaluate_orientation(self, image_bgr, orientation: str):
        scan_bundle = compute_segmented_hashes(image_bgr, self.cfg)
        shortlist = self._stage1_shortlist(image_bgr)
        scores: list[CandidateScore] = []

        for candidate in shortlist:
            card_info = candidate["card_info"]
            stage1_distance = candidate["distance"]
            key = make_card_key_from_info(card_info)
            segmented_record = self.segmented_by_key.get(key)
            if segmented_record is None:
                continue
            score = self._score_candidate(scan_bundle, segmented_record, stage1_distance, orientation)
            scores.append(score)

        scores.sort(key=lambda s: s.final_score, reverse=True)
        return scores

    def identify(self, warped_card_bgr):
        upright_scores = self._evaluate_orientation(warped_card_bgr, "upright")
        rotated = cv2.rotate(warped_card_bgr, cv2.ROTATE_180)
        rotated_scores = self._evaluate_orientation(rotated, "rotated 180")

        combined = []
        combined.extend(upright_scores)
        combined.extend(rotated_scores)
        if not combined:
            return {
                "status": "reject",
                "reason": "No candidates found in segmented DB for shortlisted cards.",
                "best": None,
                "possible": [],
            }

        combined.sort(key=lambda s: s.final_score, reverse=True)
        best = combined[0]

        if best.weighted_matches >= self.cfg.confident_min_weighted_matches:
            status = "confident"
        elif best.weighted_matches >= self.cfg.possible_min_weighted_matches:
            status = "possible"
        else:
            status = "reject"

        return {
            "status": status,
            "best": {
                "name": best.name,
                "set": best.set_code,
                "orientation": best.orientation,
                "weighted_matches": best.weighted_matches,
                "matched_regions": best.matched_regions,
                "total_regions": best.total_regions,
                "avg_region_distance": best.avg_region_distance,
                "stage1_distance": best.stage1_distance,
                "final_score": best.final_score,
            },
            "possible": [
                {
                    "name": s.name,
                    "set": s.set_code,
                    "orientation": s.orientation,
                    "weighted_matches": s.weighted_matches,
                    "matched_regions": s.matched_regions,
                    "final_score": s.final_score,
                }
                for s in combined[1 : 1 + self.cfg.possible_list_size]
            ],
        }
