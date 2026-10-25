from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class SegmentedHashConfig:
    normalized_size: Tuple[int, int] = (744, 1040)  # (width, height)
    top_half_fraction: float = 0.55
    grid_rows: int = 3
    grid_cols: int = 3

    # Hamming thresholds per region
    phash_threshold: int = 12
    dhash_threshold: int = 14

    # Stage-1 shortlist settings (existing full-card Annoy)
    stage1_k: int = 2000
    stage1_distance_window: float = 64.0
    stage1_max_candidates: int = 1000

    # How many ranked alternatives to return after the best match.
    possible_list_size: int = 15

    # Decision thresholds
    confident_min_weighted_matches: float = 4.8
    possible_min_weighted_matches: float = 2.8

    # Region weights for 3x3 top-half grid (row-major)
    region_weights: list[float] = field(
        default_factory=lambda: [
            0.35, 0.95, 0.35,
            0.85, 2.35, 0.85,
            1.00, 2.00, 0.85,
        ]
    )

    def expected_regions(self) -> int:
        return self.grid_rows * self.grid_cols

    def get_weight(self, index: int) -> float:
        if index < len(self.region_weights):
            return float(self.region_weights[index])
        return 1.0
