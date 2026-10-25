from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
from ultralytics import YOLO

import CamTesting as cam
import YOLO_Camera_Studio as ycs

from mtg_hash_v2.config import SegmentedHashConfig
from mtg_hash_v2.matcher import HybridTopHalfMatcher
from mtg_hash_v2.segmented_hashing import split_into_grid


WORKSPACE_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = WORKSPACE_DIR / "runs/obb/detect_obb_combined_v1/weights/best.pt"
DEFAULT_SOURCE_MODULE = WORKSPACE_DIR / "Expo_P.H.A.S.E.py"
DEFAULT_FULL_DB = WORKSPACE_DIR / "scryfall_phash_db_sample.json"
DEFAULT_SEGMENTED_DB = WORKSPACE_DIR / "scryfall_segmented_tophalf_db_v2.json"
DEFAULT_OUTPUT_DIR = WORKSPACE_DIR / "buggingme"
MIN_ACCEPTED_YOLO_CONF_DEFAULT = 0.88
MAX_VALID_YOLO_BOX_AREA_RATIO = 0.92

TARGET_ASPECT_WH = 63.0 / 88.0  # portrait width/height for MTG cards
MASK_APPLY_TOP_REGION_FRACTION = 0.10
MASK_APPLY_CENTER_WIDTH_FRACTION = 0.60
MASK_SAMPLE_TOP_ROWS = 3
MASK_SAMPLE_CENTER_WIDTH_FRACTION = 0.50
MASK_THRESH_STD_MULTIPLIER = 2.5
MASK_THRESH_MIN = 6.0
MASK_THRESH_MAX = 38.0
MASK_SEARCH_PAD_RATIO = 0.12
MASK_MIN_COMPONENT_AREA_RATIO = 0.02
REFINE_MIN_IOU = 0.45
REFINE_MAX_CENTER_SHIFT_RATIO = 0.35
REFINE_MIN_AREA_RATIO = 0.60
REFINE_MAX_AREA_RATIO = 1.40


@dataclass
class QualityMetrics:
    detector_confidence: float
    blur_laplacian_var: float
    warped_width: int
    warped_height: int
    warped_aspect_wh: float


@dataclass
class VariantResult:
    variant: str
    trim_ratio: float
    status: str
    best_name: str | None
    best_set: str | None
    orientation: str | None
    final_score: float | None
    weighted_matches: float | None
    matched_regions: int | None
    total_regions: int | None
    avg_region_distance: float | None


@dataclass
class ImageRunResult:
    image_path: str
    output_dir: str
    detector_confidence: float | None
    detection_ok: bool
    best_variant: VariantResult | None
    all_variants: list[VariantResult]
    notes: list[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Camera capture test: YOLO OBB detect -> tuned crop -> existing grid hash matcher with debug outputs."
    )
    parser.add_argument(
        "--camera-index",
        type=int,
        default=None,
        help="Camera index override. Omit to use the same default camera selection as YOLO_Camera_Studio.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=str(DEFAULT_MODEL_PATH),
        help="Path to trained YOLO OBB weights.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory where per-image debug outputs are saved.",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Inference size for YOLO OBB predict.",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Detection confidence threshold for YOLO.",
    )
    parser.add_argument(
        "--min-yolo-conf",
        type=float,
        default=MIN_ACCEPTED_YOLO_CONF_DEFAULT,
        help="Minimum best YOLO confidence required before masking/cropping is allowed.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Ultralytics device string, e.g. cpu or 0.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of OBB detections to test per image.",
    )
    parser.add_argument(
        "--trim-ratios",
        type=str,
        default="0.00",
        help="Comma-separated inward trim ratios to test after warp.",
    )
    parser.add_argument(
        "--source-module",
        type=str,
        default=str(DEFAULT_SOURCE_MODULE),
        help="Path to source module used by HybridTopHalfMatcher stage-1 Annoy.",
    )
    parser.add_argument(
        "--full-db",
        type=str,
        default=str(DEFAULT_FULL_DB),
        help="Full-card phash DB JSON for stage-1 Annoy shortlist.",
    )
    parser.add_argument(
        "--segmented-db",
        type=str,
        default=str(DEFAULT_SEGMENTED_DB),
        help="Segmented top-half DB JSON for grid hashing.",
    )
    parser.add_argument(
        "--strict-quality-gate",
        action="store_true",
        help="Reject low-quality warped cards before matching.",
    )
    return parser.parse_args()


def parse_trim_ratios(text: str) -> list[float]:
    values: list[float] = []
    for raw in text.split(","):
        raw = raw.strip()
        if not raw:
            continue
        value = float(raw)
        value = max(0.0, min(0.25, value))
        values.append(value)
    if not values:
        return [0.0]
    # Unique while preserving order
    deduped: list[float] = []
    for v in values:
        if v not in deduped:
            deduped.append(v)
    return deduped


def order_points(pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, dtype=np.float32)
    sums = pts.sum(axis=1)
    diffs = pts[:, 1] - pts[:, 0]
    # [top-left, top-right, bottom-right, bottom-left]
    return pts[[sums.argmin(), diffs.argmin(), sums.argmax(), diffs.argmax()]]


def obb_points_from_result(result, top_k: int) -> list[tuple[np.ndarray, float]]:
    if getattr(result, "obb", None) is None:
        return []

    obb = result.obb
    points = getattr(obb, "xyxyxyxy", None)
    conf = getattr(obb, "conf", None)

    if points is None or conf is None:
        return []

    points_np = points.cpu().numpy() if hasattr(points, "cpu") else np.asarray(points)
    conf_np = conf.cpu().numpy() if hasattr(conf, "cpu") else np.asarray(conf)

    if len(points_np) == 0:
        return []

    order = np.argsort(-conf_np)
    picks: list[tuple[np.ndarray, float]] = []
    for idx in order[: max(1, top_k)]:
        quad = order_points(points_np[idx])
        picks.append((quad, float(conf_np[idx])))
    return picks


def best_obb_conf(result) -> float | None:
    if getattr(result, "obb", None) is None:
        return None
    conf = getattr(result.obb, "conf", None)
    if conf is None:
        return None
    conf_np = conf.cpu().numpy() if hasattr(conf, "cpu") else np.asarray(conf)
    if conf_np is None or len(conf_np) == 0:
        return None
    return float(np.max(conf_np))


def best_obb_box_area_ratio(result, image_shape: tuple[int, int, int]) -> float | None:
    picks = obb_points_from_result(result, top_k=1)
    if not picks:
        return None
    quad = picks[0][0]
    h, w = image_shape[:2]
    area_img = float(max(1, h * w))
    area_box = float(cv2.contourArea(quad.astype(np.float32)))
    return float(max(0.0, min(1.0, area_box / area_img)))


def yolo_first_pick_best_result(
    model: YOLO,
    candidate_paths: list[Path],
    imgsz: int,
    conf: float,
    device: str,
) -> tuple[Path, object, float | None, list[dict]]:
    valid_paths = [p for p in candidate_paths if p.exists()]
    if not valid_paths:
        raise FileNotFoundError("No valid YOLO candidate image paths were found.")

    results = model.predict(
        source=[str(p) for p in valid_paths],
        imgsz=imgsz,
        conf=conf,
        device=device,
        save=False,
        verbose=False,
    )

    if len(results) != len(valid_paths):
        raise RuntimeError("YOLO returned an unexpected number of results for candidate inputs.")

    scored: list[tuple[float, int]] = []
    debug_rows: list[dict] = []
    for idx, (path, result) in enumerate(zip(valid_paths, results)):
        best_conf = best_obb_conf(result)
        img = cv2.imread(str(path))
        box_area_ratio = None if img is None else best_obb_box_area_ratio(result, img.shape)
        conf_score = -1.0 if best_conf is None else float(best_conf)
        scored.append((conf_score, idx))
        plotted_path = str(path.with_name(f"{path.stem}_yolo_plot.png"))
        try:
            plotted = result.plot()
            cv2.imwrite(plotted_path, plotted)
        except Exception:
            plotted_path = None
        debug_rows.append(
            {
                "candidate": str(path),
                "best_conf": None if best_conf is None else float(best_conf),
                "top_box_area_ratio": None if box_area_ratio is None else float(box_area_ratio),
                "detections": len(obb_points_from_result(result, top_k=20)),
                "plot_path": plotted_path,
            }
        )

    scored.sort(key=lambda x: x[0], reverse=True)
    best_idx = scored[0][1]
    best_conf = best_obb_conf(results[best_idx])
    return valid_paths[best_idx], results[best_idx], best_conf, debug_rows


def draw_quad(image: np.ndarray, quad: np.ndarray, color=(0, 255, 0), thickness=2) -> np.ndarray:
    out = image.copy()
    quad_i = quad.astype(np.int32)
    cv2.polylines(out, [quad_i], True, color, thickness)
    for i, p in enumerate(quad_i):
        cv2.circle(out, tuple(p), 4, (255, 0, 0), -1)
        cv2.putText(out, str(i), tuple(p + np.array([6, -6])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1, cv2.LINE_AA)
    return out


def fit_quad_in_frame_rigid(quad: np.ndarray, frame_w: int, frame_h: int) -> np.ndarray:
    """Translate a quad to fit the frame while preserving rectangle geometry."""
    q = quad.astype(np.float32).copy()
    min_x = float(np.min(q[:, 0]))
    max_x = float(np.max(q[:, 0]))
    min_y = float(np.min(q[:, 1]))
    max_y = float(np.max(q[:, 1]))

    dx = 0.0
    dy = 0.0
    if min_x < 0.0:
        dx += -min_x
    if max_x > (frame_w - 1):
        dx -= (max_x - (frame_w - 1))
    if min_y < 0.0:
        dy += -min_y
    if max_y > (frame_h - 1):
        dy -= (max_y - (frame_h - 1))

    q[:, 0] += dx
    q[:, 1] += dy
    q[:, 0] = np.clip(q[:, 0], 0, frame_w - 1)
    q[:, 1] = np.clip(q[:, 1], 0, frame_h - 1)
    return q


def top_edge_angle_deg(quad: np.ndarray) -> float:
    q = order_points(quad)
    v = q[1] - q[0]
    return float(np.degrees(np.arctan2(float(v[1]), float(v[0]))))


def quad_bbox_xyxy(quad: np.ndarray) -> tuple[float, float, float, float]:
    q = order_points(quad)
    x1 = float(np.min(q[:, 0]))
    y1 = float(np.min(q[:, 1]))
    x2 = float(np.max(q[:, 0]))
    y2 = float(np.max(q[:, 1]))
    return x1, y1, x2, y2


def bbox_iou_xyxy(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih

    area_a = max(0.0, (ax2 - ax1)) * max(0.0, (ay2 - ay1))
    area_b = max(0.0, (bx2 - bx1)) * max(0.0, (by2 - by1))
    union = area_a + area_b - inter
    if union <= 1e-6:
        return 0.0
    return float(inter / union)


def choose_refined_or_yolo_quad(yolo_quad: np.ndarray, refined_quad: np.ndarray | None) -> tuple[np.ndarray, dict]:
    """Keep YOLO as primary. Accept refinement only when it is consistent with YOLO geometry."""
    y_bbox = quad_bbox_xyxy(yolo_quad)
    y_w = max(1e-6, y_bbox[2] - y_bbox[0])
    y_h = max(1e-6, y_bbox[3] - y_bbox[1])
    y_area = y_w * y_h
    y_cx = 0.5 * (y_bbox[0] + y_bbox[2])
    y_cy = 0.5 * (y_bbox[1] + y_bbox[3])
    y_diag = max(1e-6, math.hypot(y_w, y_h))

    if refined_quad is None:
        return yolo_quad, {
            "used": "yolo",
            "reason": "no_refined_quad",
            "iou": None,
            "center_shift_ratio": None,
            "area_ratio": None,
        }

    r_bbox = quad_bbox_xyxy(refined_quad)
    r_w = max(1e-6, r_bbox[2] - r_bbox[0])
    r_h = max(1e-6, r_bbox[3] - r_bbox[1])
    r_area = r_w * r_h
    r_cx = 0.5 * (r_bbox[0] + r_bbox[2])
    r_cy = 0.5 * (r_bbox[1] + r_bbox[3])

    iou = bbox_iou_xyxy(y_bbox, r_bbox)
    center_shift = math.hypot(r_cx - y_cx, r_cy - y_cy)
    center_shift_ratio = center_shift / y_diag
    area_ratio = r_area / y_area

    if (
        iou >= REFINE_MIN_IOU
        and center_shift_ratio <= REFINE_MAX_CENTER_SHIFT_RATIO
        and REFINE_MIN_AREA_RATIO <= area_ratio <= REFINE_MAX_AREA_RATIO
    ):
        return refined_quad, {
            "used": "refined",
            "reason": "passed_consistency_gates",
            "iou": float(iou),
            "center_shift_ratio": float(center_shift_ratio),
            "area_ratio": float(area_ratio),
        }

    return yolo_quad, {
        "used": "yolo",
        "reason": "refined_failed_consistency_gates",
        "iou": float(iou),
        "center_shift_ratio": float(center_shift_ratio),
        "area_ratio": float(area_ratio),
    }


def _largest_component(mask: np.ndarray, min_area: int) -> np.ndarray | None:
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        return None

    best_label = None
    best_area = 0
    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        if area > best_area:
            best_area = area
            best_label = label

    if best_label is None:
        return None

    out = np.zeros_like(mask)
    out[labels == best_label] = 255
    return out


def refine_quad_with_top_mask(image: np.ndarray, yolo_quad: np.ndarray | None) -> tuple[np.ndarray | None, dict]:
    """YOLO-first refinement seeded from top-center pixels; apply only to top ROI band."""
    h, w = image.shape[:2]
    top_h = max(MASK_SAMPLE_TOP_ROWS + 1, int(round(h * MASK_APPLY_TOP_REGION_FRACTION)))
    top_h = min(h, top_h)

    sample_rows = max(1, min(h, MASK_SAMPLE_TOP_ROWS))
    center_band_w = max(1, int(round(w * MASK_SAMPLE_CENTER_WIDTH_FRACTION)))
    sample_x1 = max(0, (w - center_band_w) // 2)
    sample_x2 = min(w, sample_x1 + center_band_w)
    if sample_x2 <= sample_x1:
        sample_x1 = 0
        sample_x2 = w

    apply_band_w = max(1, int(round(w * MASK_APPLY_CENTER_WIDTH_FRACTION)))
    apply_x1 = max(0, (w - apply_band_w) // 2)
    apply_x2 = min(w, apply_x1 + apply_band_w)
    if apply_x2 <= apply_x1:
        apply_x1 = 0
        apply_x2 = w

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    top_sample = lab[:sample_rows, sample_x1:sample_x2, :].reshape(-1, 3).astype(np.float32)
    if top_sample.size == 0:
        top_sample = lab[:sample_rows, :, :].reshape(-1, 3).astype(np.float32)
    center = np.median(top_sample, axis=0)

    top_dist = np.linalg.norm(top_sample - center, axis=1)
    thr = float(np.mean(top_dist) + MASK_THRESH_STD_MULTIPLIER * np.std(top_dist))
    thr = float(np.clip(thr, MASK_THRESH_MIN, MASK_THRESH_MAX))

    dist_map = np.linalg.norm(lab.astype(np.float32) - center.reshape(1, 1, 3), axis=2)
    bg_mask = (dist_map <= thr).astype(np.uint8) * 255

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    bg_mask = cv2.morphologyEx(bg_mask, cv2.MORPH_OPEN, k, iterations=1)
    bg_mask = cv2.morphologyEx(bg_mask, cv2.MORPH_CLOSE, k, iterations=1)

    # Keep only top-connected background so masking grows downward from top.
    n_lbl, lbl, _stats, _ = cv2.connectedComponentsWithStats(bg_mask, connectivity=8)
    top_labels = set(int(v) for v in np.unique(lbl[0, :]) if int(v) != 0)
    bg_connected = np.zeros_like(bg_mask)
    for label in top_labels:
        bg_connected[lbl == label] = 255

    # Enforce masking only in the top 10% region and center 60% width.
    bg_connected[:, :apply_x1] = 0
    bg_connected[:, apply_x2:] = 0
    bg_connected[top_h:, :] = 0

    fg_mask = cv2.bitwise_not(bg_connected)
    fg_mask[:top_h, :] = 0
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, k, iterations=1)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, k, iterations=2)

    # Bottom edge of masked pixels defines the card angle (top edge direction).
    xs = np.arange(w, dtype=np.int32)
    bottom_pts = []
    for x in xs:
        ys = np.where(bg_connected[:, x] > 0)[0]
        if ys.size == 0:
            continue
        bottom_pts.append((float(x), float(np.max(ys))))

    line_info = None
    if len(bottom_pts) >= max(10, int(0.25 * w)):
        pts = np.array(bottom_pts, dtype=np.float32)
        fit = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01)
        vx, vy, x0, y0 = [float(v) for v in fit.reshape(-1)]
        nrm = max(1e-6, math.hypot(vx, vy))
        ux, uy = vx / nrm, vy / nrm
        if ux < 0:
            ux, uy = -ux, -uy
        # Build line endpoints across image width.
        if abs(ux) < 1e-6:
            y_left = y0
            y_right = y0
        else:
            t_left = (0.0 - x0) / ux
            t_right = ((w - 1.0) - x0) / ux
            y_left = y0 + t_left * uy
            y_right = y0 + t_right * uy
        line_info = {
            "ux": ux,
            "uy": uy,
            "x0": x0,
            "y0": y0,
            "p_left": (0.0, float(np.clip(y_left, 0, h - 1))),
            "p_right": (float(w - 1), float(np.clip(y_right, 0, h - 1))),
            "bottom_points": pts,
        }

    # Constrain refinement around YOLO quad when available; otherwise use the full ROI.
    if yolo_quad is not None:
        q = order_points(yolo_quad)
        xmin = int(np.floor(np.min(q[:, 0])))
        xmax = int(np.ceil(np.max(q[:, 0])))
        ymin = int(np.floor(np.min(q[:, 1])))
        ymax = int(np.ceil(np.max(q[:, 1])))
        bw = max(1, xmax - xmin)
        bh = max(1, ymax - ymin)
        padx = int(round(bw * MASK_SEARCH_PAD_RATIO))
        pady = int(round(bh * MASK_SEARCH_PAD_RATIO))

        sx1 = max(0, xmin - padx)
        sy1 = max(0, ymin - pady)
        sx2 = min(w, xmax + padx)
        sy2 = min(h, ymax + pady)

        search_mask = np.zeros_like(fg_mask)
        search_mask[sy1:sy2, sx1:sx2] = 255
        fg_local = cv2.bitwise_and(fg_mask, search_mask)
    else:
        sx1, sy1, sx2, sy2 = 0, 0, w, h
        fg_local = fg_mask

    min_area = int(h * w * MASK_MIN_COMPONENT_AREA_RATIO)
    comp = _largest_component(fg_local, min_area)
    if comp is None:
        comp = _largest_component(fg_mask, min_area)

    corrected_quad = None
    if comp is not None and line_info is not None:
        x0 = line_info["x0"]
        y0 = line_info["y0"]
        ux, uy = line_info["ux"], line_info["uy"]

        # Card extends downward from the fitted top axis.
        nx, ny = -uy, ux
        if ny < 0:
            nx, ny = -nx, -ny

        if yolo_quad is not None:
            q = order_points(yolo_quad)
            yolo_top_center = 0.5 * (q[0] + q[1])

            # Keep YOLO width, but set card height by MTG aspect from that width.
            yolo_w_top = float(np.linalg.norm(q[1] - q[0]))
            yolo_w_bottom = float(np.linalg.norm(q[2] - q[3]))
            width_final = max(8.0, 0.5 * (yolo_w_top + yolo_w_bottom))
            height_final = max(12.0, width_final / TARGET_ASPECT_WH)

            # Project YOLO top-center onto fitted axis so the axis becomes the new top.
            tcx, tcy = float(yolo_top_center[0]), float(yolo_top_center[1])
            t = (tcx - x0) * ux + (tcy - y0) * uy
            cx = x0 + t * ux
            cy = y0 + t * uy
        else:
            pts = line_info["bottom_points"]
            x_min_m = float(np.min(pts[:, 0]))
            x_max_m = float(np.max(pts[:, 0]))
            width_final = max(8.0, x_max_m - x_min_m)

            cx = 0.5 * (x_min_m + x_max_m)
            if abs(ux) < 1e-6:
                cy = y0
            else:
                t = (cx - x0) / ux
                cy = y0 + t * uy

            # Fallback path keeps MTG aspect when YOLO geometry is unavailable.
            height_final = width_final / TARGET_ASPECT_WH

        half_w = 0.5 * width_final
        top_left = np.array([cx - ux * half_w, cy - uy * half_w], dtype=np.float32)
        top_right = np.array([cx + ux * half_w, cy + uy * half_w], dtype=np.float32)

        bot_right = top_right + np.array([nx * height_final, ny * height_final], dtype=np.float32)
        bot_left = top_left + np.array([nx * height_final, ny * height_final], dtype=np.float32)

        quad = np.array([top_left, top_right, bot_right, bot_left], dtype=np.float32)
        corrected_quad = order_points(fit_quad_in_frame_rigid(quad, w, h))

    debug = {
        "top_h": top_h,
        "sample_rows": sample_rows,
        "sample_x1": sample_x1,
        "sample_x2": sample_x2,
        "apply_x1": apply_x1,
        "apply_x2": apply_x2,
        "threshold": thr,
        "bg_mask_raw": bg_mask,
        "dist_map": dist_map,
        "bg_connected": bg_connected,
        "fg_mask": fg_mask,
        "card_mask": fg_mask,
        "fg_component": comp,
        "card_component": comp,
        "search_rect": (sx1, sy1, sx2, sy2),
        "line_info": line_info,
    }
    return corrected_quad, debug


def warp_from_quad(image: np.ndarray, quad: np.ndarray) -> np.ndarray:
    quad = order_points(quad)

    width_a = float(np.linalg.norm(quad[2] - quad[3]))
    width_b = float(np.linalg.norm(quad[1] - quad[0]))
    height_a = float(np.linalg.norm(quad[1] - quad[2]))
    height_b = float(np.linalg.norm(quad[0] - quad[3]))

    max_width = max(2, int(round(max(width_a, width_b))))
    max_height = max(2, int(round(max(height_a, height_b))))

    dst = np.array(
        [[0, 0], [max_width - 1, 0], [max_width - 1, max_height - 1], [0, max_height - 1]],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(quad.astype(np.float32), dst)
    warped = cv2.warpPerspective(image, matrix, (max_width, max_height), flags=cv2.INTER_CUBIC)

    # Ensure portrait orientation for downstream hashing.
    if warped.shape[1] > warped.shape[0]:
        warped = cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)

    return warped


def rotate_then_crop_from_quad(image: np.ndarray, quad: np.ndarray) -> tuple[np.ndarray, dict]:
    """Rotate so quad top edge is horizontal, then crop by rotated quad bounds (no perspective warp)."""
    q = order_points(quad).astype(np.float32)
    h, w = image.shape[:2]

    top_vec = q[1] - q[0]
    angle_deg = float(np.degrees(np.arctan2(float(top_vec[1]), float(top_vec[0]))))

    center = (w * 0.5, h * 0.5)
    matrix = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    rotated = cv2.warpAffine(
        image,
        matrix,
        (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )

    q_rot = cv2.transform(np.array([q], dtype=np.float32), matrix)[0]
    q_rot = order_points(q_rot)

    x1 = int(np.floor(np.min(q_rot[:, 0])))
    y1 = int(np.floor(np.min(q_rot[:, 1])))
    x2 = int(np.ceil(np.max(q_rot[:, 0])))
    y2 = int(np.ceil(np.max(q_rot[:, 1])))

    x1 = max(0, min(w - 1, x1))
    y1 = max(0, min(h - 1, y1))
    x2 = max(x1 + 1, min(w, x2))
    y2 = max(y1 + 1, min(h, y2))

    cropped = rotated[y1:y2, x1:x2].copy()

    # Keep portrait orientation for downstream stages.
    if cropped.shape[1] > cropped.shape[0]:
        cropped = cv2.rotate(cropped, cv2.ROTATE_90_CLOCKWISE)

    debug = {
        "axis_angle_deg": angle_deg,
        "crop_xyxy": (x1, y1, x2, y2),
    }
    return cropped, debug


def _transform_points_with_affine(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    return cv2.transform(np.array([points], dtype=np.float32), matrix)[0]


def build_axis_crop_stages(
    image: np.ndarray,
    yolo_quad: np.ndarray,
    line_info: dict | None,
) -> tuple[np.ndarray, dict[str, np.ndarray], dict]:
    q = order_points(yolo_quad).astype(np.float32)
    h, w = image.shape[:2]

    top_center = 0.5 * (q[0] + q[1])
    bottom_center = 0.5 * (q[3] + q[2])
    box_center = 0.5 * (top_center + bottom_center)
    yolo_width = max(8.0, 0.5 * (float(np.linalg.norm(q[1] - q[0])) + float(np.linalg.norm(q[2] - q[3]))))

    if line_info is not None:
        ux = float(line_info["ux"])
        uy = float(line_info["uy"])
        axis_points = np.array([line_info["p_left"], line_info["p_right"]], dtype=np.float32)
    else:
        top_vec = q[1] - q[0]
        norm = max(1e-6, float(np.linalg.norm(top_vec)))
        ux = float(top_vec[0] / norm)
        uy = float(top_vec[1] / norm)
        axis_points = np.array([q[0], q[1]], dtype=np.float32)

    angle_deg = float(np.degrees(np.arctan2(uy, ux)))
    center = (w * 0.5, h * 0.5)
    matrix = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    rotated = cv2.warpAffine(
        image,
        matrix,
        (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )

    q_rot = order_points(_transform_points_with_affine(q, matrix))
    axis_rot = _transform_points_with_affine(axis_points, matrix)
    top_center_rot = _transform_points_with_affine(np.array([top_center], dtype=np.float32), matrix)[0]
    bottom_center_rot = _transform_points_with_affine(np.array([bottom_center], dtype=np.float32), matrix)[0]
    box_center_rot = _transform_points_with_affine(np.array([box_center], dtype=np.float32), matrix)[0]

    top_y = float(np.mean(axis_rot[:, 1]))
    bottom_y = float(bottom_center_rot[1])
    if bottom_y <= top_y + 2.0:
        bottom_y = float(np.max(q_rot[:, 1]))

    axis_distance = max(1.0, float(bottom_y - top_y))
    rotation_for_correction_deg = abs(angle_deg)
    correction_factor = 0.5
    # Use MTG portrait ratio: width is derived from the measured crop height.
    corrected_width = max(12.0, float(axis_distance * TARGET_ASPECT_WH))

    center_x = float(box_center_rot[0])
    left_x = center_x - (0.5 * corrected_width)
    right_x = center_x + (0.5 * corrected_width)

    top_y_i = max(0, min(h - 2, int(round(top_y))))
    bottom_y_i = max(top_y_i + 1, min(h, int(round(bottom_y))))
    left_x_i = max(0, min(w - 2, int(round(left_x))))
    right_x_i = max(left_x_i + 1, min(w, int(round(right_x))))

    stage_01 = draw_quad(image, q, color=(0, 255, 0), thickness=3)

    stage_02 = image.copy()
    p_left = tuple(int(round(v)) for v in axis_points[0])
    p_right = tuple(int(round(v)) for v in axis_points[1])
    cv2.line(stage_02, p_left, p_right, (255, 255, 255), 3)

    stage_03 = rotated.copy()
    cv2.line(stage_03, (0, top_y_i), (w - 1, top_y_i), (255, 255, 255), 2)

    stage_04 = rotated[top_y_i:, :].copy()

    stage_05 = stage_04.copy()
    cv2.line(stage_05, (left_x_i, 0), (left_x_i, stage_05.shape[0] - 1), (0, 255, 255), 2)
    cv2.line(stage_05, (right_x_i - 1, 0), (right_x_i - 1, stage_05.shape[0] - 1), (0, 255, 255), 2)

    stage_06 = stage_04.copy()
    bottom_y_rel = max(0, min(stage_06.shape[0] - 1, bottom_y_i - top_y_i))
    cv2.line(stage_06, (0, bottom_y_rel), (w - 1, bottom_y_rel), (255, 255, 255), 2)

    stage_07 = rotated[top_y_i:bottom_y_i, :].copy()

    stage_08 = stage_07.copy()
    cv2.line(stage_08, (left_x_i, 0), (left_x_i, stage_08.shape[0] - 1), (0, 255, 255), 2)
    cv2.line(stage_08, (right_x_i - 1, 0), (right_x_i - 1, stage_08.shape[0] - 1), (0, 255, 255), 2)

    final_crop = rotated[top_y_i:bottom_y_i, left_x_i:right_x_i].copy()
    if final_crop.shape[1] > final_crop.shape[0]:
        final_crop = cv2.rotate(final_crop, cv2.ROTATE_90_CLOCKWISE)

    stages = {
        "01_yolo_guess": stage_01,
        "02_top_axis": stage_02,
        "03_rotated_flat_axis": stage_03,
        "04_top_cropped": stage_04,
        "05_side_bounds_from_yolo_width": stage_05,
        "06_bottom_axis": stage_06,
        "07_top_bottom_cropped": stage_07,
        "08_side_cropped": stage_08,
        "09_final_crop": final_crop,
    }
    meta = {
        "axis_angle_deg": angle_deg,
        "top_y": top_y_i,
        "bottom_y": bottom_y_i,
        "left_x": left_x_i,
        "right_x": right_x_i,
        "yolo_width": float(yolo_width),
        "axis_distance": float(axis_distance),
        "rotation_for_correction_deg": float(rotation_for_correction_deg),
        "correction_factor": float(correction_factor),
        "corrected_width": float(corrected_width),
        "used_mask_axis": bool(line_info is not None),
    }
    return final_crop, stages, meta


def tighten_crop(image: np.ndarray, trim_ratio: float) -> np.ndarray:
    if trim_ratio <= 0.0:
        return image.copy()

    h, w = image.shape[:2]
    dx = int(round(w * trim_ratio))
    dy = int(round(h * trim_ratio))

    x1 = min(max(dx, 0), w - 2)
    y1 = min(max(dy, 0), h - 2)
    x2 = max(x1 + 1, w - dx)
    y2 = max(y1 + 1, h - dy)

    return image[y1:y2, x1:x2].copy()


def normalize_aspect_and_size(image: np.ndarray, target_w: int = 744, target_h: int = 1040) -> np.ndarray:
    h, w = image.shape[:2]
    if h < 2 or w < 2:
        return cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_AREA)

    current_aspect = w / max(h, 1)
    target_aspect = TARGET_ASPECT_WH

    # Center-crop to target aspect first to reduce residual border skew.
    if current_aspect > target_aspect:
        new_w = int(round(h * target_aspect))
        x1 = max(0, (w - new_w) // 2)
        x2 = min(w, x1 + new_w)
        adjusted = image[:, x1:x2]
    else:
        new_h = int(round(w / target_aspect))
        y1 = max(0, (h - new_h) // 2)
        y2 = min(h, y1 + new_h)
        adjusted = image[y1:y2, :]

    return cv2.resize(adjusted, (target_w, target_h), interpolation=cv2.INTER_AREA)


def laplacian_variance(image: np.ndarray) -> float:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def draw_grid_overlay(image: np.ndarray, rows: int, cols: int) -> np.ndarray:
    out = image.copy()
    h, w = out.shape[:2]

    for r in range(1, rows):
        y = int(round(r * h / rows))
        cv2.line(out, (0, y), (w - 1, y), (0, 255, 255), 1, cv2.LINE_AA)
    for c in range(1, cols):
        x = int(round(c * w / cols))
        cv2.line(out, (x, 0), (x, h - 1), (0, 255, 255), 1, cv2.LINE_AA)

    return out


def annotate_quality(image: np.ndarray, metrics: QualityMetrics) -> np.ndarray:
    out = image.copy()
    lines = [
        f"det_conf={metrics.detector_confidence:.3f}",
        f"blur_var={metrics.blur_laplacian_var:.1f}",
        f"size={metrics.warped_width}x{metrics.warped_height}",
        f"aspect_w_h={metrics.warped_aspect_wh:.3f}",
    ]
    y = 26
    for line in lines:
        cv2.putText(out, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(out, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (30, 30, 30), 1, cv2.LINE_AA)
        y += 28
    return out


def quality_gate(metrics: QualityMetrics) -> tuple[bool, list[str]]:
    reasons: list[str] = []

    # Conservative default gates; can be bypassed via CLI.
    if metrics.detector_confidence < 0.35:
        reasons.append(f"low detector confidence ({metrics.detector_confidence:.3f})")

    if metrics.blur_laplacian_var < 55.0:
        reasons.append(f"high blur (lap_var={metrics.blur_laplacian_var:.1f})")

    target = TARGET_ASPECT_WH
    if abs(metrics.warped_aspect_wh - target) > 0.12:
        reasons.append(f"aspect drift ({metrics.warped_aspect_wh:.3f} vs {target:.3f})")

    return (len(reasons) == 0), reasons


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_image(path: Path, image: np.ndarray) -> None:
    ensure_dir(path.parent)
    cv2.imwrite(str(path), image)


def build_matcher(args: argparse.Namespace) -> HybridTopHalfMatcher:
    cfg = SegmentedHashConfig()
    return HybridTopHalfMatcher(
        source_module_path=Path(args.source_module),
        full_card_db_path=Path(args.full_db),
        segmented_db_path=Path(args.segmented_db),
        cfg=cfg,
    )


def save_topbg_debug_bundle(debug_dir: Path, stem: str, roi: np.ndarray, topbg_debug: dict | None) -> None:
    if topbg_debug is None:
        return

    ensure_dir(debug_dir)

    overlay = topbg_debug.get("overlay")
    if overlay is not None:
        save_image(debug_dir / f"{stem}_topbg_overlay.png", overlay)

    bg_raw = topbg_debug.get("bg_mask_raw")
    if bg_raw is not None:
        save_image(debug_dir / f"{stem}_topbg_bg_mask_raw.png", bg_raw)

    bg_connected = topbg_debug.get("bg_connected")
    if bg_connected is not None:
        save_image(debug_dir / f"{stem}_topbg_bg_connected.png", bg_connected)

    card_mask = topbg_debug.get("card_mask")
    if card_mask is not None:
        save_image(debug_dir / f"{stem}_topbg_card_mask.png", card_mask)

    card_component = topbg_debug.get("card_component")
    if card_component is not None:
        save_image(debug_dir / f"{stem}_topbg_card_component.png", card_component)

    dist_map = topbg_debug.get("dist_map")
    if dist_map is not None:
        dist_norm = cv2.normalize(dist_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        heat = cv2.applyColorMap(dist_norm, cv2.COLORMAP_TURBO)
        save_image(debug_dir / f"{stem}_topbg_distance_heatmap.png", heat)

    # Explicit visualization of masked-out pixels (background removed from ROI).
    if bg_connected is not None:
        masked_overlay = roi.copy()
        masked_overlay[bg_connected > 0] = (0, 0, 255)
        save_image(debug_dir / f"{stem}_topbg_masked_pixels.png", masked_overlay)

    if card_component is not None:
        foreground_only = cv2.bitwise_and(roi, roi, mask=card_component)
        save_image(debug_dir / f"{stem}_topbg_foreground_only.png", foreground_only)


def capture_single_frame_from_camera(args: argparse.Namespace, output_dir: Path) -> tuple[Path, Path]:
    window_name = "GridHash Camera Capture"
    roi_window = "GridHash ROI (YOLO-Treated)"
    mask_window = "GridHash Mask (YOLO-Treated)"

    module = cam.load_source_module()

    cap, active_camera_index = ycs.configure_camera_for_yolo(
        module,
        args.camera_index,
        None,
        None,
        None,
        None,
    )

    live_controls_preset = ycs.load_live_controls_preset(ycs.LIVE_CONTROLS_PRESET_DEFAULT)
    fixed_roi_controls = ycs.get_fixed_roi_controls(live_controls_preset)

    lens_k1 = ycs.DEFAULT_LENS_K1
    lens_k2 = ycs.DEFAULT_LENS_K2
    lens_focal = ycs.DEFAULT_LENS_FOCAL_SCALE

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.namedWindow(roi_window, cv2.WINDOW_NORMAL)
    cv2.namedWindow(mask_window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1200, 760)
    cv2.resizeWindow(roi_window, 700, 900)
    cv2.resizeWindow(mask_window, 700, 900)

    captured_path: Path | None = None
    captured_yolo_input_path: Path | None = None
    try:
        while True:
            ok, frame_raw = cap.read()
            if not ok or frame_raw is None:
                continue

            frame = ycs.apply_lens_correction_custom(frame_raw, lens_k1, lens_k2, lens_focal)
            roi, roi_x1, roi_y1, roi_x2, roi_y2 = ycs.get_shifted_roi_full(
                module,
                frame,
                fixed_roi_controls["roi_shift_x"],
                fixed_roi_controls["roi_shift_y"],
                fixed_roi_controls["roi_expand_left"],
                fixed_roi_controls["roi_expand_right"],
                fixed_roi_controls["roi_expand_up"],
                fixed_roi_controls["roi_expand_down"],
            )

            processed_roi, binary_map = ycs.preprocess_for_yolo(
                roi,
                mode=ycs.DEFAULT_PREPROCESS_MODE,
                adaptive_block=ycs.DEFAULT_ADAPTIVE_BLOCK,
                adaptive_c=ycs.DEFAULT_ADAPTIVE_C,
                canny_low=0,
                canny_high=0,
                close_kernel=ycs.DEFAULT_CLOSE_KERNEL,
                dilate_iters=ycs.DEFAULT_DILATE_ITERS,
                perspective_top=ycs.DEFAULT_PERSPECTIVE_TOP_SHRINK,
                perspective_bottom=ycs.DEFAULT_PERSPECTIVE_BOTTOM_GROW,
            )

            topbg_quad, topbg_debug = refine_quad_with_top_mask(roi, None)

            preview = frame.copy()
            cv2.rectangle(preview, (roi_x1, roi_y1), (roi_x2, roi_y2), (0, 255, 255), 3)
            cv2.putText(preview, f"Camera index: {active_camera_index}", (20, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(preview, "Press S to capture YOLO-treated ROI", (20, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(preview, "Press Q or ESC to cancel", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            if topbg_quad is not None:
                quad_vis = topbg_quad.copy()
                quad_vis[:, 0] += roi_x1
                quad_vis[:, 1] += roi_y1
                cv2.polylines(preview, [quad_vis.astype(np.int32)], True, (0, 255, 0), 3)

            if topbg_debug is not None:
                line_info = topbg_debug.get("line_info")
                if line_info is not None:
                    p_left = (
                        int(round(line_info["p_left"][0] + roi_x1)),
                        int(round(line_info["p_left"][1] + roi_y1)),
                    )
                    p_right = (
                        int(round(line_info["p_right"][0] + roi_x1)),
                        int(round(line_info["p_right"][1] + roi_y1)),
                    )
                    cv2.line(preview, p_left, p_right, (255, 255, 255), 2)

            cv2.imshow(window_name, preview)
            cv2.imshow(roi_window, processed_roi)

            mask_vis = roi.copy()
            if topbg_debug is not None:
                bg_connected = topbg_debug.get("bg_connected")
                if bg_connected is not None:
                    # Show exactly what gets masked out from the top-seeded color sample.
                    mask_vis[bg_connected > 0] = (0, 0, 255)

                fg_component = topbg_debug.get("fg_component")
                if fg_component is not None:
                    contours, _ = cv2.findContours(fg_component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if contours:
                        c = max(contours, key=cv2.contourArea)
                        x, y, bw, bh = cv2.boundingRect(c)
                        cv2.rectangle(mask_vis, (x, y), (x + bw, y + bh), (255, 255, 255), 1)

                top_h = int(topbg_debug.get("top_h", int(MASK_APPLY_TOP_REGION_FRACTION * roi.shape[0])))
                apply_x1 = int(topbg_debug.get("apply_x1", int(round(roi.shape[1] * 0.20))))
                apply_x2 = int(topbg_debug.get("apply_x2", int(round(roi.shape[1] * 0.80))))
                cv2.rectangle(
                    mask_vis,
                    (max(0, apply_x1), 0),
                    (min(roi.shape[1] - 1, max(0, apply_x2 - 1)), max(0, top_h - 1)),
                    (255, 255, 0),
                    2,
                )

                sample_rows = int(topbg_debug.get("sample_rows", MASK_SAMPLE_TOP_ROWS))
                sample_x1 = int(topbg_debug.get("sample_x1", int(round(roi.shape[1] * 0.25))))
                sample_x2 = int(topbg_debug.get("sample_x2", int(round(roi.shape[1] * 0.75))))
                cv2.rectangle(
                    mask_vis,
                    (max(0, sample_x1), 0),
                    (min(roi.shape[1] - 1, max(0, sample_x2 - 1)), max(0, sample_rows - 1)),
                    (0, 255, 255),
                    2,
                )

                line_info = topbg_debug.get("line_info")
                if line_info is not None:
                    p_left = tuple(int(round(v)) for v in line_info["p_left"])
                    p_right = tuple(int(round(v)) for v in line_info["p_right"])
                    cv2.line(mask_vis, p_left, p_right, (255, 255, 255), 2)

            if topbg_quad is not None:
                cv2.polylines(mask_vis, [topbg_quad.astype(np.int32)], True, (0, 255, 0), 2)
            cv2.imshow(mask_window, mask_vis)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                raise RuntimeError("Capture canceled by user.")
            if key == ord("s"):
                ts = time.strftime("%Y%m%d_%H%M%S")
                ms = int((time.time() % 1.0) * 1000)
                capture_dir = output_dir / "captured_inputs"
                ensure_dir(capture_dir)
                captured_path = capture_dir / f"capture_{ts}_{ms:03d}.jpg"

                # Raw ROI is the canonical pipeline image.
                cv2.imwrite(str(captured_path), roi)

                # Save YOLO candidate input variant that matches studio preprocessing.
                captured_yolo_input_path = capture_dir / f"capture_{ts}_{ms:03d}_yolo_input.jpg"
                cv2.imwrite(str(captured_yolo_input_path), processed_roi)

                break
    finally:
        cap.release()
        cv2.destroyWindow(window_name)
        cv2.destroyWindow(roi_window)
        cv2.destroyWindow(mask_window)

    if captured_path is None or captured_yolo_input_path is None:
        raise RuntimeError("No frame captured.")

    print(f"[pipeline] captured image: {captured_path}")
    print(f"[pipeline] captured YOLO candidate: {captured_yolo_input_path}")
    return captured_path, captured_yolo_input_path


def summarize_variant(variant_name: str, trim_ratio: float, identify_result: dict) -> VariantResult:
    best = identify_result.get("best")
    return VariantResult(
        variant=variant_name,
        trim_ratio=trim_ratio,
        status=identify_result.get("status", "reject"),
        best_name=(best or {}).get("name"),
        best_set=(best or {}).get("set"),
        orientation=(best or {}).get("orientation"),
        final_score=(best or {}).get("final_score"),
        weighted_matches=(best or {}).get("weighted_matches"),
        matched_regions=(best or {}).get("matched_regions"),
        total_regions=(best or {}).get("total_regions"),
        avg_region_distance=(best or {}).get("avg_region_distance"),
    )


def best_variant(variants: list[VariantResult]) -> VariantResult | None:
    if not variants:
        return None

    # Prioritize by status then final score.
    status_rank = {"confident": 3, "possible": 2, "reject": 1}

    def rank(v: VariantResult):
        return (
            status_rank.get(v.status, 0),
            -9999.0 if v.final_score is None else float(v.final_score),
            -9999.0 if v.weighted_matches is None else float(v.weighted_matches),
        )

    return sorted(variants, key=rank, reverse=True)[0]


def make_regions_debug(image_norm: np.ndarray, cfg: SegmentedHashConfig) -> np.ndarray:
    top_h = int(round(image_norm.shape[0] * cfg.top_half_fraction))
    top_h = max(1, min(image_norm.shape[0], top_h))
    top = image_norm[:top_h, :]
    grid = draw_grid_overlay(top, cfg.grid_rows, cfg.grid_cols)

    # Label region indices.
    regions = split_into_grid(top, cfg.grid_rows, cfg.grid_cols)
    for reg in regions:
        x1, y1, x2, y2 = reg["bbox"]
        cx = int(round((x1 + x2) * 0.5))
        cy = int(round((y1 + y2) * 0.5))
        idx = reg["index"]
        cv2.putText(grid, str(idx), (cx - 10, cy + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(grid, str(idx), (cx - 10, cy + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1, cv2.LINE_AA)

    return grid


def process_single_image(
    image_path: Path,
    result,
    matcher: HybridTopHalfMatcher,
    trim_ratios: Iterable[float],
    output_dir: Path,
    top_k: int,
    strict_quality_gate: bool,
) -> ImageRunResult:
    image_name = image_path.stem
    image_output_dir = output_dir / image_name
    ensure_dir(image_output_dir)

    notes: list[str] = []

    orig = cv2.imread(str(image_path))
    if orig is None:
        notes.append("Could not read image with cv2.imread.")
        return ImageRunResult(
            image_path=str(image_path),
            output_dir=str(image_output_dir),
            detector_confidence=None,
            detection_ok=False,
            best_variant=None,
            all_variants=[],
            notes=notes,
        )

    picks = obb_points_from_result(result, top_k=top_k)
    if not picks:
        notes.append("No OBB detections found.")
        return ImageRunResult(
            image_path=str(image_path),
            output_dir=str(image_output_dir),
            detector_confidence=None,
            detection_ok=False,
            best_variant=None,
            all_variants=[],
            notes=notes,
        )

    shared_debug_dir = output_dir / "captured_inputs" / "debug"
    ensure_dir(shared_debug_dir)

    yolo_top_quad, best_conf = picks[0]
    topbg_quad, topbg_debug = refine_quad_with_top_mask(orig, yolo_top_quad)
    line_info = None if topbg_debug is None else topbg_debug.get("line_info")
    final_crop_debug, stage_images, final_crop_debug_info = build_axis_crop_stages(orig, yolo_top_quad, line_info)

    for stage_name, stage_image in stage_images.items():
        save_image(image_output_dir / f"{stage_name}.png", stage_image)
        save_image(shared_debug_dir / f"{image_name}_{stage_name}.png", stage_image)

    with (image_output_dir / "09_final_crop_meta.json").open("w", encoding="utf-8") as f:
        json.dump(final_crop_debug_info, f, indent=2)

    cfg = matcher.cfg
    all_variants: list[VariantResult] = []
    notes.append(f"det01: axis_angle_deg={final_crop_debug_info['axis_angle_deg']:.3f}")
    notes.append(
        "det01: crop_bounds="
        f"top={final_crop_debug_info['top_y']} bottom={final_crop_debug_info['bottom_y']} "
        f"left={final_crop_debug_info['left_x']} right={final_crop_debug_info['right_x']}"
    )

    qm = QualityMetrics(
        detector_confidence=best_conf,
        blur_laplacian_var=laplacian_variance(final_crop_debug),
        warped_width=int(final_crop_debug.shape[1]),
        warped_height=int(final_crop_debug.shape[0]),
        warped_aspect_wh=float(final_crop_debug.shape[1] / max(final_crop_debug.shape[0], 1)),
    )

    ok_quality, reasons = quality_gate(qm)
    if strict_quality_gate and not ok_quality:
        notes.append(f"det01: quality gate reject -> {', '.join(reasons)}")
    else:
        normalized = normalize_aspect_and_size(
            final_crop_debug,
            target_w=cfg.normalized_size[0],
            target_h=cfg.normalized_size[1],
        )
        identify_result = matcher.identify(normalized)
        vr = summarize_variant("det01_trim00", 0.0, identify_result)
        all_variants.append(vr)

        with (image_output_dir / "identify_result.json").open("w", encoding="utf-8") as f:
            json.dump(identify_result, f, indent=2)

    winner = best_variant(all_variants)

    report = {
        "image": str(image_path),
        "detector_confidence": best_conf,
        "winner": asdict(winner) if winner else None,
        "all_variants": [asdict(v) for v in all_variants],
        "notes": notes,
    }
    with (image_output_dir / "result_summary.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    return ImageRunResult(
        image_path=str(image_path),
        output_dir=str(image_output_dir),
        detector_confidence=best_conf,
        detection_ok=winner is not None,
        best_variant=winner,
        all_variants=all_variants,
        notes=notes,
    )


def main() -> None:
    args = parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"YOLO model not found: {model_path}")

    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)

    captured_image, captured_yolo_input = capture_single_frame_from_camera(args, output_dir)

    trim_ratios = parse_trim_ratios(args.trim_ratios)

    print(f"[pipeline] loading YOLO model: {model_path}")
    model = YOLO(str(model_path))

    print("[pipeline] loading grid hash matcher (existing HybridTopHalfMatcher)...")
    matcher = build_matcher(args)

    print("[pipeline] YOLO-first stage: evaluating capture candidates before any masking/refinement")
    # Match YOLO_Camera_Studio verify path: run on saved raw ROI first.
    yolo_candidate_paths = [captured_image]
    chosen_yolo_path, chosen_result, chosen_yolo_conf, yolo_debug_rows = yolo_first_pick_best_result(
        model=model,
        candidate_paths=yolo_candidate_paths,
        imgsz=args.imgsz,
        conf=args.conf,
        device=args.device,
    )
    print(f"[pipeline] YOLO-first selected: {chosen_yolo_path.name}")

    yolo_first_debug = {
        "stage": "yolo_first",
        "candidates": yolo_debug_rows,
        "chosen": str(chosen_yolo_path),
        "chosen_best_conf": None if chosen_yolo_conf is None else float(chosen_yolo_conf),
        "min_required_conf": float(args.min_yolo_conf),
    }
    with (output_dir / "yolo_first_selection.json").open("w", encoding="utf-8") as f:
        json.dump(yolo_first_debug, f, indent=2)

    if chosen_yolo_conf is None or float(chosen_yolo_conf) < float(args.min_yolo_conf):
        fail_payload = {
            "status": "rejected_low_confidence",
            "message": "YOLO-first confidence below required threshold; pipeline stopped before masking/cropping.",
            "chosen": str(chosen_yolo_path),
            "chosen_best_conf": None if chosen_yolo_conf is None else float(chosen_yolo_conf),
            "min_required_conf": float(args.min_yolo_conf),
            "candidates": yolo_debug_rows,
        }
        with (output_dir / "yolo_first_rejected.json").open("w", encoding="utf-8") as f:
            json.dump(fail_payload, f, indent=2)
        raise RuntimeError(
            f"YOLO confidence {chosen_yolo_conf} is below required minimum {args.min_yolo_conf}. "
            "Capture again under the known-good setup; no masking/cropping was applied."
        )

    chosen_area_ratio = None
    for row in yolo_debug_rows:
        if row.get("candidate") == str(chosen_yolo_path):
            chosen_area_ratio = row.get("top_box_area_ratio")
            break

    if chosen_area_ratio is not None and float(chosen_area_ratio) > MAX_VALID_YOLO_BOX_AREA_RATIO:
        fail_payload = {
            "status": "rejected_implausible_yolo_box",
            "message": "Chosen YOLO box is too large (near full ROI), so rotate/crop would be a no-op.",
            "chosen": str(chosen_yolo_path),
            "chosen_best_conf": None if chosen_yolo_conf is None else float(chosen_yolo_conf),
            "chosen_top_box_area_ratio": float(chosen_area_ratio),
            "max_allowed_top_box_area_ratio": float(MAX_VALID_YOLO_BOX_AREA_RATIO),
            "candidates": yolo_debug_rows,
        }
        with (output_dir / "yolo_first_rejected_implausible_box.json").open("w", encoding="utf-8") as f:
            json.dump(fail_payload, f, indent=2)
        raise RuntimeError(
            f"YOLO top box area ratio {chosen_area_ratio:.3f} exceeds max {MAX_VALID_YOLO_BOX_AREA_RATIO:.3f}; "
            "capture rejected before masking/cropping to avoid full-ROI no-op crop."
        )

    run_results: list[ImageRunResult] = []
    print(f"[pipeline] processing {captured_image.name}")
    rr = process_single_image(
        image_path=captured_image,
        result=chosen_result,
        matcher=matcher,
        trim_ratios=trim_ratios,
        output_dir=output_dir,
        top_k=args.top_k,
        strict_quality_gate=args.strict_quality_gate,
    )
    run_results.append(rr)

    if rr.best_variant is None:
        print("  - no accepted variant")
    else:
        bv = rr.best_variant
        print(
            "  - winner: "
            f"{bv.best_name} [{bv.best_set}] "
            f"status={bv.status} score={bv.final_score} "
            f"variant={bv.variant}"
        )

    summary = {
        "source": "camera_capture",
        "captured_image": str(captured_image),
        "camera_index": args.camera_index,
        "model": str(model_path),
        "output_dir": str(output_dir),
        "count": len(run_results),
        "results": [
            {
                "image_path": r.image_path,
                "output_dir": r.output_dir,
                "detector_confidence": r.detector_confidence,
                "detection_ok": r.detection_ok,
                "best_variant": asdict(r.best_variant) if r.best_variant else None,
                "notes": r.notes,
            }
            for r in run_results
        ],
    }

    summary_path = output_dir / "run_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"[pipeline] complete. Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
