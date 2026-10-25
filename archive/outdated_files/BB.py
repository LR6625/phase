"""BB.py - MTG card detector using Hough lines with 2-line fallback.

Strategy:
  1. Build an edge map from the preprocessed ROI.
  2. Detect strong horizontal/vertical line segments with HoughLinesP.
  3. Build card quad from top/bottom + left/right lines when available.
  4. If only one parallel pair is strong (2-line case), infer the missing pair
     from MTG aspect ratio to resist glare-occluded edges.
  5. Warp that box and send it to the matcher.
"""
import argparse
import time
from pathlib import Path
import math

import cv2
import numpy as np

import CamTesting as cam
from mtg_hash_v2.config import SegmentedHashConfig
from mtg_hash_v2.matcher import HybridTopHalfMatcher


WORKSPACE_DIR = Path(__file__).resolve().parent
SOURCE_MODULE_PATH = WORKSPACE_DIR / "Expo_P.H.A.S.E.py"
FULL_DB_PATH = WORKSPACE_DIR / "scryfall_phash_db_sample.json"
SEGMENTED_DB_PATH = WORKSPACE_DIR / "scryfall_segmented_tophalf_db_v2.json"

DEFAULT_ROI_SHIFT_RIGHT_RATIO = 0.06
DEFAULT_ROI_SHIFT_DOWN_RATIO = 0.00
CONTROLS_WINDOW = "BB Controls"
WINDOW_PREPROCESSED = "BB Preprocessed"
DEFAULT_PERSPECTIVE_TOP_SHRINK = 0.95
DEFAULT_PERSPECTIVE_BOTTOM_GROW = 0.95
SERVO2_HOME_ANGLE = 140
SERVO2_DROP_DELTA = 140
ENABLE_PRE_BORDER_PHOTO_NORMALIZATION = True

# MTG card portrait aspect ratio (width / height).
MTG_ASPECT_RATIO = 58 / 78          # approx 0.744
MTG_ASPECT_TOLERANCE = 0.04         # kept for consistency with AA settings

# Area and aspect gates for the inferred/fitted card box.
MIN_CONTOUR_AREA_RATIO = 0.04       # retained for parity with AA (unused here)
MAX_ASPECT_SCORE_TO_COMMIT = 0.25   # score = abs(aspect - target) / target
MIN_BOX_AREA_RATIO = 0.60

# BB-specific speed knobs. These intentionally narrow the expensive segmented pass.
AA_STAGE1_K = 250
AA_STAGE1_MAX_CANDIDATES = 80
AA_POSSIBLE_LIST_SIZE = 5
REGULAR_HASH_MAX_DISTANCE = 24.0

# Hough line localizer settings.
HOUGH_THRESHOLD = 80
HOUGH_MIN_LINE_LENGTH_RATIO = 0.25
HOUGH_MAX_LINE_GAP_RATIO = 0.05
HOUGH_ANGLE_TOLERANCE_DEG = 20.0
HOUGH_CLUSTER_MIN_GAP_RATIO = 0.15

# Directional ROI expansion (fraction of ROI width / height).
ROI_EXPAND_LEFT_RATIO = 0.02
ROI_EXPAND_RIGHT_RATIO = 0.01
ROI_EXPAND_UP_RATIO = 0.01
ROI_EXPAND_DOWN_RATIO = 0.0075

# Brightness/contrast adjustment after normalization.
# Pixel transform: out = clip(CONTRAST * in + BRIGHTNESS, 0..255)
BRIGHTNESS = 0
CONTRAST = 1.0

# Glare reduction settings (specular highlight suppression).
ENABLE_GLARE_REDUCTION = True
DEFAULT_GLARE_FIX_STRENGTH = 100
GLARE_VALUE_THRESHOLD = 240
GLARE_SAT_THRESHOLD = 65
GLARE_MIN_BLOB_AREA_RATIO = 0.00005
GLARE_MAX_BLOB_AREA_RATIO = 0.02
GLARE_DILATE_ITERS = 1
GLARE_INPAINT_RADIUS = 3


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _noop(_value):
    return None


def _odd_from_slider(value: int, min_odd: int = 3) -> int:
    value = max(min_odd, int(value))
    if value % 2 == 0:
        value += 1
    return value


def reduce_glare_specular(image, glare_fix_strength: int):
    """Suppress compact specular highlights while keeping large white regions intact."""
    if not ENABLE_GLARE_REDUCTION or glare_fix_strength <= 0:
        return image

    h, w = image.shape[:2]
    if h < 2 or w < 2:
        return image

    strength = max(0.0, min(1.0, glare_fix_strength / 100.0))

    value_threshold = int(round(GLARE_VALUE_THRESHOLD - 35.0 * strength))
    sat_threshold = int(round(GLARE_SAT_THRESHOLD + 60.0 * strength))
    max_blob_area_ratio = GLARE_MAX_BLOB_AREA_RATIO + 0.04 * strength
    dilate_iters = GLARE_DILATE_ITERS + int(round(2 * strength))
    inpaint_radius = GLARE_INPAINT_RADIUS + int(round(3 * strength))

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]

    # High-value + low-saturation pixels are typical glare candidates.
    raw_mask = ((v >= value_threshold) & (s <= sat_threshold)).astype(np.uint8) * 255
    if not np.any(raw_mask):
        return image

    # Keep only compact blobs to avoid inpainting large white card areas.
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(raw_mask, connectivity=8)
    area_total = float(h * w)
    min_area = max(1, int(area_total * GLARE_MIN_BLOB_AREA_RATIO))
    max_area = max(min_area, int(area_total * max_blob_area_ratio))

    mask = np.zeros_like(raw_mask)
    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if min_area <= area <= max_area:
            mask[labels == label] = 255

    if not np.any(mask):
        return image

    if dilate_iters > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.dilate(mask, kernel, iterations=dilate_iters)

    return cv2.inpaint(image, mask, inpaint_radius, cv2.INPAINT_TELEA)


def normalize_frame_pre_border(frame, brightness: int, contrast: float):
    corrected = cam.apply_lens_distortion_correction(frame)
    if not ENABLE_PRE_BORDER_PHOTO_NORMALIZATION:
        return corrected

    lab = cv2.cvtColor(corrected, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
    corrected = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)
    corrected = cv2.bilateralFilter(corrected, d=5, sigmaColor=35, sigmaSpace=35)

    if contrast != 1.0 or brightness != 0:
        corrected = cv2.convertScaleAbs(corrected, alpha=contrast, beta=brightness)

    return corrected


def get_shifted_trimmed_roi(module, frame, shift_right_ratio: float, shift_down_ratio: float):
    _, roi_x1, roi_y1, roi_x2, roi_y2 = module.get_center_roi(frame)
    roi_width = max(roi_x2 - roi_x1, 1)
    roi_height = max(roi_y2 - roi_y1, 1)
    shift_px = int(roi_width * max(0.0, shift_right_ratio))
    shift_py = int(roi_height * shift_down_ratio)

    frame_h, frame_w = frame.shape[:2]
    shifted_x1 = min(max(roi_x1 + shift_px, 0), max(frame_w - 2, 0))
    shifted_x2 = min(max(roi_x2 + shift_px, shifted_x1 + 1), frame_w)

    extra_left = int(roi_width * max(0.0, ROI_EXPAND_LEFT_RATIO))
    extra_right = int(roi_width * max(0.0, ROI_EXPAND_RIGHT_RATIO))
    extra_up = int(roi_height * max(0.0, ROI_EXPAND_UP_RATIO))
    extra_down = int(roi_height * max(0.0, ROI_EXPAND_DOWN_RATIO))
    shifted_x1 = max(0, shifted_x1 - extra_left)
    shifted_x2 = min(frame_w, shifted_x2 + extra_right)

    shifted_y1 = max(0, roi_y1 + shift_py)
    shifted_y2 = max(shifted_y1 + 2, roi_y2 + shift_py)
    shifted_y1 = min(shifted_y1, frame_h - 2)
    shifted_y2 = min(max(shifted_y2, shifted_y1 + 1), frame_h)
    shifted_y1 = max(0, shifted_y1 - extra_up)
    shifted_y2 = min(frame_h, max(shifted_y1 + 1, shifted_y2 + extra_down))

    roi_height_trimmed = max(shifted_y2 - shifted_y1, 1)
    trim_pixels = max(1, int(roi_height_trimmed * cam.ROI_VERTICAL_TRIM_RATIO))
    trimmed_y1 = min(shifted_y2 - 1, shifted_y1 + trim_pixels)
    trimmed_y2 = max(trimmed_y1 + 1, shifted_y2 - trim_pixels)

    trimmed_roi = frame[trimmed_y1:trimmed_y2, shifted_x1:shifted_x2]
    return trimmed_roi, shifted_x1, trimmed_y1, shifted_x2, trimmed_y2


def apply_pre_edge_perspective(roi_frame, top_shrink_ratio: float, bottom_grow_ratio: float):
    h, w = roi_frame.shape[:2]
    if h < 2 or w < 2:
        return roi_frame
    top_ratio = min(max(float(top_shrink_ratio), 0.7), 1.0)
    bottom_ratio = min(max(float(bottom_grow_ratio), 0.7), 1.0)
    top_margin = 0.5 * w * (1.0 - top_ratio)
    bottom_margin = 0.5 * w * (1.0 - bottom_ratio)
    src = cam.module_np_array([
        [top_margin, 0],
        [w - 1 - top_margin, 0],
        [w - 1 - bottom_margin, h - 1],
        [bottom_margin, h - 1],
    ])
    dst = cam.module_np_array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
    matrix = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(roi_frame, matrix, (w, h))


def preprocess_for_aa(roi_frame, canny_low, canny_high, close_kernel_size,
                      dilate_iters, perspective_top_shrink, perspective_bottom_grow,
                      glare_fix_strength):
    roi_frame = reduce_glare_specular(roi_frame, glare_fix_strength)
    corrected_roi = apply_pre_edge_perspective(roi_frame, perspective_top_shrink, perspective_bottom_grow)
    if corrected_roi.shape[:2] != roi_frame.shape[:2]:
        corrected_roi = roi_frame

    gray = cv2.cvtColor(corrected_roi, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    enhanced = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8)).apply(blurred)
    edges = cv2.Canny(enhanced, canny_low, canny_high)

    kernel_small = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (close_kernel_size, close_kernel_size))
    if dilate_iters > 0:
        edges = cv2.dilate(edges, kernel_small, iterations=dilate_iters)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel_close, iterations=1)
    return corrected_roi, edges


# ---------------------------------------------------------------------------
# Hough line helpers
# ---------------------------------------------------------------------------

def _line_from_points(px1, py1, px2, py2):
    """Return (a, b, c) with a*x + b*y = c for the infinite line through two points."""
    a = float(py2 - py1)
    b = float(px1 - px2)
    c = a * px1 + b * py1
    return a, b, c


def _intersect_lines(line1, line2):
    """Intersect two lines represented as (a, b, c). Returns (x, y) or None."""
    a1, b1, c1 = line1
    a2, b2, c2 = line2
    det = a1 * b2 - a2 * b1
    if abs(det) < 1e-6:
        return None
    x = (c1 * b2 - c2 * b1) / det
    y = (a1 * c2 - a2 * c1) / det
    return x, y


def _cluster_by_gap(positions, min_gap_ratio, image_dim):
    """Split 1-D positions into two clusters using the largest gap."""
    if len(positions) < 2:
        return None, None

    sorted_idx = np.argsort(positions)
    sorted_pos = positions[sorted_idx]

    if len(sorted_pos) == 2:
        if sorted_pos[1] - sorted_pos[0] < min_gap_ratio * image_dim:
            return None, None
        return [int(sorted_idx[0])], [int(sorted_idx[1])]

    gaps = np.diff(sorted_pos)
    split_i = int(np.argmax(gaps))
    if gaps[split_i] < min_gap_ratio * image_dim:
        return None, None

    return sorted_idx[:split_i + 1].tolist(), sorted_idx[split_i + 1:].tolist()


def _fit_line_to_segments(segments):
    """Fit one infinite line to endpoints of Hough segments using SVD."""
    segs = np.asarray(segments, dtype=np.float64)
    if segs.ndim == 1:
        segs = segs[np.newaxis, :]

    if len(segs) == 1:
        x1, y1, x2, y2 = segs[0]
        return _line_from_points(x1, y1, x2, y2)

    points = np.vstack([segs[:, :2], segs[:, 2:]])
    mean = points.mean(axis=0)
    _, _, vt = np.linalg.svd(points - mean, full_matrices=False)
    dx, dy = vt[0]
    a = -dy
    b = dx
    c = a * mean[0] + b * mean[1]
    return a, b, c


def _line_y_at_x(line, x):
    a, b, c = line
    if abs(b) < 1e-6:
        return None
    return (c - a * x) / b


def _line_x_at_y(line, y):
    a, b, c = line
    if abs(a) < 1e-6:
        return None
    return (c - b * y) / a


def _classify_single_hv_sides(line_h, line_v, w_img, h_img):
    """Classify single horizontal/vertical lines as top/bottom and left/right."""
    center_x = w_img / 2.0
    center_y = h_img / 2.0
    y_h = _line_y_at_x(line_h, center_x)
    x_v = _line_x_at_y(line_v, center_y)
    if y_h is None or x_v is None:
        return None, None
    h_side = "top" if y_h < center_y else "bottom"
    v_side = "left" if x_v < center_x else "right"
    return h_side, v_side


def find_best_mtg_quad(image, edges):
    """Find card quad from dominant Hough lines with 2-line fallback inference."""
    h_img, w_img = image.shape[:2]
    image_area = max(h_img * w_img, 1)

    min_line_length = max(1, int(min(h_img, w_img) * HOUGH_MIN_LINE_LENGTH_RATIO))
    max_line_gap = max(1, int(min(h_img, w_img) * HOUGH_MAX_LINE_GAP_RATIO))

    raw_lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=HOUGH_THRESHOLD,
        minLineLength=min_line_length,
        maxLineGap=max_line_gap,
    )
    if raw_lines is None:
        return None, float("inf"), 0.0, []

    angle_tol = math.radians(HOUGH_ANGLE_TOLERANCE_DEG)
    h_segments, h_positions = [], []
    v_segments, v_positions = [], []

    for entry in raw_lines:
        x1, y1, x2, y2 = entry[0]
        angle = abs(math.atan2(y2 - y1, x2 - x1))

        if angle <= angle_tol or angle >= (math.pi - angle_tol):
            h_segments.append((x1, y1, x2, y2))
            h_positions.append((y1 + y2) / 2.0)
        elif abs(angle - math.pi / 2) <= angle_tol:
            v_segments.append((x1, y1, x2, y2))
            v_positions.append((x1 + x2) / 2.0)

    has_h_pairs = len(h_segments) >= 2
    has_v_pairs = len(v_segments) >= 2
    has_h_any = len(h_segments) >= 1
    has_v_any = len(v_segments) >= 1
    if not has_h_any and not has_v_any:
        return None, float("inf"), 0.0, []

    line_top = None
    line_bottom = None
    line_left = None
    line_right = None
    line_h_single = None
    line_v_single = None

    if has_h_pairs:
        idx_top, idx_bottom = _cluster_by_gap(np.array(h_positions), HOUGH_CLUSTER_MIN_GAP_RATIO, h_img)
        if idx_top is not None:
            seg_arr = np.array(h_segments)
            line_a = _fit_line_to_segments(seg_arr[idx_top])
            line_b = _fit_line_to_segments(seg_arr[idx_bottom])
            y_a = _line_y_at_x(line_a, w_img / 2.0)
            y_b = _line_y_at_x(line_b, w_img / 2.0)
            if y_a is not None and y_b is not None:
                if y_a < y_b:
                    line_top, line_bottom = line_a, line_b
                else:
                    line_top, line_bottom = line_b, line_a
    if has_h_any:
        line_h_single = _fit_line_to_segments(np.array(h_segments))

    if has_v_pairs:
        idx_left, idx_right = _cluster_by_gap(np.array(v_positions), HOUGH_CLUSTER_MIN_GAP_RATIO, w_img)
        if idx_left is not None:
            seg_arr = np.array(v_segments)
            line_a = _fit_line_to_segments(seg_arr[idx_left])
            line_b = _fit_line_to_segments(seg_arr[idx_right])
            x_a = _line_x_at_y(line_a, h_img / 2.0)
            x_b = _line_x_at_y(line_b, h_img / 2.0)
            if x_a is not None and x_b is not None:
                if x_a < x_b:
                    line_left, line_right = line_a, line_b
                else:
                    line_left, line_right = line_b, line_a
    if has_v_any:
        line_v_single = _fit_line_to_segments(np.array(v_segments))

    quad = None
    card_w = 0.0
    card_h = 0.0

    if line_top is not None and line_bottom is not None and line_left is not None and line_right is not None:
        corners = [
            _intersect_lines(line_top, line_left),
            _intersect_lines(line_top, line_right),
            _intersect_lines(line_bottom, line_right),
            _intersect_lines(line_bottom, line_left),
        ]
        if all(pt is not None for pt in corners):
            quad = cam.order_points(np.array(corners, dtype=np.float32))
            w_top = np.linalg.norm(quad[1] - quad[0])
            w_bottom = np.linalg.norm(quad[2] - quad[3])
            h_left = np.linalg.norm(quad[3] - quad[0])
            h_right = np.linalg.norm(quad[2] - quad[1])
            card_w = (w_top + w_bottom) / 2.0
            card_h = (h_left + h_right) / 2.0

    elif line_top is not None and line_bottom is not None:
        center_x = w_img / 2.0
        y_top = _line_y_at_x(line_top, center_x)
        y_bottom = _line_y_at_x(line_bottom, center_x)
        if y_top is not None and y_bottom is not None:
            card_h = abs(y_bottom - y_top)
            card_w = card_h * MTG_ASPECT_RATIO
            half_w = card_w / 2.0
            quad = cam.order_points(np.array([
                (center_x - half_w, y_top),
                (center_x + half_w, y_top),
                (center_x + half_w, y_bottom),
                (center_x - half_w, y_bottom),
            ], dtype=np.float32))

    elif line_left is not None and line_right is not None:
        center_y = h_img / 2.0
        x_left = _line_x_at_y(line_left, center_y)
        x_right = _line_x_at_y(line_right, center_y)
        if x_left is not None and x_right is not None:
            card_w = abs(x_right - x_left)
            card_h = card_w / MTG_ASPECT_RATIO
            half_h = card_h / 2.0
            quad = cam.order_points(np.array([
                (x_left, center_y - half_h),
                (x_right, center_y - half_h),
                (x_right, center_y + half_h),
                (x_left, center_y + half_h),
            ], dtype=np.float32))

    elif line_h_single is not None and line_v_single is not None:
        # Single solid horizontal + single solid vertical line: infer missing edges
        # by anchoring at their intersection and expanding according to aspect ratio.
        corner = _intersect_lines(line_h_single, line_v_single)
        if corner is not None:
            cx, cy = corner
            h_side, v_side = _classify_single_hv_sides(line_h_single, line_v_single, w_img, h_img)
            if h_side is not None and v_side is not None:
                if h_side == "top":
                    avail_h = max(2.0, (h_img - 1) - cy)
                else:
                    avail_h = max(2.0, cy)

                if v_side == "left":
                    avail_w = max(2.0, (w_img - 1) - cx)
                else:
                    avail_w = max(2.0, cx)

                card_h = max(2.0, min(avail_h, avail_w / MTG_ASPECT_RATIO))
                card_w = card_h * MTG_ASPECT_RATIO

                x2 = cx + card_w if v_side == "left" else cx - card_w
                y2 = cy + card_h if h_side == "top" else cy - card_h
                quad = cam.order_points(np.array([
                    (cx, cy),
                    (x2, cy),
                    (x2, y2),
                    (cx, y2),
                ], dtype=np.float32))

    if quad is None or card_w < 1.0 or card_h < 1.0:
        return None, float("inf"), 0.0, []

    aspect = card_w / card_h
    score = abs(aspect - MTG_ASPECT_RATIO) / MTG_ASPECT_RATIO
    box_area_ratio = float((card_w * card_h) / image_area)

    if box_area_ratio < MIN_BOX_AREA_RATIO:
        return None, score, box_area_ratio, []

    if score > MAX_ASPECT_SCORE_TO_COMMIT:
        return None, score, box_area_ratio, [(score, box_area_ratio, quad)]

    return quad, score, box_area_ratio, [(score, box_area_ratio, quad)]


def card_identity_key(name, set_code):
    return ((name or "").strip().lower(), (set_code or "").strip().lower())


def get_regular_hash_best_match(matcher, warped_image):
    card_info, orientation, distance = matcher.source_module.annoy_nearest_phash_orientation(
        warped_image,
        matcher.annoy_index,
        matcher.idx_to_card,
        n=1,
    )
    if card_info is None:
        return None
    return {
        "name": card_info.get("name"),
        "set": card_info.get("set") or card_info.get("set_code") or card_info.get("set_name") or "?",
        "orientation": orientation,
        "distance": float(distance) if distance is not None else None,
    }


def should_accept_match(segmented_best, regular_best):
    if segmented_best is None:
        return False, "no segmented best"

    if float(segmented_best.get("final_score") or 0.0) > 0.0:
        return True, "positive final score"

    if regular_best is not None:
        segmented_key = card_identity_key(segmented_best.get("name"), segmented_best.get("set"))
        regular_key = card_identity_key(regular_best.get("name"), regular_best.get("set"))
        if segmented_key == regular_key:
            return True, "regular/grid best agree"

    return False, "final score <= 0 and regular/grid disagree"


def warp_best_quad(image, quad):
    """Warp the best quad using CamTesting's full perspective/expansion pipeline."""
    perspective_corrected = cam.correct_perspective_quad(quad, top_shrink_ratio=1.0, bottom_grow_ratio=1.0)
    _, leveled_box, center, level_angle = cam.level_quad_for_expansion(perspective_corrected)
    expanded_leveled_box = cam.expand_quad(leveled_box, image.shape)

    height_a = np.linalg.norm(leveled_box[1] - leveled_box[2])
    height_b = np.linalg.norm(leveled_box[0] - leveled_box[3])
    card_height = max(height_a, height_b, 1.0)
    vertical_bias = card_height * cam.EXPANSION_VERTICAL_BIAS_RATIO
    expanded_leveled_box[:, 1] = (expanded_leveled_box[:, 1] + vertical_bias).clip(0, image.shape[0] - 1)
    expanded_leveled_box = cam.order_points(expanded_leveled_box)

    expanded_box = cam.order_points(cam.rotate_points(expanded_leveled_box, level_angle, center))

    max_width = int(max(np.linalg.norm(expanded_box[2] - expanded_box[3]),
                        np.linalg.norm(expanded_box[1] - expanded_box[0])))
    max_height = int(max(np.linalg.norm(expanded_box[1] - expanded_box[2]),
                         np.linalg.norm(expanded_box[0] - expanded_box[3])))
    if max_width < 2 or max_height < 2:
        return None

    dst = cam.module_np_array([[0, 0], [max_width - 1, 0], [max_width - 1, max_height - 1], [0, max_height - 1]])
    matrix = cv2.getPerspectiveTransform(expanded_box, dst)
    return cv2.warpPerspective(image, matrix, (max_width, max_height))


# ---------------------------------------------------------------------------
# Live controls
# ---------------------------------------------------------------------------

def init_live_controls(default_shift_right_ratio, default_shift_down_ratio):
    cv2.namedWindow(CONTROLS_WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(CONTROLS_WINDOW, 520, 250)
    cv2.createTrackbar("Canny Low", CONTROLS_WINDOW, 80, 255, _noop)
    cv2.createTrackbar("Canny High", CONTROLS_WINDOW, 200, 255, _noop)
    cv2.createTrackbar("Close Kernel", CONTROLS_WINDOW, 5, 21, _noop)
    cv2.createTrackbar("Dilate Iters", CONTROLS_WINDOW, 1, 6, _noop)
    cv2.createTrackbar("ROI Shift X %", CONTROLS_WINDOW, int(default_shift_right_ratio * 100), 30, _noop)
    cv2.createTrackbar("ROI Shift Y %", CONTROLS_WINDOW, int(default_shift_down_ratio * 100), 20, _noop)
    cv2.createTrackbar("Perspective Top %", CONTROLS_WINDOW, int(DEFAULT_PERSPECTIVE_TOP_SHRINK * 100), 130, _noop)
    cv2.createTrackbar("Perspective Bot %", CONTROLS_WINDOW, int(DEFAULT_PERSPECTIVE_BOTTOM_GROW * 100), 130, _noop)
    cv2.createTrackbar("Brightness", CONTROLS_WINDOW, int(max(0, min(510, BRIGHTNESS + 255))), 510, _noop)
    cv2.createTrackbar("Contrast x100", CONTROLS_WINDOW, int(max(1, min(1000, CONTRAST * 100))), 1000, _noop)
    cv2.createTrackbar("Glare Fix %", CONTROLS_WINDOW, DEFAULT_GLARE_FIX_STRENGTH, 100, _noop)


def read_live_controls():
    canny_low = cv2.getTrackbarPos("Canny Low", CONTROLS_WINDOW)
    canny_high = cv2.getTrackbarPos("Canny High", CONTROLS_WINDOW)
    if canny_high <= canny_low:
        canny_high = min(255, canny_low + 1)
    close_kernel = _odd_from_slider(cv2.getTrackbarPos("Close Kernel", CONTROLS_WINDOW))
    dilate_iters = max(0, cv2.getTrackbarPos("Dilate Iters", CONTROLS_WINDOW))
    roi_shift_right = cv2.getTrackbarPos("ROI Shift X %", CONTROLS_WINDOW) / 100.0
    roi_shift_down = cv2.getTrackbarPos("ROI Shift Y %", CONTROLS_WINDOW) / 100.0
    persp_top = max(0.70, cv2.getTrackbarPos("Perspective Top %", CONTROLS_WINDOW) / 100.0)
    persp_bot = max(0.70, cv2.getTrackbarPos("Perspective Bot %", CONTROLS_WINDOW) / 100.0)
    brightness = int(cv2.getTrackbarPos("Brightness", CONTROLS_WINDOW) - 255)
    contrast = max(0.01, cv2.getTrackbarPos("Contrast x100", CONTROLS_WINDOW) / 100.0)
    glare_fix_strength = int(cv2.getTrackbarPos("Glare Fix %", CONTROLS_WINDOW))
    return {
        "canny_low": canny_low, "canny_high": canny_high,
        "close_kernel": close_kernel, "dilate_iters": dilate_iters,
        "roi_shift_right_ratio": roi_shift_right, "roi_shift_down_ratio": roi_shift_down,
        "perspective_top_shrink": persp_top, "perspective_bottom_grow": persp_bot,
        "brightness": brightness, "contrast": contrast, "glare_fix_strength": glare_fix_strength,
        "servo_home_angle": SERVO2_HOME_ANGLE, "servo_drop_delta": SERVO2_DROP_DELTA,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="BB - Hough-line MTG card detection.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--roi-shift-right-ratio", type=float, default=DEFAULT_ROI_SHIFT_RIGHT_RATIO)
    parser.add_argument("--roi-shift-down-ratio", type=float, default=DEFAULT_ROI_SHIFT_DOWN_RATIO)
    args = parser.parse_args()

    module = cam.load_source_module()
    cfg = SegmentedHashConfig()
    cfg.stage1_k = min(cfg.stage1_k, AA_STAGE1_K)
    cfg.stage1_max_candidates = min(cfg.stage1_max_candidates, AA_STAGE1_MAX_CANDIDATES)
    cfg.possible_list_size = min(cfg.possible_list_size, AA_POSSIBLE_LIST_SIZE)

    if not SEGMENTED_DB_PATH.exists():
        raise FileNotFoundError(f"Missing segmented DB: {SEGMENTED_DB_PATH}")

    matcher = HybridTopHalfMatcher(
        source_module_path=SOURCE_MODULE_PATH,
        full_card_db_path=FULL_DB_PATH,
        segmented_db_path=SEGMENTED_DB_PATH,
        cfg=cfg,
    )

    controller = module.DryRunController() if args.dry_run else module.StepperController(port="COM8")
    try:
        controller.open()
        controller.servo2_move_abs(SERVO2_HOME_ANGLE)
        time.sleep(0.2)
    except Exception as exc:
        if args.dry_run:
            raise
        print(f"Controller init failed ({exc}); using dry-run.", flush=True)
        controller = module.DryRunController()
        controller.open()

    cap = cam.configure_camera(module)
    print("[BB] Hough-line detector ready. q=quit  r=drop+return", flush=True)

    cv2.namedWindow(cam.WINDOW_LIVE, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.namedWindow(WINDOW_PREPROCESSED, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.namedWindow(cam.WINDOW_EDGES, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.namedWindow(cam.WINDOW_WARPED, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.resizeWindow(cam.WINDOW_LIVE, 1100, 750)
    cv2.resizeWindow(WINDOW_PREPROCESSED, 650, 950)
    cv2.resizeWindow(cam.WINDOW_EDGES, 650, 950)
    cv2.resizeWindow(cam.WINDOW_WARPED, 500, 700)
    init_live_controls(args.roi_shift_right_ratio, args.roi_shift_down_ratio)

    last_identified = "None"
    last_identification_time = 0.0
    last_wait_reason = None
    last_wait_reason_time = 0.0
    interval = 0.6

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                continue

            live_ctrl = read_live_controls()

            frame = normalize_frame_pre_border(
                frame,
                brightness=live_ctrl["brightness"],
                contrast=live_ctrl["contrast"],
            )
            roi_frame, roi_x1, roi_y1, roi_x2, roi_y2 = get_shifted_trimmed_roi(
                module, frame,
                live_ctrl["roi_shift_right_ratio"],
                live_ctrl["roi_shift_down_ratio"],
            )

            live_preview = frame.copy()
            cv2.rectangle(live_preview, (roi_x1, roi_y1), (roi_x2, roi_y2), (0, 255, 255), 3)
            module.draw_text_with_background(live_preview, "BB - Hough-Line MTG Detector", (20, 35), font_scale=0.8, thickness=2)
            module.draw_text_with_background(live_preview, f"Last identified: {last_identified}", (20, 70), font_scale=0.7, thickness=2)
            module.draw_text_with_background(
                live_preview,
                f"Canny={live_ctrl['canny_low']}/{live_ctrl['canny_high']} "
                f"CloseK={live_ctrl['close_kernel']} Dilate={live_ctrl['dilate_iters']} "
                f"ROI X={live_ctrl['roi_shift_right_ratio']:.2f} Y={live_ctrl['roi_shift_down_ratio']:.2f} "
                f"PTop={live_ctrl['perspective_top_shrink']:.2f} PBot={live_ctrl['perspective_bottom_grow']:.2f} "
                f"B={live_ctrl['brightness']} C={live_ctrl['contrast']:.2f} G={live_ctrl['glare_fix_strength']}",
                (20, 100), font_scale=0.50, thickness=2,
            )
            module.draw_text_with_background(
                live_preview,
                "q=quit  r=drop+return  v=servo test",
                (20, frame.shape[0] - 20), font_scale=0.6, thickness=2,
            )

            edge_vis = None
            warped_vis = None
            preprocessed_vis = None
            quad = None
            best_score = float("inf")
            box_area_ratio = 0.0
            reject_reason = None

            try:
                processed, edges = preprocess_for_aa(
                    roi_frame,
                    live_ctrl["canny_low"], live_ctrl["canny_high"],
                    live_ctrl["close_kernel"], live_ctrl["dilate_iters"],
                    live_ctrl["perspective_top_shrink"], live_ctrl["perspective_bottom_grow"],
                    live_ctrl["glare_fix_strength"],
                )
                preprocessed_vis = processed
                edge_vis = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

                quad, best_score, box_area_ratio, all_candidates = find_best_mtg_quad(processed, edges)

                for rank, (_, area_ratio, candidate_quad) in enumerate(all_candidates[:10]):
                    color = (100, 100, 100) if rank > 0 else (0, 255, 0)
                    thickness = 1 if rank > 0 else 2
                    cv2.polylines(edge_vis, [candidate_quad.astype("int32")], True, color, thickness)

                if quad is not None:
                    pts_live = quad.copy()
                    pts_live[:, 0] += roi_x1
                    pts_live[:, 1] += roi_y1
                    cv2.polylines(live_preview, [pts_live.astype("int32")], True, (0, 255, 0), 3)
                    module.draw_text_with_background(
                        live_preview,
                        f"Best box score: {best_score:.3f}  area={box_area_ratio * 100:.1f}%",
                        (20, 130), font_scale=0.55, thickness=2,
                    )

                    warped = warp_best_quad(processed, quad)
                    if warped is not None and warped.size > 0:
                        warped_vis = warped
                else:
                    reject_reason = (
                        f"no box within {MAX_ASPECT_SCORE_TO_COMMIT * 100:.0f}% of MTG ratio "
                        f"(best score={best_score:.3f})"
                        if best_score < float("inf")
                        else "no Hough lines found"
                    )
                    module.draw_text_with_background(
                        live_preview, reject_reason, (20, 130),
                        font_scale=0.50, thickness=2, text_color=(0, 0, 255),
                    )

                if warped_vis is None:
                    warped_vis = processed

                current_time = time.time()
                if current_time - last_identification_time >= interval:
                    if quad is None:
                        wait_reason = reject_reason or (
                            f"no box passed aspect + {MIN_BOX_AREA_RATIO * 100:.0f}% area gates"
                        )
                        if wait_reason != last_wait_reason or (current_time - last_wait_reason_time) >= 2.0:
                            print(f"[BB] waiting: {wait_reason}", flush=True)
                            last_wait_reason = wait_reason
                            last_wait_reason_time = current_time
                    else:
                        regular_best = get_regular_hash_best_match(matcher, warped_vis)
                        if regular_best is None or regular_best.get("distance") is None:
                            wait_reason = "no regular-hash candidate"
                            if wait_reason != last_wait_reason or (current_time - last_wait_reason_time) >= 2.0:
                                print(f"[BB] waiting: {wait_reason}", flush=True)
                                last_wait_reason = wait_reason
                                last_wait_reason_time = current_time
                            continue

                        if regular_best["distance"] > REGULAR_HASH_MAX_DISTANCE:
                            wait_reason = (
                                f"regular hash distance too high: {regular_best['distance']:.2f} > "
                                f"{REGULAR_HASH_MAX_DISTANCE:.2f}"
                            )
                            if wait_reason != last_wait_reason or (current_time - last_wait_reason_time) >= 2.0:
                                print(f"[BB] waiting: {wait_reason}", flush=True)
                                last_wait_reason = wait_reason
                                last_wait_reason_time = current_time
                            continue

                        t0 = time.perf_counter()
                        result = matcher.identify(warped_vis)
                        ident_ms = (time.perf_counter() - t0) * 1000.0
                        last_identification_time = current_time

                        best = result.get("best")
                        accepted, accept_reason = should_accept_match(best, regular_best)
                        if best is not None and accepted:
                            last_identified = best["name"]
                            best_set = best.get("set") or "?"
                            print("[BB] ---------------- MATCH ----------------", flush=True)
                            print(f"[BB] - identify time  : {ident_ms:.0f} ms", flush=True)
                            print(f"[BB] - accept rule    : {accept_reason}", flush=True)
                            print(f"[BB] - matcher status : {result['status']}", flush=True)
                            print(f"[BB] - best match     : {best['name']} [{best_set}]", flush=True)
                            print(f"[BB] - orientation    : {best['orientation']}", flush=True)
                            print(f"[BB] - weighted score : {best['weighted_matches']:.2f}", flush=True)
                            print(f"[BB] - regions matched: {best['matched_regions']}/{best['total_regions']}", flush=True)
                            print(f"[BB] - stage1 dist    : {best['stage1_distance']:.2f}", flush=True)
                            print(f"[BB] - avg region dist: {best['avg_region_distance']:.2f}", flush=True)
                            print(f"[BB] - final score    : {best['final_score']:.2f}", flush=True)
                            if regular_best is not None:
                                print(
                                    f"[BB] - regular hash   : {regular_best['name']} [{regular_best['set']}] "
                                    f"| {regular_best['orientation']} | d={regular_best['distance']:.2f}",
                                    flush=True,
                                )
                            possibles = result.get("possible", [])
                            if possibles:
                                print(f"[BB] - next closest ({min(len(possibles), 10)}):", flush=True)
                                for rank, possible in enumerate(possibles[:10], 1):
                                    possible_set = possible.get("set") or "?"
                                    print(
                                        f"[BB]   {rank}. {possible['name']} [{possible_set}] "
                                        f"w={possible['weighted_matches']:.2f} "
                                        f"f={possible['final_score']:.2f} {possible['orientation']}",
                                        flush=True,
                                    )
                            print("[BB] ------------------------------------------", flush=True)
                        else:
                            last_identified = "None"
                            print("[BB] ---------------- REJECT ------------------", flush=True)
                            print(f"[BB] - identify time: {ident_ms:.0f} ms", flush=True)
                            if best is not None:
                                best_set = best.get("set") or "?"
                                print(f"[BB] - segmented best : {best['name']} [{best_set}]", flush=True)
                                print(f"[BB] - final score    : {best['final_score']:.2f}", flush=True)
                            if regular_best is not None:
                                print(
                                    f"[BB] - regular hash   : {regular_best['name']} [{regular_best['set']}] "
                                    f"| {regular_best['orientation']} | d={regular_best['distance']:.2f}",
                                    flush=True,
                                )
                            print(f"[BB] - reason: {accept_reason if best is not None else result.get('reason')}", flush=True)
                            print("[BB] ------------------------------------------", flush=True)

            except Exception as exc:
                edge_vis = cv2.cvtColor(
                    cv2.Canny(cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY), 80, 200),
                    cv2.COLOR_GRAY2BGR,
                )
                warped_vis = roi_frame
                preprocessed_vis = roi_frame
                module.draw_text_with_background(
                    live_preview, f"Error: {exc}", (20, 105),
                    font_scale=0.6, thickness=2, text_color=(0, 0, 255),
                )

            cv2.imshow(cam.WINDOW_LIVE, live_preview)
            if preprocessed_vis is not None and preprocessed_vis.size > 0:
                cv2.imshow(WINDOW_PREPROCESSED, preprocessed_vis)
            if edge_vis is not None:
                cv2.imshow(cam.WINDOW_EDGES, edge_vis)
            if warped_vis is not None and warped_vis.size > 0:
                cv2.imshow(cam.WINDOW_WARPED, warped_vis)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                break
            if key == ord("v"):
                cam.run_servo2_test(controller)
            if key == ord("r"):
                home_angle = live_ctrl["servo_home_angle"]
                drop_delta = live_ctrl["servo_drop_delta"]
                down_angle = home_angle - drop_delta
                print(f"[BB] Servo drop: {home_angle} -> {down_angle}, wait 0.5s, return.", flush=True)
                controller.servo2_move_abs(down_angle)
                time.sleep(0.5)
                controller.servo2_move_abs(home_angle)
                print("[BB] Servo returned.", flush=True)

    finally:
        cap.release()
        controller.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
