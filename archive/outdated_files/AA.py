"""AA.py — MTG card detector that scores ALL candidate bounding boxes by aspect ratio
and always picks the one closest to the MTG card dimensions, then warps and identifies it.

Strategy:
  1. Build an edge map from the preprocessed ROI.
  2. Find every contour above a minimum area.
  3. Score each by how closely the minAreaRect aspect ratio matches 58:78 (portrait MTG).
  4. Commit to the single best-scoring candidate every frame — no hard rejection gates.
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
CONTROLS_WINDOW = "AA Controls"
WINDOW_PREPROCESSED = "AA Preprocessed"
DEFAULT_PERSPECTIVE_TOP_SHRINK = 0.95
DEFAULT_PERSPECTIVE_BOTTOM_GROW = 0.95
SERVO2_HOME_ANGLE = 140
SERVO2_DROP_DELTA = 140
ENABLE_PRE_BORDER_PHOTO_NORMALIZATION = True

# MTG card portrait aspect ratio (width / height) = 58 / 88 mm ≈ 0.659
# Using the common playing-card dimension that CamTesting targets.
MTG_ASPECT_RATIO = 58 / 78          # ≈ 0.744  (width / height, portrait)
MTG_ASPECT_TOLERANCE = 0.04         # Wider tolerance so slight angles still score well

# Contour size gate: candidate must cover this fraction of ROI area to be considered.
MIN_CONTOUR_AREA_RATIO = 0.04       # 4 % of ROI
# Aspect-score ceiling: candidates worse than this are ignored even if nothing better exists.
MAX_ASPECT_SCORE_TO_COMMIT = 0.25   # score = abs(aspect - target) / target; 0.25 = 25 % off
MIN_BOX_AREA_RATIO = 0.60

# AA-specific speed knobs. These intentionally narrow the expensive segmented pass.
AA_STAGE1_K = 250
AA_STAGE1_MAX_CANDIDATES = 80
AA_POSSIBLE_LIST_SIZE = 5
REGULAR_HASH_MAX_DISTANCE = 24.0

# Directional ROI expansion (fraction of ROI width / height).
ROI_EXPAND_LEFT_RATIO  = 0.04
ROI_EXPAND_RIGHT_RATIO = 0.02
ROI_EXPAND_UP_RATIO    = 0.01
ROI_EXPAND_DOWN_RATIO  = 0.0075

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
    roi_width  = max(roi_x2 - roi_x1, 1)
    roi_height = max(roi_y2 - roi_y1, 1)
    shift_px = int(roi_width  * max(0.0, shift_right_ratio))
    shift_py = int(roi_height * shift_down_ratio)

    frame_h, frame_w = frame.shape[:2]
    shifted_x1 = min(max(roi_x1 + shift_px, 0), max(frame_w - 2, 0))
    shifted_x2 = min(max(roi_x2 + shift_px, shifted_x1 + 1), frame_w)

    extra_left  = int(roi_width  * max(0.0, ROI_EXPAND_LEFT_RATIO))
    extra_right = int(roi_width  * max(0.0, ROI_EXPAND_RIGHT_RATIO))
    extra_up    = int(roi_height * max(0.0, ROI_EXPAND_UP_RATIO))
    extra_down  = int(roi_height * max(0.0, ROI_EXPAND_DOWN_RATIO))
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
    top_ratio    = min(max(float(top_shrink_ratio),  0.7), 1.0)
    bottom_ratio = min(max(float(bottom_grow_ratio), 0.7), 1.0)
    top_margin    = 0.5 * w * (1.0 - top_ratio)
    bottom_margin = 0.5 * w * (1.0 - bottom_ratio)
    src = cam.module_np_array([
        [top_margin,         0],
        [w - 1 - top_margin, 0],
        [w - 1 - bottom_margin, h - 1],
        [bottom_margin,         h - 1],
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

    gray     = cv2.cvtColor(corrected_roi, cv2.COLOR_BGR2GRAY)
    blurred  = cv2.GaussianBlur(gray, (5, 5), 0)
    enhanced = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8)).apply(blurred)
    edges    = cv2.Canny(enhanced, canny_low, canny_high)

    kernel_small = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (close_kernel_size, close_kernel_size))
    if dilate_iters > 0:
        edges = cv2.dilate(edges, kernel_small, iterations=dilate_iters)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel_close, iterations=1)
    return corrected_roi, edges


def find_best_mtg_quad(image, edges):
    """Score every contour by aspect-ratio closeness to MTG portrait dimensions.

    Returns (best_quad, best_score, scored_candidates) where scored_candidates is a
    list of (score, quad) sorted best-first, and best_quad is None if nothing passes
    the MAX_ASPECT_SCORE_TO_COMMIT ceiling.
    """
    image_area = max(image.shape[0] * image.shape[1], 1)
    min_area   = image_area * MIN_CONTOUR_AREA_RATIO

    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, float("inf"), []

    candidates = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue

        rect = cv2.minAreaRect(contour)
        w_box, h_box = rect[1]
        if w_box < 1 or h_box < 1:
            continue

        # Always express as portrait (shorter / longer).
        short_side = min(w_box, h_box)
        long_side  = max(w_box, h_box)
        aspect = short_side / long_side           # will be <= 1.0 (portrait-ish)
        box_area_ratio = float((w_box * h_box) / image_area)
        if box_area_ratio < MIN_BOX_AREA_RATIO:
            continue

        score = abs(aspect - MTG_ASPECT_RATIO) / MTG_ASPECT_RATIO
        quad  = cam.order_points(cv2.boxPoints(rect))
        candidates.append((score, area, box_area_ratio, quad))

    if not candidates:
        return None, float("inf"), 0.0, []

    # Sort: primary by aspect score, secondary by size (larger is better for ties).
    candidates.sort(key=lambda x: (x[0], -x[1]))
    best_score, _, best_area_ratio, best_quad = candidates[0]

    if best_score > MAX_ASPECT_SCORE_TO_COMMIT:
        return None, best_score, best_area_ratio, [(s, a, q) for s, _, a, q in candidates]

    return best_quad, best_score, best_area_ratio, [(s, a, q) for s, _, a, q in candidates]


def quad_area_ratio(quad, image_shape):
    image_area = float(max(image_shape[0] * image_shape[1], 1))
    quad_area = abs(cv2.contourArea(quad.astype("float32")))
    return quad_area / image_area


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

    max_width  = int(max(np.linalg.norm(expanded_box[2] - expanded_box[3]),
                         np.linalg.norm(expanded_box[1] - expanded_box[0])))
    max_height = int(max(np.linalg.norm(expanded_box[1] - expanded_box[2]),
                         np.linalg.norm(expanded_box[0] - expanded_box[3])))
    if max_width < 2 or max_height < 2:
        return None

    dst = cam.module_np_array([[0, 0], [max_width-1, 0], [max_width-1, max_height-1], [0, max_height-1]])
    matrix = cv2.getPerspectiveTransform(expanded_box, dst)
    return cv2.warpPerspective(image, matrix, (max_width, max_height))


# ---------------------------------------------------------------------------
# Live controls
# ---------------------------------------------------------------------------

def init_live_controls(default_shift_right_ratio, default_shift_down_ratio):
    cv2.namedWindow(CONTROLS_WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(CONTROLS_WINDOW, 520, 250)
    cv2.createTrackbar("Canny Low",          CONTROLS_WINDOW, 80,  255, _noop)
    cv2.createTrackbar("Canny High",         CONTROLS_WINDOW, 200, 255, _noop)
    cv2.createTrackbar("Close Kernel",       CONTROLS_WINDOW, 5,   21,  _noop)
    cv2.createTrackbar("Dilate Iters",       CONTROLS_WINDOW, 1,   6,   _noop)
    cv2.createTrackbar("ROI Shift X %",      CONTROLS_WINDOW, int(default_shift_right_ratio * 100), 30,  _noop)
    cv2.createTrackbar("ROI Shift Y %",      CONTROLS_WINDOW, int(default_shift_down_ratio  * 100), 20,  _noop)
    cv2.createTrackbar("Perspective Top %",  CONTROLS_WINDOW, int(DEFAULT_PERSPECTIVE_TOP_SHRINK   * 100), 130, _noop)
    cv2.createTrackbar("Perspective Bot %",  CONTROLS_WINDOW, int(DEFAULT_PERSPECTIVE_BOTTOM_GROW  * 100), 130, _noop)
    cv2.createTrackbar("Brightness",         CONTROLS_WINDOW, int(max(0, min(510, BRIGHTNESS + 255))), 510, _noop)
    cv2.createTrackbar("Contrast x100",      CONTROLS_WINDOW, int(max(1, min(1000, CONTRAST * 100))), 1000, _noop)
    cv2.createTrackbar("Glare Fix %",        CONTROLS_WINDOW, DEFAULT_GLARE_FIX_STRENGTH, 100, _noop)


def read_live_controls():
    canny_low  = cv2.getTrackbarPos("Canny Low",  CONTROLS_WINDOW)
    canny_high = cv2.getTrackbarPos("Canny High", CONTROLS_WINDOW)
    if canny_high <= canny_low:
        canny_high = min(255, canny_low + 1)
    close_kernel = _odd_from_slider(cv2.getTrackbarPos("Close Kernel", CONTROLS_WINDOW))
    dilate_iters = max(0, cv2.getTrackbarPos("Dilate Iters", CONTROLS_WINDOW))
    roi_shift_right = cv2.getTrackbarPos("ROI Shift X %", CONTROLS_WINDOW) / 100.0
    roi_shift_down  = cv2.getTrackbarPos("ROI Shift Y %", CONTROLS_WINDOW) / 100.0
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
    parser = argparse.ArgumentParser(description="AA — aspect-ratio-scored MTG card detection.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--roi-shift-right-ratio", type=float, default=DEFAULT_ROI_SHIFT_RIGHT_RATIO)
    parser.add_argument("--roi-shift-down-ratio",  type=float, default=DEFAULT_ROI_SHIFT_DOWN_RATIO)
    args = parser.parse_args()

    module = cam.load_source_module()
    cfg    = SegmentedHashConfig()
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
    print("[AA] Best-aspect detector ready. q=quit  r=drop+return", flush=True)

    cv2.namedWindow(cam.WINDOW_LIVE,  cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.namedWindow(WINDOW_PREPROCESSED, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.namedWindow(cam.WINDOW_EDGES, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.namedWindow(cam.WINDOW_WARPED, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.resizeWindow(cam.WINDOW_LIVE,  1100, 750)
    cv2.resizeWindow(WINDOW_PREPROCESSED, 650, 950)
    cv2.resizeWindow(cam.WINDOW_EDGES, 650, 950)
    cv2.resizeWindow(cam.WINDOW_WARPED, 500, 700)
    init_live_controls(args.roi_shift_right_ratio, args.roi_shift_down_ratio)

    last_identified      = "None"
    last_identification_time = 0.0
    last_wait_reason         = None
    last_wait_reason_time    = 0.0
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
            module.draw_text_with_background(live_preview, "AA — Best-Aspect MTG Detector", (20, 35), font_scale=0.8, thickness=2)
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

            edge_vis       = None
            warped_vis     = None
            preprocessed_vis = None
            quad           = None
            best_score     = float("inf")
            box_area_ratio = 0.0
            reject_reason  = None
            box_area_ratio = 0.0

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

                # Draw every candidate box (grey), then the winner (green).
                for rank, (score, area_ratio, cq) in enumerate(all_candidates[:10]):
                    color = (100, 100, 100) if rank > 0 else (0, 255, 0)
                    thickness = 1 if rank > 0 else 2
                    cv2.polylines(edge_vis, [cq.astype("int32")], True, color, thickness)

                if quad is not None:
                    # Draw the winner on live preview too (offset by ROI origin).
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
                        f"no box within {MAX_ASPECT_SCORE_TO_COMMIT*100:.0f}% of MTG ratio "
                        f"(best score={best_score:.3f})"
                        if best_score < float("inf")
                        else "no contours found"
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
                            print(f"[AA] waiting: {wait_reason}", flush=True)
                            last_wait_reason = wait_reason
                            last_wait_reason_time = current_time
                    else:
                        regular_best = get_regular_hash_best_match(matcher, warped_vis)
                        if regular_best is None or regular_best.get("distance") is None:
                            wait_reason = "no regular-hash candidate"
                            if wait_reason != last_wait_reason or (current_time - last_wait_reason_time) >= 2.0:
                                print(f"[AA] waiting: {wait_reason}", flush=True)
                                last_wait_reason = wait_reason
                                last_wait_reason_time = current_time
                            continue

                        if regular_best["distance"] > REGULAR_HASH_MAX_DISTANCE:
                            wait_reason = (
                                f"regular hash distance too high: {regular_best['distance']:.2f} > "
                                f"{REGULAR_HASH_MAX_DISTANCE:.2f}"
                            )
                            if wait_reason != last_wait_reason or (current_time - last_wait_reason_time) >= 2.0:
                                print(f"[AA] waiting: {wait_reason}", flush=True)
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
                            print("[AA] ---------------- MATCH ----------------", flush=True)
                            print(f"[AA] - identify time  : {ident_ms:.0f} ms", flush=True)
                            print(f"[AA] - accept rule    : {accept_reason}", flush=True)
                            print(f"[AA] - matcher status : {result['status']}", flush=True)
                            print(f"[AA] - best match     : {best['name']} [{best_set}]", flush=True)
                            print(f"[AA] - orientation    : {best['orientation']}", flush=True)
                            print(f"[AA] - weighted score : {best['weighted_matches']:.2f}", flush=True)
                            print(f"[AA] - regions matched: {best['matched_regions']}/{best['total_regions']}", flush=True)
                            print(f"[AA] - stage1 dist    : {best['stage1_distance']:.2f}", flush=True)
                            print(f"[AA] - avg region dist: {best['avg_region_distance']:.2f}", flush=True)
                            print(f"[AA] - final score    : {best['final_score']:.2f}", flush=True)
                            if regular_best is not None:
                                print(
                                    f"[AA] - regular hash   : {regular_best['name']} [{regular_best['set']}] "
                                    f"| {regular_best['orientation']} | d={regular_best['distance']:.2f}",
                                    flush=True,
                                )
                            possibles = result.get("possible", [])
                            if possibles:
                                print(f"[AA] - next closest ({min(len(possibles), 10)}):", flush=True)
                                for rank, p in enumerate(possibles[:10], 1):
                                    p_set = p.get("set") or "?"
                                    print(
                                        f"[AA]   {rank}. {p['name']} [{p_set}] "
                                        f"w={p['weighted_matches']:.2f} f={p['final_score']:.2f} {p['orientation']}",
                                        flush=True,
                                    )
                            print("[AA] ------------------------------------------", flush=True)
                        else:
                            last_identified = "None"
                            print("[AA] ---------------- REJECT ------------------", flush=True)
                            print(f"[AA] - identify time: {ident_ms:.0f} ms", flush=True)
                            if best is not None:
                                best_set = best.get("set") or "?"
                                print(f"[AA] - segmented best : {best['name']} [{best_set}]", flush=True)
                                print(f"[AA] - final score    : {best['final_score']:.2f}", flush=True)
                            if regular_best is not None:
                                print(
                                    f"[AA] - regular hash   : {regular_best['name']} [{regular_best['set']}] "
                                    f"| {regular_best['orientation']} | d={regular_best['distance']:.2f}",
                                    flush=True,
                                )
                            print(f"[AA] - reason: {accept_reason if best is not None else result.get('reason')}", flush=True)
                            print("[AA] ------------------------------------------", flush=True)

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
                print(f"[AA] Servo drop: {home_angle} -> {down_angle}, wait 0.5s, return.", flush=True)
                controller.servo2_move_abs(down_angle)
                time.sleep(0.5)
                controller.servo2_move_abs(home_angle)
                print("[AA] Servo returned.", flush=True)

    finally:
        cap.release()
        controller.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
