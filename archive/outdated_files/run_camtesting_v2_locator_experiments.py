import argparse
import math
import time
from pathlib import Path

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
CONTROLS_WINDOW = "CamTesting v2 Controls"
WINDOW_PREPROCESSED = "CamTesting v2 Preprocessed"
WINDOW_MASK = "CamTesting v2 Mask/Detector"
DEFAULT_PERSPECTIVE_TOP_SHRINK = 0.95
DEFAULT_PERSPECTIVE_BOTTOM_GROW = 0.95
SERVO2_HOME_ANGLE = 90
SERVO2_DROP_DELTA = 50
STRICT_MTG_ASPECT_RATIO = 58 / 78
STRICT_MTG_ASPECT_TOLERANCE = 0.03
CORNER_RIGHT_ANGLE_TOLERANCE_DEG = 18.0
ENABLE_PRE_BORDER_PHOTO_NORMALIZATION = True
ROI_EXTRA_WIDTH_SIDE_RATIO = 0.020
ROI_EXTRA_UP_RATIO = 0.04
MTG_HEIGHT_OVER_WIDTH = 88.0 / 63.0
MAX_CONTOUR_AREA_RATIO = 0.92
CONTOUR_BORDER_MARGIN_PX = 4


def _noop(_value):
    return None


def _odd_from_slider(value: int, min_odd: int = 3) -> int:
    value = max(min_odd, int(value))
    if value % 2 == 0:
        value += 1
    return value


def normalize_frame_pre_border(frame):
    corrected = cam.apply_lens_distortion_correction(frame)
    if not ENABLE_PRE_BORDER_PHOTO_NORMALIZATION:
        return corrected

    lab = cv2.cvtColor(corrected, cv2.COLOR_BGR2LAB)
    l_chan, a_chan, b_chan = cv2.split(lab)
    l_chan = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l_chan)
    normalized = cv2.merge((l_chan, a_chan, b_chan))
    corrected = cv2.cvtColor(normalized, cv2.COLOR_LAB2BGR)
    corrected = cv2.bilateralFilter(corrected, d=5, sigmaColor=35, sigmaSpace=35)
    return corrected


def init_live_controls(default_shift_right_ratio: float, default_shift_down_ratio: float):
    cv2.namedWindow(CONTROLS_WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(CONTROLS_WINDOW, 560, 320)

    cv2.createTrackbar("Canny Low", CONTROLS_WINDOW, 80, 255, _noop)
    cv2.createTrackbar("Canny High", CONTROLS_WINDOW, 200, 255, _noop)
    cv2.createTrackbar("Close Kernel", CONTROLS_WINDOW, 5, 21, _noop)
    cv2.createTrackbar("Dilate Iters", CONTROLS_WINDOW, 1, 6, _noop)
    cv2.createTrackbar("ROI Shift X %", CONTROLS_WINDOW, int(default_shift_right_ratio * 100), 30, _noop)
    cv2.createTrackbar("ROI Shift Y %", CONTROLS_WINDOW, int(default_shift_down_ratio * 100), 20, _noop)
    cv2.createTrackbar("Perspective Top %", CONTROLS_WINDOW, int(DEFAULT_PERSPECTIVE_TOP_SHRINK * 100), 130, _noop)
    cv2.createTrackbar("Perspective Bottom %", CONTROLS_WINDOW, int(DEFAULT_PERSPECTIVE_BOTTOM_GROW * 100), 130, _noop)
    cv2.createTrackbar("Servo Home", CONTROLS_WINDOW, SERVO2_HOME_ANGLE, 180, _noop)
    cv2.createTrackbar("Servo Drop", CONTROLS_WINDOW, SERVO2_DROP_DELTA, 120, _noop)
    cv2.createTrackbar("Mask Blur", CONTROLS_WINDOW, 21, 61, _noop)
    cv2.createTrackbar("Mask Close", CONTROLS_WINDOW, 15, 41, _noop)
    cv2.createTrackbar("Mask Open", CONTROLS_WINDOW, 7, 31, _noop)
    cv2.createTrackbar("Mask MinArea %", CONTROLS_WINDOW, 3, 20, _noop)


def read_live_controls():
    canny_low = cv2.getTrackbarPos("Canny Low", CONTROLS_WINDOW)
    canny_high = cv2.getTrackbarPos("Canny High", CONTROLS_WINDOW)
    if canny_high <= canny_low:
        canny_high = min(255, canny_low + 1)

    return {
        "canny_low": canny_low,
        "canny_high": canny_high,
        "close_kernel": _odd_from_slider(cv2.getTrackbarPos("Close Kernel", CONTROLS_WINDOW), 3),
        "dilate_iters": max(0, int(cv2.getTrackbarPos("Dilate Iters", CONTROLS_WINDOW))),
        "roi_shift_right_ratio": cv2.getTrackbarPos("ROI Shift X %", CONTROLS_WINDOW) / 100.0,
        "roi_shift_down_ratio": cv2.getTrackbarPos("ROI Shift Y %", CONTROLS_WINDOW) / 100.0,
        "perspective_top_shrink": max(0.70, cv2.getTrackbarPos("Perspective Top %", CONTROLS_WINDOW) / 100.0),
        "perspective_bottom_grow": max(0.70, cv2.getTrackbarPos("Perspective Bottom %", CONTROLS_WINDOW) / 100.0),
        "servo_home_angle": int(cv2.getTrackbarPos("Servo Home", CONTROLS_WINDOW)),
        "servo_drop_delta": int(cv2.getTrackbarPos("Servo Drop", CONTROLS_WINDOW)),
        "mask_blur": _odd_from_slider(cv2.getTrackbarPos("Mask Blur", CONTROLS_WINDOW), 5),
        "mask_close": _odd_from_slider(cv2.getTrackbarPos("Mask Close", CONTROLS_WINDOW), 3),
        "mask_open": _odd_from_slider(cv2.getTrackbarPos("Mask Open", CONTROLS_WINDOW), 1),
        "mask_min_area_ratio": max(0.01, cv2.getTrackbarPos("Mask MinArea %", CONTROLS_WINDOW) / 100.0),
    }


def get_shifted_trimmed_roi(module, frame, shift_right_ratio: float, shift_down_ratio: float):
    _, roi_x1, roi_y1, roi_x2, roi_y2 = module.get_center_roi(frame)
    roi_width = max(roi_x2 - roi_x1, 1)
    roi_height = max(roi_y2 - roi_y1, 1)
    shift_px = int(roi_width * max(0.0, shift_right_ratio))
    shift_py = int(roi_height * shift_down_ratio)

    _frame_h, frame_w = frame.shape[:2]
    shifted_x1 = min(max(roi_x1 + shift_px, 0), max(frame_w - 2, 0))
    shifted_x2 = min(max(roi_x2 + shift_px, shifted_x1 + 1), frame_w)

    shifted_y1 = max(0, roi_y1 + shift_py)
    shifted_y2 = max(shifted_y1 + 2, roi_y2 + shift_py)
    shifted_y1 = min(shifted_y1, frame.shape[0] - 2)
    shifted_y2 = min(max(shifted_y2, shifted_y1 + 1), frame.shape[0])

    roi_height = max(shifted_y2 - shifted_y1, 1)
    trim_pixels = max(1, int(roi_height * cam.ROI_VERTICAL_TRIM_RATIO))
    trimmed_y1 = min(shifted_y2 - 1, shifted_y1 + trim_pixels)
    trimmed_y2 = max(trimmed_y1 + 1, shifted_y2 - trim_pixels)

    # Expand ROI: +1.5% on left and right, +1% upward.
    trimmed_w = max(shifted_x2 - shifted_x1, 1)
    trimmed_h = max(trimmed_y2 - trimmed_y1, 1)
    extra_x = max(1, int(trimmed_w * ROI_EXTRA_WIDTH_SIDE_RATIO))
    extra_up = max(1, int(trimmed_h * ROI_EXTRA_UP_RATIO))

    shifted_x1 = max(0, shifted_x1 - extra_x)
    shifted_x2 = min(frame_w, shifted_x2 + extra_x)
    trimmed_y1 = max(0, trimmed_y1 - extra_up)

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

    src = cam.module_np_array(
        [
            [top_margin, 0],
            [w - 1 - top_margin, 0],
            [w - 1 - bottom_margin, h - 1],
            [bottom_margin, h - 1],
        ]
    )
    dst = cam.module_np_array(
        [
            [0, 0],
            [w - 1, 0],
            [w - 1, h - 1],
            [0, h - 1],
        ]
    )
    matrix = cv2.getPerspectiveTransform(src, dst)
    corrected = cv2.warpPerspective(roi_frame, matrix, (w, h))
    if corrected.shape[:2] != roi_frame.shape[:2]:
        raise RuntimeError(f"Perspective changed ROI shape from {roi_frame.shape[:2]} to {corrected.shape[:2]}")
    return corrected


def preprocess_for_detection(roi_frame, live_ctrl):
    corrected_roi = apply_pre_edge_perspective(
        roi_frame,
        top_shrink_ratio=live_ctrl["perspective_top_shrink"],
        bottom_grow_ratio=live_ctrl["perspective_bottom_grow"],
    )
    gray = cv2.cvtColor(corrected_roi, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    enhanced = clahe.apply(blurred)
    return corrected_roi, enhanced


def build_edge_mask(enhanced_gray, live_ctrl):
    edges = cv2.Canny(enhanced_gray, live_ctrl["canny_low"], live_ctrl["canny_high"])
    kernel_small = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (live_ctrl["close_kernel"], live_ctrl["close_kernel"]))
    if live_ctrl["dilate_iters"] > 0:
        edges = cv2.dilate(edges, kernel_small, iterations=live_ctrl["dilate_iters"])
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel_close, iterations=1)
    return edges


def contour_to_quad(contour):
    perimeter = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
    if len(approx) == 4:
        return cam.order_points(approx.reshape(4, 2))
    rect = cv2.minAreaRect(contour)
    return cam.order_points(cv2.boxPoints(rect))


def contour_touches_border(contour, image_shape, margin=CONTOUR_BORDER_MARGIN_PX):
    h, w = image_shape[:2]
    x, y, cw, ch = cv2.boundingRect(contour)
    if x <= margin or y <= margin:
        return True
    if (x + cw) >= (w - margin) or (y + ch) >= (h - margin):
        return True
    return False


def find_quad_from_binary_mask(image_bgr, binary_mask, min_area_ratio):
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    image_area = max(image_bgr.shape[0] * image_bgr.shape[1], 1)
    min_area = image_area * min_area_ratio
    max_area = image_area * MAX_CONTOUR_AREA_RATIO
    for contour in sorted(contours, key=cv2.contourArea, reverse=True):
        area = cv2.contourArea(contour)
        if area < min_area or area > max_area:
            continue
        if contour_touches_border(contour, image_bgr.shape):
            continue
        return contour_to_quad(contour)
    return None


def reconstruct_quad_from_top_edge(contour, image_shape):
    """Infer full card quad from top edge plus known MTG ratio (portrait)."""
    h, w = image_shape[:2]
    box = contour_to_quad(contour).astype("float32")

    top_left, top_right = box[0], box[1]
    top_vec = top_right - top_left
    top_width = float(np.linalg.norm(top_vec))
    if top_width < 12:
        return None

    # Perpendicular direction from top edge; choose the downward image direction.
    n = np.array([-top_vec[1], top_vec[0]], dtype="float32")
    n_norm = float(np.linalg.norm(n))
    if n_norm < 1e-6:
        return None
    n /= n_norm
    if n[1] < 0:
        n = -n

    target_h = top_width * MTG_HEIGHT_OVER_WIDTH
    bottom_left = top_left + (n * target_h)
    bottom_right = top_right + (n * target_h)

    quad = np.array([top_left, top_right, bottom_right, bottom_left], dtype="float32")
    quad[:, 0] = np.clip(quad[:, 0], 0, w - 1)
    quad[:, 1] = np.clip(quad[:, 1], 0, h - 1)
    return cam.order_points(quad)


def detector_mask_normalized(processed_bgr, enhanced_gray, live_ctrl):
    blur_size = live_ctrl["mask_blur"]
    heavy = cv2.GaussianBlur(enhanced_gray, (blur_size, blur_size), 0)
    background = cv2.GaussianBlur(heavy, (max(blur_size * 2 - 1, 5), max(blur_size * 2 - 1, 5)), 0)
    normalized = cv2.divide(heavy, background, scale=255)
    _thr, mask = cv2.threshold(normalized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (live_ctrl["mask_close"], live_ctrl["mask_close"]))
    kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (live_ctrl["mask_open"], live_ctrl["mask_open"]))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open, iterations=1)

    quad = find_quad_from_binary_mask(processed_bgr, mask, live_ctrl["mask_min_area_ratio"])
    debug = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    if quad is not None:
        cv2.polylines(debug, [quad.astype("int32")], True, (0, 255, 0), 2)
    return quad, debug, "mask-normalized"


def detector_mask_adaptive(processed_bgr, enhanced_gray, live_ctrl):
    blur_size = live_ctrl["mask_blur"]
    heavy = cv2.GaussianBlur(enhanced_gray, (blur_size, blur_size), 0)
    mask = cv2.adaptiveThreshold(
        heavy,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        51,
        2,
    )

    # Add a mild edge hint so border fragments survive thresholding under shadows.
    edge_hint = cv2.Canny(
        heavy,
        max(8, int(live_ctrl["canny_low"] * 0.75)),
        max(16, int(live_ctrl["canny_high"] * 0.90)),
    )
    edge_hint = cv2.dilate(edge_hint, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=1)
    mask = cv2.bitwise_or(mask, edge_hint)

    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (live_ctrl["mask_close"], live_ctrl["mask_close"]))
    adaptive_open = max(1, live_ctrl["mask_open"] - 2)
    kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (adaptive_open, adaptive_open))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    image_area = max(processed_bgr.shape[0] * processed_bgr.shape[1], 1)
    min_area = image_area * live_ctrl["mask_min_area_ratio"]
    max_area = image_area * MAX_CONTOUR_AREA_RATIO

    primary_contour = None
    for contour in sorted(contours, key=cv2.contourArea, reverse=True):
        area = cv2.contourArea(contour)
        if area < min_area or area > max_area:
            continue
        if contour_touches_border(contour, processed_bgr.shape):
            continue
        primary_contour = contour
        break

    quad = None
    if primary_contour is not None:
        # Top-edge strategy only: reconstruct full card from the top border.
        quad = reconstruct_quad_from_top_edge(primary_contour, processed_bgr.shape)

    debug = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    # Show more candidate boxes (black) for visual tuning of edge richness.
    min_box_area = image_area * max(0.001, live_ctrl["mask_min_area_ratio"] * 0.25)
    drawn = 0
    for contour in sorted(contours, key=cv2.contourArea, reverse=True):
        area = cv2.contourArea(contour)
        if area < min_box_area or area > (image_area * MAX_CONTOUR_AREA_RATIO):
            continue
        if contour_touches_border(contour, processed_bgr.shape):
            continue
        box = contour_to_quad(contour)
        cv2.polylines(debug, [box.astype("int32")], True, (0, 0, 0), 1)
        drawn += 1
        if drawn >= 12:
            break

    if primary_contour is not None:
        top_box = contour_to_quad(primary_contour)
        cv2.line(
            debug,
            tuple(top_box[0].astype("int32")),
            tuple(top_box[1].astype("int32")),
            (255, 255, 0),
            2,
        )

    if quad is not None:
        cv2.polylines(debug, [quad.astype("int32")], True, (255, 0, 255), 2)
    return quad, debug, "mask-adaptive"


def detector_edges(processed_bgr, enhanced_gray, live_ctrl):
    edges = build_edge_mask(enhanced_gray, live_ctrl)
    debug = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    quad = None
    try:
        debug, quad, _ = cam.detect_card_with_inference(processed_bgr, edges)
    except Exception:
        quad = find_quad_from_binary_mask(processed_bgr, edges, max(0.02, live_ctrl["mask_min_area_ratio"]))
        if quad is not None:
            cv2.polylines(debug, [quad.astype("int32")], True, (0, 255, 255), 2)
    return quad, debug, "edges"


def detector_canny_top_edge(processed_bgr, enhanced_gray, live_ctrl):
    """Detect top boundary as white-to-black transition in Canny map and extrapolate card."""
    edges = build_edge_mask(enhanced_gray, live_ctrl)
    debug = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    h, w = processed_bgr.shape[:2]
    # Build row-wise edge energy profile: white region then black region transition.
    row_energy = edges.mean(axis=1).astype(np.float32)
    row_energy = cv2.GaussianBlur(row_energy.reshape(-1, 1), (1, 15), 0).reshape(-1)
    if len(row_energy) < 4:
        return None, debug, "canny-top-edge"

    # Find strongest positive drop (white -> black) in upper portion of image.
    row_drop = row_energy[:-1] - row_energy[1:]
    search_start = max(2, int(h * 0.05))
    search_end = max(search_start + 2, int(h * 0.72))
    if search_end > len(row_drop):
        search_end = len(row_drop)
    if search_start >= search_end:
        return None, debug, "canny-top-edge"

    best_rel = int(np.argmax(row_drop[search_start:search_end]))
    best_y = search_start + best_rel + 1

    # Confidence check: avoid weak transitions.
    local_drop = float(row_drop[best_y - 1])
    if local_drop < 2.0:
        return None, debug, "canny-top-edge"

    # Estimate top-edge x-span from local edge density around the boundary row.
    y0 = max(0, best_y - 4)
    y1 = min(h, best_y + 5)
    band = edges[y0:y1, :]
    if band.size == 0:
        return None, debug, "canny-top-edge"
    col_energy = band.mean(axis=0).astype(np.float32)
    col_thresh = max(5.0, float(col_energy.max()) * 0.18)
    xs = np.where(col_energy >= col_thresh)[0]

    if xs.size >= 2:
        x_left = int(xs[0])
        x_right = int(xs[-1])
    else:
        x_left = int(w * 0.12)
        x_right = int(w * 0.88)

    if x_right - x_left < int(w * 0.25):
        x_left = int(w * 0.12)
        x_right = int(w * 0.88)

    p1 = np.array([x_left, best_y], dtype="float32")
    p2 = np.array([x_right, best_y], dtype="float32")
    if p1[0] > p2[0]:
        p1, p2 = p2, p1

    top_vec = p2 - p1
    top_width = float(np.linalg.norm(top_vec))
    if top_width < 10:
        return None, debug, "canny-top-edge"

    n = np.array([-top_vec[1], top_vec[0]], dtype="float32")
    n_norm = float(np.linalg.norm(n))
    if n_norm < 1e-6:
        return None, debug, "canny-top-edge"
    n /= n_norm
    if n[1] < 0:
        n = -n

    target_h = top_width * MTG_HEIGHT_OVER_WIDTH
    p3 = p2 + (n * target_h)
    p4 = p1 + (n * target_h)

    quad = np.array([p1, p2, p3, p4], dtype="float32")
    quad[:, 0] = np.clip(quad[:, 0], 0, w - 1)
    quad[:, 1] = np.clip(quad[:, 1], 0, h - 1)
    quad = cam.order_points(quad)

    # Cyan: chosen top edge from black-band transition. Magenta: extrapolated full card.
    cv2.line(debug, tuple(p1.astype("int32")), tuple(p2.astype("int32")), (255, 255, 0), 2)
    cv2.polylines(debug, [quad.astype("int32")], True, (255, 0, 255), 2)
    return quad, debug, "canny-top-edge"


def corner_angles_degrees(quad):
    pts = cam.order_points(quad).astype("float32")
    angles = []
    for i in range(4):
        p_prev = pts[(i - 1) % 4]
        p_curr = pts[i]
        p_next = pts[(i + 1) % 4]
        v1 = p_prev - p_curr
        v2 = p_next - p_curr
        n1 = float(np.linalg.norm(v1))
        n2 = float(np.linalg.norm(v2))
        if n1 < 1e-6 or n2 < 1e-6:
            angles.append(0.0)
            continue
        cos_theta = float(np.dot(v1, v2) / (n1 * n2))
        cos_theta = max(-1.0, min(1.0, cos_theta))
        angles.append(math.degrees(math.acos(cos_theta)))
    return angles


def passes_right_angle_gate(quad):
    angles = corner_angles_degrees(quad)
    min_allowed = 90.0 - CORNER_RIGHT_ANGLE_TOLERANCE_DEG
    max_allowed = 90.0 + CORNER_RIGHT_ANGLE_TOLERANCE_DEG
    return all(min_allowed <= angle <= max_allowed for angle in angles), angles


def passes_portrait_ratio_gate(quad, image_shape):
    metrics = cam.quad_geometry_metrics(quad, image_shape)
    aspect = metrics["aspect_ratio"]
    min_aspect = STRICT_MTG_ASPECT_RATIO - STRICT_MTG_ASPECT_TOLERANCE
    max_aspect = STRICT_MTG_ASPECT_RATIO + STRICT_MTG_ASPECT_TOLERANCE
    if aspect >= 1.0:
        return False, metrics
    if not (min_aspect <= aspect <= max_aspect):
        return False, metrics
    return True, metrics


def warp_card(processed_bgr, quad):
    return cam.rotate_and_crop_image_local(processed_bgr, quad)


def detect_card_experimental(processed_bgr, enhanced_gray, live_ctrl, detector_mode):
    _ = detector_mode
    return detector_canny_top_edge(processed_bgr, enhanced_gray, live_ctrl)


def main():
    parser = argparse.ArgumentParser(description="Run experimental v2 locator with multiple card-detection methods.")
    parser.add_argument("--dry-run", action="store_true", help="Use dry-run motor controller.")
    parser.add_argument("--roi-shift-right-ratio", type=float, default=DEFAULT_ROI_SHIFT_RIGHT_RATIO)
    parser.add_argument("--roi-shift-down-ratio", type=float, default=DEFAULT_ROI_SHIFT_DOWN_RATIO)
    args = parser.parse_args()

    module = cam.load_source_module()
    cfg = SegmentedHashConfig()
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
        print(f"Controller init failed ({exc}); falling back to dry-run.", flush=True)
        controller = module.DryRunController()
        controller.open()

    cap = cam.configure_camera(module)
    cv2.namedWindow(cam.WINDOW_LIVE, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.namedWindow(WINDOW_PREPROCESSED, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.namedWindow(WINDOW_MASK, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.namedWindow(cam.WINDOW_WARPED, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.resizeWindow(cam.WINDOW_LIVE, 1100, 750)
    cv2.resizeWindow(WINDOW_PREPROCESSED, 650, 950)
    cv2.resizeWindow(WINDOW_MASK, 650, 950)
    cv2.resizeWindow(cam.WINDOW_WARPED, 500, 700)
    init_live_controls(args.roi_shift_right_ratio, args.roi_shift_down_ratio)

    detector_mode = "canny-top-edge"
    last_identified = "None"
    last_identification_time = 0.0
    last_servo_home_angle = None
    last_wait_reason = None
    last_wait_reason_time = 0.0
    interval = 0.6

    print("[exp] Canny top-edge mode: detect top border then extrapolate full card.", flush=True)

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                continue

            live_ctrl = read_live_controls()
            servo_home_angle = live_ctrl["servo_home_angle"]
            if last_servo_home_angle != servo_home_angle:
                controller.servo2_move_abs(servo_home_angle)
                last_servo_home_angle = servo_home_angle

            frame = normalize_frame_pre_border(frame)
            roi_frame, roi_x1, roi_y1, roi_x2, roi_y2 = get_shifted_trimmed_roi(
                module,
                frame,
                live_ctrl["roi_shift_right_ratio"],
                live_ctrl["roi_shift_down_ratio"],
            )

            live_preview = frame.copy()
            cv2.rectangle(live_preview, (roi_x1, roi_y1), (roi_x2, roi_y2), (0, 255, 255), 3)
            module.draw_text_with_background(live_preview, "Experimental Locator Mode: canny-top-edge", (20, 35), font_scale=0.8, thickness=2)
            module.draw_text_with_background(live_preview, f"Last identified: {last_identified}", (20, 70), font_scale=0.7, thickness=2)
            module.draw_text_with_background(
                live_preview,
                (
                    f"MaskBlur={live_ctrl['mask_blur']} Close={live_ctrl['mask_close']} Open={live_ctrl['mask_open']} "
                    f"MinArea={live_ctrl['mask_min_area_ratio']:.2f} Canny={live_ctrl['canny_low']}/{live_ctrl['canny_high']}"
                ),
                (20, 105),
                font_scale=0.55,
                thickness=2,
            )
            module.draw_text_with_background(
                live_preview,
                "q=quit r=drop+return (live servo)",
                (20, frame.shape[0] - 20),
                font_scale=0.6,
                thickness=2,
            )

            preprocessed_vis = None
            detector_vis = None
            warped_vis = None

            try:
                reject_reason = None
                processed, enhanced_gray = preprocess_for_detection(roi_frame, live_ctrl)
                preprocessed_vis = processed

                quad, detector_vis, method_name = detect_card_experimental(processed, enhanced_gray, live_ctrl, detector_mode)
                if detector_vis is None:
                    detector_vis = cv2.cvtColor(enhanced_gray, cv2.COLOR_GRAY2BGR)

                module.draw_text_with_background(detector_vis, f"Detector: {method_name}", (20, 35), font_scale=0.7, thickness=2)

                if quad is not None:
                    angles_ok, angles = passes_right_angle_gate(quad)
                    if not angles_ok:
                        quad = None
                        reject_reason = (
                            "corner-angle reject: "
                            f"angles=[{angles[0]:.1f},{angles[1]:.1f},{angles[2]:.1f},{angles[3]:.1f}] "
                            f"target=90+/-{CORNER_RIGHT_ANGLE_TOLERANCE_DEG:.1f}"
                        )

                if quad is not None:
                    ratio_ok, ratio_metrics = passes_portrait_ratio_gate(quad, processed.shape)
                    if not ratio_ok:
                        quad = None
                        reject_reason = (
                            f"ratio gate reject: aspect={ratio_metrics['aspect_ratio']:.3f} "
                            f"target={STRICT_MTG_ASPECT_RATIO:.3f}+/-{STRICT_MTG_ASPECT_TOLERANCE:.3f}"
                        )

                if reject_reason:
                    module.draw_text_with_background(live_preview, reject_reason, (20, 140), font_scale=0.55, thickness=2, text_color=(0, 0, 255))

                if quad is not None:
                    warped = warp_card(processed, quad)
                    if warped is not None and warped.size > 0:
                        warped_vis = warped

                if warped_vis is None:
                    warped_vis = processed

                current_time = time.time()
                if current_time - last_identification_time >= interval:
                    if quad is None:
                        wait_reason = reject_reason or f"no valid contour with {method_name}"
                        if wait_reason != last_wait_reason or (current_time - last_wait_reason_time) >= 2.0:
                            print(f"[exp] waiting: {wait_reason}", flush=True)
                            last_wait_reason = wait_reason
                            last_wait_reason_time = current_time
                    else:
                        t0 = time.perf_counter()
                        result = matcher.identify(warped)
                        ident_ms = (time.perf_counter() - t0) * 1000.0
                        last_identification_time = current_time

                        best = result.get("best")
                        if best is not None:
                            last_identified = best["name"]
                            best_set = best.get("set") or "?"
                            print("[exp] ---------------- MATCH ----------------", flush=True)
                            print(f"[exp] - detector: {method_name}", flush=True)
                            print(f"[exp] - identify time: {ident_ms:.0f} ms", flush=True)
                            print(f"[exp] - status: {result['status']}", flush=True)
                            print(f"[exp] - best match: {best['name']} [{best_set}]", flush=True)
                            print(f"[exp] - final score: {best['final_score']:.2f}", flush=True)
                            print("[exp] --------------------------------------", flush=True)
                        else:
                            print("[exp] ---------------- MATCH ----------------", flush=True)
                            print(f"[exp] - detector: {method_name}", flush=True)
                            print(f"[exp] - identify time: {ident_ms:.0f} ms", flush=True)
                            print(f"[exp] - status: reject", flush=True)
                            print(f"[exp] - reason: {result.get('reason')}", flush=True)
                            print("[exp] --------------------------------------", flush=True)

            except Exception as exc:
                detector_vis = roi_frame
                warped_vis = roi_frame
                preprocessed_vis = roi_frame
                module.draw_text_with_background(live_preview, f"Detection error: {exc}", (20, 140), font_scale=0.55, thickness=2, text_color=(0, 0, 255))

            cv2.imshow(cam.WINDOW_LIVE, live_preview)
            if preprocessed_vis is not None and preprocessed_vis.size > 0:
                cv2.imshow(WINDOW_PREPROCESSED, preprocessed_vis)
            if detector_vis is not None and detector_vis.size > 0:
                cv2.imshow(WINDOW_MASK, detector_vis)
            if warped_vis is not None and warped_vis.size > 0:
                cv2.imshow(cam.WINDOW_WARPED, warped_vis)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                break
            if key == ord("r"):
                home_angle = live_ctrl["servo_home_angle"]
                down_angle = home_angle - live_ctrl["servo_drop_delta"]
                print(f"Servo2 manual control: moving from {home_angle} to {down_angle}, wait 1.0s, back to home.", flush=True)
                controller.servo2_move_abs(down_angle)
                time.sleep(1.0)
                controller.servo2_move_abs(home_angle)
                print("Servo2 manual control complete.", flush=True)
    finally:
        cap.release()
        controller.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()