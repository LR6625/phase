import argparse
import json
import math
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
try:
    from ultralytics import YOLO
except Exception:
    YOLO = None

import CamTesting as cam


WORKSPACE_DIR = Path(__file__).resolve().parent
DATASET_DIR_DEFAULT = WORKSPACE_DIR / "yolo_dataset"
CAMERA_PRESET_DEFAULT = WORKSPACE_DIR / "camera_preset.json"
LIVE_CONTROLS_PRESET_DEFAULT = WORKSPACE_DIR / "yolo_live_controls.json"
ROI_CONTROL_KEYS = (
    "roi_shift_x",
    "roi_shift_y",
    "roi_expand_left",
    "roi_expand_right",
    "roi_expand_up",
    "roi_expand_down",
)

WINDOW_LIVE = "YOLO Live"
WINDOW_PREPROCESSED = "YOLO ROI Preprocessed"
WINDOW_BINARY = "YOLO ROI Binary"
WINDOW_WARPED = "YOLO ROI Warped"
CONTROLS_WINDOW = "YOLO Controls"
WINDOW_CALIBRATION = "YOLO Camera Calibration"

DEFAULT_ROI_SHIFT_RIGHT_RATIO = 0.06
DEFAULT_ROI_SHIFT_DOWN_RATIO = 0.00

DEFAULT_PERSPECTIVE_TOP_SHRINK = 0.95
DEFAULT_PERSPECTIVE_BOTTOM_GROW = 0.95

# Defaults copied/tuned from today's scripts.
DEFAULT_ROI_EXPAND_LEFT = 0.02
DEFAULT_ROI_EXPAND_RIGHT = 0.01
DEFAULT_ROI_EXPAND_UP = 0.01
DEFAULT_ROI_EXPAND_DOWN = 0.012

DEFAULT_LENS_K1 = float(getattr(cam, "LENS_CORRECTION_K1", -0.03))
DEFAULT_LENS_K2 = float(getattr(cam, "LENS_CORRECTION_K2", 0.008))
DEFAULT_LENS_FOCAL_SCALE = float(getattr(cam, "LENS_FOCAL_SCALE", 1.0))

DEFAULT_BRIGHTNESS = 0
DEFAULT_CONTRAST = 1.0
DEFAULT_GLARE_FIX = 100
DEFAULT_MANUAL_EXPOSURE = -6.0
DEFAULT_MANUAL_GAIN = 0.0
DEFAULT_MANUAL_WB_TEMP = 4500.0
DEFAULT_MANUAL_FOCUS = 20.0

DEFAULT_PREPROCESS_MODE = 0
DEFAULT_ADAPTIVE_BLOCK = 21
DEFAULT_ADAPTIVE_C = 8
DEFAULT_CLOSE_KERNEL = 5
DEFAULT_DILATE_ITERS = 1

SERVO2_HOME_ANGLE = 140
SERVO2_DROP_DELTA = 140
MOTOR1_FEED_STEPS = 125
MOTOR1_SWITCH_MAX_STEPS = 200000
TT_FORWARD_SPEED = "CRAWL"
TT_FORWARD_SECONDS = 0.60
TT_REVERSE_SPEED = "MEDIUM"
TT_REVERSE_SECONDS = 0.3
TT_PAUSE_SECONDS = 0.3

# Card geometry gates.
MTG_ASPECT_RATIO = 58 / 78
MIN_BOX_AREA_RATIO = 0.60
MIN_CORNER_ANGLE_DEG_DEFAULT = 86
MAX_CORNER_ANGLE_DEG_DEFAULT = 94

# Black-band localizer settings.
BAND_SEARCH_TOP_FRACTION = 0.60
MIN_BAND_ROW_FILL = 0.10
MIN_CARD_WIDTH_RATIO = 0.35
SIDE_LOOK_FRACTION_TOP = 0.55

# Glare reduction settings.
GLARE_VALUE_THRESHOLD = 240
GLARE_SAT_THRESHOLD = 65
GLARE_MIN_BLOB_AREA_RATIO = 0.00005
GLARE_MAX_BLOB_AREA_RATIO = 0.02
GLARE_DILATE_ITERS = 1
GLARE_INPAINT_RADIUS = 3

YOLO_VERIFY_MODEL_DEFAULT = WORKSPACE_DIR / "models/best.pt"

# Top-background masking localizer settings (used on saved 's' captures).
TOP_BG_SAMPLE_FRACTION = 0.14
TOP_BG_MIN_SAMPLE_PIXELS = 10
TOP_BG_THRESH_STD_MULTIPLIER = 2.8
TOP_BG_THRESH_MIN = 8.0
TOP_BG_THRESH_MAX = 45.0
TOP_BG_MIN_CARD_AREA_RATIO = 0.10


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


def _safe_set(cap, prop_id, value):
    try:
        return bool(cap.set(prop_id, value))
    except Exception:
        return False


def _safe_get(cap, prop_id, default=None):
    try:
        val = cap.get(prop_id)
        if val is None:
            return default
        return val
    except Exception:
        return default


def _float_or_default(value, default):
    try:
        if value is None:
            return float(default)
        return float(value)
    except Exception:
        return float(default)


def apply_lens_correction_custom(frame, k1: float, k2: float, focal_scale: float):
    if frame is None or frame.size == 0:
        return frame

    k1 = float(k1)
    k2 = float(k2)
    focal_scale = max(0.1, float(focal_scale))

    if abs(k1) < 1e-9 and abs(k2) < 1e-9 and abs(focal_scale - 1.0) < 1e-9:
        return frame

    height, width = frame.shape[:2]
    fx = width * focal_scale
    fy = height * focal_scale
    cx = width * 0.5
    cy = height * 0.5

    camera_matrix = np.array(
        [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    dist_coeffs = np.array([k1, k2, 0.0, 0.0, 0.0], dtype=np.float32)
    return cv2.undistort(frame, camera_matrix, dist_coeffs)


def _open_capture_with_backends(camera_index: int):
    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    if cap.isOpened():
        return cap
    cap.release()

    cap = cv2.VideoCapture(camera_index, cv2.CAP_MSMF)
    if cap.isOpened():
        return cap
    cap.release()

    cap = cv2.VideoCapture(camera_index)
    if cap.isOpened():
        return cap
    cap.release()
    return None


def list_available_cameras(max_index: int = 8):
    cameras = []
    for idx in range(max_index + 1):
        cap = _open_capture_with_backends(idx)
        if cap is None:
            continue

        width = int(_safe_get(cap, cv2.CAP_PROP_FRAME_WIDTH, 0) or 0)
        height = int(_safe_get(cap, cv2.CAP_PROP_FRAME_HEIGHT, 0) or 0)
        ok, frame = cap.read()
        if ok and frame is not None and frame.size > 0:
            h, w = frame.shape[:2]
            if w > 0 and h > 0:
                width, height = w, h
        cap.release()
        cameras.append({"index": idx, "width": width, "height": height})
    return cameras


def resolve_camera_index(requested_index: int):
    if requested_index >= 0:
        return requested_index

    cameras = list_available_cameras(max_index=8)
    if not cameras:
        return 0

    non_zero = [c for c in cameras if c["index"] > 0]
    if non_zero:
        # Built-in webcam is commonly index 0 on Windows. Prefer non-zero by default.
        return non_zero[0]["index"]

    return cameras[0]["index"]


def load_camera_preset(preset_path: Path) -> dict:
    if not preset_path.exists():
        return {}
    try:
        data = json.loads(preset_path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    return {}


def save_camera_preset(preset_path: Path, preset: dict):
    preset_path.parent.mkdir(parents=True, exist_ok=True)
    preset_path.write_text(json.dumps(preset, indent=2), encoding="utf-8")


def load_live_controls_preset(preset_path: Path) -> dict:
    if not preset_path.exists():
        return {}
    try:
        data = json.loads(preset_path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return {k: data[k] for k in ROI_CONTROL_KEYS if k in data}
    except Exception:
        pass
    return {}


def save_live_controls_preset(preset_path: Path, controls: dict):
    controls = {k: controls[k] for k in ROI_CONTROL_KEYS if k in controls}
    preset_path.parent.mkdir(parents=True, exist_ok=True)
    preset_path.write_text(json.dumps(controls, indent=2), encoding="utf-8")


def get_fixed_roi_controls(saved: dict):
    roi_shift_x = max(0.0, min(0.40, float(saved.get("roi_shift_x", DEFAULT_ROI_SHIFT_RIGHT_RATIO))))
    roi_shift_y = max(0.0, min(0.30, float(saved.get("roi_shift_y", DEFAULT_ROI_SHIFT_DOWN_RATIO))))
    roi_expand_left = max(0.0, min(0.20, float(saved.get("roi_expand_left", DEFAULT_ROI_EXPAND_LEFT))))
    roi_expand_right = max(0.0, min(0.20, float(saved.get("roi_expand_right", DEFAULT_ROI_EXPAND_RIGHT))))
    roi_expand_up = max(0.0, min(0.20, float(saved.get("roi_expand_up", DEFAULT_ROI_EXPAND_UP))))
    roi_expand_down = max(0.0, min(0.20, float(saved.get("roi_expand_down", DEFAULT_ROI_EXPAND_DOWN))))

    return {
        "roi_shift_x": roi_shift_x,
        "roi_shift_y": roi_shift_y,
        "roi_expand_left": roi_expand_left,
        "roi_expand_right": roi_expand_right,
        "roi_expand_up": roi_expand_up,
        "roi_expand_down": roi_expand_down,
    }


def apply_live_controls_to_trackbars(focus_default: float):

    lens_k1 = DEFAULT_LENS_K1
    lens_k2 = DEFAULT_LENS_K2
    lens_focal = DEFAULT_LENS_FOCAL_SCALE
    focus = max(0.0, min(255.0, float(focus_default)))

    cv2.setTrackbarPos("Lens K1 x1000 (+200)", CONTROLS_WINDOW, int(round(lens_k1 * 1000.0)) + 200)
    cv2.setTrackbarPos("Lens K2 x1000 (+200)", CONTROLS_WINDOW, int(round(lens_k2 * 1000.0)) + 200)
    cv2.setTrackbarPos("Lens Focal x100", CONTROLS_WINDOW, int(round(lens_focal * 100.0)))
    cv2.setTrackbarPos("Focus", CONTROLS_WINDOW, int(round(focus)))


def apply_manual_camera_settings(
    cap,
    manual_exposure: Optional[float],
    manual_gain: Optional[float],
    manual_wb_temp: Optional[float],
    manual_focus: Optional[float],
):
    # Lock autos first.
    _safe_set(cap, cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
    _safe_set(cap, cv2.CAP_PROP_AUTO_WB, 0)
    _safe_set(cap, cv2.CAP_PROP_AUTOFOCUS, 0)

    if manual_exposure is not None:
        _safe_set(cap, cv2.CAP_PROP_EXPOSURE, float(manual_exposure))
    if manual_gain is not None:
        _safe_set(cap, cv2.CAP_PROP_GAIN, float(manual_gain))
    if manual_wb_temp is not None:
        _safe_set(cap, cv2.CAP_PROP_WB_TEMPERATURE, float(manual_wb_temp))
    if manual_focus is not None:
        _safe_set(cap, cv2.CAP_PROP_FOCUS, float(manual_focus))


def reduce_glare_specular(image, glare_fix_strength: int):
    if glare_fix_strength <= 0:
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

    raw_mask = ((v >= value_threshold) & (s <= sat_threshold)).astype(np.uint8) * 255
    if not np.any(raw_mask):
        return image

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


def normalize_frame_pre_border(frame, brightness: int, contrast: float, glare_fix_strength: int):
    corrected = cam.apply_lens_distortion_correction(frame)

    lab = cv2.cvtColor(corrected, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
    corrected = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)
    corrected = cv2.bilateralFilter(corrected, d=5, sigmaColor=35, sigmaSpace=35)

    if contrast != 1.0 or brightness != 0:
        corrected = cv2.convertScaleAbs(corrected, alpha=contrast, beta=brightness)

    corrected = reduce_glare_specular(corrected, glare_fix_strength)
    return corrected


def get_shifted_roi_full(
    module,
    frame,
    shift_right_ratio,
    shift_down_ratio,
    expand_left,
    expand_right,
    expand_up,
    expand_down,
):
    _, roi_x1, roi_y1, roi_x2, roi_y2 = module.get_center_roi(frame)

    roi_width = max(roi_x2 - roi_x1, 1)
    roi_height = max(roi_y2 - roi_y1, 1)

    shift_px = int(roi_width * max(0.0, shift_right_ratio))
    shift_py = int(roi_height * shift_down_ratio)

    frame_h, frame_w = frame.shape[:2]
    x1 = min(max(roi_x1 + shift_px, 0), max(frame_w - 2, 0))
    x2 = min(max(roi_x2 + shift_px, x1 + 1), frame_w)

    extra_left = int(roi_width * max(0.0, expand_left))
    extra_right = int(roi_width * max(0.0, expand_right))
    extra_up = int(roi_height * max(0.0, expand_up))
    extra_down = int(roi_height * max(0.0, expand_down))

    x1 = max(0, x1 - extra_left)
    x2 = min(frame_w, x2 + extra_right)

    y1 = max(0, roi_y1 + shift_py)
    y2 = max(y1 + 2, roi_y2 + shift_py)
    y1 = min(y1, frame_h - 2)
    y2 = min(max(y2, y1 + 1), frame_h)
    y1 = max(0, y1 - extra_up)
    y2 = min(frame_h, max(y1 + 1, y2 + extra_down))

    roi = frame[y1:y2, x1:x2]
    return roi, x1, y1, x2, y2


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


def preprocess_for_yolo(
    roi_frame,
    mode,
    adaptive_block,
    adaptive_c,
    canny_low,
    canny_high,
    close_kernel,
    dilate_iters,
    perspective_top,
    perspective_bottom,
):
    corrected_roi = apply_pre_edge_perspective(roi_frame, perspective_top, perspective_bottom)
    if corrected_roi.shape[:2] != roi_frame.shape[:2]:
        corrected_roi = roi_frame

    gray = cv2.cvtColor(corrected_roi, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    enhanced = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8)).apply(blurred)

    if mode == 0:
        block_size = _odd_from_slider(adaptive_block, min_odd=3)
        binary_map = cv2.adaptiveThreshold(
            enhanced,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            block_size,
            int(adaptive_c),
        )
    else:
        binary_map = cv2.Canny(enhanced, canny_low, canny_high)

    kernel_small = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (close_kernel, close_kernel))

    if dilate_iters > 0:
        binary_map = cv2.dilate(binary_map, kernel_small, iterations=dilate_iters)
    binary_map = cv2.morphologyEx(binary_map, cv2.MORPH_CLOSE, kernel_close, iterations=1)

    return corrected_roi, binary_map


def _largest_foreground_component(mask, min_area_ratio: float):
    h_img, w_img = mask.shape[:2]
    min_area = max(1, int(h_img * w_img * min_area_ratio))

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

    fg = np.zeros_like(mask)
    fg[labels == best_label] = 255
    return fg


def detect_card_quad_from_top_background(roi_bgr):
    """Estimate card quad by masking top-connected background color and fitting the remaining foreground."""
    h_img, w_img = roi_bgr.shape[:2]
    if h_img < 12 or w_img < 12:
        return None, None

    sample_h = max(TOP_BG_MIN_SAMPLE_PIXELS, int(round(h_img * TOP_BG_SAMPLE_FRACTION)))
    sample_h = min(h_img, sample_h)

    lab = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2LAB)
    top_sample = lab[:sample_h, :, :].reshape(-1, 3).astype(np.float32)

    center = np.median(top_sample, axis=0)
    top_dist = np.linalg.norm(top_sample - center, axis=1)
    dist_std = float(np.std(top_dist))
    dist_mean = float(np.mean(top_dist))

    threshold = dist_mean + (TOP_BG_THRESH_STD_MULTIPLIER * dist_std)
    threshold = float(np.clip(threshold, TOP_BG_THRESH_MIN, TOP_BG_THRESH_MAX))

    dist_map = np.linalg.norm(lab.astype(np.float32) - center.reshape(1, 1, 3), axis=2)
    bg_mask_raw = (dist_map <= threshold).astype(np.uint8) * 255

    # Clean raw color mask before selecting top-connected background.
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    bg_mask_clean = cv2.morphologyEx(bg_mask_raw, cv2.MORPH_OPEN, k, iterations=1)
    bg_mask_clean = cv2.morphologyEx(bg_mask_clean, cv2.MORPH_CLOSE, k, iterations=1)

    # Keep only background components touching the top edge so mask propagates downward from top.
    num_labels, labels, _stats, _ = cv2.connectedComponentsWithStats(bg_mask_clean, connectivity=8)
    top_labels = set(int(v) for v in np.unique(labels[0, :]) if int(v) != 0)
    bg_connected = np.zeros_like(bg_mask_clean)
    for label in top_labels:
        bg_connected[labels == label] = 255

    card_mask = cv2.bitwise_not(bg_connected)
    card_mask[:sample_h, :] = 0  # Card should not start in the top sampling strip.
    card_mask = cv2.morphologyEx(card_mask, cv2.MORPH_OPEN, k, iterations=1)
    card_mask = cv2.morphologyEx(card_mask, cv2.MORPH_CLOSE, k, iterations=2)

    fg = _largest_foreground_component(card_mask, TOP_BG_MIN_CARD_AREA_RATIO)
    if fg is None:
        debug = {
            "sample_h": sample_h,
            "threshold": threshold,
            "dist_map": dist_map,
            "bg_mask_raw": bg_mask_raw,
            "bg_connected": bg_connected,
            "card_mask": card_mask,
            "card_component": None,
            "overlay": roi_bgr.copy(),
        }
        return None, debug

    contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        debug = {
            "sample_h": sample_h,
            "threshold": threshold,
            "dist_map": dist_map,
            "bg_mask_raw": bg_mask_raw,
            "bg_connected": bg_connected,
            "card_mask": card_mask,
            "card_component": fg,
            "overlay": roi_bgr.copy(),
        }
        return None, debug

    contour = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(contour)
    quad = cam.order_points(cv2.boxPoints(rect))

    overlay = roi_bgr.copy()
    cv2.line(overlay, (0, sample_h), (w_img - 1, sample_h), (255, 255, 0), 2)
    cv2.polylines(overlay, [quad.astype(np.int32)], True, (0, 255, 0), 3)

    debug = {
        "sample_h": sample_h,
        "threshold": threshold,
        "dist_map": dist_map,
        "bg_mask_raw": bg_mask_raw,
        "bg_connected": bg_connected,
        "card_mask": card_mask,
        "card_component": fg,
        "overlay": overlay,
    }
    return quad, debug


def save_top_background_debug(dataset_dir: Path, stem: str, debug: dict):
    debug_dir = dataset_dir / "debug_topbg" / "train"
    debug_dir.mkdir(parents=True, exist_ok=True)

    overlay = debug.get("overlay")
    if overlay is not None:
        cv2.imwrite(str(debug_dir / f"{stem}_overlay.png"), overlay)

    bg_raw = debug.get("bg_mask_raw")
    if bg_raw is not None:
        cv2.imwrite(str(debug_dir / f"{stem}_bg_mask_raw.png"), bg_raw)

    bg_conn = debug.get("bg_connected")
    if bg_conn is not None:
        cv2.imwrite(str(debug_dir / f"{stem}_bg_connected.png"), bg_conn)

    card_mask = debug.get("card_mask")
    if card_mask is not None:
        cv2.imwrite(str(debug_dir / f"{stem}_card_mask.png"), card_mask)

    card_component = debug.get("card_component")
    if card_component is not None:
        cv2.imwrite(str(debug_dir / f"{stem}_card_component.png"), card_component)

    dist_map = debug.get("dist_map")
    if dist_map is not None:
        dist_norm = cv2.normalize(dist_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        heat = cv2.applyColorMap(dist_norm, cv2.COLORMAP_TURBO)
        cv2.imwrite(str(debug_dir / f"{stem}_distance_heatmap.png"), heat)


def _smooth_profile_1d(values, ksize: int = 9):
    ksize = _odd_from_slider(ksize, min_odd=3)
    kernel = np.ones((ksize,), dtype=np.float32) / float(ksize)
    return np.convolve(values.astype(np.float32), kernel, mode="same")


def _largest_true_run(mask_1d):
    best_start, best_end, best_len = -1, -1, 0
    run_start = None
    for i, val in enumerate(mask_1d):
        if val and run_start is None:
            run_start = i
        if (not val or i == len(mask_1d) - 1) and run_start is not None:
            run_end = i if val and i == len(mask_1d) - 1 else i - 1
            run_len = run_end - run_start + 1
            if run_len > best_len:
                best_start, best_end, best_len = run_start, run_end, run_len
            run_start = None
    if best_len <= 0:
        return None
    return best_start, best_end


def _detect_top_black_band(binary_map):
    h_img = binary_map.shape[0]
    top_h = max(4, int(h_img * BAND_SEARCH_TOP_FRACTION))
    upper = binary_map[:top_h, :]

    row_fill = np.mean(upper > 0, axis=1)
    row_fill_s = _smooth_profile_1d(row_fill, 11)

    peak_y = int(np.argmax(row_fill_s))
    peak_val = float(row_fill_s[peak_y])
    if peak_val < MIN_BAND_ROW_FILL:
        return None

    strong = row_fill_s >= max(MIN_BAND_ROW_FILL, peak_val * 0.55)
    if not strong[peak_y]:
        return None

    top_y = peak_y
    while top_y > 0 and strong[top_y - 1]:
        top_y -= 1
    bot_y = peak_y
    while bot_y < (top_h - 1) and strong[bot_y + 1]:
        bot_y += 1

    return int(top_y), int(bot_y), peak_val


def _estimate_x_bounds_from_band(binary_map, y_top, y_bot):
    h_img, w_img = binary_map.shape[:2]
    y0 = max(0, y_top - 1)
    y1 = min(h_img - 1, y_bot + 1)
    band = binary_map[y0:y1 + 1, :]
    if band.size == 0:
        return None

    col_fill = np.mean(band > 0, axis=0)
    col_fill_s = _smooth_profile_1d(col_fill, 15)
    thr = max(0.08, float(np.percentile(col_fill_s, 70)) * 0.55)
    mask = col_fill_s >= thr
    run = _largest_true_run(mask)
    if run is None:
        return None

    x_left, x_right = run
    if (x_right - x_left + 1) < int(w_img * MIN_CARD_WIDTH_RATIO):
        return None
    return int(x_left), int(x_right)


def _refine_sides_upper(binary_map, x_left, x_right, y_top, card_h_est):
    h_img, w_img = binary_map.shape[:2]
    y1 = min(h_img - 1, int(y_top + max(12.0, card_h_est * SIDE_LOOK_FRACTION_TOP)))
    y0 = max(0, min(y1 - 2, int(y_top)))

    region = binary_map[y0:y1 + 1, :]
    if region.size == 0:
        return x_left, x_right

    col_fill = np.mean(region > 0, axis=0)
    col_fill_s = _smooth_profile_1d(col_fill, 13)

    search_margin = max(6, int(0.18 * w_img))
    lx0 = max(0, x_left - search_margin)
    lx1 = min(w_img - 1, x_left + search_margin)
    rx0 = max(0, x_right - search_margin)
    rx1 = min(w_img - 1, x_right + search_margin)

    if lx1 > lx0:
        x_left = int(lx0 + np.argmax(col_fill_s[lx0:lx1 + 1]))
    if rx1 > rx0:
        x_right = int(rx0 + np.argmax(col_fill_s[rx0:rx1 + 1]))

    if x_right <= x_left:
        return None, None
    return x_left, x_right


def _corner_angles_deg(quad):
    pts = quad.astype(np.float64)
    angles = []
    for i in range(4):
        prev_pt = pts[(i - 1) % 4]
        curr_pt = pts[i]
        next_pt = pts[(i + 1) % 4]

        vec_a = prev_pt - curr_pt
        vec_b = next_pt - curr_pt
        norm_a = np.linalg.norm(vec_a)
        norm_b = np.linalg.norm(vec_b)
        if norm_a < 1e-6 or norm_b < 1e-6:
            return None

        cos_theta = float(np.dot(vec_a, vec_b) / (norm_a * norm_b))
        cos_theta = max(-1.0, min(1.0, cos_theta))
        angles.append(math.degrees(math.acos(cos_theta)))
    return angles


def find_card_quad_from_black_band(binary_map, min_corner_angle, max_corner_angle):
    h_img, w_img = binary_map.shape[:2]
    image_area = float(max(h_img * w_img, 1))

    band = _detect_top_black_band(binary_map)
    if band is None:
        return None, float("inf"), 0.0, None

    band_top, band_bottom, band_strength = band
    x_bounds = _estimate_x_bounds_from_band(binary_map, band_top, band_bottom)
    if x_bounds is None:
        return None, float("inf"), 0.0, None
    x_left, x_right = x_bounds

    width_est = float(max(2.0, x_right - x_left))
    card_h = width_est / MTG_ASPECT_RATIO
    y_top = float(band_top)
    y_bottom = y_top + card_h

    if y_bottom > (h_img - 1):
        overflow = y_bottom - (h_img - 1)
        y_top = max(0.0, y_top - overflow)
        y_bottom = y_top + card_h

    refined = _refine_sides_upper(binary_map, x_left, x_right, y_top, card_h)
    if refined != (None, None):
        x_left, x_right = refined

    card_w = float(max(2.0, x_right - x_left))
    card_h = card_w / MTG_ASPECT_RATIO
    y_bottom = y_top + card_h
    if y_bottom > (h_img - 1):
        y_bottom = float(h_img - 1)
        y_top = max(0.0, y_bottom - card_h)

    quad = cam.order_points(np.array([
        [float(x_left), float(y_top)],
        [float(x_right), float(y_top)],
        [float(x_right), float(y_bottom)],
        [float(x_left), float(y_bottom)],
    ], dtype=np.float32))

    if quad is None:
        return None, float("inf"), 0.0, None

    corner_angles = _corner_angles_deg(quad)
    if corner_angles is None:
        return None, float("inf"), 0.0, None
    if any((a < min_corner_angle or a > max_corner_angle) for a in corner_angles):
        return None, float("inf"), 0.0, None

    aspect = card_w / max(card_h, 1e-6)
    score = abs(aspect - MTG_ASPECT_RATIO) / MTG_ASPECT_RATIO
    score = float(max(0.0, score - 0.05 * min(1.0, band_strength)))

    box_area_ratio = float((card_w * card_h) / image_area)
    if box_area_ratio < MIN_BOX_AREA_RATIO:
        return None, score, box_area_ratio, None

    if score > 0.25:
        return None, score, box_area_ratio, None

    debug = {
        "band_top": int(band_top),
        "band_bottom": int(band_bottom),
        "x_left": int(x_left),
        "x_right": int(x_right),
    }
    return quad, score, box_area_ratio, debug


def warp_best_quad(image, quad):
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

    max_width = int(max(np.linalg.norm(expanded_box[2] - expanded_box[3]), np.linalg.norm(expanded_box[1] - expanded_box[0])))
    max_height = int(max(np.linalg.norm(expanded_box[1] - expanded_box[2]), np.linalg.norm(expanded_box[0] - expanded_box[3])))
    if max_width < 2 or max_height < 2:
        return None

    dst = cam.module_np_array([[0, 0], [max_width - 1, 0], [max_width - 1, max_height - 1], [0, max_height - 1]])
    matrix = cv2.getPerspectiveTransform(expanded_box, dst)
    return cv2.warpPerspective(image, matrix, (max_width, max_height))


def quad_to_yolo_bbox(quad_frame, frame_w, frame_h):
    xs = quad_frame[:, 0]
    ys = quad_frame[:, 1]
    x_min = float(np.clip(np.min(xs), 0, frame_w - 1))
    x_max = float(np.clip(np.max(xs), 0, frame_w - 1))
    y_min = float(np.clip(np.min(ys), 0, frame_h - 1))
    y_max = float(np.clip(np.max(ys), 0, frame_h - 1))

    bw = max(1.0, x_max - x_min)
    bh = max(1.0, y_max - y_min)
    cx = x_min + bw / 2.0
    cy = y_min + bh / 2.0

    return cx / frame_w, cy / frame_h, bw / frame_w, bh / frame_h


def init_live_controls():
    cv2.namedWindow(CONTROLS_WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(CONTROLS_WINDOW, 700, 240)

    cv2.createTrackbar("Lens K1 x1000 (+200)", CONTROLS_WINDOW, int(round(DEFAULT_LENS_K1 * 1000.0)) + 200, 400, _noop)
    cv2.createTrackbar("Lens K2 x1000 (+200)", CONTROLS_WINDOW, int(round(DEFAULT_LENS_K2 * 1000.0)) + 200, 400, _noop)
    cv2.createTrackbar("Lens Focal x100", CONTROLS_WINDOW, int(round(DEFAULT_LENS_FOCAL_SCALE * 100.0)), 200, _noop)
    cv2.createTrackbar("Focus", CONTROLS_WINDOW, 20, 255, _noop)


def read_live_controls(fixed_roi_controls: dict):

    lens_k1 = (cv2.getTrackbarPos("Lens K1 x1000 (+200)", CONTROLS_WINDOW) - 200) / 1000.0
    lens_k2 = (cv2.getTrackbarPos("Lens K2 x1000 (+200)", CONTROLS_WINDOW) - 200) / 1000.0
    focal_slider = cv2.getTrackbarPos("Lens Focal x100", CONTROLS_WINDOW)
    lens_focal_scale = max(0.10, focal_slider / 100.0)
    focus = float(cv2.getTrackbarPos("Focus", CONTROLS_WINDOW))

    return {
        "roi_shift_x": fixed_roi_controls["roi_shift_x"],
        "roi_shift_y": fixed_roi_controls["roi_shift_y"],
        "roi_expand_left": fixed_roi_controls["roi_expand_left"],
        "roi_expand_right": fixed_roi_controls["roi_expand_right"],
        "roi_expand_up": fixed_roi_controls["roi_expand_up"],
        "roi_expand_down": fixed_roi_controls["roi_expand_down"],
        "lens_k1": lens_k1,
        "lens_k2": lens_k2,
        "lens_focal_scale": lens_focal_scale,
        "focus": focus,
    }


def configure_camera_for_yolo(
    module,
    camera_index: int | None,
    manual_exposure: float | None,
    manual_gain: float | None,
    manual_wb_temp: float | None,
    manual_focus: float | None,
):
    # Default path: match run_camtesting_v2 exactly.
    if camera_index is None or camera_index < 0:
        cap = cam.configure_camera(module)
        selected_index = int(getattr(module, "CAMERA_INDEX", -1))
        print(
            f"[YOLO] Using CamTesting camera configuration (index {selected_index}).",
            flush=True,
        )
        return cap, selected_index

    selected_index = resolve_camera_index(camera_index)
    # Match CamTesting behavior first: default backend tends to honor USB auto exposure better.
    cap = cv2.VideoCapture(selected_index)
    if not cap.isOpened():
        cap = _open_capture_with_backends(selected_index)
    if cap is None:
        raise RuntimeError(f"Could not open camera index {camera_index}")

    # Re-try once on the same selected index if the first readback looks invalid.
    width = int(_safe_get(cap, cv2.CAP_PROP_FRAME_WIDTH, 0) or 0)
    height = int(_safe_get(cap, cv2.CAP_PROP_FRAME_HEIGHT, 0) or 0)
    if width < 640 or height < 480:
        cap.release()
        cap = cv2.VideoCapture(selected_index)
        if not cap.isOpened():
            cap = _open_capture_with_backends(selected_index)
        if cap is None:
            raise RuntimeError(f"Could not open camera index {selected_index}")

    # Follow CamTesting setup to avoid very dark startup frames.
    _safe_set(cap, cv2.CAP_PROP_FRAME_WIDTH, 3840)
    _safe_set(cap, cv2.CAP_PROP_FRAME_HEIGHT, 3032)
    _safe_set(cap, cv2.CAP_PROP_BRIGHTNESS, 130)
    _safe_set(cap, cv2.CAP_PROP_CONTRAST, 10)
    _safe_set(cap, cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)

    # Give auto exposure time to settle like CamTesting.
    time.sleep(2.0)
    for _ in range(10):
        cap.read()

    width = int(_safe_get(cap, cv2.CAP_PROP_FRAME_WIDTH, 0) or 0)
    height = int(_safe_get(cap, cv2.CAP_PROP_FRAME_HEIGHT, 0) or 0)

    print(
        f"[YOLO] Using camera index {selected_index}. "
        f"[YOLO] Camera ready at {width}x{height}. "
        "[YOLO] Camera readback: "
        f"auto_exp={_safe_get(cap, cv2.CAP_PROP_AUTO_EXPOSURE, 'n/a')} "
        f"exp={_safe_get(cap, cv2.CAP_PROP_EXPOSURE, 'n/a')} "
        f"auto_wb={_safe_get(cap, cv2.CAP_PROP_AUTO_WB, 'n/a')} "
        f"wb_temp={_safe_get(cap, cv2.CAP_PROP_WB_TEMPERATURE, 'n/a')} "
        f"autofocus={_safe_get(cap, cv2.CAP_PROP_AUTOFOCUS, 'n/a')} "
        f"focus={_safe_get(cap, cv2.CAP_PROP_FOCUS, 'n/a')} "
        f"gain={_safe_get(cap, cv2.CAP_PROP_GAIN, 'n/a')}",
        flush=True,
    )

    return cap, selected_index


def run_camera_calibration(cap, current_settings: dict):
    cv2.namedWindow(WINDOW_CALIBRATION, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.resizeWindow(WINDOW_CALIBRATION, 900, 620)

    exp_init = int(round((float(current_settings.get("manual_exposure", -6.0)) + 20.0) * 10.0))
    gain_init = int(round(float(current_settings.get("manual_gain", 0.0)) * 10.0))
    wb_init = int(round(float(current_settings.get("manual_wb_temp", 4500.0)) / 100.0))
    focus_init = int(round(float(current_settings.get("manual_focus", 20.0))))

    cv2.createTrackbar("Exposure x10 +200", WINDOW_CALIBRATION, max(0, min(400, exp_init)), 400, _noop)
    cv2.createTrackbar("Gain x10", WINDOW_CALIBRATION, max(0, min(500, gain_init)), 500, _noop)
    cv2.createTrackbar("WB Temp /100", WINDOW_CALIBRATION, max(20, min(90, wb_init)), 90, _noop)
    cv2.createTrackbar("Focus", WINDOW_CALIBRATION, max(0, min(255, focus_init)), 255, _noop)

    print("[YOLO] Calibration mode: adjust sliders, press 's' to save preset, 'q' to exit without saving.", flush=True)

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            continue

        exp_slider = cv2.getTrackbarPos("Exposure x10 +200", WINDOW_CALIBRATION)
        gain_slider = cv2.getTrackbarPos("Gain x10", WINDOW_CALIBRATION)
        wb_slider = cv2.getTrackbarPos("WB Temp /100", WINDOW_CALIBRATION)
        focus_slider = cv2.getTrackbarPos("Focus", WINDOW_CALIBRATION)

        manual_exposure = (exp_slider / 10.0) - 20.0
        manual_gain = gain_slider / 10.0
        manual_wb_temp = wb_slider * 100.0
        manual_focus = float(focus_slider)

        apply_manual_camera_settings(cap, manual_exposure, manual_gain, manual_wb_temp, manual_focus)

        preview = frame.copy()
        cv2.putText(
            preview,
            f"Exposure={manual_exposure:.1f} Gain={manual_gain:.1f} WB={manual_wb_temp:.0f} Focus={manual_focus:.0f}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )
        cv2.putText(
            preview,
            "s=save preset and continue, q=cancel",
            (20, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

        cv2.imshow(WINDOW_CALIBRATION, preview)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == 27:
            cv2.destroyWindow(WINDOW_CALIBRATION)
            return None
        if key == ord("s"):
            cv2.destroyWindow(WINDOW_CALIBRATION)
            return {
                "manual_exposure": manual_exposure,
                "manual_gain": manual_gain,
                "manual_wb_temp": manual_wb_temp,
                "manual_focus": manual_focus,
            }


def save_sample(dataset_dir: Path, frame, quad_frame, settings: dict, class_id: int = 0):
    images_dir = dataset_dir / "images" / "train"
    labels_dir = dataset_dir / "labels" / "train"
    meta_dir = dataset_dir / "meta"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)

    ts = time.strftime("%Y%m%d_%H%M%S")
    ms = int((time.time() % 1.0) * 1000)
    stem = f"card_{ts}_{ms:03d}"

    image_path = images_dir / f"{stem}.jpg"
    label_path = labels_dir / f"{stem}.txt"
    meta_path = meta_dir / f"{stem}.json"

    cv2.imwrite(str(image_path), frame)

    fh, fw = frame.shape[:2]
    if quad_frame is not None:
        x, y, w, h = quad_to_yolo_bbox(quad_frame, fw, fh)
        with open(label_path, "w", encoding="utf-8") as f:
            f.write(f"{class_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")
    else:
        # Negative sample file stays empty for YOLO convention.
        with open(label_path, "w", encoding="utf-8") as f:
            f.write("")

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(settings, f, indent=2)

    return image_path, label_path


def run_initial_motor1_homing(controller):
    print("[YOLO] Motor1 homing: down to switch, then up to switch.", flush=True)
    controller.spin_down_m1_until_switch(MOTOR1_SWITCH_MAX_STEPS)
    time.sleep(0.5)
    controller.spin_up_m1_until_switch(MOTOR1_SWITCH_MAX_STEPS)
    time.sleep(0.5)
    print("[YOLO] Motor1 homing complete.", flush=True)


def run_motor1_step_and_tt_cycle(controller):
    print(f"[YOLO] Feed cycle: motor1 up {MOTOR1_FEED_STEPS} steps, TT forward/backward.", flush=True)
    controller.spin_up_m1(MOTOR1_FEED_STEPS)

    controller.tt_motor_forward(TT_FORWARD_SPEED)
    time.sleep(TT_FORWARD_SECONDS)
    controller.tt_motor_stop()

    time.sleep(TT_PAUSE_SECONDS)

    controller.tt_motor_reverse(TT_REVERSE_SPEED)
    time.sleep(TT_REVERSE_SECONDS)
    controller.tt_motor_stop()

    time.sleep(TT_PAUSE_SECONDS)
    print("[YOLO] Feed cycle complete.", flush=True)


def run_yolo_verify_on_saved_image(verify_model, image_path: Path, debug_dir: Path, imgsz: int, conf: float, device: str):
    """Run trained YOLO on a just-saved sample image and persist a debug render."""
    debug_dir.mkdir(parents=True, exist_ok=True)

    results = verify_model.predict(
        source=str(image_path),
        imgsz=imgsz,
        conf=conf,
        device=device,
        save=False,
        verbose=False,
    )
    if not results:
        return {"best_conf": None, "count": 0, "debug_path": None}

    result = results[0]

    conf_vals = []
    if getattr(result, "obb", None) is not None and getattr(result.obb, "conf", None) is not None:
        conf_vals = list(result.obb.conf.cpu().numpy())
    elif getattr(result, "boxes", None) is not None and getattr(result.boxes, "conf", None) is not None:
        conf_vals = list(result.boxes.conf.cpu().numpy())

    best_conf = float(max(conf_vals)) if conf_vals else None
    count = len(conf_vals)

    plotted = result.plot()
    debug_path = debug_dir / f"{image_path.stem}_yolo_verify.jpg"
    cv2.imwrite(str(debug_path), plotted)

    return {"best_conf": best_conf, "count": count, "debug_path": str(debug_path)}


def main():
    parser = argparse.ArgumentParser(description="YOLO Camera Studio - collect high-quality card localization data.")
    parser.add_argument(
        "--camera-index",
        type=int,
        default=None,
        help="Camera index override. If omitted, uses CamTesting default camera config.",
    )
    parser.add_argument("--list-cameras", action="store_true")
    parser.add_argument("--dataset-dir", type=str, default=str(DATASET_DIR_DEFAULT))
    parser.add_argument("--class-id", type=int, default=0)
    parser.add_argument("--manual-exposure", type=float, default=None)
    parser.add_argument("--manual-gain", type=float, default=None)
    parser.add_argument("--manual-wb-temp", type=float, default=None)
    parser.add_argument("--manual-focus", type=float, default=None)
    parser.add_argument("--camera-preset", type=str, default=str(CAMERA_PRESET_DEFAULT))
    parser.add_argument("--live-controls-preset", type=str, default=str(LIVE_CONTROLS_PRESET_DEFAULT))
    parser.add_argument("--calibrate-camera", action="store_true")
    parser.add_argument("--verify-on-save", action="store_true", default=True, help="Run trained YOLO on each image saved with 's'.")
    parser.add_argument("--no-verify-on-save", action="store_false", dest="verify_on_save", help="Disable YOLO verification on saved samples.")
    parser.add_argument("--verify-model", type=str, default=str(YOLO_VERIFY_MODEL_DEFAULT), help="Weights path for YOLO verification model.")
    parser.add_argument("--verify-imgsz", type=int, default=640, help="Inference size used for YOLO verification.")
    parser.add_argument("--verify-conf", type=float, default=0.25, help="Confidence threshold used for YOLO verification.")
    parser.add_argument("--verify-device", type=str, default="cpu", help="Device for YOLO verification (cpu or cuda index).")
    args = parser.parse_args()

    if args.class_id != 0:
        print(f"[YOLO] For this dataset we enforce single class 'card' as class_id 0 (requested {args.class_id}).", flush=True)
        args.class_id = 0

    dataset_dir = Path(args.dataset_dir)
    preset_path = Path(args.camera_preset)
    live_controls_preset_path = Path(args.live_controls_preset)
    preset_data = load_camera_preset(preset_path)
    live_controls_preset = load_live_controls_preset(live_controls_preset_path)
    fixed_roi_controls = get_fixed_roi_controls(live_controls_preset)

    manual_exposure = args.manual_exposure if args.manual_exposure is not None else preset_data.get("manual_exposure")
    manual_gain = args.manual_gain if args.manual_gain is not None else preset_data.get("manual_gain")
    manual_wb_temp = args.manual_wb_temp if args.manual_wb_temp is not None else preset_data.get("manual_wb_temp")
    manual_focus = args.manual_focus if args.manual_focus is not None else preset_data.get("manual_focus")

    module = cam.load_source_module()
    if args.list_cameras:
        cams = list_available_cameras(max_index=8)
        if not cams:
            print("[YOLO] No cameras detected in indices 0..8", flush=True)
        else:
            print("[YOLO] Detected cameras (index: resolution):", flush=True)
            for c in cams:
                print(f"  - {c['index']}: {c['width']}x{c['height']}", flush=True)

    cap, active_camera_index = configure_camera_for_yolo(
        module,
        args.camera_index,
        manual_exposure,
        manual_gain,
        manual_wb_temp,
        manual_focus,
    )

    # If not provided via args/preset, lock to the camera's current values.
    manual_exposure = _float_or_default(
        manual_exposure if manual_exposure is not None else _safe_get(cap, cv2.CAP_PROP_EXPOSURE, None),
        DEFAULT_MANUAL_EXPOSURE,
    )
    manual_gain = _float_or_default(
        manual_gain if manual_gain is not None else _safe_get(cap, cv2.CAP_PROP_GAIN, None),
        DEFAULT_MANUAL_GAIN,
    )
    manual_wb_temp = _float_or_default(
        manual_wb_temp if manual_wb_temp is not None else _safe_get(cap, cv2.CAP_PROP_WB_TEMPERATURE, None),
        DEFAULT_MANUAL_WB_TEMP,
    )
    manual_focus = _float_or_default(
        manual_focus if manual_focus is not None else _safe_get(cap, cv2.CAP_PROP_FOCUS, None),
        DEFAULT_MANUAL_FOCUS,
    )

    if args.calibrate_camera:
        calibrated = run_camera_calibration(
            cap,
            {
                "manual_exposure": manual_exposure if manual_exposure is not None else -6.0,
                "manual_gain": manual_gain if manual_gain is not None else 0.0,
                "manual_wb_temp": manual_wb_temp if manual_wb_temp is not None else 4500.0,
                "manual_focus": manual_focus if manual_focus is not None else 20.0,
            },
        )
        if calibrated is not None:
            save_camera_preset(preset_path, calibrated)
            manual_exposure = calibrated.get("manual_exposure")
            manual_gain = calibrated.get("manual_gain")
            manual_wb_temp = calibrated.get("manual_wb_temp")
            manual_focus = calibrated.get("manual_focus")
            apply_manual_camera_settings(cap, manual_exposure, manual_gain, manual_wb_temp, manual_focus)
            print(f"[YOLO] Saved camera preset to: {preset_path}", flush=True)

    controller = module.DryRunController()
    try:
        controller = module.StepperController(port=module.SERIAL_PORT)
        controller.open()
        run_initial_motor1_homing(controller)
        controller.servo2_move_abs(SERVO2_HOME_ANGLE)
        time.sleep(0.2)
    except Exception as exc:
        print(f"[YOLO] Controller init failed ({exc}); using dry-run.", flush=True)
        controller = module.DryRunController()
        controller.open()
        try:
            run_initial_motor1_homing(controller)
        except Exception:
            pass

    cv2.namedWindow(WINDOW_LIVE, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.namedWindow(WINDOW_PREPROCESSED, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.namedWindow(WINDOW_BINARY, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.namedWindow(WINDOW_WARPED, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)

    cv2.resizeWindow(WINDOW_LIVE, 1200, 760)
    cv2.resizeWindow(WINDOW_PREPROCESSED, 650, 950)
    cv2.resizeWindow(WINDOW_BINARY, 650, 950)
    cv2.resizeWindow(WINDOW_WARPED, 500, 700)

    init_live_controls()
    focus_now = _safe_get(cap, cv2.CAP_PROP_FOCUS, None)
    focus_default = float(focus_now) if focus_now is not None and focus_now >= 0 else DEFAULT_MANUAL_FOCUS
    apply_live_controls_to_trackbars(focus_default)

    print(
        "[YOLO] ROI locked from preset: "
        f"x={fixed_roi_controls['roi_shift_x']:.3f}, y={fixed_roi_controls['roi_shift_y']:.3f}, "
        f"L={fixed_roi_controls['roi_expand_left']:.3f}, R={fixed_roi_controls['roi_expand_right']:.3f}, "
        f"U={fixed_roi_controls['roi_expand_up']:.3f}, D={fixed_roi_controls['roi_expand_down']:.3f}",
        flush=True,
    )
    print("[YOLO] Ready. Keys: q=quit, s=save ROI sample, n=save ROI negative sample, \\=feed cycle, r=servo2 drop")
    print(f"[YOLO] Active camera index: {active_camera_index}", flush=True)

    verify_model = None
    verify_debug_dir = dataset_dir / "debug_yolo_verify" / "train"
    if args.verify_on_save:
        if YOLO is None:
            print("[YOLO] Verify-on-save disabled: ultralytics import unavailable.", flush=True)
        else:
            verify_model_path = Path(args.verify_model)
            if not verify_model_path.exists():
                print(f"[YOLO] Verify-on-save disabled: model not found at {verify_model_path}", flush=True)
            else:
                try:
                    verify_model = YOLO(str(verify_model_path))
                    print(f"[YOLO] Verify-on-save ready with model: {verify_model_path}", flush=True)
                except Exception as exc:
                    print(f"[YOLO] Verify-on-save disabled: failed to load model ({exc})", flush=True)

    save_message = ""
    save_message_time = 0.0
    save_message_seconds = 1.8

    try:
        while True:
            ok, frame_raw = cap.read()
            if not ok or frame_raw is None:
                continue

            live = read_live_controls(fixed_roi_controls)

            # Focus is the only live camera property control; keep exposure/WB at camera defaults.
            _safe_set(cap, cv2.CAP_PROP_AUTOFOCUS, 0)
            _safe_set(cap, cv2.CAP_PROP_FOCUS, live["focus"])

            # Apply live lens correction controls before ROI extraction.
            frame = apply_lens_correction_custom(
                frame_raw,
                live["lens_k1"],
                live["lens_k2"],
                live["lens_focal_scale"],
            )

            roi, roi_x1, roi_y1, roi_x2, roi_y2 = get_shifted_roi_full(
                module,
                frame,
                live["roi_shift_x"],
                live["roi_shift_y"],
                live["roi_expand_left"],
                live["roi_expand_right"],
                live["roi_expand_up"],
                live["roi_expand_down"],
            )

            processed_roi, binary_map = preprocess_for_yolo(
                roi,
                mode=DEFAULT_PREPROCESS_MODE,
                adaptive_block=DEFAULT_ADAPTIVE_BLOCK,
                adaptive_c=DEFAULT_ADAPTIVE_C,
                canny_low=0,
                canny_high=0,
                close_kernel=DEFAULT_CLOSE_KERNEL,
                dilate_iters=DEFAULT_DILATE_ITERS,
                perspective_top=DEFAULT_PERSPECTIVE_TOP_SHRINK,
                perspective_bottom=DEFAULT_PERSPECTIVE_BOTTOM_GROW,
            )

            quad_roi, score, area_ratio, debug = find_card_quad_from_black_band(
                binary_map,
                min_corner_angle=float(MIN_CORNER_ANGLE_DEG_DEFAULT),
                max_corner_angle=float(MAX_CORNER_ANGLE_DEG_DEFAULT),
            )

            live_preview = frame.copy()
            cv2.rectangle(live_preview, (roi_x1, roi_y1), (roi_x2, roi_y2), (0, 255, 255), 3)

            module.draw_text_with_background(
                live_preview,
                "YOLO Camera Studio",
                (20, 35),
                font_scale=0.8,
                thickness=2,
            )
            module.draw_text_with_background(
                live_preview,
                f"Mode=Adaptive Score={score:.3f} Area={area_ratio * 100:.1f}%",
                (20, 70),
                font_scale=0.55,
                thickness=2,
            )

            if debug is not None:
                y_top = int(debug["band_top"])
                y_bot = int(debug["band_bottom"])
                x_left = int(debug["x_left"])
                x_right = int(debug["x_right"])
                cv2.line(binary_map, (0, y_top), (binary_map.shape[1] - 1, y_top), 180, 1)
                cv2.line(binary_map, (0, y_bot), (binary_map.shape[1] - 1, y_bot), 180, 1)
                cv2.line(binary_map, (x_left, 0), (x_left, binary_map.shape[0] - 1), 180, 1)
                cv2.line(binary_map, (x_right, 0), (x_right, binary_map.shape[0] - 1), 180, 1)

            warped_vis = processed_roi
            quad_frame = None

            if quad_roi is not None:
                quad_vis = quad_roi.astype(np.int32)
                cv2.polylines(binary_map, [quad_vis], True, 255, 2)

                quad_frame = quad_roi.copy()
                quad_frame[:, 0] += roi_x1
                quad_frame[:, 1] += roi_y1
                cv2.polylines(live_preview, [quad_frame.astype(np.int32)], True, (0, 255, 0), 3)

                warped = warp_best_quad(processed_roi, quad_roi)
                if warped is not None and warped.size > 0:
                    warped_vis = warped
            else:
                module.draw_text_with_background(
                    live_preview,
                    "No valid card quad found",
                    (20, 105),
                    font_scale=0.6,
                    thickness=2,
                    text_color=(0, 0, 255),
                )

            module.draw_text_with_background(
                live_preview,
                "ROI LOCKED  q=quit  s=save ROI sample  n=save ROI negative  \\=feed cycle  r=servo2 drop",
                (20, frame.shape[0] - 20),
                font_scale=0.6,
                thickness=2,
            )

            if save_message and (time.time() - save_message_time) <= save_message_seconds:
                module.draw_text_with_background(
                    live_preview,
                    save_message,
                    (20, frame.shape[0] - 55),
                    font_scale=0.7,
                    thickness=2,
                    text_color=(0, 255, 0),
                )

            cv2.imshow(WINDOW_LIVE, live_preview)
            cv2.imshow(WINDOW_PREPROCESSED, processed_roi)
            cv2.imshow(WINDOW_BINARY, binary_map)
            if warped_vis is not None and warped_vis.size > 0:
                cv2.imshow(WINDOW_WARPED, warped_vis)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("s"):
                topbg_quad, topbg_debug = detect_card_quad_from_top_background(roi)
                quad_for_save = topbg_quad if topbg_quad is not None else quad_roi

                settings = {
                    "mode": DEFAULT_PREPROCESS_MODE,
                    "adaptive_block": DEFAULT_ADAPTIVE_BLOCK,
                    "adaptive_c": DEFAULT_ADAPTIVE_C,
                    "close_kernel": DEFAULT_CLOSE_KERNEL,
                    "dilate_iters": DEFAULT_DILATE_ITERS,
                    "roi_shift_x": live["roi_shift_x"],
                    "roi_shift_y": live["roi_shift_y"],
                    "roi_expand_left": live["roi_expand_left"],
                    "roi_expand_right": live["roi_expand_right"],
                    "roi_expand_up": live["roi_expand_up"],
                    "roi_expand_down": live["roi_expand_down"],
                    "lens_k1": live["lens_k1"],
                    "lens_k2": live["lens_k2"],
                    "lens_focal_scale": live["lens_focal_scale"],
                    "perspective_top": DEFAULT_PERSPECTIVE_TOP_SHRINK,
                    "perspective_bottom": DEFAULT_PERSPECTIVE_BOTTOM_GROW,
                    "brightness": DEFAULT_BRIGHTNESS,
                    "contrast": DEFAULT_CONTRAST,
                    "glare_fix": 0,
                    "corner_min": MIN_CORNER_ANGLE_DEG_DEFAULT,
                    "corner_max": MAX_CORNER_ANGLE_DEG_DEFAULT,
                    "manual_exposure": _safe_get(cap, cv2.CAP_PROP_EXPOSURE, None),
                    "manual_gain": _safe_get(cap, cv2.CAP_PROP_GAIN, None),
                    "manual_wb_temp": _safe_get(cap, cv2.CAP_PROP_WB_TEMPERATURE, None),
                    "manual_focus": live["focus"],
                    "roi_locked": fixed_roi_controls,
                    "topbg_sample_h": int(topbg_debug["sample_h"]) if topbg_debug is not None else None,
                    "topbg_threshold": float(topbg_debug["threshold"]) if topbg_debug is not None else None,
                    "topbg_quad_found": bool(topbg_quad is not None),
                    "save_quad_source": "top_background_mask" if topbg_quad is not None else ("live_black_band" if quad_roi is not None else "none"),
                }

                image_path, label_path = save_sample(dataset_dir, roi, quad_for_save, settings, class_id=args.class_id)
                save_top_background_debug(dataset_dir, image_path.stem, topbg_debug)

                verify_result = None
                if verify_model is not None:
                    try:
                        verify_result = run_yolo_verify_on_saved_image(
                            verify_model=verify_model,
                            image_path=image_path,
                            debug_dir=verify_debug_dir,
                            imgsz=args.verify_imgsz,
                            conf=args.verify_conf,
                            device=args.verify_device,
                        )
                    except Exception as exc:
                        print(f"[YOLO] Verify-on-save failed for {image_path.name}: {exc}", flush=True)

                print(f"[YOLO] Saved sample: {image_path.name} | {label_path.name}")
                if verify_result is not None:
                    print(
                        f"[YOLO] Verify result: detections={verify_result['count']} "
                        f"best_conf={verify_result['best_conf']} "
                        f"debug={verify_result['debug_path']}",
                        flush=True,
                    )
                if quad_for_save is not None:
                    src = "topbg" if topbg_quad is not None else "live"
                    if verify_result is not None and verify_result.get("best_conf") is not None:
                        save_message = f"Saved sample: {image_path.name} ({src} quad) conf={verify_result['best_conf']:.3f}"
                    else:
                        save_message = f"Saved sample: {image_path.name} ({src} quad)"
                else:
                    save_message = f"Saved sample (no card label): {image_path.name}"
                save_message_time = time.time()
            if key == ord("n"):
                settings = {
                    "mode": DEFAULT_PREPROCESS_MODE,
                    "close_kernel": DEFAULT_CLOSE_KERNEL,
                    "dilate_iters": DEFAULT_DILATE_ITERS,
                    "lens_k1": live["lens_k1"],
                    "lens_k2": live["lens_k2"],
                    "lens_focal_scale": live["lens_focal_scale"],
                    "perspective_top": DEFAULT_PERSPECTIVE_TOP_SHRINK,
                    "perspective_bottom": DEFAULT_PERSPECTIVE_BOTTOM_GROW,
                    "manual_exposure": _safe_get(cap, cv2.CAP_PROP_EXPOSURE, None),
                    "manual_gain": _safe_get(cap, cv2.CAP_PROP_GAIN, None),
                    "manual_wb_temp": _safe_get(cap, cv2.CAP_PROP_WB_TEMPERATURE, None),
                    "manual_focus": live["focus"],
                    "roi_locked": fixed_roi_controls,
                    "note": "negative_sample",
                }
                image_path, label_path = save_sample(dataset_dir, roi, None, settings, class_id=args.class_id)
                print(f"[YOLO] Saved negative: {image_path.name} | {label_path.name}")
                save_message = f"Saved negative: {image_path.name}"
                save_message_time = time.time()
            if key == ord("\\"):
                run_motor1_step_and_tt_cycle(controller)
            if key == ord("r"):
                down_angle = SERVO2_HOME_ANGLE - SERVO2_DROP_DELTA
                print(f"[YOLO] Servo2 drop: {SERVO2_HOME_ANGLE} -> {down_angle}, wait 1s, return.", flush=True)
                controller.servo2_move_abs(down_angle)
                time.sleep(1.0)
                controller.servo2_move_abs(SERVO2_HOME_ANGLE)
                print("[YOLO] Servo2 returned.", flush=True)

    finally:
        cap.release()
        controller.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
