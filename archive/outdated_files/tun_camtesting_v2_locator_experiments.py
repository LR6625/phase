import argparse
import time
from pathlib import Path
import math

import cv2

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
DEFAULT_PERSPECTIVE_TOP_SHRINK = 0.95
DEFAULT_PERSPECTIVE_BOTTOM_GROW = 0.95
SERVO2_HOME_ANGLE = 90
SERVO2_DROP_DELTA = 50
STRICT_MTG_ASPECT_RATIO = 58 / 78
STRICT_MTG_ASPECT_TOLERANCE = 0.03
CORNER_RIGHT_ANGLE_TOLERANCE_DEG = 2
ENABLE_PRE_BORDER_PHOTO_NORMALIZATION = True
ROI_EXTRA_WIDTH_SIDE_RATIO = 0.04
ROI_EXTRA_HEIGHT_RATIO = 0.02
ROI_EXTRA_BOTTOM_RATIO = 0.01
DESKEW_MIN_LINE_LENGTH_RATIO = 0.35
DESKEW_MAX_CANDIDATE_ANGLE_DEG = 35.0
DESKEW_MIN_APPLY_ANGLE_DEG = 1.2
DESKEW_MAX_APPLY_ANGLE_DEG = 22.0


def _noop(_value):
    return None


def _odd_from_slider(value: int, min_odd: int = 3) -> int:
    value = max(min_odd, int(value))
    if value % 2 == 0:
        value += 1
    return value


def normalize_frame_pre_border(frame):
    """Apply camera/photo corrections before ROI and border detection."""
    # 1) Correct lens/fisheye distortion first.
    corrected = cam.apply_lens_distortion_correction(frame)

    if not ENABLE_PRE_BORDER_PHOTO_NORMALIZATION:
        return corrected

    # 2) Mild illumination normalization to reduce lighting bias.
    lab = cv2.cvtColor(corrected, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
    normalized = cv2.merge((l, a, b))
    corrected = cv2.cvtColor(normalized, cv2.COLOR_LAB2BGR)

    # 3) Gentle denoise to stabilize edge maps.
    corrected = cv2.bilateralFilter(corrected, d=5, sigmaColor=35, sigmaSpace=35)
    return corrected


def init_live_controls(default_shift_right_ratio: float, default_shift_down_ratio: float):
    cv2.namedWindow(CONTROLS_WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(CONTROLS_WINDOW, 520, 250)

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


def read_live_controls():
    canny_low = cv2.getTrackbarPos("Canny Low", CONTROLS_WINDOW)
    canny_high = cv2.getTrackbarPos("Canny High", CONTROLS_WINDOW)
    if canny_high <= canny_low:
        canny_high = min(255, canny_low + 1)

    close_kernel = _odd_from_slider(cv2.getTrackbarPos("Close Kernel", CONTROLS_WINDOW), min_odd=3)
    dilate_iters = max(0, int(cv2.getTrackbarPos("Dilate Iters", CONTROLS_WINDOW)))
    roi_shift_right_ratio = cv2.getTrackbarPos("ROI Shift X %", CONTROLS_WINDOW) / 100.0
    roi_shift_down_ratio = cv2.getTrackbarPos("ROI Shift Y %", CONTROLS_WINDOW) / 100.0
    perspective_top_shrink = max(0.70, cv2.getTrackbarPos("Perspective Top %", CONTROLS_WINDOW) / 100.0)
    perspective_bottom_grow = max(0.70, cv2.getTrackbarPos("Perspective Bottom %", CONTROLS_WINDOW) / 100.0)
    servo_home_angle = int(cv2.getTrackbarPos("Servo Home", CONTROLS_WINDOW))
    servo_drop_delta = int(cv2.getTrackbarPos("Servo Drop", CONTROLS_WINDOW))

    return {
        "canny_low": canny_low,
        "canny_high": canny_high,
        "close_kernel": close_kernel,
        "dilate_iters": dilate_iters,
        "roi_shift_right_ratio": roi_shift_right_ratio,
        "roi_shift_down_ratio": roi_shift_down_ratio,
        "perspective_top_shrink": perspective_top_shrink,
        "perspective_bottom_grow": perspective_bottom_grow,
        "servo_home_angle": servo_home_angle,
        "servo_drop_delta": servo_drop_delta,
    }


def warp_with_perspective_tuning(image, quad, top_shrink_ratio: float, bottom_grow_ratio: float):
    """Warp card using CamTesting perspective correction with tunable top/bottom scaling."""
    import numpy as np

    perspective_corrected = cam.correct_perspective_quad(
        quad,
        top_shrink_ratio=top_shrink_ratio,
        bottom_grow_ratio=bottom_grow_ratio,
    )
    _, leveled_box, center, level_angle = cam.level_quad_for_expansion(perspective_corrected)
    expanded_leveled_box = cam.expand_quad(leveled_box, image.shape)

    height_a = np.linalg.norm(leveled_box[1] - leveled_box[2])
    height_b = np.linalg.norm(leveled_box[0] - leveled_box[3])
    card_height = max(height_a, height_b, 1.0)
    vertical_bias = card_height * cam.EXPANSION_VERTICAL_BIAS_RATIO
    expanded_leveled_box[:, 1] = expanded_leveled_box[:, 1] + vertical_bias
    expanded_leveled_box[:, 1] = expanded_leveled_box[:, 1].clip(0, image.shape[0] - 1)
    expanded_leveled_box = cam.order_points(expanded_leveled_box)

    expanded_box = cam.order_points(cam.rotate_points(expanded_leveled_box, level_angle, center))

    width_a = np.linalg.norm(expanded_box[2] - expanded_box[3])
    width_b = np.linalg.norm(expanded_box[1] - expanded_box[0])
    max_width = int(max(width_a, width_b))

    height_a = np.linalg.norm(expanded_box[1] - expanded_box[2])
    height_b = np.linalg.norm(expanded_box[0] - expanded_box[3])
    max_height = int(max(height_a, height_b))

    dst = cam.module_np_array(
        [
            [0, 0],
            [max_width - 1, 0],
            [max_width - 1, max_height - 1],
            [0, max_height - 1],
        ]
    )
    matrix = cv2.getPerspectiveTransform(expanded_box, dst)
    return cv2.warpPerspective(image, matrix, (max_width, max_height))


def apply_pre_edge_perspective(roi_frame, top_shrink_ratio: float, bottom_grow_ratio: float):
    """Apply perspective normalization to the ROI before edge detection.

    Ratios above 1.0 require extrapolation beyond image bounds, so we clamp to 1.0
    for this pre-edge normalization step.
    """
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
    return cv2.warpPerspective(roi_frame, matrix, (w, h))


def preprocess_for_v2(
    roi_frame,
    canny_low: int,
    canny_high: int,
    close_kernel_size: int,
    dilate_iters: int,
    perspective_top_shrink: float,
    perspective_bottom_grow: float,
):
    """Original edge map from CamTesting preprocessing."""
    # Perspective normalization is applied before edge detection.
    corrected_roi = apply_pre_edge_perspective(
        roi_frame,
        top_shrink_ratio=perspective_top_shrink,
        bottom_grow_ratio=perspective_bottom_grow,
    )
    if corrected_roi.shape[:2] != roi_frame.shape[:2]:
        raise RuntimeError(
            f"Pre-edge perspective changed ROI shape from {roi_frame.shape[:2]} to {corrected_roi.shape[:2]}"
        )

    gray = cv2.cvtColor(corrected_roi, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    enhanced = clahe.apply(blurred)

    edges = cv2.Canny(enhanced, canny_low, canny_high)

    kernel_small = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (close_kernel_size, close_kernel_size))
    if dilate_iters > 0:
        edges = cv2.dilate(edges, kernel_small, iterations=dilate_iters)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel_close, iterations=1)

    deskew_angle = estimate_deskew_angle_from_edges(edges)
    if abs(deskew_angle) >= DESKEW_MIN_APPLY_ANGLE_DEG:
        corrected_roi = rotate_image_keep_size(corrected_roi, -deskew_angle, border_mode=cv2.BORDER_REPLICATE)
        edges = rotate_image_keep_size(edges, -deskew_angle, border_mode=cv2.BORDER_CONSTANT)

    return corrected_roi, edges, deskew_angle


def rotate_image_keep_size(image, angle_deg: float, border_mode=cv2.BORDER_CONSTANT):
    """Rotate an image around center while preserving original output size."""
    h, w = image.shape[:2]
    center = (w * 0.5, h * 0.5)
    matrix = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    return cv2.warpAffine(
        image,
        matrix,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=border_mode,
        borderValue=0,
    )


def estimate_deskew_angle_from_edges(edges):
    """Estimate dominant near-horizontal line angle from edge map."""
    h, w = edges.shape[:2]
    min_line_len = max(40, int(min(h, w) * DESKEW_MIN_LINE_LENGTH_RATIO))
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=math.pi / 180.0,
        threshold=70,
        minLineLength=min_line_len,
        maxLineGap=20,
    )
    if lines is None:
        return 0.0

    weighted_sum = 0.0
    weight_total = 0.0
    for segment in lines.reshape(-1, 4):
        x1, y1, x2, y2 = map(float, segment)
        dx = x2 - x1
        dy = y2 - y1
        length = math.hypot(dx, dy)
        if length < min_line_len:
            continue

        angle = math.degrees(math.atan2(dy, dx))
        while angle > 90.0:
            angle -= 180.0
        while angle < -90.0:
            angle += 180.0

        if abs(angle) > DESKEW_MAX_CANDIDATE_ANGLE_DEG:
            continue

        weighted_sum += angle * length
        weight_total += length

    if weight_total <= 1e-6:
        return 0.0

    dominant_angle = weighted_sum / weight_total
    if abs(dominant_angle) > DESKEW_MAX_APPLY_ANGLE_DEG:
        return 0.0
    return float(dominant_angle)


def fallback_quad_from_edges(image, edges):
    """Fallback: pick the largest plausible contour if quad inference is too strict."""
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    image_area = max(image.shape[0] * image.shape[1], 1)
    min_area = image_area * 0.028

    for contour in sorted(contours, key=cv2.contourArea, reverse=True):
        area = cv2.contourArea(contour)
        if area < min_area:
            continue
        rect = cv2.minAreaRect(contour)
        return cam.order_points(cv2.boxPoints(rect))
    return None


def passes_portrait_ratio_gate(quad, image_shape):
    """Apply stricter portrait-only MTG ratio filtering to reduce false detections."""
    metrics = cam.quad_geometry_metrics(quad, image_shape)
    aspect = metrics["aspect_ratio"]
    min_aspect = STRICT_MTG_ASPECT_RATIO - STRICT_MTG_ASPECT_TOLERANCE
    max_aspect = STRICT_MTG_ASPECT_RATIO + STRICT_MTG_ASPECT_TOLERANCE

    if aspect >= 1.0:
        return False, metrics
    if not (min_aspect <= aspect <= max_aspect):
        return False, metrics
    return True, metrics


def corner_angles_degrees(quad):
    """Return the 4 interior corner angles for an ordered quadrilateral."""
    import numpy as np

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
    """Ensure all 4 corners are approximately right angles."""
    angles = corner_angles_degrees(quad)
    min_allowed = 90.0 - CORNER_RIGHT_ANGLE_TOLERANCE_DEG
    max_allowed = 90.0 + CORNER_RIGHT_ANGLE_TOLERANCE_DEG
    ok = all(min_allowed <= a <= max_allowed for a in angles)
    return ok, angles


def build_topline_projection_from_quad(quad):
    """Build a debug top edge and projected card quad from a detected quad."""
    import numpy as np

    pts = cam.order_points(quad).astype("float32")
    tl, tr, br, bl = pts

    left_vec = bl - tl
    right_vec = br - tr
    left_len = float(max(1e-6, np.linalg.norm(left_vec)))
    right_len = float(max(1e-6, np.linalg.norm(right_vec)))
    side_len = (left_len + right_len) * 0.5

    left_dir = left_vec / left_len
    right_dir = right_vec / right_len
    side_dir = left_dir + right_dir
    side_norm = float(max(1e-6, np.linalg.norm(side_dir)))
    side_dir = side_dir / side_norm

    p_tl = tl
    p_tr = tr
    p_br = tr + side_dir * side_len
    p_bl = tl + side_dir * side_len
    projected = cam.order_points(np.array([p_tl, p_tr, p_br, p_bl], dtype="float32"))
    return p_tl, p_tr, projected.astype("int32")


def get_shifted_trimmed_roi(module, frame, shift_right_ratio: float, shift_down_ratio: float):
    _, roi_x1, roi_y1, roi_x2, roi_y2 = module.get_center_roi(frame)
    roi_width = max(roi_x2 - roi_x1, 1)
    roi_height = max(roi_y2 - roi_y1, 1)
    shift_px = int(roi_width * max(0.0, shift_right_ratio))
    shift_py = int(roi_height * shift_down_ratio)

    _frame_h, frame_w = frame.shape[:2]
    shifted_x1 = min(max(roi_x1 + shift_px, 0), max(frame_w - 2, 0))
    shifted_x2 = min(max(roi_x2 + shift_px, shifted_x1 + 1), frame_w)

    # Slightly widen ROI to include a bit more left/right border context.
    extra_x = max(1, int(roi_width * ROI_EXTRA_WIDTH_SIDE_RATIO))
    shifted_x1 = max(0, shifted_x1 - extra_x)
    shifted_x2 = min(frame_w, shifted_x2 + extra_x)

    shifted_y1 = max(0, roi_y1 + shift_py)
    shifted_y2 = max(shifted_y1 + 2, roi_y2 + shift_py)
    shifted_y1 = min(shifted_y1, frame.shape[0] - 2)
    shifted_y2 = min(max(shifted_y2, shifted_y1 + 1), frame.shape[0])

    frame_h = frame.shape[0]
    base_h = max(shifted_y2 - shifted_y1, 1)

    # Expand ROI height by a configurable amount and bias extra expansion downward.
    extra_height_px = max(0, int(base_h * max(0.0, ROI_EXTRA_HEIGHT_RATIO)))
    extra_bottom_px = max(0, int(base_h * max(0.0, ROI_EXTRA_BOTTOM_RATIO)))
    extra_top_px = extra_height_px // 2
    extra_height_bottom_px = extra_height_px - extra_top_px

    shifted_y1 = max(0, shifted_y1 - extra_top_px)
    shifted_y2 = min(frame_h, shifted_y2 + extra_height_bottom_px + extra_bottom_px)
    shifted_y2 = max(shifted_y1 + 1, shifted_y2)

    roi_height = max(shifted_y2 - shifted_y1, 1)
    trim_pixels = max(1, int(roi_height * cam.ROI_VERTICAL_TRIM_RATIO))
    trimmed_y1 = min(shifted_y2 - 1, shifted_y1 + trim_pixels)
    trimmed_y2 = max(trimmed_y1 + 1, shifted_y2 - trim_pixels)

    trimmed_roi = frame[trimmed_y1:trimmed_y2, shifted_x1:shifted_x2]
    return trimmed_roi, shifted_x1, trimmed_y1, shifted_x2, trimmed_y2


def main():
    parser = argparse.ArgumentParser(description="Run CamTesting with segmented top-half matcher v2.")
    parser.add_argument("--dry-run", action="store_true", help="Use dry-run motor controller.")
    parser.add_argument(
        "--roi-shift-right-ratio",
        type=float,
        default=DEFAULT_ROI_SHIFT_RIGHT_RATIO,
        help="Shift detection ROI to the right by this fraction of ROI width (default: 0.06).",
    )
    parser.add_argument(
        "--roi-shift-down-ratio",
        type=float,
        default=DEFAULT_ROI_SHIFT_DOWN_RATIO,
        help="Shift detection ROI down by this fraction of ROI height (default: 0.00).",
    )
    args = parser.parse_args()

    module = cam.load_source_module()
    cfg = SegmentedHashConfig()

    if not SEGMENTED_DB_PATH.exists():
        raise FileNotFoundError(
            f"Missing segmented DB: {SEGMENTED_DB_PATH}. Run build_segmented_tophalf_db_v2.py first."
        )

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
        print(f"Controller init failed on hardware ({exc}); falling back to dry-run controller.", flush=True)
        controller = module.DryRunController()
        controller.open()

    cap = cam.configure_camera(module)
    print(f"[v2] ROI shift right ratio: {args.roi_shift_right_ratio:.3f}", flush=True)
    print(f"[v2] ROI shift down ratio: {args.roi_shift_down_ratio:.3f}", flush=True)
    print("[v2] Live controls window enabled for edge + ROI tuning.", flush=True)

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
    last_servo_home_angle = None
    last_wait_reason = None
    last_wait_reason_time = 0.0
    interval = 0.6

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                continue

            live_ctrl = read_live_controls()

            # Live servo adjustment from controls window.
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
            module.draw_text_with_background(live_preview, "CamTesting v2 Segmented Matcher", (20, 35), font_scale=0.8, thickness=2)
            module.draw_text_with_background(live_preview, f"Last identified: {last_identified}", (20, 70), font_scale=0.7, thickness=2)
            module.draw_text_with_background(
                live_preview,
                (
                    f"Canny={live_ctrl['canny_low']}/{live_ctrl['canny_high']} "
                    f"CloseK={live_ctrl['close_kernel']} Dilate={live_ctrl['dilate_iters']} "
                    f"ROI X={live_ctrl['roi_shift_right_ratio']:.2f} Y={live_ctrl['roi_shift_down_ratio']:.2f} "
                    f"PTop={live_ctrl['perspective_top_shrink']:.2f} PBot={live_ctrl['perspective_bottom_grow']:.2f} "
                    f"ServoHome={live_ctrl['servo_home_angle']} Drop={live_ctrl['servo_drop_delta']}"
                ),
                (20, 100),
                font_scale=0.55,
                thickness=2,
            )
            module.draw_text_with_background(
                live_preview,
                "q=quit v=servo test r=drop+return (uses live ServoHome/ServoDrop)",
                (20, frame.shape[0] - 20),
                font_scale=0.6,
                thickness=2,
            )

            edge_vis = None
            warped_vis = None
            preprocessed_vis = None
            try:
                reject_reason = None
                processed, edges, deskew_angle = preprocess_for_v2(
                    roi_frame,
                    live_ctrl["canny_low"],
                    live_ctrl["canny_high"],
                    live_ctrl["close_kernel"],
                    live_ctrl["dilate_iters"],
                    live_ctrl["perspective_top_shrink"],
                    live_ctrl["perspective_bottom_grow"],
                )
                preprocessed_vis = processed
                edge_vis = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

                if abs(deskew_angle) >= DESKEW_MIN_APPLY_ANGLE_DEG:
                    module.draw_text_with_background(
                        live_preview,
                        f"Auto-level: {-deskew_angle:+.1f} deg",
                        (20, 155),
                        font_scale=0.55,
                        thickness=2,
                    )

                try:
                    debug_image, quad, _ = cam.detect_card_with_inference(processed, edges)
                    edge_vis = debug_image
                except Exception:
                    quad = fallback_quad_from_edges(processed, edges)
                    if quad is not None:
                        cv2.polylines(edge_vis, [quad.astype("int32")], True, (0, 255, 255), 2)

                if quad is not None:
                    angles_ok, angles = passes_right_angle_gate(quad)
                    if not angles_ok:
                        quad = None
                        reject_reason = (
                            "corner-angle reject: "
                            f"angles=[{angles[0]:.1f},{angles[1]:.1f},{angles[2]:.1f},{angles[3]:.1f}] "
                            f"target=90+/-{CORNER_RIGHT_ANGLE_TOLERANCE_DEG:.1f}"
                        )
                        module.draw_text_with_background(
                            live_preview,
                            reject_reason,
                            (20, 130),
                            font_scale=0.55,
                            thickness=2,
                            text_color=(0, 0, 255),
                        )

                debug_top_tl = None
                debug_top_tr = None
                debug_projected = None

                if quad is not None:
                    ratio_ok, ratio_metrics = passes_portrait_ratio_gate(quad, processed.shape)
                    if not ratio_ok:
                        quad = None
                        reject_reason = (
                            f"ratio gate reject: aspect={ratio_metrics['aspect_ratio']:.3f} "
                            f"target={STRICT_MTG_ASPECT_RATIO:.3f}+/-{STRICT_MTG_ASPECT_TOLERANCE:.3f}"
                        )
                        module.draw_text_with_background(
                            live_preview,
                            reject_reason,
                            (20, 130),
                            font_scale=0.55,
                            thickness=2,
                            text_color=(0, 0, 255),
                        )

                if quad is not None:
                    debug_top_tl, debug_top_tr, debug_projected = build_topline_projection_from_quad(quad)

                    cv2.line(
                        edge_vis,
                        (int(debug_top_tl[0]), int(debug_top_tl[1])),
                        (int(debug_top_tr[0]), int(debug_top_tr[1])),
                        (255, 255, 0),
                        2,
                    )
                    cv2.polylines(edge_vis, [debug_projected], True, (255, 0, 255), 2)

                    if preprocessed_vis is not None and preprocessed_vis.size > 0:
                        cv2.line(
                            preprocessed_vis,
                            (int(debug_top_tl[0]), int(debug_top_tl[1])),
                            (int(debug_top_tr[0]), int(debug_top_tr[1])),
                            (255, 255, 0),
                            2,
                        )
                        cv2.polylines(preprocessed_vis, [debug_projected], True, (255, 0, 255), 2)

                    top_a = (int(debug_top_tl[0] + roi_x1), int(debug_top_tl[1] + roi_y1))
                    top_b = (int(debug_top_tr[0] + roi_x1), int(debug_top_tr[1] + roi_y1))
                    cv2.line(live_preview, top_a, top_b, (255, 255, 0), 3)
                    projected_live = debug_projected.copy()
                    projected_live[:, 0] += roi_x1
                    projected_live[:, 1] += roi_y1
                    cv2.polylines(live_preview, [projected_live.astype("int32")], True, (255, 0, 255), 3)

                if quad is not None:
                    warped = warp_with_perspective_tuning(
                        processed,
                        quad,
                        top_shrink_ratio=1.0,
                        bottom_grow_ratio=1.0,
                    )
                    if warped is not None and warped.size > 0:
                        warped_vis = warped

                # Keep warped window alive even if no contour is found this frame.
                if warped_vis is None:
                    warped_vis = processed

                current_time = time.time()
                if current_time - last_identification_time >= interval:
                    if warped_vis is None or quad is None:
                        wait_reason = reject_reason or "no valid contour/quad yet"
                        if wait_reason != last_wait_reason or (current_time - last_wait_reason_time) >= 2.0:
                            print(f"[v2] waiting: {wait_reason}", flush=True)
                            last_wait_reason = wait_reason
                            last_wait_reason_time = current_time
                    else:
                        t_ident_start = time.perf_counter()
                        result = matcher.identify(warped)
                        ident_ms = (time.perf_counter() - t_ident_start) * 1000.0
                        last_identification_time = current_time

                        best = result.get("best")
                        if best is not None:
                            last_identified = best["name"]
                            best_set = best.get("set") or "?"
                            print("[v2] ---------------- MATCH ----------------", flush=True)
                            print(f"[v2] - identify time: {ident_ms:.0f} ms", flush=True)
                            print(f"[v2] - status: {result['status']}", flush=True)
                            print(f"[v2] - best match: {best['name']} [{best_set}]", flush=True)
                            print(f"[v2] - orientation: {best['orientation']}", flush=True)
                            print(f"[v2] - weighted score: {best['weighted_matches']:.2f}", flush=True)
                            print(f"[v2] - regions matched: {best['matched_regions']}/{best['total_regions']}", flush=True)
                            print(f"[v2] - stage1 distance: {best['stage1_distance']:.2f}", flush=True)
                            print(f"[v2] - avg region distance: {best['avg_region_distance']:.2f}", flush=True)
                            print(f"[v2] - final score: {best['final_score']:.2f}", flush=True)
                            possibles = result.get("possible", [])
                            if possibles:
                                print(f"[v2] - next closest (showing {min(len(possibles), 15)}):", flush=True)
                                for rank, p in enumerate(possibles[:15], start=1):
                                    p_set = p.get("set") or "?"
                                    print(
                                        f"[v2]   {rank}. {p['name']} [{p_set}] | w={p['weighted_matches']:.2f} | "
                                        f"f={p['final_score']:.2f} | {p['orientation']}",
                                        flush=True,
                                    )
                            print("[v2] --------------------------------------", flush=True)
                        else:
                            last_identified = "None"
                            print("[v2] ---------------- MATCH ----------------", flush=True)
                            print(f"[v2] - identify time: {ident_ms:.0f} ms", flush=True)
                            print("[v2] - status: reject", flush=True)
                            print(f"[v2] - reason: {result.get('reason')}", flush=True)
                            print("[v2] --------------------------------------", flush=True)

            except Exception as exc:
                edge_vis = cv2.cvtColor(cv2.Canny(cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY), 80, 200), cv2.COLOR_GRAY2BGR)
                warped_vis = roi_frame
                preprocessed_vis = roi_frame
                module.draw_text_with_background(live_preview, f"No card contour: {exc}", (20, 105), font_scale=0.6, thickness=2, text_color=(0, 0, 255))

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
                print(
                    f"Servo2 manual control: moving from {home_angle} to {down_angle}, "
                    "wait 1.0s, back to home.",
                    flush=True,
                )
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
