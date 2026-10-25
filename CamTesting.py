import importlib.util
import json
import os
import sys
import time
from collections import Counter, deque
from pathlib import Path

import cv2


WORKSPACE_DIR = Path(__file__).resolve().parent
SOURCE_PATH = WORKSPACE_DIR / "Expo_P.H.A.S.E.py"
WINDOW_LIVE = "CamTesting Live Feed"
WINDOW_EDGES = "CamTesting Edges"
WINDOW_WARPED = "CamTesting Warped Card"
CARD_CROP_EXPANSION_RATIO = 0.11
EXPANSION_VERTICAL_BIAS_RATIO = 0.015
ROI_VERTICAL_TRIM_RATIO = 0.02
ENABLE_LENS_CORRECTION = True
LENS_CORRECTION_K1 = -0.03
LENS_CORRECTION_K2 = 0.008
LENS_FOCAL_SCALE = 1.00
MTG_CARD_ASPECT_RATIO = 63.0 / 88.0
ASPECT_RATIO_TOLERANCE = 0.04
OPPOSITE_SIDE_TOLERANCE = 0.18
CARD_AREA_MIN_RATIO = 0.12
CARD_AREA_MAX_RATIO = 0.85
LIVE_CONFIDENT_DISTANCE_MAX = 5.0
LIVE_MARGIN_MIN = 1.5
LIVE_HISTORY_SIZE = 8
LIVE_STABLE_MIN_COUNT = 4
LIVE_STABLE_MIN_FRACTION = 0.60


def load_source_module():
    spec = importlib.util.spec_from_file_location("exfcc_copy_module", SOURCE_PATH)
    module = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        raise RuntimeError(f"Could not load source module from {SOURCE_PATH}")
    spec.loader.exec_module(module)
    return module


def preprocess_frame(module, frame):
    image = module.resize_to_scryfall_normal(frame)
    blurred = cv2.GaussianBlur(image, (3, 3), 0)
    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    gray_enhanced = clahe.apply(gray)
    edges = cv2.Canny(gray_enhanced, 80, 200)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges = cv2.dilate(edges, kernel, iterations=1)
    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, close_kernel, iterations=1)
    return image, gray_enhanced, edges


def apply_lens_distortion_correction(frame):
    import numpy as np

    if not ENABLE_LENS_CORRECTION:
        return frame

    height, width = frame.shape[:2]
    fx = width * LENS_FOCAL_SCALE
    fy = height * LENS_FOCAL_SCALE
    cx = width * 0.5
    cy = height * 0.5

    camera_matrix = np.array(
        [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    dist_coeffs = np.array([LENS_CORRECTION_K1, LENS_CORRECTION_K2, 0.0, 0.0, 0.0], dtype=np.float32)
    return cv2.undistort(frame, camera_matrix, dist_coeffs)


def order_points(pts):
    pts = module_np_array(pts)
    sums = pts.sum(axis=1)
    diffs = pts[:, 1] - pts[:, 0]
    return pts[[sums.argmin(), diffs.argmin(), sums.argmax(), diffs.argmax()]]


def module_np_array(pts):
    import numpy as np

    return np.asarray(pts, dtype="float32")


def expand_quad(quad, image_shape, expansion_ratio=CARD_CROP_EXPANSION_RATIO):
    ordered = order_points(quad)
    center = ordered.mean(axis=0)
    expanded = center + (ordered - center) * (1.0 + expansion_ratio)
    height, width = image_shape[:2]
    expanded[:, 0] = expanded[:, 0].clip(0, width - 1)
    expanded[:, 1] = expanded[:, 1].clip(0, height - 1)
    return order_points(expanded)


def rotate_points(points, angle_radians, center):
    import numpy as np

    points = module_np_array(points)
    center = module_np_array(center)
    cos_a = np.cos(angle_radians)
    sin_a = np.sin(angle_radians)
    rotation = np.array([[cos_a, -sin_a], [sin_a, cos_a]], dtype="float32")
    return (points - center) @ rotation.T + center


def level_quad_for_expansion(quad):
    """Rotate quad so bottom edge is horizontal before expansion, then return both frames."""
    import numpy as np

    ordered = order_points(quad)
    center = ordered.mean(axis=0)
    bottom_left = ordered[3]
    bottom_right = ordered[2]
    edge_vec = bottom_right - bottom_left
    if np.linalg.norm(edge_vec) < 1e-6:
        return ordered, ordered, center, 0.0

    angle = np.arctan2(edge_vec[1], edge_vec[0])
    leveled = order_points(rotate_points(ordered, -angle, center))
    return ordered, leveled, center, angle


def correct_perspective_quad(quad, top_shrink_ratio=1.01, bottom_grow_ratio=0.99):
    """Correct perspective distortion by shrinking top and growing bottom edge."""
    import numpy as np

    ordered = order_points(quad)
    top_left = ordered[0]
    top_right = ordered[1]
    bottom_right = ordered[2]
    bottom_left = ordered[3]

    top_mid = (top_left + top_right) * 0.5
    bottom_mid = (bottom_right + bottom_left) * 0.5

    corrected_top_left = top_mid + (top_left - top_mid) * top_shrink_ratio
    corrected_top_right = top_mid + (top_right - top_mid) * top_shrink_ratio
    corrected_bottom_left = bottom_mid + (bottom_left - bottom_mid) * bottom_grow_ratio
    corrected_bottom_right = bottom_mid + (bottom_right - bottom_mid) * bottom_grow_ratio

    return np.array(
        [corrected_top_left, corrected_top_right, corrected_bottom_right, corrected_bottom_left],
        dtype="float32",
    )


def quad_geometry_metrics(quad, image_shape):
    import numpy as np

    ordered = order_points(quad)
    top = float(np.linalg.norm(ordered[1] - ordered[0]))
    right = float(np.linalg.norm(ordered[2] - ordered[1]))
    bottom = float(np.linalg.norm(ordered[2] - ordered[3]))
    left = float(np.linalg.norm(ordered[3] - ordered[0]))

    width = max((top + bottom) * 0.5, 1e-6)
    height = max((left + right) * 0.5, 1e-6)
    aspect_ratio = width / height

    top_bottom_delta = abs(top - bottom) / max(width, 1e-6)
    left_right_delta = abs(left - right) / max(height, 1e-6)

    quad_area = float(cv2.contourArea(ordered.astype("float32")))
    image_area = float(max(image_shape[0] * image_shape[1], 1))
    area_ratio = quad_area / image_area

    return {
        "aspect_ratio": aspect_ratio,
        "top_bottom_delta": top_bottom_delta,
        "left_right_delta": left_right_delta,
        "area_ratio": area_ratio,
    }


def is_valid_mtg_card_geometry(metrics):
    min_aspect = MTG_CARD_ASPECT_RATIO - ASPECT_RATIO_TOLERANCE
    max_aspect = MTG_CARD_ASPECT_RATIO + ASPECT_RATIO_TOLERANCE
    if not (min_aspect <= metrics["aspect_ratio"] <= max_aspect):
        return False
    if metrics["top_bottom_delta"] > OPPOSITE_SIDE_TOLERANCE:
        return False
    if metrics["left_right_delta"] > OPPOSITE_SIDE_TOLERANCE:
        return False
    if not (CARD_AREA_MIN_RATIO <= metrics["area_ratio"] <= CARD_AREA_MAX_RATIO):
        return False
    return True


def get_stable_live_prediction(history):
    if len(history) < LIVE_STABLE_MIN_COUNT:
        return None

    names = [entry["name"] for entry in history]
    counts = Counter(names)
    top_name, top_count = counts.most_common(1)[0]
    fraction = top_count / len(history)
    if fraction < LIVE_STABLE_MIN_FRACTION:
        return None
    return top_name


def infer_fourth_corner_from_three(points):
    import numpy as np

    points = np.asarray(points, dtype="float32")
    if len(points) != 3:
        return None

    best_quad = None
    best_score = -1.0

    for anchor_idx in range(3):
        anchor = points[anchor_idx]
        others = [points[idx] for idx in range(3) if idx != anchor_idx]
        vec_a = others[0] - anchor
        vec_b = others[1] - anchor
        norm_a = np.linalg.norm(vec_a)
        norm_b = np.linalg.norm(vec_b)
        if norm_a < 1e-3 or norm_b < 1e-3:
            continue

        cosine = abs(np.dot(vec_a, vec_b) / (norm_a * norm_b))
        right_angle_score = 1.0 - cosine
        if right_angle_score < 0.55:
            continue

        inferred = others[0] + others[1] - anchor
        quad = order_points(np.vstack([anchor, others[0], inferred, others[1]]))
        score = right_angle_score * min(norm_a, norm_b)
        if score > best_score:
            best_score = score
            best_quad = quad

    return best_quad


def detect_card_with_inference(image, edges):
    import numpy as np

    min_area = image.shape[0] * image.shape[1] * 0.03
    detection_passes = [
        {"kernel": (5, 5), "eps": 0.016},
        {"kernel": (5, 5), "eps": 0.020},
        {"kernel": (7, 7), "eps": 0.024},
    ]

    all_candidates = []
    fallback_contours = []

    for pass_index, pass_cfg in enumerate(detection_passes):
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, pass_cfg["kernel"])
        edges_closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=1)
        contours, _ = cv2.findContours(edges_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue

        fallback_contours.extend(contours)
        for contour in sorted(contours, key=cv2.contourArea, reverse=True):
            area = cv2.contourArea(contour)
            if area < min_area:
                continue

            perimeter = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, pass_cfg["eps"] * perimeter, True)
            rect = cv2.minAreaRect(contour)
            rect_box = order_points(cv2.boxPoints(rect))
            candidate_quad = rect_box

            if len(approx) == 4:
                candidate_quad = order_points(approx.reshape(4, 2))
            elif len(approx) == 3:
                inferred_quad = infer_fourth_corner_from_three(approx.reshape(3, 2))
                if inferred_quad is not None:
                    candidate_quad = inferred_quad

            x, y, w, h = cv2.boundingRect(candidate_quad.astype(np.int32))
            aspect_ratio = w / h if h else 0
            if not 0.45 <= aspect_ratio <= 0.95:
                continue

            rect_area = max(w * h, 1)
            fill_ratio = area / rect_area
            if fill_ratio < 0.45:
                continue

            geom_metrics = quad_geometry_metrics(candidate_quad, image.shape)
            if not is_valid_mtg_card_geometry(geom_metrics):
                continue

            corner_score = 1.0 if len(approx) == 4 else 0.9 if len(approx) == 3 else 0.75 if len(approx) in (5, 6) else 0.4
            geom_score = max(0.0, 1.0 - (abs(geom_metrics["aspect_ratio"] - MTG_CARD_ASPECT_RATIO) / ASPECT_RATIO_TOLERANCE))
            quality = fill_ratio * corner_score * (0.6 + 0.4 * geom_score)
            all_candidates.append(
                {
                    "quad": candidate_quad,
                    "area": area,
                    "quality": quality,
                    "pass_index": pass_index,
                    "metrics": geom_metrics,
                }
            )

    selected_quad = None
    if all_candidates:
        best_quality = max(candidate["quality"] for candidate in all_candidates)
        good_candidates = [
            candidate for candidate in all_candidates if candidate["quality"] >= best_quality * 0.90
        ]
        selected = min(good_candidates, key=lambda candidate: (candidate["area"], candidate["pass_index"]))
        selected_quad = selected["quad"]
        selected_metrics = selected["metrics"]
    elif fallback_contours:
        selected_metrics = None
        for contour in sorted(fallback_contours, key=cv2.contourArea, reverse=True):
            rect = cv2.minAreaRect(contour)
            rect_quad = order_points(cv2.boxPoints(rect))
            geom_metrics = quad_geometry_metrics(rect_quad, image.shape)
            if is_valid_mtg_card_geometry(geom_metrics):
                selected_quad = rect_quad
                selected_metrics = geom_metrics
                break
        if selected_quad is None:
            raise ValueError("No valid MTG-like card geometry found.")
    else:
        raise ValueError("No contours found while detecting card.")

    debug_image = image.copy()
    cv2.polylines(debug_image, [selected_quad.astype("int32")], True, (0, 255, 0), 3)
    return debug_image, selected_quad, selected_metrics


def rotate_and_crop_image_local(image, contour):
    import numpy as np

    perspective_corrected = correct_perspective_quad(contour)
    _, leveled_box, center, level_angle = level_quad_for_expansion(perspective_corrected)
    expanded_leveled_box = expand_quad(leveled_box, image.shape)

    # Shift expanded crop down slightly to reduce extra top space and recover bottom border.
    height_a = np.linalg.norm(leveled_box[1] - leveled_box[2])
    height_b = np.linalg.norm(leveled_box[0] - leveled_box[3])
    card_height = max(height_a, height_b, 1.0)
    vertical_bias = card_height * EXPANSION_VERTICAL_BIAS_RATIO
    expanded_leveled_box[:, 1] = expanded_leveled_box[:, 1] + vertical_bias
    expanded_leveled_box[:, 1] = expanded_leveled_box[:, 1].clip(0, image.shape[0] - 1)
    expanded_leveled_box = order_points(expanded_leveled_box)

    expanded_box = order_points(rotate_points(expanded_leveled_box, level_angle, center))

    width_a = np.linalg.norm(expanded_box[2] - expanded_box[3])
    width_b = np.linalg.norm(expanded_box[1] - expanded_box[0])
    max_width = int(max(width_a, width_b))

    height_a = np.linalg.norm(expanded_box[1] - expanded_box[2])
    height_b = np.linalg.norm(expanded_box[0] - expanded_box[3])
    max_height = int(max(height_a, height_b))

    dst = module_np_array([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1],
    ])

    matrix = cv2.getPerspectiveTransform(expanded_box, dst)
    warped = cv2.warpPerspective(image, matrix, (max_width, max_height))
    return warped


def configure_camera(module):
    print("Opening camera for CamTesting...", flush=True)
    cap = cv2.VideoCapture(module.CAMERA_INDEX)
    if not cap.isOpened():
        raise RuntimeError("Could not open camera")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 3840)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 3032)
    cap.set(cv2.CAP_PROP_BRIGHTNESS, 130)
    cap.set(cv2.CAP_PROP_CONTRAST, 10)
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)

    time.sleep(2)
    for _ in range(10):
        cap.read()

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Camera ready at {width}x{height}", flush=True)
    return cap


def get_trimmed_center_roi(module, frame):
    roi_frame, roi_x1, roi_y1, roi_x2, roi_y2 = module.get_center_roi(frame)
    roi_height = max(roi_y2 - roi_y1, 1)
    trim_pixels = max(1, int(roi_height * ROI_VERTICAL_TRIM_RATIO))

    trimmed_y1 = min(roi_y2 - 1, roi_y1 + trim_pixels)
    trimmed_y2 = max(trimmed_y1 + 1, roi_y2 - trim_pixels)

    trimmed_roi = frame[trimmed_y1:trimmed_y2, roi_x1:roi_x2]
    return trimmed_roi, roi_x1, trimmed_y1, roi_x2, trimmed_y2


def save_current_roi(module, frame):
    cropped, roi_x1, roi_y1, roi_x2, roi_y2 = get_trimmed_center_roi(module, frame)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    capture_path = WORKSPACE_DIR / "Captured_Cards" / f"camtest_{timestamp}.jpg"
    cv2.imwrite(str(capture_path), cropped)
    return capture_path, cropped


def run_servo2_test(controller):
    print("Running servo2 test: home 140, down to 0, hold 0.5 second, return to 140.", flush=True)
    controller.servo2_move_abs(140)
    controller.servo2_move_abs(0)
    time.sleep(0.5)
    controller.servo2_move_abs(140)


def identify_warped_card_live(module, warped_image, annoy_index, idx_to_card):
    """Identify a warped card by always selecting the smallest available Hamming distance."""
    rotated_image = cv2.rotate(warped_image, cv2.ROTATE_180)
    upright_matches = module.annoy_nearest_phash(warped_image, annoy_index, idx_to_card, n=5)
    rotated_matches = module.annoy_nearest_phash(rotated_image, annoy_index, idx_to_card, n=5)

    candidates = []
    for card_info, distance in upright_matches:
        candidates.append((card_info, "upright", float(distance)))
    for card_info, distance in rotated_matches:
        candidates.append((card_info, "rotated 180", float(distance)))

    if not candidates:
        return {
            "card_info": None,
            "orientation": None,
            "best_distance": None,
            "second_distance": None,
        }

    candidates.sort(key=lambda item: item[2])
    best_card, best_orientation, best_distance = candidates[0]
    second_distance = candidates[1][2] if len(candidates) > 1 else None
    return {
        "card_info": best_card,
        "orientation": best_orientation,
        "best_distance": best_distance,
        "second_distance": second_distance,
    }


def main():
    module = load_source_module()
    dry_run_mode = "--dry-run" in sys.argv
    print(f"CamTesting dry run mode: {dry_run_mode}", flush=True)
    print("Controls: q=quit, r=recognize current ROI, v=test servo2", flush=True)

    controller = module.DryRunController() if dry_run_mode else module.StepperController(port=module.SERIAL_PORT)
    try:
        controller.open()
        controller.servo2_move(35)
        time.sleep(0.5)
    except Exception as exc:
        if dry_run_mode:
            raise
        print(f"Controller init failed on hardware ({exc}); falling back to dry-run controller.", flush=True)
        controller = module.DryRunController()
        controller.open()


    with open(WORKSPACE_DIR / "scryfall_phash_db_sample.json", "r", encoding="utf-8") as handle:
        card_hash_db = json.load(handle)
    annoy_index, idx_to_card = module.load_or_build_annoy_index(card_hash_db)

    cap = configure_camera(module)

    cv2.namedWindow(WINDOW_LIVE, cv2.WINDOW_NORMAL)
    cv2.namedWindow(WINDOW_EDGES, cv2.WINDOW_NORMAL)
    cv2.namedWindow(WINDOW_WARPED, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_LIVE, 1100, 750)
    cv2.resizeWindow(WINDOW_EDGES, 900, 700)
    cv2.resizeWindow(WINDOW_WARPED, 500, 700)
    last_identified_card = "None"
    last_live_identification_time = 0.0
    live_identification_interval = 0.8
    last_reported_card = None
    live_history = deque(maxlen=LIVE_HISTORY_SIZE)
    last_stable_card = "None"

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                continue

            frame = apply_lens_distortion_correction(frame)

            roi_frame, roi_x1, roi_y1, roi_x2, roi_y2 = get_trimmed_center_roi(module, frame)
            live_preview = frame.copy()
            cv2.rectangle(live_preview, (roi_x1, roi_y1), (roi_x2, roi_y2), (0, 255, 255), 3)
            module.draw_text_with_background(live_preview, "CamTesting Live View", (20, 35), font_scale=0.8, thickness=2)
            module.draw_text_with_background(live_preview, f"Last identified: {last_identified_card}", (20, 70), font_scale=0.7, thickness=2)
            module.draw_text_with_background(live_preview, "q=quit r=recognize ROI v=servo2 test", (20, frame.shape[0] - 20), font_scale=0.6, thickness=2)

            edge_vis = None
            warped_vis = None
            try:
                processed, _, edges = preprocess_frame(module, roi_frame)
                debug_image, quad, geometry_metrics = detect_card_with_inference(processed, edges)
                warped = rotate_and_crop_image_local(processed, quad)
                edge_vis = debug_image
                warped_vis = warped
                module.draw_text_with_background(live_preview, "Card contour found", (20, 105), font_scale=0.7, thickness=2, text_color=(0, 255, 0))

                current_time = time.time()
                if current_time - last_live_identification_time >= live_identification_interval:
                    match_result = identify_warped_card_live(module, warped, annoy_index, idx_to_card)
                    last_live_identification_time = current_time

                    card_info = match_result["card_info"]
                    orientation = match_result["orientation"]
                    distance = match_result["best_distance"]
                    second_distance = match_result["second_distance"]

                    if card_info:
                        candidate_name = card_info.get("name", "Unknown")
                        margin = second_distance - distance if second_distance is not None else 999.0
                        is_confident = distance <= LIVE_CONFIDENT_DISTANCE_MAX and margin >= LIVE_MARGIN_MIN

                        if is_confident:
                            live_history.append({"name": candidate_name, "distance": distance})
                            stable_name = get_stable_live_prediction(live_history)
                            if stable_name is not None:
                                last_stable_card = stable_name
                                last_identified_card = last_stable_card
                            else:
                                last_identified_card = candidate_name
                        else:
                            last_identified_card = candidate_name

                        if last_identified_card != last_reported_card:
                            print(
                                f"Live identified card: {last_identified_card} | orientation={orientation} | distance={distance} | margin={margin:.2f} | aspect={geometry_metrics['aspect_ratio']:.3f}",
                                flush=True,
                            )
                            last_reported_card = last_identified_card
                    else:
                        last_identified_card = "None"
            except Exception as exc:
                edge_vis = cv2.cvtColor(cv2.Canny(cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY), 80, 200), cv2.COLOR_GRAY2BGR)
                warped_vis = None
                module.draw_text_with_background(live_preview, f"No card contour: {exc}", (20, 105), font_scale=0.6, thickness=2, text_color=(0, 0, 255))

            cv2.imshow(WINDOW_LIVE, live_preview)
            if edge_vis is not None:
                cv2.imshow(WINDOW_EDGES, edge_vis)
            if warped_vis is not None and warped_vis.size > 0:
                cv2.imshow(WINDOW_WARPED, warped_vis)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                print("CamTesting closed by user.", flush=True)
                break
            if key == ord("v"):
                run_servo2_test(controller)
            if key == ord("r"):
                print("Servo2 manual control: home 140, down to 0, wait 0.5s, back to 140.", flush=True)
                controller.servo2_move_abs(140)
                controller.servo2_move_abs(0)
                time.sleep(0.5)
                controller.servo2_move_abs(140)
                print("Servo2 manual control complete.", flush=True)
    finally:
        cap.release()
        controller.close()
        cv2.destroyAllWindows()
        print("CamTesting closed.", flush=True)


if __name__ == "__main__":
    main()
