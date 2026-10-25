import argparse
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np


WINDOW_NAME = "YOLO Corner Annotator"
SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
REVIEW_MODES = ("all", "negative", "skipped", "nonpositive")


def order_points_clockwise(points: np.ndarray) -> np.ndarray:
    """Return points ordered as TL, TR, BR, BL."""
    pts = np.asarray(points, dtype=np.float32)
    sums = pts.sum(axis=1)
    diffs = pts[:, 1] - pts[:, 0]
    ordered = np.zeros((4, 2), dtype=np.float32)
    ordered[0] = pts[np.argmin(sums)]  # top-left
    ordered[2] = pts[np.argmax(sums)]  # bottom-right
    ordered[1] = pts[np.argmin(diffs)]  # top-right
    ordered[3] = pts[np.argmax(diffs)]  # bottom-left
    return ordered


def quad_area(points: np.ndarray) -> float:
    return float(abs(cv2.contourArea(points.astype(np.float32))))


def is_convex_quad(points: np.ndarray) -> bool:
    hull = cv2.convexHull(points.astype(np.float32))
    return len(hull) == 4


def normalize_points(points: np.ndarray, w: int, h: int) -> np.ndarray:
    norm = points.copy().astype(np.float32)
    norm[:, 0] /= float(max(w, 1))
    norm[:, 1] /= float(max(h, 1))
    norm = np.clip(norm, 0.0, 1.0)
    return norm


def yolo_bbox_from_quad(points: np.ndarray, w: int, h: int) -> Tuple[float, float, float, float]:
    xs = points[:, 0]
    ys = points[:, 1]
    x_min = float(np.clip(np.min(xs), 0, w - 1))
    x_max = float(np.clip(np.max(xs), 0, w - 1))
    y_min = float(np.clip(np.min(ys), 0, h - 1))
    y_max = float(np.clip(np.max(ys), 0, h - 1))

    bw = max(1.0, x_max - x_min)
    bh = max(1.0, y_max - y_min)
    cx = x_min + bw / 2.0
    cy = y_min + bh / 2.0

    return (
        cx / float(max(w, 1)),
        cy / float(max(h, 1)),
        bw / float(max(w, 1)),
        bh / float(max(h, 1)),
    )


def list_images(images_dir: Path) -> List[Path]:
    all_images = []
    for p in images_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS:
            all_images.append(p)
    all_images.sort()
    return all_images


def label_path_for(image_path: Path, images_dir: Path, labels_dir: Path) -> Path:
    rel = image_path.relative_to(images_dir)
    return (labels_dir / rel).with_suffix(".txt")


def read_label_text(path: Path) -> Optional[str]:
    if not path.exists():
        return None
    return path.read_text(encoding="utf-8").strip()


def get_annotation_status(
    image_path: Path,
    images_dir: Path,
    labels_obb_dir: Path,
    labels_bbox_dir: Path,
) -> str:
    obb_path = label_path_for(image_path, images_dir, labels_obb_dir)
    bbox_path = label_path_for(image_path, images_dir, labels_bbox_dir)
    obb_text = read_label_text(obb_path)
    bbox_text = read_label_text(bbox_path)

    if obb_text or bbox_text:
        return "positive"
    if obb_text == "" or bbox_text == "":
        return "negative"
    return "skipped"


def matches_review_mode(status: str, review_mode: str) -> bool:
    if review_mode == "all":
        return True
    if review_mode == "negative":
        return status == "negative"
    if review_mode == "skipped":
        return status == "skipped"
    if review_mode == "nonpositive":
        return status in {"negative", "skipped"}
    return True


def filter_images_for_review(
    images: List[Path],
    images_dir: Path,
    labels_obb_dir: Path,
    labels_bbox_dir: Path,
    review_mode: str,
) -> List[Path]:
    return [
        image_path
        for image_path in images
        if matches_review_mode(
            get_annotation_status(image_path, images_dir, labels_obb_dir, labels_bbox_dir),
            review_mode,
        )
    ]


def parse_saved_obb_points(
    image_path: Path,
    images_dir: Path,
    labels_obb_dir: Path,
) -> List[Tuple[int, int]]:
    obb_path = label_path_for(image_path, images_dir, labels_obb_dir)
    obb_text = read_label_text(obb_path)
    if not obb_text:
        return []

    tokens = obb_text.split()
    if len(tokens) != 9:
        return []

    try:
        values = [float(token) for token in tokens[1:]]
    except ValueError:
        return []

    img = cv2.imread(str(image_path))
    if img is None:
        return []
    h, w = img.shape[:2]

    points = []
    for idx in range(0, len(values), 2):
        x = int(round(np.clip(values[idx], 0.0, 1.0) * max(w - 1, 1)))
        y = int(round(np.clip(values[idx + 1], 0.0, 1.0) * max(h - 1, 1)))
        points.append((x, y))
    return points


def count_statuses(
    images: List[Path],
    images_dir: Path,
    labels_obb_dir: Path,
    labels_bbox_dir: Path,
) -> Tuple[int, int, int]:
    positives = 0
    negatives = 0
    skipped = 0
    for image_path in images:
        status = get_annotation_status(image_path, images_dir, labels_obb_dir, labels_bbox_dir)
        if status == "positive":
            positives += 1
        elif status == "negative":
            negatives += 1
        else:
            skipped += 1
    return positives, negatives, skipped


def ensure_parent(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)


def save_labels(
    image_path: Path,
    images_dir: Path,
    labels_obb_dir: Path,
    labels_bbox_dir: Path,
    class_id: int,
    quad: Optional[np.ndarray],
):
    obb_path = label_path_for(image_path, images_dir, labels_obb_dir)
    bbox_path = label_path_for(image_path, images_dir, labels_bbox_dir)
    ensure_parent(obb_path)
    ensure_parent(bbox_path)

    if quad is None:
        obb_path.write_text("", encoding="utf-8")
        bbox_path.write_text("", encoding="utf-8")
        return obb_path, bbox_path

    img = cv2.imread(str(image_path))
    if img is None:
        raise RuntimeError(f"Could not read image for saving labels: {image_path}")
    h, w = img.shape[:2]

    ordered = order_points_clockwise(quad)
    ordered_norm = normalize_points(ordered, w, h)
    obb_vals = " ".join(f"{v:.6f}" for v in ordered_norm.reshape(-1))
    obb_line = f"{class_id} {obb_vals}\n"
    obb_path.write_text(obb_line, encoding="utf-8")

    x, y, bw, bh = yolo_bbox_from_quad(ordered, w, h)
    bbox_line = f"{class_id} {x:.6f} {y:.6f} {bw:.6f} {bh:.6f}\n"
    bbox_path.write_text(bbox_line, encoding="utf-8")

    return obb_path, bbox_path


def print_help():
    print("\n=== YOLO Corner Annotator Controls ===")
    print("Mouse Left Click  : Add corner point (4 points max)")
    print("u                 : Undo last point")
    print("c                 : Clear all points")
    print("s                 : Save current labels")
    print("n                 : Mark as negative (empty labels) and next")
    print("SPACE             : Save and next")
    print("a                 : Previous image")
    print("d                 : Next image")
    print("h                 : Print this help")
    print("q or ESC          : Quit")
    print("======================================\n")


class AnnotatorState:
    def __init__(self):
        self.points: List[Tuple[int, int]] = []


def draw_overlay(base_img: np.ndarray, points: List[Tuple[int, int]], status_text: str) -> np.ndarray:
    vis = base_img.copy()

    for idx, (x, y) in enumerate(points):
        cv2.circle(vis, (x, y), 5, (0, 255, 255), -1)
        cv2.putText(vis, str(idx + 1), (x + 8, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    if len(points) >= 2:
        poly_pts = np.array(points, dtype=np.int32)
        cv2.polylines(vis, [poly_pts], False, (255, 200, 0), 2)

    if len(points) == 4:
        quad = order_points_clockwise(np.array(points, dtype=np.float32))
        quad_i = quad.astype(np.int32)
        cv2.polylines(vis, [quad_i], True, (0, 255, 0), 2)

        x_min = int(np.min(quad[:, 0]))
        x_max = int(np.max(quad[:, 0]))
        y_min = int(np.min(quad[:, 1]))
        y_max = int(np.max(quad[:, 1]))
        cv2.rectangle(vis, (x_min, y_min), (x_max, y_max), (0, 180, 255), 2)

    cv2.putText(vis, status_text, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(vis, "h=help  s=save  n=negative  space=save+next  a/d=prev/next  q=quit", (12, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    return vis


def main():
    parser = argparse.ArgumentParser(description="4-corner annotation tool for YOLO OBB + YOLO bbox.")
    parser.add_argument("--images-dir", required=True, type=str, help="Directory containing images to annotate.")
    parser.add_argument("--labels-obb-dir", type=str, default="yolo_dataset/labels_obb/train", help="Output directory for YOLO OBB labels.")
    parser.add_argument("--labels-bbox-dir", type=str, default="yolo_dataset/labels/train", help="Output directory for standard YOLO bbox labels.")
    parser.add_argument("--class-id", type=int, default=0, help="Class id for card labels.")
    parser.add_argument("--min-quad-area-ratio", type=float, default=0.01, help="Reject quads with area ratio below this.")
    parser.add_argument("--review", choices=REVIEW_MODES, default="all", help="Filter review set by current saved label state.")
    args = parser.parse_args()

    images_dir = Path(args.images_dir).resolve()
    labels_obb_dir = Path(args.labels_obb_dir).resolve()
    labels_bbox_dir = Path(args.labels_bbox_dir).resolve()

    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")

    all_images = list_images(images_dir)
    if not all_images:
        raise RuntimeError(f"No images found in {images_dir}")

    images = filter_images_for_review(all_images, images_dir, labels_obb_dir, labels_bbox_dir, args.review)
    if not images:
        raise RuntimeError(f"No images matched review mode '{args.review}' in {images_dir}")

    print_help()
    positives, negatives, skipped = count_statuses(all_images, images_dir, labels_obb_dir, labels_bbox_dir)
    print(f"Loaded {len(images)} images from {images_dir} (review={args.review})")
    print(f"Current status counts -> positive={positives} negative={negatives} skipped={skipped}")
    print(f"OBB labels -> {labels_obb_dir}")
    print(f"BBox labels -> {labels_bbox_dir}")

    state = AnnotatorState()
    idx = 0

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    def load_points_for_current_image():
        state.points = parse_saved_obb_points(images[idx], images_dir, labels_obb_dir)

    def refresh_review_images(old_idx: int):
        nonlocal images, idx
        if args.review == "all":
            return
        images = filter_images_for_review(all_images, images_dir, labels_obb_dir, labels_bbox_dir, args.review)
        if not images:
            print(f"Review set '{args.review}' is complete.")
            cv2.destroyAllWindows()
            raise SystemExit(0)
        idx = min(old_idx, len(images) - 1)
        load_points_for_current_image()

    load_points_for_current_image()

    def on_mouse(event, x, y, flags, param):
        del flags, param
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(state.points) < 4:
                state.points.append((int(x), int(y)))
            else:
                print("Already have 4 points. Press c to clear or u to undo.")

    cv2.setMouseCallback(WINDOW_NAME, on_mouse)

    while True:
        image_path = images[idx]
        img = cv2.imread(str(image_path))
        if img is None:
            print(f"Skipping unreadable image: {image_path}")
            idx = (idx + 1) % len(images)
            state.points.clear()
            continue

        h, w = img.shape[:2]
        saved_status = get_annotation_status(image_path, images_dir, labels_obb_dir, labels_bbox_dir)
        status = f"{idx + 1}/{len(images)} | {image_path.name} | saved={saved_status} | points={len(state.points)}"
        vis = draw_overlay(img, state.points, status)
        cv2.imshow(WINDOW_NAME, vis)

        key = cv2.waitKey(20) & 0xFF

        if key in (ord("q"), 27):
            break

        if key == ord("h"):
            print_help()

        elif key == ord("u"):
            if state.points:
                state.points.pop()

        elif key == ord("c"):
            state.points.clear()

        elif key == ord("a"):
            idx = (idx - 1) % len(images)
            load_points_for_current_image()

        elif key == ord("d"):
            idx = (idx + 1) % len(images)
            load_points_for_current_image()

        elif key == ord("n"):
            old_idx = idx
            obb_path, bbox_path = save_labels(
                image_path,
                images_dir,
                labels_obb_dir,
                labels_bbox_dir,
                args.class_id,
                quad=None,
            )
            print(f"NEG saved: {obb_path} | {bbox_path}")
            idx = (idx + 1) % len(images)
            load_points_for_current_image()
            refresh_review_images(old_idx)

        elif key == ord("s") or key == ord(" "):
            if len(state.points) != 4:
                print("Need exactly 4 points for a card label. Use 'n' to mark a negative.")
                continue

            quad = order_points_clockwise(np.array(state.points, dtype=np.float32))

            if not is_convex_quad(quad):
                print("Rejected: points are not a convex quad.")
                continue

            area_ratio = quad_area(quad) / float(max(h * w, 1))
            if area_ratio < args.min_quad_area_ratio:
                print(f"Rejected: quad area ratio too small ({area_ratio:.4f}).")
                continue

            old_idx = idx
            obb_path, bbox_path = save_labels(
                image_path,
                images_dir,
                labels_obb_dir,
                labels_bbox_dir,
                args.class_id,
                quad,
            )
            print(f"Saved: {obb_path} | {bbox_path}")

            if key == ord(" "):
                idx = (idx + 1) % len(images)
                load_points_for_current_image()
            refresh_review_images(old_idx)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
