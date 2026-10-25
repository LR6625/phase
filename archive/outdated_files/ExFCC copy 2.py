"""
Magic: The Gathering Card Recognition and Motor Control System
================================================================
This script is the combination of the ipynb and the allmotors,py files, to be tested as of now:
1. Motor control functionality (StepperController from AllMotors.py)
2. Card image recognition system (from Main copy.ipynb)

All image operations use OpenCV/NumPy vectorized code for optimal performance.
Hash database search uses Annoy for fast nearest-neighbor queries.
"""
print("hello world", flush=True)
# ==============================================================================
# IMPORTS
# ==============================================================================
from itertools import cycle
import time
from datetime import datetime
from typing import Optional, Literal
from pathlib import Path
import importlib.util
import msvcrt
import sys
import serial
import threading
try:
    import torch
except Exception:
    torch = None

# Image processing and ML libraries
import cv2
import numpy as np
import os
import imagehash
from PIL import Image
import json
import requests
import annoy
import re
import gradio as gr
import glob
import shutil
from annoy import AnnoyIndex
from ultralytics import YOLO
from mtg_hash_v2.config import SegmentedHashConfig
from mtg_hash_v2.matcher import HybridTopHalfMatcher
import YOLO_Camera_Studio as ycs


def _load_yolo_locator_module():
    """Load the local pipeline copy explicitly so ExFCC uses the latest mask/axis logic."""
    pipeline_path = Path(__file__).resolve().parent / "test_yolo_gridhash_pipeline copy.py"
    if not pipeline_path.exists():
        raise FileNotFoundError(f"Pipeline module not found: {pipeline_path}")

    spec = importlib.util.spec_from_file_location("test_yolo_gridhash_pipeline_copy", str(pipeline_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module spec from: {pipeline_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(spec.name, None)
        raise
    return module


yolo_locator = _load_yolo_locator_module()


# ==============================================================================
# STEPPER CONTROLLER CLASS (from AllMotors.py)
# ==============================================================================
class StepperController:
    def __init__(self, port: str, baudrate: int = 9600, timeout: float = 120.0) -> None:
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self._ser: Optional[serial.Serial] = None
        self._io_lock = threading.Lock()

    def open(self) -> None:
        if self._ser and self._ser.is_open:
            return
        self._ser = serial.Serial(self.port, self.baudrate, timeout=self.timeout)
        time.sleep(2.0)
        self._ser.reset_input_buffer()
        self._ser.reset_output_buffer()

    def close(self) -> None:
        if self._ser and self._ser.is_open:
            self._ser.close()

    def _send(self, line: str) -> str:
        with self._io_lock:
            if not self._ser or not self._ser.is_open:
                self.open()
            clean_line = line.strip()
            command = clean_line + "\n"
            is_switch_query = clean_line.upper() == "GET_SWITCH"
            if not is_switch_query:
                print(f"> Sending: {clean_line}", flush=True)
            self._ser.write(command.encode("utf-8"))
            response = self._ser.readline().decode("utf-8").strip()
            if not is_switch_query:
                print(f"< Received: {response}", flush=True)
            return response

    def _spin(self, direction: Literal["CW", "CCW"], motor: Literal["M1", "M2", "BOTH"], steps: int) -> str:
        if steps < 0:
            steps = -steps
        return self._send(f"SPIN_{direction}:{motor}:{steps}")

    # Motor 1 control (release motor)
    def spin_down_m1(self, steps: int) -> str:
        """Rotate motor 1 clockwise"""
        return self._spin("CW", "M1", steps)

    def spin_up_m1(self, steps: int) -> str:
        """Rotate motor 1 counter-clockwise"""
        return self._spin("CCW", "M1", steps)

    def spin_up_m1_until_switch(self, max_steps: int = 200000) -> str:
        """Rotate motor 1 counter-clockwise until the limit switch is pressed"""
        if max_steps <= 0:
            max_steps = 200000
        return self._send(f"SPIN_UP_TO_SWITCH:M1:{max_steps}")

    def spin_down_m1_until_switch(self, max_steps: int = 200000) -> str:
        """Rotate motor 1 clockwise until the limit switch is pressed"""
        if max_steps <= 0:
            max_steps = 200000
        return self._send(f"SPIN_DOWN_TO_SWITCH:M1:{max_steps}")

    # Motor 2 control (being the motor post-card identification)
    def spin_down_m2(self, steps: int) -> str:
        """Rotate motor 2 clockwise"""
        return self._spin("CW", "M2", steps)

    def spin_up_m2(self, steps: int) -> str:
        """Rotate motor 2 counter-clockwise"""
        return self._spin("CCW", "M2", steps)

    def spin_down_m2_async(self, steps: int) -> str:
        """Start motor 2 clockwise asynchronously (firmware support required)."""
        if steps < 0:
            steps = -steps
        response = self._send(f"SPIN_CW_ASYNC:M2:{steps}")
        if response in ("ERR:UNKNOWN", "ERR:BADARGS"):
            return self.spin_down_m2(steps)
        return response

    def spin_up_m2_async(self, steps: int) -> str:
        """Start motor 2 counter-clockwise asynchronously (firmware support required)."""
        if steps < 0:
            steps = -steps
        response = self._send(f"SPIN_CCW_ASYNC:M2:{steps}")
        if response in ("ERR:UNKNOWN", "ERR:BADARGS"):
            return self.spin_up_m2(steps)
        return response

    def is_m2_busy(self) -> bool:
        """Return True while asynchronous motor 2 motion is still active."""
        response = self._send("GET_M2_BUSY")
        if response == "BUSY":
            return True
        if response == "IDLE":
            return False
        return False

    def wait_for_m2_idle(self, max_wait_seconds: float = 30.0, poll_interval: float = 0.12) -> bool:
        """Wait until asynchronous M2 motion is complete."""
        start = time.time()
        while True:
            if not self.is_m2_busy():
                return True
            if max_wait_seconds > 0 and (time.time() - start) >= max_wait_seconds:
                return False
            time.sleep(max(0.005, poll_interval))

    # Both motors control
    def spin_down_both(self, steps: int) -> str:
        """Rotate both motors clockwise simultaneously"""
        return self._spin("CW", "BOTH", steps)

    def spin_up_both(self, steps: int) -> str:
        """Rotate both motors counter-clockwise simultaneously"""
        return self._spin("CCW", "BOTH", steps)

    # Legacy methods for backward compatibility - can be deleted
    def spin_down(self, steps: int) -> str:
        """Rotate motor 1 clockwise (legacy)"""
        return self.spin_down_m1(steps)

    def spin_up(self, steps: int) -> str:
        """Rotate motor 1 counter-clockwise (legacy)"""
        return self.spin_up_m1(steps)

    def sleep(self, milliseconds: int) -> str:
        if milliseconds < 0:
            milliseconds = 0
        return self._send(f"SLEEP:{milliseconds}")

    # TT Motor control (L298N driver)
    def tt_motor_forward(self, speed: Literal["SLOW", "MEDIUM", "FAST"] = "MEDIUM") -> str:
        """Run TT motor forward at specified speed"""
        return self._send(f"TT_FORWARD:{speed}")

    def tt_motor_forward_for(self, milliseconds: int, speed: Literal["CRAWL", "SLOW", "MEDIUM", "FAST"] = "MEDIUM") -> str:
        """Run TT motor forward for a fixed duration and auto-stop in firmware."""
        if milliseconds < 0:
            milliseconds = 0
        return self._send(f"TT_FORWARD_MS:{speed}:{milliseconds}")

    def tt_motor_reverse(self, speed: Literal["SLOW", "MEDIUM", "FAST"] = "MEDIUM") -> str:
        """Run TT motor in reverse at specified speed"""
        return self._send(f"TT_REVERSE:{speed}")

    def tt_motor_reverse_for(self, milliseconds: int, speed: Literal["CRAWL", "SLOW", "MEDIUM", "FAST"] = "MEDIUM") -> str:
        """Run TT motor in reverse for a fixed duration and auto-stop in firmware."""
        if milliseconds < 0:
            milliseconds = 0
        return self._send(f"TT_REVERSE_MS:{speed}:{milliseconds}")

    def tt_motor_stop(self) -> str:
        """Stop TT motor"""
        return self._send("TT_STOP")

    def is_switch_pressed(self) -> bool:
        """Check if limit switch is pressed"""
        response = self._send("GET_SWITCH")
        return response == "PRESSED"

    # MG90s servo control -- Note; servo 1 is the servo right above storage, 2 is the one right below imaging
    def servo1_move(self, angle: int) -> str:
        """Move servo 1 by signed offset from center (-90 to 90). Negative = down, positive = up."""
        return self._send(f"SERVO1:{angle}")

    def servo2_move(self, angle: int) -> str:
        """Move servo 2 by signed offset from center (-90 to 90). Negative = down, positive = up."""
        return self._send(f"SERVO2:{angle}")

    def servo1_move_abs(self, angle: int) -> str:
        """Move servo 1 to absolute angle (0-180). Use servo1_move() for signed offsets."""
        return self._send(f"SERVO1_ABS:{angle}")

    def servo2_move_abs(self, angle: int) -> str:
        """Move servo 2 to absolute angle (0-180). Use servo2_move() for signed offsets."""
        return self._send(f"SERVO2_ABS:{angle}")

    def servo_move(self, angle: int) -> str:
        """Move servo 1 by signed offset from center (-90 to 90). Negative = down, positive = up."""
        return self.servo1_move(angle)

    def servo1_rotate_90_and_back(self, delay_seconds: float = 0.5) -> None:
        """Rotate servo 1 up 90 degrees, pause, then down 90 degrees (returns to center)."""
        self.servo1_move(90)
        time.sleep(max(0.0, delay_seconds))
        self.servo1_move(-90)
        time.sleep(max(0.0, delay_seconds))
        self.servo1_move(0)


class DryRunController:
    """Controller mock for safe dry runs without hardware movement."""

    def open(self) -> None:
        print("[DRY-RUN] Controller open", flush=True)

    def close(self) -> None:
        print("[DRY-RUN] Controller close", flush=True)

    def _log(self, message: str) -> str:
        print(f"[DRY-RUN] {message}", flush=True)
        return "DRY_RUN"

    def spin_down_m1(self, steps: int) -> str:
        return self._log(f"spin_down_m1({steps})")

    def spin_up_m1(self, steps: int) -> str:
        return self._log(f"spin_up_m1({steps})")

    def spin_down_m1_until_switch(self, max_steps: int = 200000) -> str:
        return self._log(f"spin_down_m1_until_switch({max_steps})")

    def spin_up_m1_until_switch(self, max_steps: int = 200000) -> str:
        return self._log(f"spin_up_m1_until_switch({max_steps})")

    def spin_down_m2(self, steps: int) -> str:
        return self._log(f"spin_down_m2({steps})")

    def spin_up_m2(self, steps: int) -> str:
        return self._log(f"spin_up_m2({steps})")

    def spin_down_m2_async(self, steps: int) -> str:
        return self._log(f"spin_down_m2_async({steps})")

    def spin_up_m2_async(self, steps: int) -> str:
        return self._log(f"spin_up_m2_async({steps})")

    def is_m2_busy(self) -> bool:
        self._log("is_m2_busy()")
        return False

    def wait_for_m2_idle(self, max_wait_seconds: float = 30.0, poll_interval: float = 0.12) -> bool:
        self._log(f"wait_for_m2_idle(max_wait_seconds={max_wait_seconds}, poll_interval={poll_interval})")
        return True

    def tt_motor_forward(self, speed: Literal["SLOW", "MEDIUM", "FAST"] = "MEDIUM") -> str:
        return self._log(f"tt_motor_forward({speed})")

    def tt_motor_forward_for(self, milliseconds: int, speed: Literal["CRAWL", "SLOW", "MEDIUM", "FAST"] = "MEDIUM") -> str:
        return self._log(f"tt_motor_forward_for({milliseconds}, {speed})")

    def tt_motor_reverse(self, speed: Literal["SLOW", "MEDIUM", "FAST"] = "MEDIUM") -> str:
        return self._log(f"tt_motor_reverse({speed})")

    def tt_motor_reverse_for(self, milliseconds: int, speed: Literal["CRAWL", "SLOW", "MEDIUM", "FAST"] = "MEDIUM") -> str:
        return self._log(f"tt_motor_reverse_for({milliseconds}, {speed})")

    def tt_motor_stop(self) -> str:
        return self._log("tt_motor_stop()")

    def servo1_move(self, angle: int) -> str:
        return self._log(f"servo1_move({angle})")

    def servo2_move(self, angle: int) -> str:
        return self._log(f"servo2_move({angle})")

    def servo1_move_abs(self, angle: int) -> str:
        return self._log(f"servo1_move_abs({angle})")

    def servo2_move_abs(self, angle: int) -> str:
        return self._log(f"servo2_move_abs({angle})")

    def servo1_rotate_90_and_back(self, delay_seconds: float = 0.5) -> None:
        self._log(f"servo1_rotate_90_and_back(delay_seconds={delay_seconds})")

    def is_switch_pressed(self) -> bool:
        return False

# ==============================================================================
# MOTOR CONTROL HELPER FUNCTIONS (from AllMotors.py)
# ==============================================================================
def check_abort():
    """Check if 'p' key was pressed to abort. Returns True if abort requested."""
    if msvcrt.kbhit():
        key = msvcrt.getch().decode('utf-8').lower()
        if key == 'p':
            return True
    return False

def sleep_with_abort(duration):
    """Sleep for duration seconds, checking for abort every 50ms. Returns True if aborted."""
    start_time = time.time()
    while time.time() - start_time < duration:
        if check_abort():
            print("\n⚠️  ABORT REQUESTED - Stopping sequence...\n")
            return True
        time.sleep(0.05)
    return False

def run_motor_with_abort(motor_func, duration, controller):
    """Run a motor for duration seconds with abort checking. Returns True if aborted."""
    start_time = time.time()
    while time.time() - start_time < duration:
        if check_abort():
            controller.tt_motor_stop()
            print("\n⚠️  ABORT REQUESTED - Stopping sequence...\n")
            return True
        time.sleep(0.05)
    return False

# ==============================================================================
# IMAGE PROCESSING FUNCTIONS (from Main copy.ipynb)
# ==============================================================================

# Global configuration
DEBUG_DIR = 'Debugging Images'
os.makedirs(DEBUG_DIR, exist_ok=True)
IDENTIFIED_DIR = 'Identified_Cards'
os.makedirs(IDENTIFIED_DIR, exist_ok=True)
FAILED_DIR = 'reallybugginme'
os.makedirs(FAILED_DIR, exist_ok=True)

# Scryfall "normal" image size
SCRYFALL_NORMAL_SIZE = (488, 680)  # (width, height)

# Annoy index configuration
ANNOY_INDEX_PATH = "phash_annoy_index.ann"
PHASH_IDX_TO_CARD_PATH = "phash_idx_to_card.json"
ANNOY_DIM = 64
CARD_CROP_EXPANSION_RATIO = 0.03
YOLO_LOCATOR_MODEL_PATH = Path("runs/obb/detect_obb_combined_v1/weights/best.pt")
YOLO_LOCATOR_IMGSZ = 640
YOLO_LOCATOR_CONF = 0.25
YOLO_LOCATOR_DEVICE = "auto"
GRID_MATCH_SOURCE_MODULE_PATH = Path("Expo_P.H.A.S.E.py")
GRID_MATCH_FULL_DB_PATH = Path("scryfall_phash_db_sample.json")
GRID_MATCH_SEGMENTED_DB_PATH = Path("scryfall_segmented_tophalf_db_v2.json")
SAVE_LOCATOR_DEBUG_ARTIFACTS = False
SAVE_GRID_IDENTIFY_RESULT_JSON = False

_YOLO_LOCATOR_MODEL = None
_YOLO_LOCATOR_DEVICE_RESOLVED = None
_GRID_MATCHER = None
_LAST_GRID_IDENTIFY_RESULT = None
_SCRYFALL_SESSION = None
_SCRYFALL_PRICE_CACHE = {}
SCRYFALL_REQUEST_TIMEOUT_SECONDS = 4.0
SERVO2_HOME_ANGLE = 140
SERVO2_DROP_DELTA = 140
SERVO2_DROP_WAIT_SECONDS = 0.8
CARD_PRESENCE_CHECK_INTERVAL_SECONDS = 0.5
CARD_CAPTURE_SETTLE_SECONDS = 0.5
RELEASE_ATTEMPT_MIN_INTERVAL_SECONDS = 1.5
LOCATOR_FAILURE_MARKER = "__locator_failed__"
AUTO_RUN_WORKBOOK_PATH = Path("Identified_Cards") / "automatic_runs.xlsx"
LAST_RECOGNITION_TIMINGS = {}
APP_RUNTIME_START_TS = None
M2_ROUTE_STEPS = 2050
# Match test_motor2_only.py default behavior for best M2 smoothness.
# "sync" uses SPIN_CW/SPIN_CCW (blocking); "async" uses async commands + idle wait.
M2_ROUTE_COMMAND_MODE = "sync"
M2_ROUTE_ASYNC_MAX_WAIT_SECONDS = 40.0
M2_ROUTE_ASYNC_POLL_SECONDS = 0.20
M2_ROUTE_SETTLE_SECONDS = 0.2


def resolve_yolo_device():
    configured = str(YOLO_LOCATOR_DEVICE).strip().lower()
    if configured != "auto":
        return YOLO_LOCATOR_DEVICE
    if torch is not None:
        try:
            if torch.cuda.is_available():
                return 0
        except Exception:
            pass
    return "cpu"


def get_yolo_locator_model():
    global _YOLO_LOCATOR_MODEL, _YOLO_LOCATOR_DEVICE_RESOLVED
    if _YOLO_LOCATOR_MODEL is None:
        if not YOLO_LOCATOR_MODEL_PATH.exists():
            raise FileNotFoundError(f"YOLO locator model not found: {YOLO_LOCATOR_MODEL_PATH}")
        print(f"Loading YOLO locator model: {YOLO_LOCATOR_MODEL_PATH}", flush=True)
        _YOLO_LOCATOR_MODEL = YOLO(str(YOLO_LOCATOR_MODEL_PATH))
    if _YOLO_LOCATOR_DEVICE_RESOLVED is None:
        _YOLO_LOCATOR_DEVICE_RESOLVED = resolve_yolo_device()
        print(f"YOLO locator device resolved to: {_YOLO_LOCATOR_DEVICE_RESOLVED}", flush=True)
    return _YOLO_LOCATOR_MODEL


def get_grid_matcher():
    global _GRID_MATCHER
    if _GRID_MATCHER is None:
        cfg = SegmentedHashConfig()
        _GRID_MATCHER = HybridTopHalfMatcher(
            source_module_path=GRID_MATCH_SOURCE_MODULE_PATH,
            full_card_db_path=GRID_MATCH_FULL_DB_PATH,
            segmented_db_path=GRID_MATCH_SEGMENTED_DB_PATH,
            cfg=cfg,
        )
    return _GRID_MATCHER


def get_scryfall_session():
    global _SCRYFALL_SESSION
    if _SCRYFALL_SESSION is None:
        _SCRYFALL_SESSION = requests.Session()
    return _SCRYFALL_SESSION


def prewarm_recognition_pipeline():
    """Load heavy recognition components before the first card to avoid cold-start spikes."""
    t0 = time.perf_counter()
    try:
        model = get_yolo_locator_model()
        dummy = np.zeros((YOLO_LOCATOR_IMGSZ, YOLO_LOCATOR_IMGSZ, 3), dtype=np.uint8)
        model.predict(
            source=[dummy],
            imgsz=YOLO_LOCATOR_IMGSZ,
            conf=YOLO_LOCATOR_CONF,
            device=_YOLO_LOCATOR_DEVICE_RESOLVED,
            save=False,
            verbose=False,
        )
    except Exception as exc:
        print(f"YOLO prewarm skipped: {exc}", flush=True)

    try:
        get_grid_matcher()
    except Exception as exc:
        print(f"Grid matcher prewarm skipped: {exc}", flush=True)

    elapsed = time.perf_counter() - t0
    print(f"Recognition pipeline prewarm completed in {elapsed:.3f}s", flush=True)


def locate_card_with_yolo_axis_pipeline(image_bgr):
    """Use finalized YOLO-first + axis crop process to locate and crop the card."""
    model = get_yolo_locator_model()

    results = model.predict(
        source=[image_bgr],
        imgsz=YOLO_LOCATOR_IMGSZ,
        conf=YOLO_LOCATOR_CONF,
        device=_YOLO_LOCATOR_DEVICE_RESOLVED,
        save=False,
        verbose=False,
    )
    if not results:
        raise ValueError("YOLO locator returned no results.")

    result = results[0]
    picks = yolo_locator.obb_points_from_result(result, top_k=1)
    if not picks:
        raise ValueError("No YOLO OBB card detection found.")

    yolo_quad = picks[0][0]
    topbg_quad, topbg_debug = yolo_locator.refine_quad_with_top_mask(image_bgr, yolo_quad)
    line_info = None if topbg_debug is None else topbg_debug.get("line_info")

    final_crop, stage_images, meta = yolo_locator.build_axis_crop_stages(image_bgr, yolo_quad, line_info)

    if SAVE_LOCATOR_DEBUG_ARTIFACTS:
        if topbg_debug is not None:
            yolo_locator.save_topbg_debug_bundle(Path(DEBUG_DIR), "locator", image_bgr, topbg_debug)

        # Save only the finalized location pipeline stages for quick visual verification.
        for stage_name, stage_img in stage_images.items():
            cv2.imwrite(os.path.join(DEBUG_DIR, f"locator_{stage_name}.png"), stage_img)

        with open(os.path.join(DEBUG_DIR, "locator_meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

    return final_crop, meta

def clear_debug_images(debug_dir):
    """Clear out old debug images before processing a new card"""
    for f in glob.glob(os.path.join(debug_dir, '*.png')):
        try:
            os.remove(f)
        except Exception as e:
            print(f'Could not delete {f}: {e}', flush=True)

def resize_to_scryfall_normal(image):
    """
    Resize image so that the shorter side matches the Scryfall normal size,
    preserving aspect ratio and maximizing retained resolution.
    """
    target_w, target_h = SCRYFALL_NORMAL_SIZE
    h, w = image.shape[:2]
    scale = max(target_w / w, target_h / h)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return resized

def preprocess_image(image_path):
    """Load and preprocess the image with enhanced edge detection"""
    # Loading the image
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found at path: {image_path}")

    # Resize to Scryfall normal size (preserving aspect ratio, no loss of resolution)
    image = resize_to_scryfall_normal(image)
    
    # Apply slight Gaussian blur to reduce noise while preserving edges
    blurred = cv2.GaussianBlur(image, (3, 3), 0)

    # Convert to grayscale
    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
    
    # Apply CLAHE to enhance contrast and make dark borders stand out
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    gray_enhanced = clahe.apply(gray)

    # Apply edge detection with higher thresholds to reduce noise
    edges = cv2.Canny(gray_enhanced, 80, 200)
    
    # Moderately strengthen edges so card borders connect without over-closing the shape.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges = cv2.dilate(edges, kernel, iterations=1)
    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, close_kernel, iterations=1)

    return image, gray_enhanced, edges


def order_points(pts):
    """Order quadrilateral points as top-left, top-right, bottom-right, bottom-left."""
    pts = np.asarray(pts, dtype="float32")
    sums = pts.sum(axis=1)
    diffs = np.diff(pts, axis=1).reshape(-1)
    return np.array([
        pts[np.argmin(sums)],
        pts[np.argmin(diffs)],
        pts[np.argmax(sums)],
        pts[np.argmax(diffs)],
    ], dtype="float32")


def expand_quad(quad, image_shape, expansion_ratio=CARD_CROP_EXPANSION_RATIO):
    """Expand a detected quadrilateral outward around its center by a fixed ratio."""
    ordered = order_points(quad)
    center = ordered.mean(axis=0)
    expanded = center + (ordered - center) * (1.0 + expansion_ratio)

    height, width = image_shape[:2]
    expanded[:, 0] = np.clip(expanded[:, 0], 0, width - 1)
    expanded[:, 1] = np.clip(expanded[:, 1], 0, height - 1)
    return order_points(expanded)


def infer_fourth_corner_from_three(points):
    """Infer a fourth corner when three strong corners of a card rectangle are visible."""
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

def detect_card(image, edges):
    """Detect the card contour with a moderate preference for rectangle-like shapes."""
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    edges_closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=1)
    
    contours, _ = cv2.findContours(edges_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("No contours found while detecting card.")

    min_area = image.shape[0] * image.shape[1] * 0.05
    selected_quad = None
    selected_contour = None
    best_score = -1.0

    for contour in sorted(contours, key=cv2.contourArea, reverse=True):
        area = cv2.contourArea(contour)
        if area < min_area:
            continue

        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.018 * perimeter, True)
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
        corner_score = 1.0 if len(approx) == 4 else 0.9 if len(approx) == 3 else 0.75 if len(approx) in (5, 6) else 0.4
        score = area * fill_ratio * corner_score

        if fill_ratio < 0.45:
            continue

        if score > best_score:
            best_score = score
            selected_quad = candidate_quad
            selected_contour = contour

    if selected_quad is None:
        selected_contour = max(contours, key=cv2.contourArea)
        rect = cv2.minAreaRect(selected_contour)
        selected_quad = order_points(cv2.boxPoints(rect))

    debug_image = image.copy()
    cv2.polylines(debug_image, [selected_quad.astype(np.int32)], True, (0, 255, 0), 3)

    return debug_image, selected_quad

def rotate_and_crop_image(image, contour):
    """
    Rotate and crop the image to align edges and ensure correct orientation
    Ensures the detected edges form a flat rectangle by rotating the image
    """
    ordered_box = order_points(contour)
    expanded_box = expand_quad(ordered_box, image.shape)

    # Compute width and height of the expanded crop region.
    widthA = np.linalg.norm(expanded_box[2] - expanded_box[3])
    widthB = np.linalg.norm(expanded_box[1] - expanded_box[0])
    maxWidth = int(max(widthA, widthB))

    heightA = np.linalg.norm(expanded_box[1] - expanded_box[2])
    heightB = np.linalg.norm(expanded_box[0] - expanded_box[3])
    maxHeight = int(max(heightA, heightB))

    # Destination points for the upright rectangle
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]
    ], dtype="float32")

    # Perspective transform to get the upright, cropped card
    M = cv2.getPerspectiveTransform(expanded_box, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    # If the card is sideways (width > height), rotate it upright
    if warped.shape[0] < warped.shape[1]:
        warped = cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)

    return warped

def crop_physical_regions(card_img):
    """
    Crop two regions from the card image based on physical measurements.
    Returns (top_crop, bottom_crop, debug_img)
    """
    h, w = card_img.shape[:2]
    card_width_mm = 63.0
    px_per_mm = w / card_width_mm

    # Top crop (upright)
    right_edge_px = int(round(w - 4.6 * px_per_mm))
    top_edge_px = int(round(4.6 * px_per_mm))
    left_edge_px = int(round(right_edge_px - 24 * px_per_mm))
    bottom_edge_px = int(round(top_edge_px + 4.5 * px_per_mm))

    top_crop = card_img[top_edge_px:bottom_edge_px, left_edge_px:right_edge_px]

    # Bottom crop (inverted)
    left_edge_px_b = int(round(4.6 * px_per_mm))
    bottom_edge_px_b = int(round(h - 4.6 * px_per_mm))
    right_edge_px_b = int(round(left_edge_px_b + 24 * px_per_mm))
    top_edge_px_b = int(round(bottom_edge_px_b - 4.5 * px_per_mm))

    bottom_crop = card_img[top_edge_px_b:bottom_edge_px_b, left_edge_px_b:right_edge_px_b]
    # Rotate the bottom crop 180 degrees before saving
    bottom_crop_rotated = cv2.rotate(bottom_crop, cv2.ROTATE_180)

    # Debug image with rectangles drawn
    debug_img = card_img.copy()
    cv2.rectangle(debug_img, (left_edge_px, top_edge_px), (right_edge_px, bottom_edge_px), (0,255,0), 2)
    cv2.rectangle(debug_img, (left_edge_px_b, top_edge_px_b), (right_edge_px_b, bottom_edge_px_b), (0,0,255), 2)
    return top_crop, bottom_crop_rotated, debug_img

def segment_symbols(crop_img, debug_prefix="symbol"):
    """
    Segments individual mana symbols from the cropped mana cost region
    Based on the contrast of the threshold of circular mana symbols.
    Returns a list of symbol images and a debug image with bounding boxes.
    """
    gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
    # Adaptive thresholding for better separation of symbols
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 15
    )
    # Morphological closing to connect symbol edges
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    # Find contours (external only)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    symbol_imgs = []
    bounding_boxes = [cv2.boundingRect(c) for c in contours]
    sorted_items = sorted(zip(bounding_boxes, contours), key=lambda b: b[0][0])
    sorted_contours = [c for _, c in sorted_items]
    sorted_boxes = [b for b, _ in sorted_items]

    debug_img = crop_img.copy()
    symbol_found = False
    for idx, (c, (x, y, w, h)) in enumerate(zip(sorted_contours, sorted_boxes)):
        # Filter out noise, keep likely circular shapes (mana symbols)
        if w > 10 and h > 10 and w < crop_img.shape[1] * 0.7 and h < crop_img.shape[0] * 0.9:
            aspect_ratio = w / h
            area = cv2.contourArea(c)
            # Heuristic: aspect ratio near 1 and area not too small
            if 0.6 < aspect_ratio < 1.4 and area > 0.4 * np.pi * (min(w, h) / 2) ** 2:
                symbol = crop_img[y:y+h, x:x+w]
                # Saving each symbol image in the debugging folder
                cv2.imwrite(os.path.join(DEBUG_DIR, f"{debug_prefix}_{idx}.png"), symbol)
                symbol_imgs.append(symbol)
                cv2.rectangle(debug_img, (x, y), (x+w, y+h), (0, 0, 255), 2)
                symbol_found = True
    # Only save the debug bounding boxes image if at least one symbol was found
    if symbol_found:
        cv2.imwrite(os.path.join(DEBUG_DIR, f"{debug_prefix}_bounding_boxes.png"), debug_img)
    return symbol_imgs

def compute_phash(image):
    """Compute perceptual hash for a given image using PIL and imagehash."""
    pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    return str(imagehash.phash(pil_img))

def phash_to_vec(phash_str):
    """
    Converts the hex perceptual hash string to a 64-dim float vector (0/1).
    This enables vectorized operations with ANNOY.
    """
    h = imagehash.hex_to_hash(phash_str)
    return h.hash.flatten().astype(np.float32)

def load_or_build_annoy_index(card_hash_db):
    """Load or build Annoy index and idx_to_card mapping"""
    annoy_index = AnnoyIndex(ANNOY_DIM, 'hamming')

    if os.path.exists(ANNOY_INDEX_PATH) and os.path.exists(PHASH_IDX_TO_CARD_PATH):
        # Load Annoy index and idx_to_card mapping
        annoy_index.load(ANNOY_INDEX_PATH)
        with open(PHASH_IDX_TO_CARD_PATH, "r", encoding="utf-8") as f:
            idx_to_card = json.load(f)
        idx_to_card = {int(k): v for k, v in idx_to_card.items()}
    else:
        # Build Annoy index and idx_to_card mapping, then save them
        # NOTE: This only needs to happen once, the ANNOY database should already be created
        phash_to_idx = {}
        idx_to_card = {}
        for idx, entry in enumerate(card_hash_db):
            phash = entry.get('phash')
            if phash is None:
                continue
            vec = phash_to_vec(phash)
            annoy_index.add_item(idx, vec)
            phash_to_idx[phash] = idx
            idx_to_card[idx] = entry
        annoy_index.build(100)
        annoy_index.save(ANNOY_INDEX_PATH)
        with open(PHASH_IDX_TO_CARD_PATH, "w", encoding="utf-8") as f:
            json.dump(idx_to_card, f)

    return annoy_index, idx_to_card

def annoy_nearest_phash(image, annoy_index, idx_to_card, n=3):
    """
    Find the nearest card(s) in the Annoy index to the image's phash.
    Returns a list of (card_info, distance).
    """
    img_hash = compute_phash(image)
    img_vec = phash_to_vec(img_hash)
    idxs, dists = annoy_index.get_nns_by_vector(img_vec, n, include_distances=True)
    results = []
    for idx, dist in zip(idxs, dists):
        card = idx_to_card[idx]
        results.append((card, dist))
    return results

def annoy_nearest_phash_orientation(image, annoy_index, idx_to_card, n=3):
    """
    Find the nearest card(s) in the Annoy index to both upright and 180-rotated image.
    Returns best match, orientation, and distance.
    """
    img_hash = compute_phash(image)
    img_vec = phash_to_vec(img_hash)
    rotated_img = cv2.rotate(image, cv2.ROTATE_180)
    rotated_hash = compute_phash(rotated_img)
    rotated_vec = phash_to_vec(rotated_hash)

    idxs_upright, dists_upright = annoy_index.get_nns_by_vector(img_vec, n, include_distances=True)
    idxs_rot, dists_rot = annoy_index.get_nns_by_vector(rotated_vec, n, include_distances=True)

    best = None
    best_dist = float('inf')
    best_orientation = None
    for idx, dist in zip(idxs_upright, dists_upright):
        if dist < best_dist:
            best = idx
            best_dist = dist
            best_orientation = 'upright'
    for idx, dist in zip(idxs_rot, dists_rot):
        if dist < best_dist:
            best = idx
            best_dist = dist
            best_orientation = 'rotated 180'
    if best is not None:
        return idx_to_card[best], best_orientation, best_dist
    else:
        return None, None, None

def get_prices_from_any_printing(card_name):
    """Search for prices from any printing of a card by name."""
    try:
        session = get_scryfall_session()
        # Search for the card by exact name
        search_url = f"https://api.scryfall.com/cards/search?q=!%22{card_name}%22&order=released"
        response = session.get(search_url, timeout=SCRYFALL_REQUEST_TIMEOUT_SECONDS)
        if response.status_code == 200:
            data = response.json()
            if data.get('data'):
                # Try each printing until we find one with prices
                for card in data['data']:
                    prices = card.get('prices', {})
                    # Check for USD or EUR prices
                    if prices.get('usd') or prices.get('eur'):
                        return prices
        return {}
    except Exception as e:
        print(f"Error searching for other printings of {card_name}: {e}", flush=True)
        return {}

def get_scryfall_prices(card_info):
    """Retrieve price information for the identified card from Scryfall"""
    global _SCRYFALL_PRICE_CACHE

    session = get_scryfall_session()
    scryfall_uri = card_info.get('scryfall_uri', '')

    m = re.search(r'/card/([^/]+)/([^/?]+)', scryfall_uri)
    if m:
        set_code, collector_number = m.group(1), m.group(2)
        api_url = f"https://api.scryfall.com/cards/{set_code}/{collector_number}"
    else:
        # As a last resort, try to use the full scryfall_uri (replace domain)
        api_url = scryfall_uri.replace('scryfall.com/card/', 'api.scryfall.com/cards/')

    cache_key = api_url or scryfall_uri or card_info.get('name', '')
    if cache_key in _SCRYFALL_PRICE_CACHE:
        return dict(_SCRYFALL_PRICE_CACHE[cache_key])

    response = session.get(api_url, timeout=SCRYFALL_REQUEST_TIMEOUT_SECONDS)
    if response.status_code == 200:
        data = response.json()
        prices = data.get('prices', {})
        # If this printing has no prices, search other printings
        if not prices.get('usd') and not prices.get('eur'):
            card_name = card_info.get('name', '')
            if card_name:
                print(f"No prices for this printing. Searching other printings of {card_name}...", flush=True)
                prices = get_prices_from_any_printing(card_name)
                if prices:
                    print(f"Found prices from another printing of {card_name}.", flush=True)
        _SCRYFALL_PRICE_CACHE[cache_key] = dict(prices or {})
        return prices
    else:
        print(f"Failed to retrieve prices from {api_url} (status {response.status_code}). Searching other printings...", flush=True)
        card_name = card_info.get('name', '')
        if card_name:
            prices = get_prices_from_any_printing(card_name)
            _SCRYFALL_PRICE_CACHE[cache_key] = dict(prices or {})
            return prices
        _SCRYFALL_PRICE_CACHE[cache_key] = {}
        return {}

def get_cad_price(prices, usd_to_cad=1.37, eur_to_cad=1.57, default_price=None):
    """Convert USD or EUR prices to CAD. Returns (cad_price, source_currency) or (default_price, 'default')."""
    if prices.get('usd'):
        try:
            return round(float(prices['usd']) * usd_to_cad, 2), 'usd'
        except Exception:
            pass
    if prices.get('eur'):
        try:
            return round(float(prices['eur']) * eur_to_cad, 2), 'eur'
        except Exception:
            pass
    # If no prices found and default is provided, use it
    if default_price is not None:
        return default_price, 'default'
    return None, None

def print_card_prices(prices, usd_to_cad=1.37, eur_to_cad=1.47, default_price=None):
    """Print card prices in requested order: CAD, USD, EUR, then foils"""
    cad_price, cad_source = get_cad_price(prices, usd_to_cad, eur_to_cad, default_price=default_price)

    print("Card prices:", flush=True)
    if cad_price is not None:
        if cad_source == 'default':
            print(f"CAD: ${cad_price} (default fallback)", flush=True)
        else:
            print(f"CAD: ${cad_price} (converted from {cad_source.upper()})", flush=True)
    if prices.get('usd'):
        print(f"USD: ${prices['usd']}", flush=True)
    if prices.get('eur'):
        print(f"EUR: €{prices['eur']}", flush=True)
    if prices.get('usd_foil'):
        print(f"USD Foil: ${prices['usd_foil']}", flush=True)
    if prices.get('usd_etched'):
        print(f"USD Etched: ${prices['usd_etched']}", flush=True)
    if prices.get('eur_foil'):
        print(f"EUR Foil: €{prices['eur_foil']}", flush=True)
    if prices.get('tix'):
        print(f"MTGO Tix: {prices['tix']}", flush=True)


def sanitize_filename_component(value):
    """Convert arbitrary text to a filesystem-safe name component."""
    safe_value = re.sub(r'[^A-Za-z0-9._-]+', '_', str(value)).strip('._')
    return safe_value or 'unknown'


def _load_openpyxl_symbols():
    """Lazy-import openpyxl so runtime environment controls dependency resolution."""
    try:
        import openpyxl  # type: ignore
        return openpyxl.Workbook, openpyxl.load_workbook
    except Exception as exc:
        raise RuntimeError("openpyxl is required for automatic run Excel logging.") from exc


def resolve_card_info_from_db(card_hash_db, name, set_code):
    """Find full card info (including scryfall_uri) from DB by name/set."""
    if not name:
        return None

    name_l = str(name).strip().lower()
    set_l = str(set_code).strip().lower() if set_code else None

    for entry in card_hash_db:
        e_name = str(entry.get('name', '')).strip().lower()
        e_set = str(entry.get('set', '')).strip().lower()
        if e_name != name_l:
            continue
        if set_l is None or e_set == set_l:
            return entry

    # Fallback to name-only if exact set isn't present.
    for entry in card_hash_db:
        e_name = str(entry.get('name', '')).strip().lower()
        if e_name == name_l:
            return entry
    return None


def archive_identified_card_artifacts(capture_path, card_info, orientation, distance, prices):
    """Save the recognized capture and all debug images into a per-card folder."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    card_name = sanitize_filename_component(card_info.get('name', 'Unknown'))
    artifact_dir = os.path.join(IDENTIFIED_DIR, f"{timestamp}_{card_name}")
    os.makedirs(artifact_dir, exist_ok=True)

    if capture_path and os.path.exists(capture_path):
        shutil.copy2(capture_path, os.path.join(artifact_dir, os.path.basename(capture_path)))

    for debug_path in glob.glob(os.path.join(DEBUG_DIR, '*.png')):
        shutil.copy2(debug_path, os.path.join(artifact_dir, os.path.basename(debug_path)))

    metadata = {
        'name': card_info.get('name', 'Unknown'),
        'set': card_info.get('set', ''),
        'orientation': orientation,
        'distance': distance,
        'scryfall_uri': card_info.get('scryfall_uri', ''),
        'prices': prices or {},
        'capture_path': capture_path,
    }

    metadata_path = os.path.join(artifact_dir, 'metadata.json')
    with open(metadata_path, 'w', encoding='utf-8') as handle:
        json.dump(metadata, handle, indent=2)

    print(f"Saved identified card artifacts to: {artifact_dir}", flush=True)
    return artifact_dir


def archive_failed_card_artifacts(capture_path, reason, extra=None):
    """Save failed recognition capture and current debug artifacts."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    failure_dir = os.path.join(FAILED_DIR, f"{timestamp}_failed")
    os.makedirs(failure_dir, exist_ok=True)

    excluded_debug_names = {
        'bottom_crop.png',
        'debug_crops.png',
        'debug crops.png',
    }

    if capture_path and os.path.exists(capture_path):
        shutil.copy2(capture_path, os.path.join(failure_dir, os.path.basename(capture_path)))

    for debug_path in glob.glob(os.path.join(DEBUG_DIR, '*.png')):
        debug_name = os.path.basename(debug_path)
        if debug_name.lower() in excluded_debug_names:
            continue
        shutil.copy2(debug_path, os.path.join(failure_dir, debug_name))
    for debug_json in glob.glob(os.path.join(DEBUG_DIR, '*.json')):
        shutil.copy2(debug_json, os.path.join(failure_dir, os.path.basename(debug_json)))

    metadata = {
        'reason': reason,
        'capture_path': capture_path,
    }
    if extra is not None:
        metadata['extra'] = extra

    with open(os.path.join(failure_dir, 'failure_metadata.json'), 'w', encoding='utf-8') as handle:
        json.dump(metadata, handle, indent=2)

    print(f"Saved failed card artifacts to: {failure_dir}", flush=True)
    return failure_dir


def _build_daily_run_sheet_name(existing_names):
    """Build sheet name using format YY.MM.DD. - N where N is today's run count."""
    day_prefix = datetime.now().strftime("%y.%m.%d.")
    pattern = re.compile(rf"^{re.escape(day_prefix)}\s-\s(\d+)$")

    max_run = 0
    for name in existing_names:
        match = pattern.match(str(name))
        if not match:
            continue
        try:
            max_run = max(max_run, int(match.group(1)))
        except Exception:
            pass

    next_run = max_run + 1
    while True:
        candidate = f"{day_prefix} - {next_run}"
        if candidate not in existing_names:
            return candidate
        next_run += 1


def _save_workbook_resilient(wb, workbook_path, retries=4, retry_delay=0.4):
    """Save workbook with retries; if locked, fall back to a timestamped sibling file."""
    path = Path(workbook_path)

    for attempt in range(retries):
        try:
            wb.save(str(path))
            return str(path)
        except PermissionError:
            if attempt < retries - 1:
                print(
                    f"Workbook is busy/locked: {path}. Retrying save ({attempt + 1}/{retries})...",
                    flush=True,
                )
                time.sleep(retry_delay)
                continue

            fallback_name = f"{path.stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}{path.suffix}"
            fallback_path = path.with_name(fallback_name)
            wb.save(str(fallback_path))
            print(
                f"Workbook remained locked. Saved logging output to fallback file: {fallback_path}",
                flush=True,
            )
            return str(fallback_path)


def print_recognition_timing_summary(total_seconds, timings):
    """Print capture-to-result total and per-step timings for performance tuning."""
    try:
        print(f"Recognition timing | capture_to_result={float(total_seconds):.3f}s", flush=True)
    except Exception:
        print(f"Recognition timing | capture_to_result={total_seconds}", flush=True)

    if not isinstance(timings, dict) or not timings:
        print("Recognition timing details unavailable.", flush=True)
        return

    ordered_keys = [
        'load_image',
        'locate_card_pipeline',
        'crop_regions_and_debug_save',
        'grid_identify',
        'price_fetch',
        'total',
    ]
    for key in ordered_keys:
        if key in timings:
            print(f"  - {key}: {float(timings[key]):.3f}s", flush=True)


def init_automatic_run_sheet(workbook_path=AUTO_RUN_WORKBOOK_PATH):
    """Create a fresh sheet for this automatic run and return logging metadata."""
    Workbook, load_workbook = _load_openpyxl_symbols()
    workbook_path = Path(workbook_path)
    workbook_path.parent.mkdir(parents=True, exist_ok=True)

    if workbook_path.exists():
        wb = load_workbook(str(workbook_path))
    else:
        wb = Workbook()
        # Remove default empty sheet so each run is explicitly named.
        if "Sheet" in wb.sheetnames and len(wb.sheetnames) == 1 and wb["Sheet"].max_row == 1 and wb["Sheet"].max_column == 1:
            del wb["Sheet"]

    sheet_name = _build_daily_run_sheet_name(set(wb.sheetnames))
    ws = wb.create_sheet(title=sheet_name)
    ws.append([
        "Card Name",
        "Camera Image Link",
        "Scryfall API Link",
        "Current Price (CAD)",
        "Match Confidence",
    ])
    saved_path = _save_workbook_resilient(wb, workbook_path)
    print(f"Automatic run logging sheet created: {saved_path} [{sheet_name}]", flush=True)
    return {
        'workbook_path': str(saved_path),
        'sheet_name': sheet_name,
    }


def append_automatic_run_entry(run_sheet_info, card_name, capture_path, scryfall_uri, cad_price, confidence):
    """Append one identified card row to the current automatic-run sheet."""
    _Workbook, load_workbook = _load_openpyxl_symbols()
    if not run_sheet_info:
        return

    workbook_path = run_sheet_info.get('workbook_path')
    sheet_name = run_sheet_info.get('sheet_name')
    if not workbook_path or not sheet_name:
        return

    wb = load_workbook(workbook_path)
    if sheet_name not in wb.sheetnames:
        wb.close()
        return

    ws = wb[sheet_name]
    image_link = str(Path(capture_path).resolve()) if capture_path else ""
    ws.append([
        card_name or "Unknown",
        image_link,
        scryfall_uri or "",
        cad_price,
        confidence,
    ])

    row_idx = ws.max_row
    if image_link:
        image_cell = ws.cell(row=row_idx, column=2)
        image_cell.hyperlink = image_link
        image_cell.style = "Hyperlink"
    if scryfall_uri:
        scry_cell = ws.cell(row=row_idx, column=3)
        scry_cell.hyperlink = scryfall_uri
        scry_cell.style = "Hyperlink"

    saved_path = _save_workbook_resilient(wb, workbook_path)
    run_sheet_info['workbook_path'] = str(saved_path)
    wb.close()


def write_automatic_run_stats(run_sheet_info, total_runtime_seconds, identified_count, failure_count):
    """Write run-level summary stats to the side of the automatic-run sheet."""
    _Workbook, load_workbook = _load_openpyxl_symbols()
    if not run_sheet_info:
        return

    workbook_path = run_sheet_info.get('workbook_path')
    sheet_name = run_sheet_info.get('sheet_name')
    if not workbook_path or not sheet_name:
        return

    wb = load_workbook(workbook_path)
    if sheet_name not in wb.sheetnames:
        wb.close()
        return

    ws = wb[sheet_name]
    ws["H1"] = "Run Summary"
    ws["H2"] = "Total Runtime (s)"
    ws["I2"] = round(float(total_runtime_seconds), 3)
    ws["H3"] = "Identified Cards"
    ws["I3"] = int(identified_count)
    ws["H4"] = "Failures To Identify"
    ws["I4"] = int(failure_count)

    saved_path = _save_workbook_resilient(wb, workbook_path)
    run_sheet_info['workbook_path'] = str(saved_path)
    wb.close()


def read_latest_match_confidence():
    """Read confidence from the latest grid identification debug payload."""
    global _LAST_GRID_IDENTIFY_RESULT

    if _LAST_GRID_IDENTIFY_RESULT is not None:
        try:
            best = (_LAST_GRID_IDENTIFY_RESULT or {}).get('best') or {}
            final_score = best.get('final_score')
            weighted_matches = best.get('weighted_matches')
            if final_score is not None:
                return float(final_score)
            if weighted_matches is not None:
                return float(weighted_matches)
        except Exception:
            pass

    debug_path = os.path.join(DEBUG_DIR, "grid_identify_result.json")
    if not os.path.exists(debug_path):
        return None

    try:
        with open(debug_path, 'r', encoding='utf-8') as handle:
            payload = json.load(handle)
        best = payload.get('best') or {}
        final_score = best.get('final_score')
        weighted_matches = best.get('weighted_matches')
        if final_score is not None:
            return float(final_score)
        if weighted_matches is not None:
            return float(weighted_matches)
    except Exception as exc:
        print(f"Could not read confidence from debug payload: {exc}", flush=True)
    return None

# ==============================================================================
# CARD RECOGNITION PIPELINE
# ==============================================================================
def recognize_card(image_path, card_hash_db, annoy_index, idx_to_card):
    """
    Complete card recognition pipeline
    Returns card_info, orientation, distance, and prices
    """
    global LAST_RECOGNITION_TIMINGS, _LAST_GRID_IDENTIFY_RESULT
    timings = {}
    t_total_start = time.perf_counter()

    # Clear old debug images
    clear_debug_images(DEBUG_DIR)

    # Step 1: Load captured ROI image
    t0 = time.perf_counter()
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found at path: {image_path}")
    timings['load_image'] = time.perf_counter() - t0

    # Step 2: Locate card using finalized YOLO + axis pipeline.
    # If YOLO cannot find a valid OBB, treat as non-identification and continue loop.
    t0 = time.perf_counter()
    try:
        rotated_cropped_image, locator_meta = locate_card_with_yolo_axis_pipeline(image)
    except Exception as exc:
        timings['locate_card_pipeline'] = time.perf_counter() - t0
        timings['total'] = time.perf_counter() - t_total_start
        LAST_RECOGNITION_TIMINGS = timings
        print(f"Locator failed: {exc}", flush=True)
        return None, LOCATOR_FAILURE_MARKER, None, None
    timings['locate_card_pipeline'] = time.perf_counter() - t0
    print(
        "Locator applied | "
        f"angle={locator_meta.get('axis_angle_deg')} "
        f"width={locator_meta.get('corrected_width')}",
        flush=True,
    )

    if rotated_cropped_image.shape[0] > 0 and rotated_cropped_image.shape[1] > 0:
        cv2.imwrite(os.path.join(DEBUG_DIR, 'rotated_cropped_image.png'), rotated_cropped_image)
    else:
        timings['total'] = time.perf_counter() - t_total_start
        LAST_RECOGNITION_TIMINGS = timings
        print('Error: Cropped image has invalid dimensions.', flush=True)
        return None, None, None, None

    # Step 4: Save expanded card crop and debug crop regions
    t0 = time.perf_counter()
    top_crop, bottom_crop, debug_img = crop_physical_regions(rotated_cropped_image)

    # Save debug images
    cv2.imwrite(os.path.join(DEBUG_DIR, "debug_crops.png"), debug_img)
    cv2.imwrite(os.path.join(DEBUG_DIR, "top_crop.png"), top_crop)
    cv2.imwrite(os.path.join(DEBUG_DIR, "bottom_crop.png"), bottom_crop)
    timings['crop_regions_and_debug_save'] = time.perf_counter() - t0

    # Step 5: Grid hashing with upright + 180-degree orientation handling.
    t0 = time.perf_counter()
    matcher = get_grid_matcher()
    identify_result = matcher.identify(rotated_cropped_image)
    _LAST_GRID_IDENTIFY_RESULT = identify_result
    if SAVE_GRID_IDENTIFY_RESULT_JSON:
        with open(os.path.join(DEBUG_DIR, "grid_identify_result.json"), "w", encoding="utf-8") as f:
            json.dump(identify_result, f, indent=2)
    timings['grid_identify'] = time.perf_counter() - t0

    best = identify_result.get("best")
    status = identify_result.get("status", "reject")
    card_info = None
    orientation = None
    distance = None

    if best:
        final_score = float(best.get("final_score") or 0.0)
        weighted_matches = float(best.get("weighted_matches") or 0.0)
        matched_regions = best.get("matched_regions")
        total_regions = best.get("total_regions")
        stage1_distance = best.get("stage1_distance")
        orientation = best.get("orientation")
        distance = best.get("avg_region_distance")
        if weighted_matches >= 1.5:
            card_info = resolve_card_info_from_db(card_hash_db, best.get("name"), best.get("set"))
            if card_info is None:
                card_info = {
                    "name": best.get("name", "Unknown"),
                    "set": best.get("set", ""),
                    "scryfall_uri": "",
                }

            print(
                f"Grid match: {best.get('name', 'Unknown')} ({best.get('set', '')}) | "
                f"status={status} | orientation={orientation} | "
                f"score={best.get('final_score')} | weighted={best.get('weighted_matches')} | "
                f"regions={matched_regions}/{total_regions} | stage1={stage1_distance}",
                flush=True,
            )
            print(f"Scryfall URI: {card_info.get('scryfall_uri', '')}", flush=True)
        else:
            print(
                f"Grid best rejected by weighted gate: {best.get('name', 'Unknown')} ({best.get('set', '')}) "
                f"| orientation={orientation} | weighted={best.get('weighted_matches')} < 1.5 "
                f"| score={best.get('final_score')} | status={status} "
                f"| regions={matched_regions}/{total_regions} | stage1={stage1_distance}",
                flush=True,
            )
    else:
        print(f"No grid match found (status={status}).", flush=True)

    # Step 6: Get prices if card was identified
    prices = None
    if card_info:
        t0 = time.perf_counter()
        prices = get_scryfall_prices(card_info)
        print_card_prices(prices, default_price=0.37)
        timings['price_fetch'] = time.perf_counter() - t0
    else:
        timings['price_fetch'] = 0.0

    timings['total'] = time.perf_counter() - t_total_start
    LAST_RECOGNITION_TIMINGS = timings

    return card_info, orientation, distance, prices

# ==============================================================================
# GRADIO UI
# ==============================================================================
def test_ranges(*args):
    """Helper function for Gradio control panel"""
    uppers = list(args)
    lowers = [0] + uppers[:-1]
    details = []
    for i in range(6):
        details.append(f"Box {i+1}: {lowers[i]} to {uppers[i]}")
    return "\n".join(details)

def get_current_frame():
    """Get the current frame for Gradio display"""
    global current_frame
    with frame_lock:
        if current_frame is not None:
            return current_frame
    # Return a blank frame if none available
    return np.zeros((480, 640, 3), dtype=np.uint8)

def get_debug_images():
    """Get list of current debugging images"""
    debug_image_paths = sorted(glob.glob(os.path.join(DEBUG_DIR, "*.png")))
    if debug_image_paths:
        return debug_image_paths
    return []

def create_gradio_interface():
    """Create and return the Gradio interface for debug images only"""
    with gr.Blocks() as demo:
        gr.Markdown("# MTG Card Recognition Debug Images")
        gr.Markdown("## Card Recognition Debug Images")
        gr.Markdown("Images generated during the card recognition process.")
        gr.Markdown("**Note:** Live camera feed is displayed in a separate OpenCV window.")
        
        gallery = gr.Gallery(label="Debug Images", columns=3, height="auto")
        refresh_btn = gr.Button("🔄 Refresh Debug Images")
        
        def refresh_images():
            images = get_debug_images()
            return images if images else []
        
        refresh_btn.click(refresh_images, None, gallery)
        demo.load(refresh_images, None, gallery)

    return demo

# ==============================================================================
# CAMERA DETECTION FUNCTIONS
# ==============================================================================

CAMERA_INDEX = 0  # USB camera (0 = built-in webcam, 1+ = USB cameras)
MIN_CARD_AREA = 5000
MAX_CARD_AREA = 2000000  # Increased for high-res camera

# Global variables for video streaming
current_frame = None
frame_lock = threading.Lock()
detection_active = True
EDGE_THRESHOLD_LOW = 50
EDGE_THRESHOLD_HIGH = 150
_RUNTIME_ROI_CONTROLS = None


def get_runtime_roi_controls():
    """Load the exact fixed ROI controls used by test_yolo_gridhash_pipeline."""
    global _RUNTIME_ROI_CONTROLS
    if _RUNTIME_ROI_CONTROLS is None:
        live_controls_preset = ycs.load_live_controls_preset(ycs.LIVE_CONTROLS_PRESET_DEFAULT)
        _RUNTIME_ROI_CONTROLS = ycs.get_fixed_roi_controls(live_controls_preset)
    return _RUNTIME_ROI_CONTROLS


def get_runtime_frame_and_roi(frame):
    """Match test_yolo_gridhash_pipeline camera frame preprocessing and ROI extraction."""
    corrected = ycs.apply_lens_correction_custom(
        frame,
        ycs.DEFAULT_LENS_K1,
        ycs.DEFAULT_LENS_K2,
        ycs.DEFAULT_LENS_FOCAL_SCALE,
    )
    controls = get_runtime_roi_controls()
    roi, roi_x1, roi_y1, roi_x2, roi_y2 = ycs.get_shifted_roi_full(
        sys.modules[__name__],
        corrected,
        controls["roi_shift_x"],
        controls["roi_shift_y"],
        controls["roi_expand_left"],
        controls["roi_expand_right"],
        controls["roi_expand_up"],
        controls["roi_expand_down"],
    )
    return corrected, roi, roi_x1, roi_y1, roi_x2, roi_y2

def get_center_roi(frame, roi_width_percent=0.20, roi_height_percent=0.34):
    """
    Extract center region of interest from frame - shifted left 3.4% and down 3.4%.
    Width: 20%, Height: 34%
    Returns (roi, roi_x1, roi_y1, roi_x2, roi_y2)
    """
    h, w = frame.shape[:2]
    
    roi_w = int(w * roi_width_percent)
    roi_h = int(h * roi_height_percent)
    roi_x1 = (w - roi_w) // 2 - int(w * 0.027)  # Shifted left by 3.4%
    roi_y1 = (h - roi_h) // 2 + int(h * 0.027)  # Shifted down by 3.4%
    roi_x2 = roi_x1 + roi_w
    roi_y2 = roi_y1 + roi_h
    
    roi = frame[roi_y1:roi_y2, roi_x1:roi_x2]
    return roi, roi_x1, roi_y1, roi_x2, roi_y2

def draw_text_with_background(frame, text, position, font_scale=0.6, thickness=1, 
                               text_color=(255, 255, 255), bg_color=(0, 0, 0)):
    """Draw text with a background rectangle for better visibility"""
    font = cv2.FONT_HERSHEY_SIMPLEX
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = position
    # Draw background rectangle
    cv2.rectangle(frame, (x - 5, y - text_height - 5), 
                  (x + text_width + 5, y + baseline + 5), bg_color, -1)
    # Draw text
    cv2.putText(frame, text, (x, y), font, font_scale, text_color, thickness)

def detect_card_in_frame(frame):
    """
    Detect if a card is present in the runtime ROI using YOLO.
    Uses the same model family as localization so presence check and box source match.
    Returns (has_card, debug_info_dict)
    """
    _, roi, roi_x1, roi_y1, roi_x2, roi_y2 = get_runtime_frame_and_roi(frame)

    debug_info = {
        'detected': False,
        'bbox': None,
        'obb_quad': None,
        'confidence': None,
        'detections': 0,
        'source': 'yolo',
    }

    try:
        model = get_yolo_locator_model()
        results = model.predict(
            source=[roi],
            imgsz=YOLO_LOCATOR_IMGSZ,
            conf=YOLO_LOCATOR_CONF,
            device=_YOLO_LOCATOR_DEVICE_RESOLVED,
            save=False,
            verbose=False,
        )
    except Exception as exc:
        debug_info['error'] = str(exc)
        return False, debug_info

    if not results:
        return False, debug_info

    result = results[0]
    picks = yolo_locator.obb_points_from_result(result, top_k=1)
    debug_info['detections'] = len(picks)
    if not picks:
        return False, debug_info

    quad, conf = picks[0]
    quad = np.asarray(quad, dtype=np.float32)
    x_min = int(np.clip(np.min(quad[:, 0]), 0, roi.shape[1] - 1))
    x_max = int(np.clip(np.max(quad[:, 0]), 0, roi.shape[1] - 1))
    y_min = int(np.clip(np.min(quad[:, 1]), 0, roi.shape[0] - 1))
    y_max = int(np.clip(np.max(quad[:, 1]), 0, roi.shape[0] - 1))

    w = max(0, x_max - x_min)
    h = max(0, y_max - y_min)
    if w <= 0 or h <= 0:
        return False, debug_info

    # Convert ROI-relative detection back to full corrected-frame coordinates.
    debug_info['detected'] = True
    debug_info['confidence'] = float(conf)
    debug_info['obb_quad'] = (quad + np.array([roi_x1, roi_y1], dtype=np.float32)).tolist()
    debug_info['bbox'] = (roi_x1 + x_min, roi_y1 + y_min, w, h)
    return True, debug_info

def wait_for_card_detection(cap):
    """
    Wait for a card to be detected by the camera.
    Returns True when a card is detected.
    Displays camera feed with ROI visualization.
    """
    if not cap or not cap.isOpened():
        print("ERROR: Could not open camera", flush=True)
        return False
    
    last_detection_time = 0
    detection_interval = 0.8
    
    while True:  # No timeout - runs indefinitely
        ret, frame = cap.read()
        
        if not ret:
            continue
        
        current_time = time.time()
        corrected_frame, _, roi_x1, roi_y1, roi_x2, roi_y2 = get_runtime_frame_and_roi(frame)
        debug_frame = corrected_frame.copy()
        
        # Draw ROI rectangle
        cv2.rectangle(debug_frame, (roi_x1, roi_y1), (roi_x2, roi_y2), (255, 255, 0), 2)
        
        # Check for card detection
        card_detected = False
        if current_time - last_detection_time > detection_interval:
            has_card, _ = detect_card_in_frame(frame)
            if has_card:
                card_detected = True
                cv2.putText(debug_frame, "CARD DETECTED!", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
            last_detection_time = current_time
        
        cv2.putText(debug_frame, "Press Q to abort", (50, frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        cv2.imshow('Waiting for Card - Center region highlighted', debug_frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            print("Card detection aborted by user", flush=True)
            return False
        
        if card_detected:
            print("Card detected! Waiting 0.2 seconds before capture...", flush=True)
            time.sleep(0.2)
            return True
def engage_startup_processes():
    try:
        print("creating controller...", flush=True)
        controller = StepperController(port="COM8")
        print("Controller created", flush=True)        
        controller.open()
        print("Controller opened", flush=True)
            
        # Set servo 2 to home and keep it there
        print(f"Setting servo 2 to home ({SERVO2_HOME_ANGLE} degrees)...", flush=True)
        servo2_move_abs_with_fresh_check(controller, SERVO2_HOME_ANGLE, cap=None)
        print(f"Servo 2 positioned at home ({SERVO2_HOME_ANGLE} degrees)", flush=True)
    except Exception as e:
        print(f"Error connecting to controller: {e}", flush=True)
        exit(1)

    print("opening camera...", flush=True)
    cap, active_camera_index = ycs.configure_camera_for_yolo(
        sys.modules[__name__],
        None,
        None,
        None,
        None,
        None,
    )
    print(f"Camera configured through YOLO_Camera_Studio (index {active_camera_index})", flush=True)

    print("loading hash database...", flush=True)
    with open('scryfall_phash_db_sample.json', 'r', encoding='utf-8') as f:
        card_hash_db = json.load(f)
    annoy_index, idx_to_card = load_or_build_annoy_index(card_hash_db)
    print("camera and database ready", flush=True)



def continuous_card_detection_and_processing(controller, cap, card_hash_db, annoy_index, idx_to_card):
    """
    Continuously detect cards, capture images, and process them through recognition system.
    Displays video feed in OpenCV window.
    """
    global detection_active
    print("Starting continuous card detection and processing...", flush=True)
    
    if not cap or not cap.isOpened():
        print("ERROR: Could not open camera", flush=True)
        return False
    
    last_detection_time = 0
    detection_interval = 0.8
    
    # Create OpenCV window
    cv2.namedWindow('Card Detection Feed', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Card Detection Feed', 800, 600)
    
    # Initial motor setup sequence
    print("Running initial motor setup...", flush=True)
    controller.spin_down_m1_until_switch()
    time.sleep(1)
    controller.spin_up_m1_until_switch()
    time.sleep(1)
    
    # Counter for tracking motor cycles
    motor_cycle_count = 0
    last_motor_time = 0  # Set to 0 to trigger immediately
    motor_interval = 2.0  # Time between motor cycles
    
    try:
        while detection_active:
            ret, frame = cap.read()
            
            if not ret:
                continue
            
            current_time = time.time()
            corrected_frame, _, roi_x1, roi_y1, roi_x2, roi_y2 = get_runtime_frame_and_roi(frame)
            debug_frame = corrected_frame.copy()
            
            # Draw ROI rectangle (bright cyan, thick line)
            cv2.rectangle(debug_frame, (roi_x1, roi_y1), (roi_x2, roi_y2), (0, 255, 255), 5)
            # Add label to ROI - this area is what gets captured
            draw_text_with_background(debug_frame, "CAPTURE ZONE (Only this area is saved)", (roi_x1, roi_y1 - 10), 
                                     font_scale=0.7, thickness=2, text_color=(0, 255, 255), bg_color=(0, 0, 0))
            
            # Run motor sequence periodically
            if current_time - last_motor_time > motor_interval:
                print(f"Running motor cycle {motor_cycle_count + 1}...", flush=True)
                
                # Motor 1 up 250 steps
                controller.spin_up_m1(250)
                
                # TT motor forward slow for 0.5s
                controller.tt_motor_forward("SLOW")
                time.sleep(0.5)
                controller.tt_motor_stop()
                
                # Sleep 0.3s
                time.sleep(0.3)
                
                # TT motor backward medium for 0.3s
                controller.tt_motor_reverse("MEDIUM")
                time.sleep(0.3)
                controller.tt_motor_stop()
                
                # Sleep 0.3s
                time.sleep(0.3)
                
                # Check for abort
                if check_abort():
                    print("\n⚠️  ABORT REQUESTED - Stopping motor sequence...\n", flush=True)
                    detection_active = False
                    break
                
                motor_cycle_count += 1
                last_motor_time = current_time
            
            # Check for card detection
            card_detected = False
            detection_info = ""
            if current_time - last_detection_time > detection_interval:
                has_card, debug_info = detect_card_in_frame(frame)
                detection_info = f"Area: {debug_info['area']} | AR: {debug_info['aspect_ratio']} | Contours: {debug_info['contours']}"
                if has_card:
                    card_detected = True
                    draw_text_with_background(debug_frame, "CARD DETECTED! WOWZA!!", (50, 80), 
                                             font_scale=1.2, thickness=3, text_color=(0, 255, 0), bg_color=(0, 0, 0))
                    
                    # Draw bright green bounding box around detected card
                    if debug_info['bbox'] is not None:
                        bbox_x, bbox_y, bbox_w, bbox_h = debug_info['bbox']
                        # Draw outer box (bright green, thick)
                        cv2.rectangle(debug_frame, (bbox_x, bbox_y), (bbox_x + bbox_w, bbox_y + bbox_h), 
                                    (0, 255, 0), 6)
                        # Draw inner box (yellow, medium)
                        cv2.rectangle(debug_frame, (bbox_x + 3, bbox_y + 3), 
                                    (bbox_x + bbox_w - 3, bbox_y + bbox_h - 3), (0, 255, 255), 3)
                    
                    print(f"Card detected! {detection_info}", flush=True)
                last_detection_time = current_time
            
            # Display debug info with backgrounds for visibility
            draw_text_with_background(debug_frame, f"Motor Cycles: {motor_cycle_count}", 
                                     (50, frame.shape[0] - 80), font_scale=0.6, thickness=2)
            draw_text_with_background(debug_frame, detection_info if detection_info else "Waiting for card...", 
                                     (50, frame.shape[0] - 50), font_scale=0.5, thickness=1, text_color=(255, 255, 0))
            draw_text_with_background(debug_frame, "Card Detection Active - Press 'p' to abort", 
                                     (50, frame.shape[0] - 20), font_scale=0.6, thickness=2)
            
            # Display in OpenCV window
            cv2.imshow('Card Detection Feed', debug_frame)
            
            if card_detected:
                print("Card detected! Waiting 0.5 seconds before capture...", flush=True)
                time.sleep(0.5)
                
                # Capture image and crop to ROI only
                ret, capture_frame = cap.read()
                if ret:
                    # Get ROI coordinates and crop the captured frame
                    corrected_capture, cropped_frame, roi_x1, roi_y1, roi_x2, roi_y2 = get_runtime_frame_and_roi(capture_frame)
                    
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    capture_path = os.path.join('Captured_Cards', f"card_{timestamp}.jpg")
                    cv2.imwrite(capture_path, cropped_frame)  # Save only the ROI
                    print(f"Image captured (ROI only): {capture_path}", flush=True)
                    print(f"Cropped to ROI size: {cropped_frame.shape[1]}x{cropped_frame.shape[0]}", flush=True)
                    
                    # Process through card recognition (using cropped ROI image)
                    print("Processing captured image through card recognition...", flush=True)
                    card_info, orientation, distance, prices = recognize_card(
                        capture_path, card_hash_db, annoy_index, idx_to_card
                    )
                    
                    if card_info:
                        print(f"Card identified: {card_info.get('name', 'Unknown')}", flush=True)
                        
                        # Execute servo sequence
                        print("Executing servo sequence...", flush=True)
                        # SERVO 2 DISABLED - CANNOT MOVE
                        # controller.servo2_move(-50)
                        # time.sleep(0.5)
                        # controller.servo2_move(90)
                    else:
                        print("Card not recognized", flush=True)
            
            # Handle window events
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # 'q' or ESC to quit
                detection_active = False
                break
    
    finally:
        detection_active = False
        cv2.destroyAllWindows()
        print("Detection stopped", flush=True)


def manual_stop_requested(controller):
    """Stop only on explicit keyboard stop keys outside the OpenCV window."""

    try:
        if msvcrt.kbhit():
            key = msvcrt.getch().decode('utf-8', errors='ignore').lower()
            if key in ('q', 'p', '\x1b'):
                print(f"Manual stop requested: keyboard key '{key}'.", flush=True)
                return True
    except Exception:
        pass

    return False


def sleep_with_manual_stop(seconds, controller, interval=0.05):
    """Sleep in short intervals so stop signals can be honored quickly."""
    start = time.time()
    while time.time() - start < seconds:
        if manual_stop_requested(controller):
            return True
        time.sleep(interval)
    return False


def servo2_move_abs_with_fresh_check(controller, angle, cap=None, check_delay=0.0):
    """Compatibility wrapper: move servo2 to absolute angle (no per-move card check)."""
    controller.servo2_move_abs(angle)
    return None


def envservorelease(controller, bottom_sleep_seconds=SERVO2_DROP_WAIT_SECONDS, cap=None):
    """Release servo2 in environment: go to drop angle, wait at bottom, then return home."""
    drop_angle = max(0, min(180, SERVO2_HOME_ANGLE - SERVO2_DROP_DELTA))
    servo2_move_abs_with_fresh_check(controller, drop_angle, cap=cap)
    time.sleep(max(0.0, bottom_sleep_seconds))
    servo2_move_abs_with_fresh_check(controller, SERVO2_HOME_ANGLE, cap=cap)


def run_servo2_drop_cycle(controller, wait_seconds=SERVO2_DROP_WAIT_SECONDS, cap=None):
    """Backward-compatible alias for the environment servo release helper."""
    envservorelease(controller, wait_seconds, cap=cap)


def run_card_release_attempt(controller):
    """Attempt a single card release using TT motor sequence with one M1 lift."""
    controller.spin_up_m1(192)
    controller.tt_motor_forward_for(700, "SLOW")
    time.sleep(0.7)
    time.sleep(0.3)

    controller.tt_motor_reverse_for(300, "MEDIUM")
    time.sleep(0.3)
    time.sleep(0.3)


def spin_m1_until_switch_or_key(controller, direction, step_chunk=350):
    """Spin motor 1 in one direction until switch or any key is pressed."""
    direction_label = "down" if direction == "down" else "up"
    print(f"Motor 1 spinning {direction_label}. Press any key or trigger switch to stop.", flush=True)

    while True:
        try:
            if controller.is_switch_pressed():
                print(f"Motor 1 {direction_label} stopped by switch.", flush=True)
                return
        except Exception:
            pass

        try:
            if msvcrt.kbhit():
                key = msvcrt.getch().decode('utf-8', errors='ignore')
                print(f"Motor 1 {direction_label} stopped by keyboard key '{key}'.", flush=True)
                return
        except Exception:
            pass

        if direction == "down":
            controller.spin_down_m1(step_chunk)
        else:
            controller.spin_up_m1(step_chunk)

        time.sleep(0.03)


def clear_keyboard_buffer():
    """Drain pending keypresses so startup stop keys do not terminate the main loop."""
    try:
        while msvcrt.kbhit():
            msvcrt.getch()
    except Exception:
        pass


def wait_for_start_signal(cap, window_name):
    """Show live feed and wait for manual ('s') or automatic ('a') start."""
    print("Waiting for start signal. Press 's' for manual mode, 'a' for automatic mode, or 'q'/ESC to cancel.", flush=True)

    while True:
        ret, live_frame = cap.read()
        if not ret or live_frame is None:
            continue

        corrected_frame, _, roi_x1, roi_y1, roi_x2, roi_y2 = get_runtime_frame_and_roi(live_frame)
        preview = corrected_frame.copy()
        cv2.rectangle(preview, (roi_x1, roi_y1), (roi_x2, roi_y2), (0, 255, 255), 3)
        draw_text_with_background(preview, "Live Feed Ready", (20, 35), font_scale=0.8, thickness=2)
        draw_text_with_background(preview, "Press 's' = manual, 'a' = automatic", (20, 70), font_scale=0.8, thickness=2)
        draw_text_with_background(preview, "Press 'q' or ESC to cancel", (20, live_frame.shape[0] - 20), font_scale=0.6, thickness=2)
        cv2.imshow(window_name, preview)

        window_key = cv2.waitKey(1) & 0xFF
        if window_key in (ord('s'), ord('S')):
            clear_keyboard_buffer()
            print("Manual mode selected from camera window.", flush=True)
            return "manual"
        if window_key in (ord('a'), ord('A')):
            clear_keyboard_buffer()
            print("Automatic mode selected from camera window.", flush=True)
            return "automatic"
        if window_key == ord('q') or window_key == 27:
            print("Start canceled from camera window.", flush=True)
            return None

        try:
            if msvcrt.kbhit():
                key = msvcrt.getch().decode('utf-8', errors='ignore').lower()
                if key == 's':
                    clear_keyboard_buffer()
                    print("Manual mode selected from keyboard.", flush=True)
                    return "manual"
                if key == 'a':
                    clear_keyboard_buffer()
                    print("Automatic mode selected from keyboard.", flush=True)
                    return "automatic"
                if key == 'q':
                    print("Start canceled from keyboard.", flush=True)
                    return None
        except Exception:
            pass


def wait_for_route_thread_completion(route_thread, route_result, controller, wait_reason):
    """Wait for an active route thread to finish, honoring manual stop keys."""
    if route_thread is None:
        return True

    while route_thread.is_alive():
        if manual_stop_requested(controller):
            print(f"Stop requested while waiting for routing thread ({wait_reason}).", flush=True)
            return False
        time.sleep(0.05)

    route_thread.join()
    if route_result.get('error') is not None:
        raise route_result['error']
    return True


def start_async_route_thread(controller, route_info, state):
    """Start route execution in a background thread so release logic can continue."""
    route_result = {'error': None}

    def _route_worker():
        try:
            route_identified_card(controller, route_info, state)
        except Exception as exc:
            route_result['error'] = exc

    route_thread = threading.Thread(target=_route_worker, daemon=True)
    route_thread.start()
    return route_thread, route_result


def automatic_release_identify_and_sort(controller, cap, card_hash_db, annoy_index, idx_to_card):
    """
    Automatic mode:
    - continuously run release attempts until a card is seen under camera
    - identify card with the existing pipeline
    - release via servo2 only when prior route thread has fully completed
    - route current card asynchronously while beginning next release attempts
    - after first successful identification, stop after 5 consecutive empty releases
    """
    state = {
        'isreadytorecieve_under_camera': True,
        'isreadytorecieve_m2_container': True,
        'iscardpresent': False,
    }
    identified_card_names = []
    first_card_identified = False
    empty_release_streak_after_first_identification = 0
    route_thread = None
    route_result = None
    window_name = "Live Camera Feed"
    next_card_presence_check_time = 0.0
    last_release_attempt_start_time = None
    try:
        run_sheet_info = init_automatic_run_sheet()
    except Exception as exc:
        print(f"Automatic run sheet initialization failed, continuing without Excel logging: {exc}", flush=True)
        run_sheet_info = None
    failed_identification_count = 0
    automatic_mode_start_ts = time.time()
    first_card_detected_under_camera_ts = None

    print("Starting automatic release-identify-sort loop", flush=True)
    print("Automatic mode: release attempts run continuously until card detection succeeds.", flush=True)
    print("Stop with hardware switch or q/ESC in the camera window.", flush=True)

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1000, 700)

    try:
        while True:
            ret_live, live_frame = cap.read()
            if ret_live and live_frame is not None:
                now = time.time()
                if now >= next_card_presence_check_time:
                    has_card_periodic, _ = detect_card_in_frame(live_frame)
                    state['iscardpresent'] = has_card_periodic
                    state['isreadytorecieve_under_camera'] = not has_card_periodic
                    next_card_presence_check_time = now + CARD_PRESENCE_CHECK_INTERVAL_SECONDS

                corrected_live, _, roi_x1, roi_y1, roi_x2, roi_y2 = get_runtime_frame_and_roi(live_frame)
                preview = corrected_live.copy()
                cv2.rectangle(preview, (roi_x1, roi_y1), (roi_x2, roi_y2), (0, 255, 255), 3)
                draw_text_with_background(preview, "Live Feed (Auto)", (20, 35), font_scale=0.8, thickness=2)
                draw_text_with_background(preview, f"Identified: {len(identified_card_names)}", (20, 70), font_scale=0.7, thickness=2)
                draw_text_with_background(
                    preview,
                    f"Empty release streak (post-first-identification): {empty_release_streak_after_first_identification}/5",
                    (20, 105),
                    font_scale=0.6,
                    thickness=2,
                )
                draw_text_with_background(preview, "q/ESC = stop", (20, live_frame.shape[0] - 20), font_scale=0.6, thickness=2)
                cv2.imshow(window_name, preview)

                window_key = cv2.waitKey(1) & 0xFF
                if window_key == ord('q') or window_key == 27:
                    print("Stopping automatic mode from camera window keypress.", flush=True)
                    break

            if manual_stop_requested(controller):
                print("Stopping automatic mode on manual request.", flush=True)
                break

            if route_thread is not None and not route_thread.is_alive():
                route_thread.join()
                if route_result is not None and route_result.get('error') is not None:
                    raise route_result['error']
                route_thread = None
                route_result = None

            if last_release_attempt_start_time is not None:
                elapsed = time.time() - last_release_attempt_start_time
                if elapsed < RELEASE_ATTEMPT_MIN_INTERVAL_SECONDS:
                    remaining = RELEASE_ATTEMPT_MIN_INTERVAL_SECONDS - elapsed
                    if sleep_with_manual_stop(remaining, controller):
                        print("Stopping automatic mode on manual request.", flush=True)
                        break

            print("Release one card: running M1 + TT release attempt.", flush=True)
            last_release_attempt_start_time = time.time()
            run_card_release_attempt(controller)
            card_entered = wait_for_card_entry_after_release(cap, controller)

            if card_entered is None:
                print("Stopping automatic mode on manual request.", flush=True)
                break

            if not card_entered:
                state['isreadytorecieve_under_camera'] = True
                print("No card detected in imaging area after release attempt.", flush=True)

                if first_card_identified:
                    empty_release_streak_after_first_identification += 1
                    if empty_release_streak_after_first_identification >= 5:
                        print("Finished the stack", flush=True)
                        break
                continue

            empty_release_streak_after_first_identification = 0
            state['iscardpresent'] = True
            state['isreadytorecieve_under_camera'] = False
            if first_card_detected_under_camera_ts is None:
                first_card_detected_under_camera_ts = time.time()
            print("Card detected in imaging area. Capturing and identifying...", flush=True)

            if sleep_with_manual_stop(CARD_CAPTURE_SETTLE_SECONDS, controller):
                print("Stopping automatic mode on manual request.", flush=True)
                break

            capture_path, _ = capture_roi_image(cap)
            if not capture_path:
                print("Capture failed. Continuing automatic release attempts.", flush=True)
                continue

            capture_to_result_start = time.perf_counter()
            card_info, orientation, distance, prices = recognize_card(
                capture_path, card_hash_db, annoy_index, idx_to_card
            )
            capture_to_result_total = time.perf_counter() - capture_to_result_start
            print_recognition_timing_summary(capture_to_result_total, LAST_RECOGNITION_TIMINGS)

            if not card_info:
                failed_identification_count += 1
                if orientation == LOCATOR_FAILURE_MARKER:
                    print("Skipping failure archive: YOLO OBB locator did not find card borders.", flush=True)
                else:
                    archive_failed_card_artifacts(
                        capture_path,
                        reason="card_not_identified",
                        extra={
                            'orientation': orientation,
                            'distance': distance,
                        },
                    )
                pending_route = build_post_release_route(prices=None, identified=False)
                print(
                    "Card could not be identified. Releasing from environment then routing asynchronously.",
                    flush=True,
                )
            else:
                card_name = card_info.get('name', 'Unknown')
                identified_card_names.append(card_name)
                first_card_identified = True
                print(f"Card identified: {card_name}", flush=True)
                archive_identified_card_artifacts(capture_path, card_info, orientation, distance, prices)
                cad_price, _cad_source = get_cad_price(prices if prices else {}, default_price=0.37)
                confidence = read_latest_match_confidence()
                append_automatic_run_entry(
                    run_sheet_info,
                    card_name=card_name,
                    capture_path=capture_path,
                    scryfall_uri=card_info.get('scryfall_uri', ''),
                    cad_price=cad_price,
                    confidence=confidence,
                )
                pending_route = build_post_release_route(prices, identified=True)

            if not wait_for_route_thread_completion(
                route_thread,
                route_result if route_result is not None else {'error': None},
                controller,
                wait_reason="servo2 safety interlock",
            ):
                break
            route_thread = None
            route_result = None

            print("Releasing card from environment with servo2.", flush=True)
            envservorelease(controller, cap=cap)

            route_thread, route_result = start_async_route_thread(controller, pending_route, state)

        if route_thread is not None:
            wait_for_route_thread_completion(
                route_thread,
                route_result if route_result is not None else {'error': None},
                controller,
                wait_reason="shutdown",
            )
    finally:
        if first_card_detected_under_camera_ts is not None:
            total_runtime_seconds = max(0.0, time.time() - first_card_detected_under_camera_ts)
        else:
            total_runtime_seconds = 0.0
        write_automatic_run_stats(
            run_sheet_info,
            total_runtime_seconds=total_runtime_seconds,
            identified_count=len(identified_card_names),
            failure_count=failed_identification_count,
        )
        cv2.destroyWindow(window_name)

    print("Automatic loop ended.", flush=True)
    return identified_card_names


def read_frame_and_detect_card(cap):
    """Read one frame and run card detection."""
    ret, frame = cap.read()
    if not ret or frame is None:
        return False, None
    has_card, _ = detect_card_in_frame(frame)
    return has_card, frame


def capture_roi_image(cap):
    """Capture current frame and return ROI crop + save path."""
    ret, capture_frame = cap.read()
    if not ret or capture_frame is None:
        return None, None

    _, cropped_frame, roi_x1, roi_y1, roi_x2, roi_y2 = get_runtime_frame_and_roi(capture_frame)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    capture_path = os.path.join('Captured_Cards', f"card_{timestamp}.jpg")
    cv2.imwrite(capture_path, cropped_frame)
    return capture_path, cropped_frame


def wait_for_card_entry_after_release(cap, controller):
    """
    Check after release for card entry:
    - check at +0.3s
    - then every 0.1s until total 0.6s
    """
    checkpoints = [0.3, 0.4, 0.5, 0.6]
    start = time.time()

    for checkpoint in checkpoints:
        remaining = checkpoint - (time.time() - start)
        if remaining > 0 and sleep_with_manual_stop(remaining, controller):
            return None
        has_card, _ = read_frame_and_detect_card(cap)
        if has_card:
            return True
    return False


def wait_for_switch_release_then_press(controller, poll_interval=0.05):
    """Require a full switch cycle: release, fresh press, then release again."""
    print("Startup switch gate: waiting for switch to release.", flush=True)
    while True:
        if manual_stop_requested(controller):
            return False
        try:
            if not controller.is_switch_pressed():
                break
        except Exception:
            pass
        time.sleep(poll_interval)

    print("Startup switch gate: waiting for switch to be pressed again.", flush=True)
    while True:
        if manual_stop_requested(controller):
            return False
        try:
            if controller.is_switch_pressed():
                print("Startup switch gate: second press detected.", flush=True)
                break
        except Exception:
            pass
        time.sleep(poll_interval)

    # Avoid immediate continuation while the second press is still being held.
    print("Startup switch gate: waiting for second press release.", flush=True)
    while True:
        if manual_stop_requested(controller):
            return False
        try:
            if not controller.is_switch_pressed():
                return True
        except Exception:
            pass
        time.sleep(poll_interval)


def build_post_release_route(prices, identified=True):
    """Classify the card into one of the three storage routes after it leaves the environment."""
    # If card is identified but has no prices, use default of $0.37 CAD
    default_price = 0.37 if identified else None
    cad_price, cad_source = get_cad_price(prices if prices else {}, default_price=default_price)

    if (not identified) or cad_price is None or cad_price <= 0:
        return {
            'route': 'unidentified',
            'cad_price': cad_price,
            'price_source': cad_source,
        }

    if cad_price <= 0.39:
        return {
            'route': 'low_value',
            'cad_price': cad_price,
            'price_source': cad_source,
        }

    return {
        'route': 'high_value',
        'cad_price': cad_price,
        'price_source': cad_source,
    }


def route_move_m2(controller, direction, steps=M2_ROUTE_STEPS):
    """Move M2 using the route command mode that mirrors the standalone tester behavior."""
    mode = str(M2_ROUTE_COMMAND_MODE).strip().lower()
    if steps < 0:
        steps = -steps

    if direction == 'down':
        if mode == 'async':
            controller.spin_down_m2_async(steps)
            controller.wait_for_m2_idle(
                max_wait_seconds=M2_ROUTE_ASYNC_MAX_WAIT_SECONDS,
                poll_interval=M2_ROUTE_ASYNC_POLL_SECONDS,
            )
        else:
            controller.spin_down_m2(steps)
        return

    if direction == 'up':
        if mode == 'async':
            controller.spin_up_m2_async(steps)
            controller.wait_for_m2_idle(
                max_wait_seconds=M2_ROUTE_ASYNC_MAX_WAIT_SECONDS,
                poll_interval=M2_ROUTE_ASYNC_POLL_SECONDS,
            )
        else:
            controller.spin_up_m2(steps)
        return

    raise ValueError(f"Unknown M2 route direction: {direction}")


def route_identified_card(controller, route_info, state):
    """Execute the storage motion once the just-released card has cleared the imaging area."""
    route_name = route_info.get('route', 'unidentified')
    cad_price = route_info.get('cad_price')
    price_source = route_info.get('price_source', 'unknown')
    state['isreadytorecieve_m2_container'] = False

    try:
        if route_name == 'low_value':
            price_note = f" (source: {price_source})" if price_source != 'unknown' else ""
            print(
                f"Routing low-value non-foil card (CAD=${cad_price:.2f}{price_note}) to bin 1 via servo1.",
                flush=True,
            )
            controller.servo1_rotate_90_and_back(delay_seconds=0.3)
            return

        if route_name == 'high_value':
            price_note = f" (source: {price_source})" if price_source != 'unknown' else ""
            print(
                f"Routing higher-value non-foil card (CAD=${cad_price:.2f}{price_note}) to bin 2 via M2 down {M2_ROUTE_STEPS} ({M2_ROUTE_COMMAND_MODE}).",
                flush=True,
            )
            route_move_m2(controller, 'down', M2_ROUTE_STEPS)
            time.sleep(M2_ROUTE_SETTLE_SECONDS)
            controller.servo1_rotate_90_and_back(delay_seconds=0.3)
            time.sleep(M2_ROUTE_SETTLE_SECONDS)
            route_move_m2(controller, 'up', M2_ROUTE_STEPS)
            return

        if cad_price is None:
            cad_text = "unknown"
        else:
            cad_text = f"{cad_price:.2f}"
        price_note = f" (source: {price_source})" if price_source != 'unknown' else ""
        print(
            f"Routing unidentified/non-positive-price card (CAD=${cad_text}{price_note}) to bin 3 via M2 up {M2_ROUTE_STEPS} ({M2_ROUTE_COMMAND_MODE}).",
            flush=True,
        )
        route_move_m2(controller, 'up', M2_ROUTE_STEPS)
        time.sleep(M2_ROUTE_SETTLE_SECONDS)
        controller.servo1_rotate_90_and_back(delay_seconds=0.3)
        time.sleep(M2_ROUTE_SETTLE_SECONDS)
        route_move_m2(controller, 'down', M2_ROUTE_STEPS)
    finally:
        state['iscardpresent'] = False
        state['isreadytorecieve_m2_container'] = True


def run_startup_and_motor_diagnostics(controller, cap):
    """
    Run one-pass diagnostics:
    - verify camera frame read
    - test each motor/servo briefly
    - return actuators to expected resting positions
    """
    print("\n=== Startup Diagnostics ===", flush=True)

    if not cap or not cap.isOpened():
        print("Camera check failed: camera is not open.", flush=True)
        return False

    ret, frame = cap.read()
    if not ret or frame is None:
        print("Camera check failed: could not read frame.", flush=True)
        return False

    print(f"Camera frame check passed: {frame.shape[1]}x{frame.shape[0]}", flush=True)

    try:
        print("Testing M1: up then back down...", flush=True)
        controller.spin_down_m1(4000)
        time.sleep(0.35)
        controller.spin_up_m1(4000)
        time.sleep(0.35)

        # Disabled for current bring-up:
        # print("Testing M2: down then back up...", flush=True)
        # controller.spin_down_m2(2050)
        # time.sleep(0.35)
        # controller.spin_up_m2(2050)
        # time.sleep(0.35)

        print("Testing TT motor: forward and reverse...", flush=True)
        controller.tt_motor_forward("SLOW")
        time.sleep(0.9)
        controller.tt_motor_stop()
        time.sleep(1)
        controller.tt_motor_reverse("SLOW")
        time.sleep(0.9)
        controller.tt_motor_stop()
        time.sleep(0.2)

        # Disabled for current bring-up:
        # print("Testing servo 1 sweep and return to center...", flush=True)
        # controller.servo1_rotate_90_and_back(delay_seconds=0.3)
        # time.sleep(0.3)

        print("Testing servo 2 movement and return to expected position...", flush=True)
        envservorelease(controller, cap=cap)
        time.sleep(0.3)

        print("Diagnostics complete: all motors exercised and returned.", flush=True)
        return True
    except Exception as exc:
        print(f"Diagnostics failed while testing actuators: {exc}", flush=True)
        return False


def continuous_release_identify_and_sort(controller, cap, card_hash_db, annoy_index, idx_to_card):
    """
    Key-driven loop:
    - run startup motor homing (M1 down-to-switch, then up-to-switch)
    - wait for manual release key
    - after release, detect + localize + identify
    - after identification, keep checking every ~0.4s for card presence
    - print "ready to rumble" when no card is under camera
    Stops when switch is pressed or q/ESC is pressed in the camera window.
    """
    state = {
        'isreadytorecieve_under_camera': True,
        'isreadytorecieve_m2_container': True,
        'iscardpresent': False,
    }
    identified_card_names = []
    stop_requested = False
    waiting_for_release_key = True
    monitoring_until_clear = False
    has_printed_waiting_message = False
    pending_post_release_route = None
    window_name = "Live Camera Feed"
    next_card_presence_check_time = 0.0
    last_release_attempt_start_time = None

    print("Starting key-driven release-identify-sort loop", flush=True)
    print("Press '\\' in the camera window to release a card.", flush=True)
    print("Stop with hardware switch or q/ESC in the camera window.", flush=True)
    print("Live preview enabled.", flush=True)

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1000, 700)

    selected_mode = wait_for_start_signal(cap, window_name)
    if selected_mode is None:
        cv2.destroyWindow(window_name)
        return identified_card_names

    print("Startup sequence: M1 down until switch, wait for switch cycle, M1 up until switch, then TT motor loop.", flush=True)
    controller.spin_down_m1_until_switch()
    if not wait_for_switch_release_then_press(controller):
        print("Startup canceled while waiting for switch cycle.", flush=True)
        cv2.destroyWindow(window_name)
        return identified_card_names
    controller.spin_up_m1_until_switch()
    time.sleep(0.2)
    # TT motor behavior begins only after the full startup switch cycle + M1 up is complete.

    if selected_mode == "automatic":
        cv2.destroyWindow(window_name)
        return automatic_release_identify_and_sort(controller, cap, card_hash_db, annoy_index, idx_to_card)

    while True:
        ret_live, live_frame = cap.read()
        if ret_live and live_frame is not None:
            now = time.time()
            if now >= next_card_presence_check_time:
                has_card_periodic, _ = detect_card_in_frame(live_frame)
                state['iscardpresent'] = has_card_periodic
                state['isreadytorecieve_under_camera'] = not has_card_periodic
                print(f"Card presence check (0.5s): under_camera={has_card_periodic}", flush=True)
                next_card_presence_check_time = now + CARD_PRESENCE_CHECK_INTERVAL_SECONDS

                if monitoring_until_clear:
                    if has_card_periodic:
                        if not has_printed_waiting_message:
                            print("Card still under camera. Waiting...", flush=True)
                            has_printed_waiting_message = True
                    else:
                        if pending_post_release_route is not None:
                            route_identified_card(controller, pending_post_release_route, state)
                            pending_post_release_route = None
                        print("ready to rumble", flush=True)
                        monitoring_until_clear = False
                        waiting_for_release_key = True
                        has_printed_waiting_message = False

            corrected_live, _, roi_x1, roi_y1, roi_x2, roi_y2 = get_runtime_frame_and_roi(live_frame)
            preview = corrected_live.copy()
            cv2.rectangle(preview, (roi_x1, roi_y1), (roi_x2, roi_y2), (0, 255, 255), 3)
            draw_text_with_background(preview, "Live Feed", (20, 35), font_scale=0.8, thickness=2)
            draw_text_with_background(preview, f"Identified: {len(identified_card_names)}", (20, 70), font_scale=0.7, thickness=2)
            draw_text_with_background(preview, "\\ = release | r = servo2 test | q/ESC = stop", (20, live_frame.shape[0] - 20), font_scale=0.6, thickness=2)
            cv2.imshow(window_name, preview)

            window_key = cv2.waitKey(1) & 0xFF
            if window_key == ord('\\'):
                print("Manual release key '\\' pressed.", flush=True)
                if waiting_for_release_key:
                    if last_release_attempt_start_time is not None:
                        elapsed = time.time() - last_release_attempt_start_time
                        if elapsed < RELEASE_ATTEMPT_MIN_INTERVAL_SECONDS:
                            remaining = RELEASE_ATTEMPT_MIN_INTERVAL_SECONDS - elapsed
                            if sleep_with_manual_stop(remaining, controller):
                                print("Stopping process on manual request.", flush=True)
                                stop_requested = True
                                break
                    last_release_attempt_start_time = time.time()
                    run_card_release_attempt(controller)
                    waiting_for_release_key = False
                    monitoring_until_clear = False
                    has_printed_waiting_message = False
                elif monitoring_until_clear:
                    if state.get('iscardpresent', False):
                        print("Release blocked: card still under camera. Clear environment first.", flush=True)
                    else:
                        print("Environment clear on periodic check. Releasing next card.", flush=True)
                        monitoring_until_clear = False
                        waiting_for_release_key = False
                        has_printed_waiting_message = False
                        if last_release_attempt_start_time is not None:
                            elapsed = time.time() - last_release_attempt_start_time
                            if elapsed < RELEASE_ATTEMPT_MIN_INTERVAL_SECONDS:
                                remaining = RELEASE_ATTEMPT_MIN_INTERVAL_SECONDS - elapsed
                                if sleep_with_manual_stop(remaining, controller):
                                    print("Stopping process on manual request.", flush=True)
                                    stop_requested = True
                                    break
                        last_release_attempt_start_time = time.time()
                        run_card_release_attempt(controller)
                else:
                    print("Release request ignored: current card cycle still active.", flush=True)
                continue
            if window_key == ord('r'):
                print("Manual servo2 key 'r' pressed.", flush=True)
                envservorelease(controller, cap=cap)
                continue
            if window_key == ord('q') or window_key == 27:
                print("Stopping process from camera window keypress.", flush=True)
                stop_requested = True
                break

        if manual_stop_requested(controller):
            print("Stopping process on manual request.", flush=True)
            stop_requested = True
            break

        if monitoring_until_clear:
            # Periodic 0.5s card checks are handled in the live-frame branch above.
            time.sleep(0.01)
            continue

        if waiting_for_release_key:
            # Idle state: wait for '\\' key in window handler.
            time.sleep(0.01)
            continue

        card_entered = wait_for_card_entry_after_release(cap, controller)
        if card_entered is None:
            print("Stopping process on manual request.", flush=True)
            break
        if not card_entered:
            state['isreadytorecieve_under_camera'] = True
            print("No card detected after release window. Press '\\' to try again.", flush=True)
            waiting_for_release_key = True
            continue

        state['iscardpresent'] = True
        state['isreadytorecieve_under_camera'] = False
        print("Card detected in imaging area. Capturing and identifying...", flush=True)

        if sleep_with_manual_stop(CARD_CAPTURE_SETTLE_SECONDS, controller):
            print("Stopping process on manual request.", flush=True)
            break

        capture_path, _ = capture_roi_image(cap)
        if not capture_path:
            print("Capture failed. Press '\\' to retry release.", flush=True)
            waiting_for_release_key = True
            continue

        capture_to_result_start = time.perf_counter()
        card_info, orientation, distance, prices = recognize_card(
            capture_path, card_hash_db, annoy_index, idx_to_card
        )
        capture_to_result_total = time.perf_counter() - capture_to_result_start
        print_recognition_timing_summary(capture_to_result_total, LAST_RECOGNITION_TIMINGS)

        if not card_info:
            if orientation == LOCATOR_FAILURE_MARKER:
                print("Skipping failure archive: YOLO OBB locator did not find card borders.", flush=True)
            else:
                archive_failed_card_artifacts(
                    capture_path,
                    reason="card_not_identified",
                    extra={
                        'orientation': orientation,
                        'distance': distance,
                    },
                )
            pending_post_release_route = build_post_release_route(prices=None, identified=False)
            print(
                "Card could not be identified. Releasing from environment and waiting for clear before routing.",
                flush=True,
            )
            envservorelease(controller, cap=cap)
            monitoring_until_clear = True
            waiting_for_release_key = False
            has_printed_waiting_message = False
            continue

        card_name = card_info.get('name', 'Unknown')
        identified_card_names.append(card_name)
        print(f"Card identified: {card_name}", flush=True)
        archive_identified_card_artifacts(capture_path, card_info, orientation, distance, prices)

        pending_post_release_route = build_post_release_route(prices, identified=True)
        print("Releasing identified card from environment with servo2.", flush=True)
        envservorelease(controller, cap=cap)
        monitoring_until_clear = True
        waiting_for_release_key = False
        has_printed_waiting_message = False

    print("Perpetual loop ended.", flush=True)
    cv2.destroyWindow(window_name)
    return identified_card_names
# ==============================================================================
# MAIN EXECUTION
# ==============================================================================
if __name__ == "__main__":
    APP_RUNTIME_START_TS = time.time()
    print("Script started!", flush=True)
    dry_run_mode = "--dry-run" in sys.argv
    print(f"Dry run mode: {dry_run_mode}", flush=True)

    # Example usage from AllMotors.py
    try:
        # I WANT TO PUT engage_startup_processes() here but I would have to check for debugging.
        print("creating controller...", flush=True)
        controller = DryRunController() if dry_run_mode else StepperController(port="COM8")
        print("Controller created", flush=True)
        controller.open()
        print("Controller opened", flush=True)
        
        # Set servo 2 to home and keep it there
        print(f"Setting servo 2 to home ({SERVO2_HOME_ANGLE} degrees)...", flush=True)
        servo2_move_abs_with_fresh_check(controller, SERVO2_HOME_ANGLE, cap=None)
        print(f"Servo 2 positioned at home ({SERVO2_HOME_ANGLE} degrees)", flush=True)
    except Exception as e:
        print(f"Error connecting to controller: {e}", flush=True)
        exit(1)

    print("opening camera...", flush=True)
    cap, active_camera_index = ycs.configure_camera_for_yolo(
        sys.modules[__name__],
        None,
        None,
        None,
        None,
        None,
    )
    actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    actual_brightness = cap.get(cv2.CAP_PROP_BRIGHTNESS)
    actual_contrast = cap.get(cv2.CAP_PROP_CONTRAST)
    print(f"Camera initialized at {int(actual_width)}x{int(actual_height)} (index {active_camera_index})", flush=True)
    print(f"Brightness: {int(actual_brightness)}, Contrast: {int(actual_contrast)}", flush=True)

    print("loading hash database...", flush=True)
    with open('scryfall_phash_db_sample.json', 'r', encoding='utf-8') as f:
        card_hash_db = json.load(f)
    annoy_index, idx_to_card = load_or_build_annoy_index(card_hash_db)
    print("camera and database ready", flush=True)
    prewarm_recognition_pipeline()
    time.sleep(2)
    print("Setup has succeeded")

    # diagnostics_ok = run_startup_and_motor_diagnostics(controller, cap)
    # if not diagnostics_ok:
    #     print("Startup diagnostics reported a failure. Exiting before loop execution.", flush=True)

    # Gradio interface commented out - using OpenCV window only
    # print("Creating Gradio interface...", flush=True)
    # demo = create_gradio_interface()
    
    # Run requested perpetual release/detect/identify/sort loop
    identified_cards = []
    try:
        identified_cards = continuous_release_identify_and_sort(controller, cap, card_hash_db, annoy_index, idx_to_card)
        # What does this involve?
        """
        1. Attempt card release, verify if any card has entered the environment
            If Yes: 
                - Image and identify the card
                - Once it's information is obtained, release card from environment
                - As soon as servo is rotating back to closed, repeat 1.
            
            If No: Repeat 1

        2. If 1. 'Yes' has occured and all is successful, compute the position that the stepper motor must achieve, corresponding to the value of the card
                - Note, about 2050 steps between one box and another
            
            Once card has entered the container above the storage, rotate the stepper motor to go above the relevant box,
            then spin forward and backwards the servo to release

            Spin both the servo and the stepper back to the origional position, repeat step 2.

        NOTE: in essence the idea is that step one is all about releasing and interpreting the card
              step 2 is about taking that card and putting it into storage. This loop should keep running and running and running until it takes __ to ___ attempts to release a card, and no cards are identified in the imaging area. I imagine something like 3 or 4 attempts would suffice, never enough to damage anything or spin anything out of place.

        """

    finally:
        detection_active = False
        cap.release()
        controller.close()
        print("\nIdentified card names this run:", flush=True)
        if identified_cards:
            for idx, card_name in enumerate(identified_cards, start=1):
                print(f"{idx}. {card_name}", flush=True)
        else:
            print("No cards were identified before termination.", flush=True)
        print("Application closed", flush=True)
    
    # Alternative: Use Gradio interface with background detection
    # print("Creating Gradio interface...", flush=True)
    # demo = create_gradio_interface()
    # detection_thread = threading.Thread(
    #     target=continuous_card_detection_and_processing,
    #     args=(controller, cap, card_hash_db, annoy_index, idx_to_card),
    #     daemon=True
    # )
    # detection_thread.start()
    # print("Card detection started in background", flush=True)
    # try:
    #     print("Launching Gradio interface...", flush=True)
    #     demo.launch(share=False, inbrowser=True)
    # finally:
    #     detection_active = False
    #     detection_thread.join(timeout=2)
    #     cap.release()
    #     controller.close()
    #     print("Application closed", flush=True)
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    """
    controller = StepperController(port="COM8")
    controller.open()
    abort_requested = False
    
    try:
        time.sleep(4)
        controller.servo_rotate_90_and_back(1)
        time.sleep(2)
        controller.spin_down_m2(2050)
        controller.servo_rotate_90_and_back(1)
        time.sleep(1)
        controller.spin_up_m2(2050)
        time.sleep(1)
        controller.spin_down_m2(4100)
        controller.servo_rotate_90_and_back(1)
        time.sleep(1)
        controller.spin_up_m2(4100)
    finally:    
        controller.close()


    # Example usage from Main copy.ipynb

    # Define the path to the Image_storage folder
    image_storage_path = 'Image_storage/'
    image_path = image_storage_path + 'FinalSetupTest1.jpg'

    # Load the Scryfall perceptual hash database
    with open('scryfall_phash_db_sample.json', 'r', encoding='utf-8') as f:
        card_hash_db = json.load(f)

    # Load or build Annoy index
    annoy_index, idx_to_card = load_or_build_annoy_index(card_hash_db)

    # Recognize the card
    card_info, orientation, distance, prices = recognize_card(
        image_path, card_hash_db, annoy_index, idx_to_card
    )
    print(card_info, prices, flush=True)
    # Launch Gradio interface
    demo = create_gradio_interface()
    # demo.launch()  # Uncomment to launch Gradio UI


    print("=" * 70, flush=True)
    print("Magic: The Gathering Card Recognition and Motor Control System", flush=True)
    print("=" * 70, flush=True)
    print("\nThis file combines:", flush=True)
    print("  1. Motor control functionality (StepperController)", flush=True)
    print("  2. Card image recognition system", flush=True)
    print("\nUncomment the example usage sections in __main__ to run.", flush=True)
    print("=" * 70, flush=True)
    """