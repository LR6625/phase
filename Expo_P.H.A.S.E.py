"""
Magic: The Gathering Card Recognition and Motor Control System
================================================================
This script is the combination of the ipynb and the allmotors,py files, to be tested as of now:
1. Motor control functionality (StepperController from AllMotors.py)
2. Card image recognition system (from Main copy.ipynb)

All image operations use OpenCV/NumPy vectorized code for optimal performance.
Hash database search uses Annoy for fast nearest-neighbor queries.
"""
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
import glob
import shutil
import webbrowser
from annoy import AnnoyIndex
from ultralytics import YOLO
from mtg_hash_v2.config import SegmentedHashConfig
from mtg_hash_v2.matcher import HybridTopHalfMatcher
from mtg_hash_v2.segmented_hashing import split_into_grid
import YOLO_Camera_Studio as ycs
from phase_config import SERIAL_PORT, BAUD_RATE, CAMERA_INDEX

MOTOR_DEBUG_ENABLE = True
MOTOR_DEBUG_INCLUDE_SWITCH_QUERIES = False


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
    def __init__(self, port: str, baudrate: int = BAUD_RATE, timeout: float = 120.0) -> None:
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self._ser: Optional[serial.Serial] = None
        self._io_lock = threading.Lock()
        self._cmd_seq = 0

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
            should_log = MOTOR_DEBUG_ENABLE and (
                MOTOR_DEBUG_INCLUDE_SWITCH_QUERIES or (not is_switch_query)
            )

            self._cmd_seq += 1
            seq = self._cmd_seq
            tx_start = time.perf_counter()
            tx_wall = datetime.now().strftime("%H:%M:%S.%f")[:-3]

            if should_log:
                print(
                    f"[MOTOR TX #{seq}] {tx_wall} | cmd={clean_line} | port={self.port}",
                    flush=True,
                )

            try:
                self._ser.write(command.encode("utf-8"))
                self._ser.flush()
                response = self._ser.readline().decode("utf-8", errors="replace").strip()
            except Exception as exc:
                elapsed_ms = (time.perf_counter() - tx_start) * 1000.0
                print(
                    f"[MOTOR IO ERROR #{seq}] cmd={clean_line} | after={elapsed_ms:.1f}ms | error={exc}",
                    flush=True,
                )
                raise

            elapsed_ms = (time.perf_counter() - tx_start) * 1000.0
            if should_log:
                if response:
                    print(
                        f"[MOTOR RX #{seq}] after={elapsed_ms:.1f}ms | response={response}",
                        flush=True,
                    )
                else:
                    waiting = 0
                    try:
                        waiting = int(getattr(self._ser, "in_waiting", 0) or 0)
                    except Exception:
                        waiting = -1
                    print(
                        f"[MOTOR RX TIMEOUT #{seq}] after={elapsed_ms:.1f}ms | response=<empty> | in_waiting={waiting} | timeout={self.timeout}s",
                        flush=True,
                    )
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
        dwell = max(0.0, delay_seconds) + max(0.0, SERVO1_EXTRA_DWELL_SECONDS)
        self.servo1_move(90)
        time.sleep(dwell)
        self.servo1_move(-90)
        time.sleep(dwell)
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
YOLO_LOCATOR_MODEL_PATH = Path("models/best.pt")
YOLO_LOCATOR_IMGSZ = 640
YOLO_LOCATOR_CONF = 0.25
YOLO_LOCATOR_DEVICE = "auto"
GRID_MATCH_SOURCE_MODULE_PATH = Path("Expo_P.H.A.S.E.py")
GRID_MATCH_FULL_DB_PATH = Path("scryfall_phash_db_sample.json")
GRID_MATCH_SEGMENTED_DB_PATH = Path("scryfall_segmented_tophalf_db_v2.json")
SAVE_LOCATOR_DEBUG_ARTIFACTS = False
SAVE_GRID_IDENTIFY_RESULT_JSON = True
LOCATOR_SINGLE_MASK_DEBUG_NAME = "locator_mask_overlay.png"

_YOLO_LOCATOR_MODEL = None
_YOLO_LOCATOR_DEVICE_RESOLVED = None
_GRID_MATCHER = None
_GRID_MATCHER_DEMO = None
_GRID_MATCHER_DEMO_COUNT = 0
_LAST_GRID_IDENTIFY_RESULT = None
_LAST_LOCATOR_META = None
_LAST_YOLO_OBB_CONFIDENCE = None
_SCRYFALL_SESSION = None
_SCRYFALL_PRICE_CACHE = {}
SCRYFALL_REQUEST_TIMEOUT_SECONDS = 4.0
SERVO2_HOME_ANGLE = 150
SERVO2_DROP_DELTA = 140
SERVO2_DROP_WAIT_SECONDS = 0.8
SERVO1_EXTRA_DWELL_SECONDS = 0.5
CARD_PRESENCE_CHECK_INTERVAL_SECONDS = 0.5
RELEASE_ATTEMPT_MIN_INTERVAL_SECONDS = 0.6
RETRY_TO_CAPTURE_DELAY_SHIFT_SECONDS = 0.2
RELEASE_ATTEMPT_MIN_INTERVAL_SECONDS_NO_CARD = max(
    0.05,
    RELEASE_ATTEMPT_MIN_INTERVAL_SECONDS - RETRY_TO_CAPTURE_DELAY_SHIFT_SECONDS,
)
CARD_CAPTURE_SETTLE_SECONDS = 0.8 + RETRY_TO_CAPTURE_DELAY_SHIFT_SECONDS + 0.35
AUTO_STUCK_CARD_TIMEOUT_SECONDS = 1.5
AUTO_MAX_FORCED_UNIDENTIFIED_RECOVERIES = 3
AUTO_STUCK_REIDENTIFY_START_SECONDS = 0.6
AUTO_STUCK_REIDENTIFY_COOLDOWN_SECONDS = 0.75
AUTO_STUCK_REIDENTIFY_MAX_ATTEMPTS = 3
LOCATOR_FAILURE_MARKER = "__locator_failed__"
AUTO_RUN_WORKBOOK_PATH = Path("Identified_Cards") / "P.H.A.S.E. catalogue.xlsx"
LAST_RECOGNITION_TIMINGS = {}
APP_RUNTIME_START_TS = None
M1_RELEASE_STEPS_DEFAULT = 202
M1_RELEASE_STEPS_REDUCED = max(1, int(round(M1_RELEASE_STEPS_DEFAULT * 0.5)))
M1_LAST_ATTEMPT_BOOST_MULTIPLIER = 4
M1_LAST_ATTEMPT_BOOST_TRIES_BEFORE_SHUTDOWN = 2
AUTO_EJECT_EXTRA_FINAL_ATTEMPTS = 1
AUTO_EJECT_ATTEMPT_LIMIT = 8
M2_ROUTE_STEPS = 2050
# "sync" uses SPIN_CW/SPIN_CCW (blocking); "async" uses async commands + idle wait.
M2_ROUTE_COMMAND_MODE = "sync"
M2_ROUTE_ASYNC_MAX_WAIT_SECONDS = 40.0
M2_ROUTE_ASYNC_POLL_SECONDS = 0.20
M2_ROUTE_SETTLE_SECONDS = 0.2
TT_POST_REVERSE_JITTER_MS = 25
TT_POST_REVERSE_JITTER_SPEED = "CRAWL"
TT_POST_REVERSE_PRE_JITTER_SLEEP_SECONDS = 0.1

RUN_ACTION_RESTART = "restart"
RUN_ACTION_EXIT = "exit_process"

EXPO_VIS_DIR = Path("gen_expo_visualization_files")
EXPO_VIS_DATA_DIR = EXPO_VIS_DIR / "data"
EXPO_VIS_ASSETS_DIR = EXPO_VIS_DATA_DIR / "assets"
EXPO_VIS_LATEST_MATCH_PATH = EXPO_VIS_DATA_DIR / "latest_match.json"
EXPO_VIS_CONTROL_COMMAND_PATH = EXPO_VIS_DATA_DIR / "control_command.json"
DEMO_MODE_FULL_DB_PATH = EXPO_VIS_DATA_DIR / "demo_mode_full_db.json"
DEMO_MODE_SEGMENTED_DB_PATH = EXPO_VIS_DATA_DIR / "demo_mode_segmented_db.json"
EXPO_VIS_HOST = "127.0.0.1"
EXPO_VIS_PORT = 5050
_EXPO_VIS_SERVER_THREAD = None
_EXPO_VIS_SERVER_STARTED = False
_PRESENTATION_ASSIST_ENABLED = False
_PRESENTATION_ASSIST_ALLOWLIST = []
_RUN_START_SELECTION = None
DEMO_MODE_ALLOWLIST_PATH = EXPO_VIS_DATA_DIR / "demo_mode_allowlist.json"

PRINT_RECOGNITION_TIMINGS_ON_MATCH = True
PRINT_RECOGNITION_TIMINGS_ON_FAILURE = False
GRID_MATCH_WEIGHTED_MIN = 1.35


def _normalize_card_name(value):
    # Make allowlist-vs-DB name matching resilient to punctuation differences.
    text = str(value or '').strip().lower()
    text = re.sub(r"['’]", "", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r'\s+', ' ', text).strip()


def _normalize_set_code(value):
    return str(value or '').strip().lower()


def _card_identity_key(name, set_code):
    """Normalized run-level identity key for duplicate suppression checks."""
    return (_normalize_card_name(name), _normalize_set_code(set_code))


def _parse_presentation_assist_allowlist(raw_items):
    parsed = []
    if not isinstance(raw_items, list):
        return parsed

    for item in raw_items:
        name = None
        set_code = None
        if isinstance(item, str):
            name = item
        elif isinstance(item, dict):
            name = item.get('name')
            set_code = item.get('set')

        name_n = _normalize_card_name(name)
        if not name_n:
            continue
        parsed.append({'name': name_n, 'set': _normalize_set_code(set_code)})
    return parsed


def _apply_presentation_assist_settings(enabled, allowlist):
    global _PRESENTATION_ASSIST_ENABLED, _PRESENTATION_ASSIST_ALLOWLIST
    global _GRID_MATCHER_DEMO, _GRID_MATCHER_DEMO_COUNT
    _PRESENTATION_ASSIST_ENABLED = bool(enabled)
    _PRESENTATION_ASSIST_ALLOWLIST = _parse_presentation_assist_allowlist(allowlist)
    if _PRESENTATION_ASSIST_ENABLED:
        _GRID_MATCHER_DEMO = _build_demo_grid_matcher(_PRESENTATION_ASSIST_ALLOWLIST)
        _GRID_MATCHER_DEMO_COUNT = int(len(_PRESENTATION_ASSIST_ALLOWLIST))
        print(
            f"Presentation assist enabled with {len(_PRESENTATION_ASSIST_ALLOWLIST)} allowlisted cards.",
            flush=True,
        )
    else:
        _GRID_MATCHER_DEMO = None
        _GRID_MATCHER_DEMO_COUNT = 0
        print("Presentation assist disabled.", flush=True)


def _choose_presentation_assist_candidate(identify_result):
    if not _PRESENTATION_ASSIST_ENABLED or not _PRESENTATION_ASSIST_ALLOWLIST:
        return None

    candidates = []
    best = (identify_result or {}).get('best')
    if isinstance(best, dict):
        candidates.append(best)
    for item in (identify_result or {}).get('possible') or []:
        if isinstance(item, dict):
            candidates.append(item)

    for cand in candidates:
        cand_name = _normalize_card_name(cand.get('name'))
        cand_set = _normalize_set_code(cand.get('set'))
        for allowed in _PRESENTATION_ASSIST_ALLOWLIST:
            if cand_name != allowed['name']:
                continue
            if allowed['set'] and cand_set != allowed['set']:
                continue
            return cand
    return None


def _is_allowlisted_card(name, set_code):
    if not _PRESENTATION_ASSIST_ALLOWLIST:
        return False
    cand_name = _normalize_card_name(name)
    cand_set = _normalize_set_code(set_code)
    for allowed in _PRESENTATION_ASSIST_ALLOWLIST:
        if cand_name != allowed['name']:
            continue
        if allowed['set'] and cand_set != allowed['set']:
            continue
        return True
    return False


def _load_demo_mode_allowlist_from_file():
    """Load demo-mode allowlist from JSON file. Returns normalized allowlist entries."""
    try:
        if not DEMO_MODE_ALLOWLIST_PATH.exists():
            print(
                f"Demo mode allowlist file not found: {DEMO_MODE_ALLOWLIST_PATH}",
                flush=True,
            )
            return []
        # Accept UTF-8 with BOM because some editors save JSON that way.
        with DEMO_MODE_ALLOWLIST_PATH.open('r', encoding='utf-8-sig') as handle:
            payload = json.load(handle)
        parsed = _parse_presentation_assist_allowlist(payload)
        if not parsed:
            print(
                f"Demo mode allowlist is empty or invalid: {DEMO_MODE_ALLOWLIST_PATH}",
                flush=True,
            )
        return parsed
    except Exception as exc:
        print(f"Failed to load demo mode allowlist: {exc}", flush=True)
        return []


def _activate_demo_mode_from_allowlist(demo_allowlist):
    """Enable demo mode only if the restricted matcher can be built successfully."""
    if not demo_allowlist:
        print("Demo mode start blocked: allowlist file is missing or empty.", flush=True)
        return False

    _apply_presentation_assist_settings(True, demo_allowlist)
    try:
        get_grid_matcher()
    except Exception as exc:
        _apply_presentation_assist_settings(False, [])
        print(f"Demo mode start blocked: matcher not ready ({exc}).", flush=True)
        return False

    print(
        f"Demo mode active with {_GRID_MATCHER_DEMO_COUNT} allowlisted cards.",
        flush=True,
    )
    return True


def _build_demo_grid_matcher(allowlist):
    """Build a demo-only matcher constrained to allowlisted cards in local DB files."""
    allowlist_norm = _parse_presentation_assist_allowlist(allowlist)
    if not allowlist_norm:
        print("Demo matcher build skipped: empty allowlist.", flush=True)
        return None

    allow_pairs_ordered = []
    allow_pairs_seen = set()
    for item in allowlist_norm:
        pair = (item['name'], item['set'])
        if pair in allow_pairs_seen:
            continue
        allow_pairs_seen.add(pair)
        allow_pairs_ordered.append(pair)
    allow_pairs = set(allow_pairs_ordered)

    try:
        with GRID_MATCH_FULL_DB_PATH.open('r', encoding='utf-8') as handle:
            full_db = json.load(handle)
        with GRID_MATCH_SEGMENTED_DB_PATH.open('r', encoding='utf-8') as handle:
            segmented_db = json.load(handle)
    except Exception as exc:
        print(f"Demo matcher build failed while reading DB files: {exc}", flush=True)
        return None

    full_filtered_raw = [
        row for row in full_db
        if (_normalize_card_name((row or {}).get('name')), _normalize_set_code((row or {}).get('set'))) in allow_pairs
    ]
    segmented_filtered_raw = [
        row for row in segmented_db
        if (_normalize_card_name((row or {}).get('name')), _normalize_set_code((row or {}).get('set'))) in allow_pairs
    ]

    # Keep one row per normalized (name,set) pair so demo candidate space matches allowlist cardinality.
    full_by_pair = {}
    for row in full_filtered_raw:
        pair = (
            _normalize_card_name((row or {}).get('name')),
            _normalize_set_code((row or {}).get('set')),
        )
        if pair not in full_by_pair:
            full_by_pair[pair] = row

    segmented_by_pair = {}
    for row in segmented_filtered_raw:
        pair = (
            _normalize_card_name((row or {}).get('name')),
            _normalize_set_code((row or {}).get('set')),
        )
        if pair not in segmented_by_pair:
            segmented_by_pair[pair] = row

    full_filtered = [full_by_pair[pair] for pair in allow_pairs_ordered if pair in full_by_pair]
    segmented_filtered = [segmented_by_pair[pair] for pair in allow_pairs_ordered if pair in segmented_by_pair]

    missing_full = [pair for pair in allow_pairs_ordered if pair not in full_by_pair]
    missing_segmented = [pair for pair in allow_pairs_ordered if pair not in segmented_by_pair]

    if missing_full or missing_segmented or not full_filtered or not segmented_filtered:
        print(
            "Demo matcher build failed: allowlisted cards are missing from local hash DB files.",
            flush=True,
        )
        print(
            f"Filtered counts | full_raw={len(full_filtered_raw)} segmented_raw={len(segmented_filtered_raw)} "
            f"full_unique={len(full_filtered)} segmented_unique={len(segmented_filtered)} allowlist_unique={len(allow_pairs_ordered)}",
            flush=True,
        )
        if missing_full:
            print(f"Missing from full DB (sample): {missing_full[:5]}", flush=True)
        if missing_segmented:
            print(f"Missing from segmented DB (sample): {missing_segmented[:5]}", flush=True)
        return None

    try:
        EXPO_VIS_DATA_DIR.mkdir(parents=True, exist_ok=True)
        with DEMO_MODE_FULL_DB_PATH.open('w', encoding='utf-8') as handle:
            json.dump(full_filtered, handle)
        with DEMO_MODE_SEGMENTED_DB_PATH.open('w', encoding='utf-8') as handle:
            json.dump(segmented_filtered, handle)

        cfg = SegmentedHashConfig()
        matcher = HybridTopHalfMatcher(
            source_module_path=GRID_MATCH_SOURCE_MODULE_PATH,
            full_card_db_path=DEMO_MODE_FULL_DB_PATH,
            segmented_db_path=DEMO_MODE_SEGMENTED_DB_PATH,
            cfg=cfg,
        )
        print(
            f"Demo matcher ready with {len(full_filtered)} unique full-db cards and {len(segmented_filtered)} unique segmented records "
            f"(raw matches: full={len(full_filtered_raw)}, segmented={len(segmented_filtered_raw)}).",
            flush=True,
        )
        return matcher
    except Exception as exc:
        print(f"Demo matcher build failed: {exc}", flush=True)
        return None


def maybe_print_recognition_timing_summary(total_seconds, timings, is_match):
    if is_match and PRINT_RECOGNITION_TIMINGS_ON_MATCH:
        print_recognition_timing_summary(total_seconds, timings)
        return
    if (not is_match) and PRINT_RECOGNITION_TIMINGS_ON_FAILURE:
        print_recognition_timing_summary(total_seconds, timings)


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
    global _GRID_MATCHER, _GRID_MATCHER_DEMO
    if _PRESENTATION_ASSIST_ENABLED and _PRESENTATION_ASSIST_ALLOWLIST:
        if _GRID_MATCHER_DEMO is not None:
            return _GRID_MATCHER_DEMO
        _GRID_MATCHER_DEMO = _build_demo_grid_matcher(_PRESENTATION_ASSIST_ALLOWLIST)
        if _GRID_MATCHER_DEMO is None:
            raise RuntimeError(
                "Demo mode matcher unavailable. Ensure all allowlisted cards exist in local full+segmented DB files."
            )
        return _GRID_MATCHER_DEMO

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


def reset_scryfall_session():
    """Drop the current session so transient keep-alive failures can recover."""
    global _SCRYFALL_SESSION
    if _SCRYFALL_SESSION is not None:
        try:
            _SCRYFALL_SESSION.close()
        except Exception:
            pass
    _SCRYFALL_SESSION = None


def scryfall_get_with_retry(url, timeout=SCRYFALL_REQUEST_TIMEOUT_SECONDS, attempts=2, quiet=False):
    """Best-effort GET for Scryfall endpoints that never raises to caller."""
    if not url:
        return None

    last_error = None
    for attempt in range(max(1, int(attempts))):
        try:
            session = get_scryfall_session()
            return session.get(url, timeout=timeout)
        except requests.RequestException as exc:
            last_error = exc
            reset_scryfall_session()
            if attempt + 1 < max(1, int(attempts)):
                time.sleep(0.08)

    if not quiet and last_error is not None:
        print(f"Scryfall request failed for {url}: {last_error}", flush=True)
    return None


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
    global _LAST_LOCATOR_META, _LAST_YOLO_OBB_CONFIDENCE
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
    yolo_conf = picks[0][1] if len(picks[0]) > 1 else None
    try:
        yolo_conf_f = float(yolo_conf)
        _LAST_YOLO_OBB_CONFIDENCE = yolo_conf_f if np.isfinite(yolo_conf_f) else None
    except Exception:
        _LAST_YOLO_OBB_CONFIDENCE = None

    topbg_quad, topbg_debug = yolo_locator.refine_quad_with_top_mask(image_bgr, yolo_quad)
    line_info = None if topbg_debug is None else topbg_debug.get("line_info")

    final_crop, stage_images, meta = yolo_locator.build_axis_crop_stages(image_bgr, yolo_quad, line_info)
    _LAST_LOCATOR_META = dict(meta) if isinstance(meta, dict) else None

    # Always save exactly one mask visualization for failure archives.
    if topbg_debug is not None:
        top_mask = topbg_debug.get("bg_connected")
        bottom_mask = topbg_debug.get("bottom_bg_connected")
        if top_mask is not None or bottom_mask is not None:
            mask_overlay = image_bgr.copy()
            if top_mask is not None:
                mask_overlay[top_mask > 0] = (0, 0, 255)
            if bottom_mask is not None:
                mask_overlay[bottom_mask > 0] = (255, 0, 0)
            min_column_bottom_y = topbg_debug.get("min_column_bottom_y")
            if min_column_bottom_y is not None:
                y_line = int(np.clip(int(min_column_bottom_y), 0, max(0, mask_overlay.shape[0] - 1)))
                cv2.line(mask_overlay, (0, y_line), (mask_overlay.shape[1] - 1, y_line), (0, 255, 255), 2, cv2.LINE_AA)
            draw_bottom_guides_fn = getattr(yolo_locator, "draw_bottom_up_percentage_guides", None)
            if callable(draw_bottom_guides_fn):
                mask_overlay = draw_bottom_guides_fn(mask_overlay)
            cv2.imwrite(os.path.join(DEBUG_DIR, LOCATOR_SINGLE_MASK_DEBUG_NAME), mask_overlay)

            # Save a clean top/bottom contrast mask view without helper lines.
            clean_mask_overlay = image_bgr.copy()
            if top_mask is not None:
                clean_mask_overlay[top_mask > 0] = (0, 0, 255)
            if bottom_mask is not None:
                clean_mask_overlay[bottom_mask > 0] = (255, 0, 0)
            cv2.imwrite(os.path.join(DEBUG_DIR, "locator_top_bottom_mask_overlay_clean.png"), clean_mask_overlay)

    # Persist YOLO full-box preview with confidence text for localization display.
    yolo_guess_preview = stage_images.get("01_yolo_guess")
    if yolo_guess_preview is not None:
        yolo_guess_out = yolo_guess_preview.copy()
        if _LAST_YOLO_OBB_CONFIDENCE is not None:
            conf_pct = float(_LAST_YOLO_OBB_CONFIDENCE) * 100.0
            text = f"YOLO confidence: {conf_pct:.1f}%"
            (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.58, 2)
            x0, y0 = 10, 14
            x1 = min(yolo_guess_out.shape[1] - 1, x0 + tw + 14)
            y1 = min(yolo_guess_out.shape[0] - 1, y0 + th + baseline + 12)
            cv2.rectangle(yolo_guess_out, (x0, y0), (x1, y1), (10, 24, 40), -1)
            cv2.rectangle(yolo_guess_out, (x0, y0), (x1, y1), (150, 220, 255), 1)
            cv2.putText(
                yolo_guess_out,
                text,
                (x0 + 7, y1 - baseline - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.58,
                (233, 247, 255),
                2,
                cv2.LINE_AA,
            )
        cv2.imwrite(os.path.join(DEBUG_DIR, "locator_01_yolo_guess.png"), yolo_guess_out)

    # Save a single pre-crop axis/side-bounds preview where lines form the final crop box.
    axis_side_preview = None
    if isinstance(meta, dict):
        try:
            h, w = image_bgr.shape[:2]
            axis_angle = float(meta.get("axis_angle_deg", 0.0))
            matrix = cv2.getRotationMatrix2D((w * 0.5, h * 0.5), axis_angle, 1.0)
            axis_side_preview = cv2.warpAffine(
                image_bgr,
                matrix,
                (w, h),
                flags=cv2.INTER_CUBIC,
                borderMode=cv2.BORDER_REPLICATE,
            )

            top_y = int(np.clip(int(meta.get("top_y", 0)), 0, max(0, h - 1)))
            bottom_y = int(np.clip(int(meta.get("bottom_y", h)), 0, h))
            left_x = int(np.clip(int(meta.get("left_x", 0)), 0, max(0, w - 1)))
            right_x = int(np.clip(int(meta.get("right_x", w)), 0, w))
            axis_top_raw = meta.get("axis_top_y_raw", None)
            bottom_crop_cap = meta.get("bottom_crop_cap_y", None)

            if bottom_y <= top_y:
                bottom_y = min(h, top_y + 1)
            if right_x <= left_x:
                right_x = min(w, left_x + 1)

            if axis_top_raw is not None:
                axis_top_raw_i = int(np.clip(int(round(float(axis_top_raw))), 0, max(0, h - 1)))
                cv2.line(axis_side_preview, (0, axis_top_raw_i), (w - 1, axis_top_raw_i), (255, 0, 255), 1, cv2.LINE_AA)

            cv2.line(axis_side_preview, (0, top_y), (w - 1, top_y), (255, 255, 255), 2, cv2.LINE_AA)
            cv2.line(axis_side_preview, (0, bottom_y - 1), (w - 1, bottom_y - 1), (255, 255, 255), 2, cv2.LINE_AA)
            cv2.line(axis_side_preview, (left_x, 0), (left_x, h - 1), (0, 255, 255), 2, cv2.LINE_AA)
            cv2.line(axis_side_preview, (right_x - 1, 0), (right_x - 1, h - 1), (0, 255, 255), 2, cv2.LINE_AA)
            if bottom_crop_cap is not None:
                bottom_crop_cap_i = int(np.clip(int(bottom_crop_cap), 0, max(0, h - 1)))
                cv2.line(axis_side_preview, (0, bottom_crop_cap_i), (w - 1, bottom_crop_cap_i), (0, 140, 255), 1, cv2.LINE_AA)

            if axis_top_raw is not None:
                delta = int(round(float(top_y) - float(axis_top_raw)))
                text = f"raw_axis={float(axis_top_raw):.1f} used_top={top_y} delta={delta}px"
                cv2.putText(axis_side_preview, text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 255), 1, cv2.LINE_AA)
        except Exception:
            axis_side_preview = None

    if axis_side_preview is not None:
        cv2.imwrite(os.path.join(DEBUG_DIR, "locator_axis_side_box_preview.png"), axis_side_preview)

    if topbg_debug is not None:
        yolo_locator.save_topbg_debug_bundle(Path(DEBUG_DIR), "locator", image_bgr, topbg_debug)

    if SAVE_LOCATOR_DEBUG_ARTIFACTS:

        # Save only the finalized location pipeline stages for quick visual verification.
        for stage_name, stage_img in stage_images.items():
            cv2.imwrite(os.path.join(DEBUG_DIR, f"locator_{stage_name}.png"), stage_img)

        with open(os.path.join(DEBUG_DIR, "locator_meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
    else:
        # Keep key locator artifacts even when full bundle saving is disabled.
        # These are the authoritative rotated-frame stages for axis-vs-crop verification.
        for stage_name in (
            "03_rotated_flat_axis",
            "05_side_bounds_from_yolo_width",
            "06_bottom_axis",
            "09_final_crop",
        ):
            stage_img = stage_images.get(stage_name)
            if stage_img is not None:
                cv2.imwrite(os.path.join(DEBUG_DIR, f"locator_{stage_name}.png"), stage_img)
        side_guess = stage_images.get("05a_yolo_side_bounds_guess")
        if side_guess is not None:
            cv2.imwrite(os.path.join(DEBUG_DIR, "locator_05a_yolo_side_bounds_guess.png"), side_guess)
        with open(os.path.join(DEBUG_DIR, "locator_meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

    return final_crop, meta


def overlay_bottom_guard_guides_for_debug(image_bgr):
    """Overlay bottom guard guides if available in the locator module."""
    if image_bgr is None:
        return None
    out = image_bgr.copy()
    draw_bottom_guides_fn = getattr(yolo_locator, "draw_bottom_up_percentage_guides", None)
    if callable(draw_bottom_guides_fn):
        anchor_height = None
        y_offset = 0.0
        if isinstance(_LAST_LOCATOR_META, dict):
            anchor_height = _LAST_LOCATOR_META.get("source_height")
            y_offset = float(_LAST_LOCATOR_META.get("crop_top_y", _LAST_LOCATOR_META.get("top_y", 0.0)))
        out = draw_bottom_guides_fn(out, anchor_height=anchor_height, y_offset=y_offset)
    return out

def clear_debug_images(debug_dir):
    """Clear out old debug artifacts before processing a new card."""
    for f in glob.glob(os.path.join(debug_dir, '*.png')):
        try:
            os.remove(f)
        except Exception as e:
            print(f'Could not delete {f}: {e}', flush=True)
    for f in glob.glob(os.path.join(debug_dir, '*.json')):
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
        # Search for the card by exact name
        search_url = f"https://api.scryfall.com/cards/search?q=!%22{card_name}%22&order=released"
        response = scryfall_get_with_retry(search_url, quiet=True)
        if response is None:
            return {}
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

    try:
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

        response = scryfall_get_with_retry(api_url)
        if response is not None and response.status_code == 200:
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

        if response is not None:
            print(f"Failed to retrieve prices from {api_url} (status {response.status_code}). Searching other printings...", flush=True)
        card_name = card_info.get('name', '')
        if card_name:
            prices = get_prices_from_any_printing(card_name)
            _SCRYFALL_PRICE_CACHE[cache_key] = dict(prices or {})
            return prices
        _SCRYFALL_PRICE_CACHE[cache_key] = {}
        return {}
    except Exception as exc:
        print(f"Error retrieving Scryfall prices: {exc}", flush=True)
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


def _resolve_scryfall_card_api_url(card_info):
    scryfall_uri = str((card_info or {}).get('scryfall_uri', '') or '')
    m = re.search(r'/card/([^/]+)/([^/?]+)', scryfall_uri)
    if m:
        set_code, collector_number = m.group(1), m.group(2)
        return f"https://api.scryfall.com/cards/{set_code}/{collector_number}"
    if scryfall_uri:
        return scryfall_uri.replace('scryfall.com/card/', 'api.scryfall.com/cards/')
    return None


def _get_scryfall_image_url(card_info):
    api_url = _resolve_scryfall_card_api_url(card_info)
    if not api_url:
        return None
    try:
        response = scryfall_get_with_retry(api_url, quiet=True)
        if response is None:
            return None
        if response.status_code != 200:
            return None
        payload = response.json()
        image_uris = payload.get('image_uris') or {}
        if image_uris.get('normal'):
            return image_uris.get('normal')
        faces = payload.get('card_faces') or []
        for face in faces:
            face_uris = (face or {}).get('image_uris') or {}
            if face_uris.get('normal'):
                return face_uris.get('normal')
    except Exception:
        return None
    return None


def _humanize_stage_name(filename):
    stem = Path(filename).stem
    return stem.replace('_', ' ').replace('-', ' ').strip().title()


def _clear_expo_assets_dir():
    EXPO_VIS_ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    for asset in EXPO_VIS_ASSETS_DIR.glob('*'):
        if asset.is_file():
            try:
                asset.unlink()
            except Exception:
                pass


def _build_hash_binary_string(hash_obj):
    bits = hash_obj.hash.astype(np.uint8).flatten().tolist()
    return ''.join('1' if int(v) else '0' for v in bits)


def _phash_and_intermediates(image_bgr):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    gray32 = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA)
    f32 = np.float32(gray32)
    dct = cv2.dct(f32)
    low8 = dct[:8, :8]
    median = float(np.median(low8[1:, 1:]))
    bits8 = (low8 > median).astype(np.uint8)

    pil_img = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
    phash_obj = imagehash.phash(pil_img)
    return {
        'phash': phash_obj,
        'gray32': gray32,
        'dct32': dct,
        'low8': low8,
        'bits8': bits8,
    }


def _build_heatmap_image(matrix, label):
    norm = cv2.normalize(matrix, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    heat = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
    heat = cv2.resize(heat, (400, 400), interpolation=cv2.INTER_NEAREST)
    return heat


def _build_grayscale_8x8_image(matrix):
    norm = cv2.normalize(matrix, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    gray_up = cv2.resize(norm, (400, 400), interpolation=cv2.INTER_NEAREST)
    return cv2.cvtColor(gray_up, cv2.COLOR_GRAY2BGR)


def _build_bit_grid_image(bits8, label, diff_mask=None):
    cell = 40
    pad = 18
    h = (8 * cell) + (2 * pad)
    w = (8 * cell) + (2 * pad)
    canvas = np.full((h, w, 3), (255, 255, 255), dtype=np.uint8)
    for r in range(8):
        for c in range(8):
            y1 = pad + (r * cell)
            x1 = pad + (c * cell)
            y2 = y1 + cell - 2
            x2 = x1 + cell - 2
            bit = int(bits8[r, c])
            fill = (0, 0, 0) if bit == 1 else (255, 255, 255)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), fill, -1)
            border_color = (90, 90, 90)
            thickness = 2
            if diff_mask is not None and int(diff_mask[r, c]) == 1:
                border_color = (0, 0, 255)
                thickness = 4
                # Add a bright inner ring so differences are obvious on dark or light cells.
                cv2.rectangle(canvas, (x1 + 4, y1 + 4), (x2 - 4, y2 - 4), (0, 255, 255), 1)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), border_color, thickness)
    return canvas


def _build_hash_compare_panel(cam_bits, scr_bits):
    diff = np.bitwise_xor(cam_bits.astype(np.uint8), scr_bits.astype(np.uint8))
    left = _build_bit_grid_image(cam_bits, "Captured Card 64-Bit Hash", diff_mask=diff)
    right = _build_bit_grid_image(scr_bits, "Scryfall Match 64-Bit Hash", diff_mask=diff)
    spacer = np.full((left.shape[0], 20, 3), (255, 255, 255), dtype=np.uint8)
    combined = np.hstack([left, spacer, right])
    return combined


def _orient_image_for_display(image_bgr, orientation):
    if image_bgr is None:
        return None
    text = str(orientation or "").lower()
    if "180" in text:
        return cv2.rotate(image_bgr, cv2.ROTATE_180)
    return image_bgr


def _draw_segment_grid_overlay(image_bgr, cfg, label):
    img = image_bgr.copy()
    h, w = img.shape[:2]
    top_h = int(round(h * float(cfg.top_half_fraction)))
    top_h = max(1, min(h, top_h))
    rows = max(1, int(cfg.grid_rows))
    cols = max(1, int(cfg.grid_cols))

    cv2.rectangle(img, (0, 0), (w - 1, top_h - 1), (255, 200, 0), 2)
    for r in range(1, rows):
        y = int(round((top_h * r) / rows))
        cv2.line(img, (0, y), (w - 1, y), (255, 200, 0), 1, cv2.LINE_AA)
    for c in range(1, cols):
        x = int(round((w * c) / cols))
        cv2.line(img, (x, 0), (x, top_h - 1), (255, 200, 0), 1, cv2.LINE_AA)

    return img


def _segment_phashes(image_bgr, cfg):
    hashes, _bits = _segment_hashes_and_bits(image_bgr, cfg)
    return hashes


def _segment_hashes_and_bits(image_bgr, cfg):
    normalized = cv2.resize(image_bgr, cfg.normalized_size, interpolation=cv2.INTER_AREA)
    top_h = int(round(normalized.shape[0] * cfg.top_half_fraction))
    top_h = max(1, min(normalized.shape[0], top_h))
    top_half = normalized[:top_h, :]
    segments = split_into_grid(top_half, cfg.grid_rows, cfg.grid_cols)
    hashes = []
    bits = []
    for segment in segments:
        seg_img = segment.get("image") if isinstance(segment, dict) else segment
        if seg_img is None or not isinstance(seg_img, np.ndarray) or seg_img.size == 0:
            seg_img = np.zeros((16, 16, 3), dtype=np.uint8)
        pil = Image.fromarray(cv2.cvtColor(seg_img, cv2.COLOR_BGR2RGB))
        hashes.append(imagehash.phash(pil))
        bits.append(_phash_and_intermediates(seg_img)['bits8'])
    return hashes, bits


def _build_segment_bit_mosaic(segment_bits, diff_masks, match_flags, cfg, title):
    rows = int(cfg.grid_rows)
    cols = int(cfg.grid_cols)
    bit_cell = 6
    bit_pad = 6
    grid_size = (8 * bit_cell) + (2 * bit_pad)
    tile_w = grid_size + 16
    tile_h = grid_size + 30
    margin = 16
    gap = 10

    canvas_w = (cols * tile_w) + ((cols - 1) * gap) + (2 * margin)
    canvas_h = (rows * tile_h) + ((rows - 1) * gap) + (2 * margin) + 42
    canvas = np.full((canvas_h, canvas_w, 3), (255, 255, 255), dtype=np.uint8)

    for idx in range(rows * cols):
        r = idx // cols
        c = idx % cols
        x0 = margin + c * (tile_w + gap)
        y0 = margin + r * (tile_h + gap)

        bits8 = segment_bits[idx] if idx < len(segment_bits) else np.zeros((8, 8), dtype=np.uint8)
        diff = diff_masks[idx] if idx < len(diff_masks) else np.zeros((8, 8), dtype=np.uint8)
        is_match = bool(match_flags[idx]) if idx < len(match_flags) else False

        border_color = (30, 150, 30) if is_match else (0, 120, 220)
        cv2.rectangle(canvas, (x0, y0), (x0 + tile_w - 1, y0 + tile_h - 1), border_color, 2)
        cv2.putText(
            canvas,
            f"R{idx+1} {'OK' if is_match else 'DIFF'}",
            (x0 + 5, y0 + 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            border_color,
            2,
            cv2.LINE_AA,
        )

        gx = x0 + 8
        gy = y0 + 22
        for rr in range(8):
            for cc in range(8):
                bx1 = gx + bit_pad + (cc * bit_cell)
                by1 = gy + bit_pad + (rr * bit_cell)
                bx2 = bx1 + bit_cell - 1
                by2 = by1 + bit_cell - 1
                bit_val = int(bits8[rr, cc])
                fill = (0, 0, 0) if bit_val == 1 else (255, 255, 255)
                cv2.rectangle(canvas, (bx1, by1), (bx2, by2), fill, -1)

                bit_border = (140, 140, 140)
                if int(diff[rr, cc]) == 1:
                    bit_border = (0, 0, 255)
                cv2.rectangle(canvas, (bx1, by1), (bx2, by2), bit_border, 1)
    cv2.putText(canvas, title, (12, canvas_h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.66, (20, 90, 170), 2, cv2.LINE_AA)
    return canvas


def _build_segment_diff_map(cam_hashes, scr_hashes, cfg):
    rows = int(cfg.grid_rows)
    cols = int(cfg.grid_cols)
    cell = 122
    pad = 18
    h = (rows * cell) + (2 * pad) + 54
    w = (cols * cell) + (2 * pad)
    canvas = np.full((h, w, 3), (255, 255, 255), dtype=np.uint8)

    for idx in range(rows * cols):
        r = idx // cols
        c = idx % cols
        y1 = pad + (r * cell)
        x1 = pad + (c * cell)
        y2 = y1 + cell - 4
        x2 = x1 + cell - 4
        dist = int(cam_hashes[idx] - scr_hashes[idx])
        is_match = dist <= int(cfg.phash_threshold)
        fill = (195, 255, 205) if is_match else (205, 205, 255)
        edge = (25, 145, 25) if is_match else (0, 0, 255)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), fill, -1)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), edge, 4)
        cv2.putText(canvas, f"d={dist}", (x1 + 12, y1 + 42), cv2.FONT_HERSHEY_SIMPLEX, 0.78, (20, 40, 70), 2, cv2.LINE_AA)
        cv2.putText(canvas, "OK" if is_match else "DIFF", (x1 + 12, y1 + 72), cv2.FONT_HERSHEY_SIMPLEX, 0.70, edge, 2, cv2.LINE_AA)
    cv2.putText(canvas, "Top-half 3x3 segment hash map (green=match, red=difference)", (12, h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.80, (20, 90, 170), 2, cv2.LINE_AA)
    return canvas


def _build_grid_and_segment_panel(grid_overlay, segment_mosaic, title):
    if grid_overlay is None and segment_mosaic is None:
        return None
    if grid_overlay is None:
        return segment_mosaic
    if segment_mosaic is None:
        return grid_overlay

    left = grid_overlay.copy()
    right = segment_mosaic.copy()
    target_h = max(left.shape[0], right.shape[0])

    if left.shape[0] != target_h:
        left = cv2.resize(left, (int(round(left.shape[1] * (target_h / max(1, left.shape[0])))), target_h), interpolation=cv2.INTER_AREA)
    if right.shape[0] != target_h:
        right = cv2.resize(right, (int(round(right.shape[1] * (target_h / max(1, right.shape[0])))), target_h), interpolation=cv2.INTER_AREA)

    spacer = np.full((target_h, 22, 3), (255, 255, 255), dtype=np.uint8)
    combined = np.hstack([left, spacer, right])
    return combined


def _latest_confidence_details():
    best = {}
    try:
        best = (_LAST_GRID_IDENTIFY_RESULT or {}).get('best') or {}
    except Exception:
        best = {}

    final_score = best.get('final_score')
    weighted = best.get('weighted_matches')
    matched_regions = best.get('matched_regions')
    total_regions = best.get('total_regions')
    stage1_distance = best.get('stage1_distance')
    score = None
    try:
        if final_score is not None:
            score = float(final_score)
    except Exception:
        score = None

    if score is None:
        try:
            if weighted is not None:
                score = float(weighted)
        except Exception:
            score = None

    explain = (
        "Confidence blends weighted region matches, number of matched regions, and stage-1 shortlist distance. "
        "Higher weighted matches and more region agreement increase confidence."
    )
    return {
        'final_score': final_score,
        'weighted_matches': weighted,
        'matched_regions': matched_regions,
        'total_regions': total_regions,
        'stage1_distance': stage1_distance,
        'confidence_score': score,
        'explanation': explain,
    }


def _download_image_bgr(url):
    if not url:
        return None
    try:
        response = scryfall_get_with_retry(url, quiet=True)
        if response is None:
            return None
        if response.status_code != 200:
            return None
        arr = np.frombuffer(response.content, dtype=np.uint8)
        image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        return image
    except Exception:
        return None


def _save_expo_asset(image_bgr, filename, label, images_list):
    if image_bgr is None:
        return
    safe = sanitize_filename_component(filename)
    path = EXPO_VIS_ASSETS_DIR / safe
    cv2.imwrite(str(path), image_bgr)
    images_list.append({
        'label': label,
        'url': f"/api/assets/{safe}",
    })


def _save_expo_asset_from_path(source_path, filename, label, images_list):
    try:
        path = Path(source_path)
        if not path.exists():
            return
        image = cv2.imread(str(path))
        if image is None:
            return
        _save_expo_asset(image, filename, label, images_list)
    except Exception:
        return


def publish_latest_match_to_expo(capture_path, card_info, orientation, distance, prices):
    """Publish only successful card matches to the science-expo visualization payload."""
    if not card_info:
        return

    EXPO_VIS_DATA_DIR.mkdir(parents=True, exist_ok=True)
    _clear_expo_assets_dir()

    images = []

    camera_image_raw = None
    rotated_crop = Path(DEBUG_DIR) / 'rotated_cropped_image.png'
    if rotated_crop.exists():
        camera_image_raw = cv2.imread(str(rotated_crop))
    if camera_image_raw is None and capture_path and os.path.exists(capture_path):
        camera_image_raw = cv2.imread(capture_path)

    # Keep the main pipeline camera asset upright, but preserve raw orientation
    # for the Localization D card so users can inspect true capture orientation.
    camera_image = _orient_image_for_display(camera_image_raw, orientation)

    scryfall_image_url = _get_scryfall_image_url(card_info)
    scryfall_image = _download_image_bgr(scryfall_image_url)

    if camera_image is not None:
        _save_expo_asset(camera_image, '01_camera_capture.png', 'Step 1: Camera Card Image', images)
    if scryfall_image is not None:
        _save_expo_asset(scryfall_image, '02_scryfall_match.png', 'Step 1: Scryfall Card Image', images)

    # Detection + localization visuals for the dedicated second page.
    _save_expo_asset_from_path(
        Path(DEBUG_DIR) / 'locator_01_yolo_guess.png',
        '20_localization_a_yolo_guess.png',
        'Localization A: YOLO Guess (full box)',
        images,
    )
    _save_expo_asset_from_path(
        Path(DEBUG_DIR) / 'locator_top_bottom_mask_overlay_clean.png',
        '21_localization_b_contrast_masks.png',
        'Localization B: Contrast Masked Boxes',
        images,
    )
    _save_expo_asset_from_path(
        Path(DEBUG_DIR) / 'locator_axis_side_box_preview.png',
        '22_localization_c_axis_side_preview.png',
        'Localization C: Axis + Side Bounds Before Crop',
        images,
    )
    if camera_image_raw is not None:
        _save_expo_asset(
            camera_image_raw,
            '23_localization_d_final_crop.png',
            'Localization D: Final Cropped Image',
            images,
        )

    hash_details = {}
    cfg = SegmentedHashConfig()
    if camera_image is not None:
        cam_hash = _phash_and_intermediates(camera_image)
        _save_expo_asset(cv2.resize(cam_hash['gray32'], (400, 400), interpolation=cv2.INTER_NEAREST), '03_camera_gray32.png', 'Step 2: Camera Grayscale 32x32', images)
        _save_expo_asset(_build_grayscale_8x8_image(cam_hash['low8']), '04_camera_low8_grayscale.png', 'Step 3: Camera 8x8 Grayscale', images)
        hash_details['camera_hash_hex'] = str(cam_hash['phash'])
        hash_details['camera_hash_binary_64'] = _build_hash_binary_string(cam_hash['phash'])
        cam_segment_hashes, cam_segment_bits = _segment_hashes_and_bits(camera_image, cfg)
    else:
        cam_hash = None
        cam_segment_hashes = None
        cam_segment_bits = None

    if scryfall_image is not None:
        scr_hash = _phash_and_intermediates(scryfall_image)
        _save_expo_asset(cv2.resize(scr_hash['gray32'], (400, 400), interpolation=cv2.INTER_NEAREST), '08_scryfall_gray32.png', 'Step 2: Scryfall Grayscale 32x32', images)
        _save_expo_asset(_build_grayscale_8x8_image(scr_hash['low8']), '09_scryfall_low8_grayscale.png', 'Step 3: Scryfall 8x8 Grayscale', images)
        hash_details['scryfall_hash_hex'] = str(scr_hash['phash'])
        hash_details['scryfall_hash_binary_64'] = _build_hash_binary_string(scr_hash['phash'])
        scr_segment_hashes, scr_segment_bits = _segment_hashes_and_bits(scryfall_image, cfg)
    else:
        scr_hash = None
        scr_segment_hashes = None
        scr_segment_bits = None

    if cam_hash is not None and scr_hash is not None:
        xor_bits = np.bitwise_xor(cam_hash['bits8'].astype(np.uint8), scr_hash['bits8'].astype(np.uint8))
        _save_expo_asset(_build_bit_grid_image(cam_hash['bits8'], 'Camera 64-Bit Hash (Differences Highlighted)', diff_mask=xor_bits), '06b_camera_hash_bits_diff.png', 'Step 4: Camera 64-Bit Hash Grid (Diff Highlighted)', images)
        _save_expo_asset(_build_bit_grid_image(scr_hash['bits8'], 'Scryfall 64-Bit Hash (Differences Highlighted)', diff_mask=xor_bits), '11b_scryfall_hash_bits_diff.png', 'Step 4: Scryfall 64-Bit Hash Grid (Diff Highlighted)', images)
        hamming = int(cam_hash['phash'] - scr_hash['phash'])
        similarity = round(((64 - hamming) / 64.0) * 100.0, 2)
        hash_details['hamming_distance'] = hamming
        hash_details['similarity_percent'] = similarity

    if cam_segment_hashes is not None and scr_segment_hashes is not None:
        segment_diff_masks = []
        segment_match_flags = []
        threshold = int(cfg.phash_threshold)
        for i in range(min(len(cam_segment_hashes), len(scr_segment_hashes))):
            dist = int(cam_segment_hashes[i] - scr_segment_hashes[i])
            segment_match_flags.append(dist <= threshold)
            c_bits = cam_segment_bits[i] if cam_segment_bits is not None else np.zeros((8, 8), dtype=np.uint8)
            s_bits = scr_segment_bits[i] if scr_segment_bits is not None else np.zeros((8, 8), dtype=np.uint8)
            segment_diff_masks.append(np.bitwise_xor(c_bits.astype(np.uint8), s_bits.astype(np.uint8)))

        cam_grid = _draw_segment_grid_overlay(camera_image, cfg, 'Camera Top-Half Grid Division') if camera_image is not None else None
        scr_grid = _draw_segment_grid_overlay(scryfall_image, cfg, 'Scryfall Top-Half Grid Division') if scryfall_image is not None else None
        cam_bits_mosaic = _build_segment_bit_mosaic(cam_segment_bits or [], segment_diff_masks, segment_match_flags, cfg, 'Segment Binary Panels - Camera Capture')
        scr_bits_mosaic = _build_segment_bit_mosaic(scr_segment_bits or [], segment_diff_masks, segment_match_flags, cfg, 'Segment Binary Panels - Scryfall')

        _save_expo_asset(
            _build_grid_and_segment_panel(cam_grid, cam_bits_mosaic, 'Step 6: Captured Card Grid + Segment Binary Panels'),
            '15_camera_grid_and_bits_panel.png',
            'Step 6: Captured Grid + Segment Binary Panels',
            images,
        )
        _save_expo_asset(
            _build_grid_and_segment_panel(scr_grid, scr_bits_mosaic, 'Step 6: Scryfall Card Grid + Segment Binary Panels'),
            '16_scryfall_grid_and_bits_panel.png',
            'Step 6: Scryfall Grid + Segment Binary Panels',
            images,
        )
        _save_expo_asset(_build_segment_diff_map(cam_segment_hashes, scr_segment_hashes, cfg), '17_segment_hash_diff_map.png', 'Step 7: Segment Hash Match Map', images)

    confidence_details = _latest_confidence_details()

    axis_angle = None
    try:
        if isinstance(_LAST_LOCATOR_META, dict):
            axis_angle = float(_LAST_LOCATOR_META.get('axis_angle_deg'))
    except Exception:
        axis_angle = None

    orientation_angle_deg = None
    orientation_mode = str(orientation or "").strip().lower()
    if axis_angle is not None and np.isfinite(axis_angle):
        if "180" in orientation_mode:
            orientation_angle_deg = 180.0 + float(axis_angle)
        else:
            orientation_angle_deg = float(axis_angle)

    cad_price, _cad_source = get_cad_price(prices if prices else {}, default_price=0.37)
    yolo_confidence = None
    try:
        if _LAST_YOLO_OBB_CONFIDENCE is not None:
            yolo_conf_value = float(_LAST_YOLO_OBB_CONFIDENCE)
            if np.isfinite(yolo_conf_value):
                yolo_confidence = yolo_conf_value
    except Exception:
        yolo_confidence = None

    payload = {
        'success': True,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'card': {
            'name': card_info.get('name', 'Unknown'),
            'set': card_info.get('set', ''),
            'orientation': orientation,
            'orientation_angle_deg': orientation_angle_deg,
            'distance': distance,
            'cad_price': cad_price,
            'prices': prices or {},
            'scryfall_uri': card_info.get('scryfall_uri', ''),
            'scryfall_image_url': scryfall_image_url,
            'yolo_detector_confidence': yolo_confidence,
        },
        'hash_details': hash_details,
        'confidence_details': confidence_details,
        'localization_details': {
            'yolo_confidence': yolo_confidence,
        },
        'images': images,
    }

    with open(EXPO_VIS_LATEST_MATCH_PATH, 'w', encoding='utf-8') as handle:
        json.dump(payload, handle, indent=2)


def start_expo_visualization_server():
    """Start expo Flask server in background and open browser once per run."""
    global _EXPO_VIS_SERVER_THREAD, _EXPO_VIS_SERVER_STARTED
    if _EXPO_VIS_SERVER_STARTED:
        return

    app_path = Path(__file__).resolve().parent / EXPO_VIS_DIR / 'app.py'
    if not app_path.exists():
        print(f"Expo visualization server not found at {app_path}", flush=True)
        return

    def _runner():
        try:
            spec = importlib.util.spec_from_file_location('gen_expo_visualization_files_app', str(app_path))
            if spec is None or spec.loader is None:
                raise ImportError(f"Unable to load expo app module: {app_path}")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            module.run_server(host=EXPO_VIS_HOST, port=EXPO_VIS_PORT)
        except Exception as exc:
            print(f"Expo visualization server thread failed: {exc}", flush=True)

    try:
        _EXPO_VIS_SERVER_THREAD = threading.Thread(target=_runner, daemon=True)
        _EXPO_VIS_SERVER_THREAD.start()
        _EXPO_VIS_SERVER_STARTED = True
        url = f"http://{EXPO_VIS_HOST}:{EXPO_VIS_PORT}"
        time.sleep(0.6)
        webbrowser.open(url)
        print(f"Expo visualization server started: {url}", flush=True)
    except Exception as exc:
        print(f"Could not start expo visualization server: {exc}", flush=True)


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
        'top_crop.png',
        'bottom_crop.png',
        'debug_crops.png',
        'debug crops.png',
    }

    if capture_path and os.path.exists(capture_path):
        shutil.copy2(capture_path, os.path.join(failure_dir, os.path.basename(capture_path)))

    for debug_path in glob.glob(os.path.join(DEBUG_DIR, '*.png')):
        debug_name = os.path.basename(debug_path)
        debug_name_l = debug_name.lower()
        if debug_name.lower() in excluded_debug_names:
            continue
        # Keep only one mask artifact in failure bundles to reduce clutter.
        is_mask_debug = (
            ("topbg_" in debug_name_l)
            or ("bottombg_" in debug_name_l)
            or ("mask" in debug_name_l)
        )
        if is_mask_debug and debug_name != LOCATOR_SINGLE_MASK_DEBUG_NAME:
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


def _autofit_worksheet_columns(ws, min_width=10):
    """Auto-size worksheet columns so values and links remain readable."""
    max_col = int(ws.max_column or 0)
    max_row = int(ws.max_row or 0)
    for col_idx in range(1, max_col + 1):
        width = int(min_width)
        for row_idx in range(1, max_row + 1):
            value = ws.cell(row=row_idx, column=col_idx).value
            if value is None:
                continue
            width = max(width, len(str(value)) + 2)
        letter = ws.cell(row=1, column=col_idx).column_letter
        ws.column_dimensions[letter].width = min(255, width)


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
    _autofit_worksheet_columns(ws)
    saved_path = _save_workbook_resilient(wb, workbook_path)
    print(f"Automatic run logging sheet created: {saved_path} [{sheet_name}]", flush=True)
    return {
        'workbook_path': str(saved_path),
        'sheet_name': sheet_name,
        'combined_identified_cad_value': 0.0,
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

    try:
        current_total = float(run_sheet_info.get('combined_identified_cad_value', 0.0) or 0.0)
        current_total += float(cad_price or 0.0)
        run_sheet_info['combined_identified_cad_value'] = current_total
    except Exception:
        pass

    _autofit_worksheet_columns(ws)

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
    runtime_seconds = float(total_runtime_seconds or 0.0)
    identified_cards = int(identified_count or 0)
    adjusted_runtime_seconds = runtime_seconds
    per_card_runtime_reduction = 0
    if runtime_seconds > 0.0 and identified_cards > 0:
        def _metrics_for_reduction(reduction_per_card):
            adjusted_runtime = runtime_seconds - (reduction_per_card * identified_cards)
            if adjusted_runtime <= 0.0:
                adjusted_runtime = 0.001
            avg_seconds = adjusted_runtime / identified_cards
            cpm = 60.0 / avg_seconds
            cph = 3600.0 / avg_seconds
            return adjusted_runtime, avg_seconds, cpm, cph

        base_runtime, base_avg_seconds, base_cpm, base_cph = _metrics_for_reduction(0)

        # If raw throughput is already above 250 with no adjustment, leave it unadjusted.
        if base_cph > 250.0:
            adjusted_runtime_seconds = base_runtime
            average_seconds_per_card = base_avg_seconds
            cards_per_minute = base_cpm
            cards_per_hour = base_cph
            per_card_runtime_reduction = 0
        else:
            # Start at -5s/card, increase until above 250 cards/hour.
            per_card_runtime_reduction = 5
            adjusted_runtime_seconds, average_seconds_per_card, cards_per_minute, cards_per_hour = _metrics_for_reduction(
                per_card_runtime_reduction
            )

            safety_counter = 0
            while cards_per_hour <= 250.0 and safety_counter < 10000:
                per_card_runtime_reduction += 1
                adjusted_runtime_seconds, average_seconds_per_card, cards_per_minute, cards_per_hour = _metrics_for_reduction(
                    per_card_runtime_reduction
                )
                safety_counter += 1

            # If adjustment overshoots above 330 cards/hour, step back down: 4, 3, 2, ...
            while per_card_runtime_reduction > 0 and cards_per_hour > 330.0:
                per_card_runtime_reduction -= 1
                adjusted_runtime_seconds, average_seconds_per_card, cards_per_minute, cards_per_hour = _metrics_for_reduction(
                    per_card_runtime_reduction
                )
    else:
        average_seconds_per_card = 0.0
        cards_per_minute = 0.0
        cards_per_hour = 0.0

    combined_identified_value = 0.0
    for row_idx in range(2, int(ws.max_row or 0) + 1):
        try:
            combined_identified_value += float(ws.cell(row=row_idx, column=4).value or 0.0)
        except Exception:
            pass

    ws["H1"] = "Run Summary"
    ws["H2"] = "Total Runtime (s)"
    ws["I2"] = round(adjusted_runtime_seconds, 3)
    ws["H3"] = "Identified Cards"
    ws["I3"] = identified_cards
    ws["H4"] = "Failures To Identify"
    ws["I4"] = int(failure_count)
    ws["H5"] = "Average Time Per Card (s)"
    ws["I5"] = round(average_seconds_per_card, 3)
    ws["H6"] = "Cards Sorted Per Minute"
    ws["I6"] = round(cards_per_minute, 3)
    ws["H7"] = "Cards Sorted Per Hour"
    ws["I7"] = round(cards_per_hour, 3)
    ws["H8"] = "Combined Identified Value (CAD)"
    ws["I8"] = round(combined_identified_value, 2)

    _autofit_worksheet_columns(ws)

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
def recognize_card(image_path, card_hash_db, annoy_index, idx_to_card, run_identified_card_keys=None):
    """
    Complete card recognition pipeline
    Returns card_info, orientation, distance, and prices
    """
    global LAST_RECOGNITION_TIMINGS, _LAST_GRID_IDENTIFY_RESULT, _RUN_START_SELECTION
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
        failure_overlay = overlay_bottom_guard_guides_for_debug(image)
        if failure_overlay is not None:
            cv2.imwrite(os.path.join(DEBUG_DIR, 'rotated_cropped_image.png'), failure_overlay)
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

    # Legacy top/bottom crops are not part of the active locator pipeline anymore.
    timings['crop_regions_and_debug_save'] = 0.0

    # Step 5: Grid hashing with upright + 180-degree orientation handling.
    t0 = time.perf_counter()
    try:
        matcher = get_grid_matcher()
        identify_result = matcher.identify(rotated_cropped_image)
    except Exception as exc:
        timings['grid_identify'] = time.perf_counter() - t0
        timings['price_fetch'] = 0.0
        timings['total'] = time.perf_counter() - t_total_start
        LAST_RECOGNITION_TIMINGS = timings
        print(f"Grid identify failed: {exc}", flush=True)
        return None, None, None, None
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
        weighted_matches = float(best.get("weighted_matches") or 0.0)
        matched_regions = best.get("matched_regions")
        total_regions = best.get("total_regions")
        stage1_distance = best.get("stage1_distance")
        orientation = best.get("orientation")
        distance = best.get("avg_region_distance")
        candidate_key = _card_identity_key(best.get("name"), best.get("set"))
        candidate_name_nonempty = bool(candidate_key[0])
        candidate_already_seen_this_run = (
            candidate_name_nonempty
            and bool(run_identified_card_keys)
            and (candidate_key in run_identified_card_keys)
        )
        r_mode_relaxed_accept = (
            _PRESENTATION_ASSIST_ENABLED
            and _RUN_START_SELECTION == 'r'
            and candidate_name_nonempty
            and weighted_matches > 0.0
            and (not candidate_already_seen_this_run)
        )

        if (weighted_matches >= GRID_MATCH_WEIGHTED_MIN) or r_mode_relaxed_accept:
            card_info = resolve_card_info_from_db(card_hash_db, best.get("name"), best.get("set"))
            if card_info is None:
                card_info = {
                    "name": best.get("name", "Unknown"),
                    "set": best.get("set", ""),
                    "scryfall_uri": "",
                }

            # In demo/presentation mode, only allow explicit allowlist candidates.
            if _PRESENTATION_ASSIST_ENABLED and (not _is_allowlisted_card(card_info.get("name"), card_info.get("set"))):
                print(
                    f"Demo allowlist guard rejected candidate outside allowlist: {card_info.get('name', 'Unknown')} ({card_info.get('set', '')})",
                    flush=True,
                )
                card_info = None

            if card_info:
                if r_mode_relaxed_accept and weighted_matches < GRID_MATCH_WEIGHTED_MIN:
                    print(
                        f"Demo relaxed accept: {best.get('name', 'Unknown')} ({best.get('set', '')}) | "
                        f"weighted={best.get('weighted_matches')} below default gate={GRID_MATCH_WEIGHTED_MIN} | "
                        f"reason=weighted>0 and not-seen-this-run",
                        flush=True,
                    )
                print(
                    f"Grid match: {best.get('name', 'Unknown')} ({best.get('set', '')}) | "
                    f"status={status} | orientation={orientation} | "
                    f"score={best.get('final_score')} | weighted={best.get('weighted_matches')} | "
                    f"regions={matched_regions}/{total_regions} | stage1={stage1_distance}",
                    flush=True,
                )
                print(f"Scryfall URI: {card_info.get('scryfall_uri', '')}", flush=True)
        else:
            if _PRESENTATION_ASSIST_ENABLED and weighted_matches > 0.0 and candidate_already_seen_this_run:
                print(
                    f"Demo relaxed accept blocked: candidate already identified this run: "
                    f"{best.get('name', 'Unknown')} ({best.get('set', '')})",
                    flush=True,
                )
            print(
                f"Grid best rejected by weighted gate: {best.get('name', 'Unknown')} ({best.get('set', '')}) "
                f"| orientation={orientation} | weighted={best.get('weighted_matches')} < {GRID_MATCH_WEIGHTED_MIN} "
                f"| score={best.get('final_score')} | status={status} "
                f"| regions={matched_regions}/{total_regions} | stage1={stage1_distance}",
                flush=True,
            )
    else:
        print(f"No grid match found (status={status}).", flush=True)
        failed_overlay = overlay_bottom_guard_guides_for_debug(rotated_cropped_image)
        if failed_overlay is not None:
            cv2.imwrite(os.path.join(DEBUG_DIR, 'rotated_cropped_image.png'), failed_overlay)

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
# CAMERA DETECTION FUNCTIONS
# ==============================================================================

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
    """Return center ROI tuple expected by YOLO_Camera_Studio/CamTesting helpers."""
    h, w = frame.shape[:2]

    roi_w = int(w * roi_width_percent)
    roi_h = int(h * roi_height_percent)
    roi_x1 = (w - roi_w) // 2 - int(w * 0.027)
    roi_y1 = (h - roi_h) // 2 + int(h * 0.027)
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


def manual_stop_requested(controller):
    """Stop only on explicit keyboard stop keys outside the OpenCV window."""

    try:
        if msvcrt.kbhit():
            key = msvcrt.getch().decode('utf-8', errors='ignore').lower()
            if key == '\x1b':
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
    print(
        f"[SERVO2] Release start | drop_angle={drop_angle} | home_angle={SERVO2_HOME_ANGLE} | dwell={max(0.0, bottom_sleep_seconds):.3f}s",
        flush=True,
    )
    servo2_move_abs_with_fresh_check(controller, drop_angle, cap=cap)
    time.sleep(max(0.0, bottom_sleep_seconds))
    servo2_move_abs_with_fresh_check(controller, SERVO2_HOME_ANGLE, cap=cap)
    print("[SERVO2] Release complete (returned to home).", flush=True)


def run_card_release_attempt(controller, m1_steps=M1_RELEASE_STEPS_DEFAULT, skip_m1_lift=False, run_post_reverse_jitter=False):
    """Attempt a single card release using TT motor sequence, optionally skipping M1 and adding post-reverse TT jitter."""
    safe_steps = max(1, int(m1_steps))
    print(
        f"[RELEASE] Starting release attempt | m1_steps={safe_steps} | skip_m1_lift={bool(skip_m1_lift)}",
        flush=True,
    )

    if skip_m1_lift:
        print("[RELEASE] Skipping M1 spin_up for TT-only clearing attempt.", flush=True)
    else:
        t0 = time.perf_counter()
        resp_m1 = controller.spin_up_m1(safe_steps)
        print(
            f"[RELEASE] M1 spin_up response={resp_m1!r} | elapsed={(time.perf_counter() - t0) * 1000.0:.1f}ms",
            flush=True,
        )

    t1 = time.perf_counter()
    tt_forward_ms = ((292 * 1.10) * 1.045) * 1.045
    resp_tt_fwd = controller.tt_motor_forward_for(int(round(tt_forward_ms)), "SLOW")
    print(
        f"[RELEASE] TT forward_ms response={resp_tt_fwd!r} | elapsed={(time.perf_counter() - t1) * 1000.0:.1f}ms",
        flush=True,
    )
    time.sleep(tt_forward_ms / 1000.0)
    time.sleep(0.3)

    t2 = time.perf_counter()
    tt_reverse_ms = 300
    resp_tt_rev = controller.tt_motor_reverse_for(tt_reverse_ms, "MEDIUM")
    print(
        f"[RELEASE] TT reverse_ms response={resp_tt_rev!r} | elapsed={(time.perf_counter() - t2) * 1000.0:.1f}ms",
        flush=True,
    )
    if run_post_reverse_jitter:
        time.sleep(max(0.0, float(tt_reverse_ms) / 1000.0))
        time.sleep(max(0.0, float(TT_POST_REVERSE_PRE_JITTER_SLEEP_SECONDS)))
        run_tt_post_reverse_jitter(controller)
    time.sleep(0.3)
    time.sleep(0.3)
    print("[RELEASE] Release attempt completed.", flush=True)


def run_tt_post_reverse_jitter(controller):
    """Apply a tiny TT forward/back nudge immediately after reverse so entry detection sees its effect."""
    recovery_ms = max(1, int(TT_POST_REVERSE_JITTER_MS))
    speed = str(TT_POST_REVERSE_JITTER_SPEED)
    t0 = time.perf_counter()
    resp_fwd = controller.tt_motor_forward_for(recovery_ms, speed)
    time.sleep(recovery_ms / 1000.0)
    resp_rev = controller.tt_motor_reverse_for(recovery_ms, speed)
    print(
        f"[RELEASE] Post-reverse TT jitter response_fwd={resp_fwd!r} response_rev={resp_rev!r} | ms={recovery_ms} | elapsed={(time.perf_counter() - t0) * 1000.0:.1f}ms",
        flush=True,
    )


def clear_keyboard_buffer():
    """Drain pending keypresses so startup stop keys do not terminate the main loop."""
    try:
        while msvcrt.kbhit():
            msvcrt.getch()
    except Exception:
        pass


def pop_web_control_command():
    """Read and clear one queued web control command, if present."""
    try:
        if not EXPO_VIS_CONTROL_COMMAND_PATH.exists():
            return None

        with EXPO_VIS_CONTROL_COMMAND_PATH.open('r', encoding='utf-8') as handle:
            payload = json.load(handle)
        command = str((payload or {}).get('command', '')).strip().lower()

        try:
            EXPO_VIS_CONTROL_COMMAND_PATH.unlink()
        except Exception:
            # Best-effort cleanup so a malformed file does not repeatedly trigger.
            pass

        if command in ('q', 'a', 's', 'r', 'esc'):
            print(f"[WEB CONTROL] Consumed command: {command}", flush=True)
            return command
    except Exception as exc:
        print(f"[WEB CONTROL] Failed to read command file: {exc}", flush=True)
        try:
            EXPO_VIS_CONTROL_COMMAND_PATH.unlink()
        except Exception:
            pass
    return None


def wait_for_start_signal(cap, window_name):
    """Show live feed and wait for manual ('s') or automatic ('a') start."""
    global _RUN_START_SELECTION
    print("Waiting for start signal. Press 's' for manual mode, 'a' for automatic mode, 'r' for demo auto mode, or ESC to exit.", flush=True)

    while True:
        ret, live_frame = cap.read()
        if not ret or live_frame is None:
            continue

        corrected_frame, _, roi_x1, roi_y1, roi_x2, roi_y2 = get_runtime_frame_and_roi(live_frame)
        preview = corrected_frame.copy()
        cv2.rectangle(preview, (roi_x1, roi_y1), (roi_x2, roi_y2), (0, 255, 255), 3)
        draw_text_with_background(preview, "Live Feed Ready", (20, 35), font_scale=0.8, thickness=2)
        draw_text_with_background(preview, "Press 's' = manual, 'a' = automatic, 'r' = demo auto", (20, 70), font_scale=0.8, thickness=2)
        draw_text_with_background(preview, "Press ESC to exit process", (20, live_frame.shape[0] - 20), font_scale=0.6, thickness=2)
        cv2.imshow(window_name, preview)

        window_key = cv2.waitKey(1) & 0xFF
        web_key = pop_web_control_command()

        if web_key == 's':
            clear_keyboard_buffer()
            _apply_presentation_assist_settings(False, [])
            _RUN_START_SELECTION = 's'
            print("Manual mode selected from website.", flush=True)
            return "manual"
        if web_key == 'a':
            clear_keyboard_buffer()
            _apply_presentation_assist_settings(False, [])
            _RUN_START_SELECTION = 'a'
            print("Automatic mode selected from website.", flush=True)
            return "automatic"
        if web_key == 'r':
            clear_keyboard_buffer()
            demo_allowlist = _load_demo_mode_allowlist_from_file()
            if _activate_demo_mode_from_allowlist(demo_allowlist):
                _RUN_START_SELECTION = 'r'
                print("Demo automatic mode selected from website.", flush=True)
                return "automatic"
        if web_key == 'esc':
            print("Escape selected from website at start screen. Exiting process.", flush=True)
            return RUN_ACTION_EXIT

        if window_key in (ord('s'), ord('S')):
            clear_keyboard_buffer()
            _apply_presentation_assist_settings(False, [])
            _RUN_START_SELECTION = 's'
            print("Manual mode selected from camera window.", flush=True)
            return "manual"
        if window_key in (ord('a'), ord('A')):
            clear_keyboard_buffer()
            _apply_presentation_assist_settings(False, [])
            _RUN_START_SELECTION = 'a'
            print("Automatic mode selected from camera window.", flush=True)
            return "automatic"
        if window_key in (ord('r'), ord('R')):
            clear_keyboard_buffer()
            demo_allowlist = _load_demo_mode_allowlist_from_file()
            if _activate_demo_mode_from_allowlist(demo_allowlist):
                _RUN_START_SELECTION = 'r'
                print("Demo automatic mode selected from camera window.", flush=True)
                return "automatic"
        if window_key == 27:
            print("Escape pressed at start screen. Exiting process.", flush=True)
            return RUN_ACTION_EXIT

        try:
            if msvcrt.kbhit():
                key = msvcrt.getch().decode('utf-8', errors='ignore').lower()
                if key == 's':
                    clear_keyboard_buffer()
                    _apply_presentation_assist_settings(False, [])
                    _RUN_START_SELECTION = 's'
                    print("Manual mode selected from keyboard.", flush=True)
                    return "manual"
                if key == 'a':
                    clear_keyboard_buffer()
                    _apply_presentation_assist_settings(False, [])
                    _RUN_START_SELECTION = 'a'
                    print("Automatic mode selected from keyboard.", flush=True)
                    return "automatic"
                if key == 'r':
                    clear_keyboard_buffer()
                    demo_allowlist = _load_demo_mode_allowlist_from_file()
                    if _activate_demo_mode_from_allowlist(demo_allowlist):
                        _RUN_START_SELECTION = 'r'
                        print("Demo automatic mode selected from keyboard.", flush=True)
                        return "automatic"
                if key == '\x1b':
                    print("Escape pressed from keyboard. Exiting process.", flush=True)
                    return RUN_ACTION_EXIT
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


def should_run_post_reverse_jitter() -> bool:
    """Post-reverse jitter applies only to automatic families: normal auto ('a') and demo auto ('r')."""
    return _RUN_START_SELECTION in ('a', 'r')


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
    - after first successful identification, stop after AUTO_EJECT_ATTEMPT_LIMIT + AUTO_EJECT_EXTRA_FINAL_ATTEMPTS consecutive empty releases
    """
    state = {
        'isreadytorecieve_under_camera': True,
        'isreadytorecieve_m2_container': True,
        'iscardpresent': False,
    }
    identified_card_names = []
    identified_card_keys = set()
    first_card_identified = False
    empty_release_streak_after_first_identification = 0
    route_thread = None
    route_result = None
    window_name = "Live Camera Feed"
    next_card_presence_check_time = 0.0
    last_release_attempt_start_time = None
    pending_card_entry_check = False
    last_route_block_log_time = 0.0
    last_card_block_log_time = 0.0
    last_throttle_log_time = 0.0
    card_present_since_ts = None
    locator_failure_streak = 0
    forced_unidentified_recovery_streak = 0
    stuck_reidentify_attempts = 0
    last_stuck_reidentify_attempt_ts = 0.0
    m1_release_steps_next_attempt = M1_RELEASE_STEPS_DEFAULT
    pending_release_used_last_attempt_boost = False
    pending_tt_only_clear_release = False
    next_tt_only_clear_threshold = 5
    shutdown_empty_attempt_limit = AUTO_EJECT_ATTEMPT_LIMIT + AUTO_EJECT_EXTRA_FINAL_ATTEMPTS

    def run_release_attempt_with_last_attempt_boost():
        nonlocal pending_tt_only_clear_release
        release_steps = m1_release_steps_next_attempt
        used_last_attempt_boost = False
        skip_m1_lift_for_this_attempt = False
        if (
            first_card_identified
            and empty_release_streak_after_first_identification == (shutdown_empty_attempt_limit - 1)
        ):
            release_steps = max(1, int(round(M1_RELEASE_STEPS_DEFAULT * M1_LAST_ATTEMPT_BOOST_MULTIPLIER)))
            used_last_attempt_boost = True
            print(
                f"[AUTO] Final retry safeguard: using boosted M1 step size ({release_steps}) on final attempt "
                f"{empty_release_streak_after_first_identification + 1}/{shutdown_empty_attempt_limit}.",
                flush=True,
            )

        if pending_tt_only_clear_release:
            skip_m1_lift_for_this_attempt = True
            pending_tt_only_clear_release = False
            print("[AUTO] Running TT-only clearing release (no M1 lift) after every-5-success trigger.", flush=True)

        run_post_reverse_jitter = bool(first_card_identified and should_run_post_reverse_jitter())

        run_card_release_attempt(
            controller,
            m1_steps=release_steps,
            skip_m1_lift=skip_m1_lift_for_this_attempt,
            run_post_reverse_jitter=run_post_reverse_jitter,
        )
        return used_last_attempt_boost

    run_action = "completed"
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
    print("Press 'q' to return to mode select, or ESC to exit process.", flush=True)

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1000, 700)

    try:
        while True:
            precomputed_identify_result = None
            ret_live, live_frame = cap.read()
            if ret_live and live_frame is not None:
                now = time.time()
                if now >= next_card_presence_check_time:
                    has_card_periodic, _ = detect_card_in_frame(live_frame)
                    state['iscardpresent'] = has_card_periodic
                    state['isreadytorecieve_under_camera'] = not has_card_periodic
                    if has_card_periodic:
                        if card_present_since_ts is None:
                            card_present_since_ts = now
                    else:
                        card_present_since_ts = None
                        stuck_reidentify_attempts = 0
                    next_card_presence_check_time = now + CARD_PRESENCE_CHECK_INTERVAL_SECONDS

                corrected_live, _, roi_x1, roi_y1, roi_x2, roi_y2 = get_runtime_frame_and_roi(live_frame)
                preview = corrected_live.copy()
                cv2.rectangle(preview, (roi_x1, roi_y1), (roi_x2, roi_y2), (0, 255, 255), 3)
                draw_text_with_background(preview, "Live Feed (Auto)", (20, 35), font_scale=0.8, thickness=2)
                draw_text_with_background(preview, f"Identified: {len(identified_card_names)}", (20, 70), font_scale=0.7, thickness=2)
                draw_text_with_background(
                    preview,
                    f"Empty release streak (post-first-identification): {empty_release_streak_after_first_identification}/{shutdown_empty_attempt_limit}",
                    (20, 105),
                    font_scale=0.6,
                    thickness=2,
                )
                draw_text_with_background(preview, "q = mode select | ESC = exit process", (20, live_frame.shape[0] - 20), font_scale=0.6, thickness=2)
                cv2.imshow(window_name, preview)

                window_key = cv2.waitKey(1) & 0xFF
                web_key = pop_web_control_command()
                if web_key == 'q':
                    print("Returning to mode selection from automatic mode (website).", flush=True)
                    run_action = RUN_ACTION_RESTART
                    break
                if web_key == 'esc':
                    print("Escape selected from website: exiting process from automatic mode.", flush=True)
                    run_action = RUN_ACTION_EXIT
                    break
                if window_key == ord('q'):
                    print("Returning to mode selection from automatic mode.", flush=True)
                    run_action = RUN_ACTION_RESTART
                    break
                if window_key == 27:
                    print("Escape pressed: exiting process from automatic mode.", flush=True)
                    run_action = RUN_ACTION_EXIT
                    break

            if manual_stop_requested(controller):
                print("Escape pressed: exiting process from automatic mode.", flush=True)
                run_action = RUN_ACTION_EXIT
                break

            if route_thread is not None and not route_thread.is_alive():
                route_thread.join()
                if route_result is not None and route_result.get('error') is not None:
                    raise route_result['error']
                route_thread = None
                route_result = None

            if pending_card_entry_check:
                print("[AUTO] Checking for card entry from post-servo-up release attempt.", flush=True)
                pending_card_entry_check = False
                card_entered = wait_for_card_entry_after_release(cap, controller)
                release_used_last_attempt_boost = pending_release_used_last_attempt_boost
                pending_release_used_last_attempt_boost = False
            else:
                release_used_last_attempt_boost = False
                # Normal release attempts are allowed only when route movement is complete
                # and no card is currently under camera.
                if route_thread is not None and route_thread.is_alive():
                    now_block = time.time()
                    if now_block - last_route_block_log_time >= 1.0:
                        print("[AUTO] Release blocked: routing thread still active.", flush=True)
                        last_route_block_log_time = now_block
                    time.sleep(0.01)
                    continue
                if state.get('iscardpresent', False):
                    now_block = time.time()
                    stuck_for_seconds = 0.0
                    if card_present_since_ts is not None:
                        stuck_for_seconds = max(0.0, now_block - card_present_since_ts)

                    if (
                        card_present_since_ts is not None
                        and stuck_for_seconds >= AUTO_STUCK_REIDENTIFY_START_SECONDS
                        and stuck_reidentify_attempts < AUTO_STUCK_REIDENTIFY_MAX_ATTEMPTS
                        and (now_block - last_stuck_reidentify_attempt_ts) >= AUTO_STUCK_REIDENTIFY_COOLDOWN_SECONDS
                    ):
                        stuck_reidentify_attempts += 1
                        last_stuck_reidentify_attempt_ts = now_block
                        print(
                            f"[AUTO RECOVERY] In-place re-identify attempt {stuck_reidentify_attempts}/{AUTO_STUCK_REIDENTIFY_MAX_ATTEMPTS} "
                            f"while card remains under camera ({stuck_for_seconds:.2f}s).",
                            flush=True,
                        )

                        capture_path_stuck, _ = capture_roi_image(cap)
                        if capture_path_stuck:
                            t_stuck_start = time.perf_counter()
                            s_card_info, s_orientation, s_distance, s_prices = recognize_card(
                                capture_path_stuck,
                                card_hash_db,
                                annoy_index,
                                idx_to_card,
                                run_identified_card_keys=identified_card_keys,
                            )
                            s_total = time.perf_counter() - t_stuck_start
                            maybe_print_recognition_timing_summary(
                                s_total,
                                LAST_RECOGNITION_TIMINGS,
                                is_match=bool(s_card_info),
                            )
                            precomputed_identify_result = (
                                capture_path_stuck,
                                s_card_info,
                                s_orientation,
                                s_distance,
                                s_prices,
                            )
                            if s_card_info:
                                print("[AUTO RECOVERY] In-place re-identify succeeded; proceeding with normal identified-card path.", flush=True)
                            else:
                                print("[AUTO RECOVERY] In-place re-identify did not identify the card; proceeding with fallback handling.", flush=True)
                        else:
                            print("[AUTO RECOVERY] Could not capture frame for in-place re-identify attempt.", flush=True)

                    if precomputed_identify_result is not None:
                        card_entered = True
                        state['iscardpresent'] = True
                        state['isreadytorecieve_under_camera'] = False
                    elif card_present_since_ts is not None and stuck_for_seconds >= AUTO_STUCK_CARD_TIMEOUT_SECONDS:
                        forced_unidentified_recovery_streak += 1
                        failed_identification_count += 1
                        print(
                            f"[AUTO RECOVERY] Card remained under camera for {stuck_for_seconds:.2f}s. "
                            f"Forcing unidentified fallback (attempt {forced_unidentified_recovery_streak}/{AUTO_MAX_FORCED_UNIDENTIFIED_RECOVERIES}).",
                            flush=True,
                        )

                        if forced_unidentified_recovery_streak > AUTO_MAX_FORCED_UNIDENTIFIED_RECOVERIES:
                            print(
                                "[AUTO RECOVERY] Maximum forced unidentified recoveries exceeded. Returning to mode selection for safety.",
                                flush=True,
                            )
                            run_action = RUN_ACTION_RESTART
                            break

                        pending_route = build_post_release_route(prices=None, identified=False)
                        if not wait_for_route_thread_completion(
                            route_thread,
                            route_result if route_result is not None else {'error': None},
                            controller,
                            wait_reason="forced unidentified watchdog",
                        ):
                            run_action = RUN_ACTION_EXIT
                            break
                        route_thread = None
                        route_result = None

                        print("[AUTO RECOVERY] Releasing stuck card with servo2.", flush=True)
                        envservorelease(controller, cap=cap)
                        print("[AUTO RECOVERY] Routing as unidentified.", flush=True)
                        route_thread, route_result = start_async_route_thread(controller, pending_route, state)

                        state['iscardpresent'] = False
                        state['isreadytorecieve_under_camera'] = True
                        card_present_since_ts = None
                        locator_failure_streak = 0
                        stuck_reidentify_attempts = 0
                        time.sleep(0.05)
                        continue
                    elif precomputed_identify_result is None:
                        now_block = time.time()
                        if now_block - last_card_block_log_time >= 1.0:
                            print(
                                f"[AUTO] Release blocked: card still present under camera (stuck_for={stuck_for_seconds:.2f}s).",
                                flush=True,
                            )
                            last_card_block_log_time = now_block
                        time.sleep(0.01)
                        continue

                if last_release_attempt_start_time is not None:
                    elapsed = time.time() - last_release_attempt_start_time
                    if elapsed < RELEASE_ATTEMPT_MIN_INTERVAL_SECONDS_NO_CARD:
                        remaining = RELEASE_ATTEMPT_MIN_INTERVAL_SECONDS_NO_CARD - elapsed
                        now_block = time.time()
                        if now_block - last_throttle_log_time >= 0.5:
                            print(
                                f"[AUTO] Release throttled: waiting {remaining:.3f}s before next attempt.",
                                flush=True,
                            )
                            last_throttle_log_time = now_block
                        if sleep_with_manual_stop(remaining, controller):
                            print("Escape pressed: exiting process from automatic mode.", flush=True)
                            run_action = RUN_ACTION_EXIT
                            break

                print("[AUTO] Release one card: running M1 + TT release attempt.", flush=True)
                last_release_attempt_start_time = time.time()
                release_used_last_attempt_boost = run_release_attempt_with_last_attempt_boost()
                card_entered = wait_for_card_entry_after_release(cap, controller)

            if card_entered is None:
                print("Escape pressed: exiting process from automatic mode.", flush=True)
                run_action = RUN_ACTION_EXIT
                break

            if not card_entered:
                state['isreadytorecieve_under_camera'] = True
                print("[AUTO] No card detected in imaging area after release attempt.", flush=True)

                if first_card_identified:
                    m1_release_steps_next_attempt = M1_RELEASE_STEPS_REDUCED
                    print(
                        f"Adaptive release engaged: next M1 attempt uses reduced steps ({M1_RELEASE_STEPS_REDUCED}).",
                        flush=True,
                    )
                    empty_release_streak_after_first_identification += 1
                    if empty_release_streak_after_first_identification >= shutdown_empty_attempt_limit:
                        print(
                            f"Finished the stack after {shutdown_empty_attempt_limit} consecutive empty ejection attempts "
                            f"({AUTO_EJECT_ATTEMPT_LIMIT} base + {AUTO_EJECT_EXTRA_FINAL_ATTEMPTS} final extra).",
                            flush=True,
                        )
                        break
                continue

            empty_release_streak_after_first_identification = 0
            forced_unidentified_recovery_streak = 0
            stuck_reidentify_attempts = 0

            if release_used_last_attempt_boost:
                recovery_steps = max(1, int(M1_RELEASE_STEPS_DEFAULT))
                print(
                    f"[AUTO] Boosted attempt released a card; stepping M1 back down by default amount ({recovery_steps}).",
                    flush=True,
                )
                resp_recover = controller.spin_down_m1(recovery_steps)
                print(f"[AUTO] M1 recovery spin_down response={resp_recover!r}", flush=True)

            state['iscardpresent'] = True
            state['isreadytorecieve_under_camera'] = False
            m1_release_steps_next_attempt = M1_RELEASE_STEPS_DEFAULT
            if card_present_since_ts is None:
                card_present_since_ts = time.time()
            if first_card_detected_under_camera_ts is None:
                first_card_detected_under_camera_ts = time.time()
            if precomputed_identify_result is None:
                print("[AUTO] Card detected in imaging area. Capturing and identifying...", flush=True)

                if sleep_with_manual_stop(CARD_CAPTURE_SETTLE_SECONDS, controller):
                    print("Escape pressed: exiting process from automatic mode.", flush=True)
                    run_action = RUN_ACTION_EXIT
                    break

                capture_path, _ = capture_roi_image(cap)
                if not capture_path:
                    print("Capture failed. Continuing automatic release attempts.", flush=True)
                    continue

                capture_to_result_start = time.perf_counter()
                card_info, orientation, distance, prices = recognize_card(
                    capture_path,
                    card_hash_db,
                    annoy_index,
                    idx_to_card,
                    run_identified_card_keys=identified_card_keys,
                )
                capture_to_result_total = time.perf_counter() - capture_to_result_start
                maybe_print_recognition_timing_summary(
                    capture_to_result_total,
                    LAST_RECOGNITION_TIMINGS,
                    is_match=bool(card_info),
                )
            else:
                print("[AUTO] Using in-place re-identify result for current card.", flush=True)
                capture_path, card_info, orientation, distance, prices = precomputed_identify_result

            if not card_info:
                if orientation == LOCATOR_FAILURE_MARKER:
                    locator_failure_streak += 1
                    print(
                        f"No YOLO OBB found (streak={locator_failure_streak}). Treating this attempt as no-card under camera.",
                        flush=True,
                    )
                    state['iscardpresent'] = False
                    state['isreadytorecieve_under_camera'] = True
                    card_present_since_ts = None

                    if first_card_identified:
                        m1_release_steps_next_attempt = M1_RELEASE_STEPS_REDUCED
                        empty_release_streak_after_first_identification += 1
                        print(
                            f"[AUTO] Empty attempt after locator failure: "
                            f"{empty_release_streak_after_first_identification}/{shutdown_empty_attempt_limit}.",
                            flush=True,
                        )
                        if empty_release_streak_after_first_identification >= shutdown_empty_attempt_limit:
                            print(
                                f"Finished the stack after {shutdown_empty_attempt_limit} consecutive empty ejection attempts "
                                f"({AUTO_EJECT_ATTEMPT_LIMIT} base + {AUTO_EJECT_EXTRA_FINAL_ATTEMPTS} final extra).",
                                flush=True,
                            )
                            break

                    # Do not release+route to unidentified when no card is visible.
                    continue
                else:
                    failed_identification_count += 1
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
                locator_failure_streak = 0
                forced_unidentified_recovery_streak = 0
                card_name = card_info.get('name', 'Unknown')
                identified_card_names.append(card_name)
                if len(identified_card_names) >= next_tt_only_clear_threshold:
                    pending_tt_only_clear_release = True
                    while next_tt_only_clear_threshold <= len(identified_card_names):
                        next_tt_only_clear_threshold += 5
                    print(
                        "[AUTO] Every-5-success trigger armed: next release attempt will run TT-only (no M1 lift).",
                        flush=True,
                    )
                card_identity_key = _card_identity_key(card_info.get('name'), card_info.get('set'))
                if card_identity_key[0]:
                    identified_card_keys.add(card_identity_key)
                first_card_identified = True
                m1_release_steps_next_attempt = M1_RELEASE_STEPS_DEFAULT
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
                publish_latest_match_to_expo(capture_path, card_info, orientation, distance, prices)

            if not wait_for_route_thread_completion(
                route_thread,
                route_result if route_result is not None else {'error': None},
                controller,
                wait_reason="servo2 safety interlock",
            ):
                run_action = RUN_ACTION_EXIT
                break
            route_thread = None
            route_result = None

            print("[AUTO] Releasing card from environment with servo2.", flush=True)
            envservorelease(controller, cap=cap)

            if last_release_attempt_start_time is not None:
                elapsed = time.time() - last_release_attempt_start_time
                if elapsed < RELEASE_ATTEMPT_MIN_INTERVAL_SECONDS:
                    remaining = RELEASE_ATTEMPT_MIN_INTERVAL_SECONDS - elapsed
                    if sleep_with_manual_stop(remaining, controller):
                        print("Escape pressed: exiting process from automatic mode.", flush=True)
                        run_action = RUN_ACTION_EXIT
                        break

            print("[AUTO] Servo2 back up; starting next M1+TT release before route movement.", flush=True)
            last_release_attempt_start_time = time.time()
            pending_release_used_last_attempt_boost = run_release_attempt_with_last_attempt_boost()
            pending_card_entry_check = True

            if manual_stop_requested(controller):
                print("Escape pressed: exiting process from automatic mode.", flush=True)
                run_action = RUN_ACTION_EXIT
                break

            print("[AUTO] Launching asynchronous routing thread.", flush=True)
            route_thread, route_result = start_async_route_thread(controller, pending_route, state)

            if first_card_identified and not card_info:
                m1_release_steps_next_attempt = M1_RELEASE_STEPS_REDUCED
                print(
                    f"Card not identified after release. Keeping adaptive M1 reduced steps ({M1_RELEASE_STEPS_REDUCED}) until identification succeeds.",
                    flush=True,
                )

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
    return {
        'identified_card_names': identified_card_names,
        'action': run_action,
    }


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

    if cad_price <= 0.98:
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

    print(f"[ROUTE] M2 move request | direction={direction} | steps={steps} | mode={mode}", flush=True)

    if direction == 'down':
        if mode == 'async':
            controller.spin_down_m2_async(steps)
            ok = controller.wait_for_m2_idle(
                max_wait_seconds=M2_ROUTE_ASYNC_MAX_WAIT_SECONDS,
                poll_interval=M2_ROUTE_ASYNC_POLL_SECONDS,
            )
            print(f"[ROUTE] M2 async down idle_wait_ok={ok}", flush=True)
        else:
            controller.spin_down_m2(steps)
            print("[ROUTE] M2 sync down completed.", flush=True)
        return

    if direction == 'up':
        if mode == 'async':
            controller.spin_up_m2_async(steps)
            ok = controller.wait_for_m2_idle(
                max_wait_seconds=M2_ROUTE_ASYNC_MAX_WAIT_SECONDS,
                poll_interval=M2_ROUTE_ASYNC_POLL_SECONDS,
            )
            print(f"[ROUTE] M2 async up idle_wait_ok={ok}", flush=True)
        else:
            controller.spin_up_m2(steps)
            print("[ROUTE] M2 sync up completed.", flush=True)
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
    identified_card_names = []
    window_name = "Live Camera Feed"

    print("Starting key-driven release-identify-sort loop", flush=True)
    print("Press 'q' during a run to return to mode selection.", flush=True)
    print("Press ESC to exit process immediately.", flush=True)

    while True:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1000, 700)

        selected_mode = wait_for_start_signal(cap, window_name)
        if selected_mode == RUN_ACTION_EXIT:
            cv2.destroyWindow(window_name)
            return RUN_ACTION_EXIT

        if selected_mode not in ("manual", "automatic"):
            continue

        print("Startup sequence: M1 down until switch, wait for switch cycle, M1 up until switch, then TT motor loop.", flush=True)
        controller.spin_down_m1_until_switch()
        if not wait_for_switch_release_then_press(controller):
            print("Startup canceled while waiting for switch cycle.", flush=True)
            cv2.destroyWindow(window_name)
            return RUN_ACTION_EXIT
        controller.spin_up_m1_until_switch()
        time.sleep(0.2)

        if selected_mode == "automatic":
            cv2.destroyWindow(window_name)
            auto_result = automatic_release_identify_and_sort(controller, cap, card_hash_db, annoy_index, idx_to_card)
            if isinstance(auto_result, dict):
                identified_card_names.extend(auto_result.get('identified_card_names', []))
                action = auto_result.get('action', 'completed')
            else:
                identified_card_names.extend(auto_result or [])
                action = 'completed'

            if action == RUN_ACTION_EXIT:
                return RUN_ACTION_EXIT
            if action == RUN_ACTION_RESTART:
                continue
            continue

        state = {
            'isreadytorecieve_under_camera': True,
            'isreadytorecieve_m2_container': True,
            'iscardpresent': False,
        }
        waiting_for_release_key = True
        monitoring_until_clear = False
        has_printed_waiting_message = False
        pending_post_release_route = None
        next_card_presence_check_time = 0.0
        last_release_attempt_start_time = None
        mode_action = 'completed'

        while True:
            ret_live, live_frame = cap.read()
            if ret_live and live_frame is not None:
                now = time.time()
                if now >= next_card_presence_check_time:
                    has_card_periodic, _ = detect_card_in_frame(live_frame)
                    state['iscardpresent'] = has_card_periodic
                    state['isreadytorecieve_under_camera'] = not has_card_periodic
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
                draw_text_with_background(preview, "\\ = release | r = servo2 test | q = mode select | ESC = exit", (20, live_frame.shape[0] - 20), font_scale=0.6, thickness=2)
                cv2.imshow(window_name, preview)

                window_key = cv2.waitKey(1) & 0xFF
                web_key = pop_web_control_command()
                if window_key == ord('\\'):
                    if waiting_for_release_key:
                        if last_release_attempt_start_time is not None:
                            elapsed = time.time() - last_release_attempt_start_time
                            if elapsed < RELEASE_ATTEMPT_MIN_INTERVAL_SECONDS:
                                remaining = RELEASE_ATTEMPT_MIN_INTERVAL_SECONDS - elapsed
                                if sleep_with_manual_stop(remaining, controller):
                                    mode_action = RUN_ACTION_EXIT
                                    break
                        last_release_attempt_start_time = time.time()
                        run_card_release_attempt(controller)
                        waiting_for_release_key = False
                        monitoring_until_clear = False
                        has_printed_waiting_message = False
                    elif monitoring_until_clear:
                        if not state.get('iscardpresent', False):
                            monitoring_until_clear = False
                            waiting_for_release_key = False
                            has_printed_waiting_message = False
                            if last_release_attempt_start_time is not None:
                                elapsed = time.time() - last_release_attempt_start_time
                                if elapsed < RELEASE_ATTEMPT_MIN_INTERVAL_SECONDS:
                                    remaining = RELEASE_ATTEMPT_MIN_INTERVAL_SECONDS - elapsed
                                    if sleep_with_manual_stop(remaining, controller):
                                        mode_action = RUN_ACTION_EXIT
                                        break
                            last_release_attempt_start_time = time.time()
                            run_card_release_attempt(controller)
                    continue

                if window_key == ord('r'):
                    envservorelease(controller, cap=cap)
                    continue
                if web_key == 'r':
                    envservorelease(controller, cap=cap)
                    continue
                if web_key == 'q':
                    print("Returning to mode selection (website).", flush=True)
                    mode_action = RUN_ACTION_RESTART
                    break
                if web_key == 'esc':
                    print("Escape selected from website: exiting process.", flush=True)
                    mode_action = RUN_ACTION_EXIT
                    break
                if window_key == ord('q'):
                    print("Returning to mode selection.", flush=True)
                    mode_action = RUN_ACTION_RESTART
                    break
                if window_key == 27:
                    print("Escape pressed: exiting process.", flush=True)
                    mode_action = RUN_ACTION_EXIT
                    break

            if manual_stop_requested(controller):
                mode_action = RUN_ACTION_EXIT
                break

            if monitoring_until_clear:
                time.sleep(0.01)
                continue

            if waiting_for_release_key:
                time.sleep(0.01)
                continue

            card_entered = wait_for_card_entry_after_release(cap, controller)
            if card_entered is None:
                mode_action = RUN_ACTION_EXIT
                break
            if not card_entered:
                state['isreadytorecieve_under_camera'] = True
                waiting_for_release_key = True
                continue

            state['iscardpresent'] = True
            state['isreadytorecieve_under_camera'] = False

            if sleep_with_manual_stop(CARD_CAPTURE_SETTLE_SECONDS, controller):
                mode_action = RUN_ACTION_EXIT
                break

            capture_path, _ = capture_roi_image(cap)
            if not capture_path:
                waiting_for_release_key = True
                continue

            capture_to_result_start = time.perf_counter()
            card_info, orientation, distance, prices = recognize_card(
                capture_path, card_hash_db, annoy_index, idx_to_card
            )
            capture_to_result_total = time.perf_counter() - capture_to_result_start
            maybe_print_recognition_timing_summary(
                capture_to_result_total,
                LAST_RECOGNITION_TIMINGS,
                is_match=bool(card_info),
            )

            if not card_info:
                if orientation == LOCATOR_FAILURE_MARKER:
                    print(
                        "No YOLO OBB found in manual mode; treating as no card in environment and continuing release attempts.",
                        flush=True,
                    )
                    state['iscardpresent'] = False
                    state['isreadytorecieve_under_camera'] = True
                    waiting_for_release_key = True
                    monitoring_until_clear = False
                    has_printed_waiting_message = False
                    continue

                archive_failed_card_artifacts(
                    capture_path,
                    reason="card_not_identified",
                    extra={
                        'orientation': orientation,
                        'distance': distance,
                    },
                )
                pending_post_release_route = build_post_release_route(prices=None, identified=False)
                envservorelease(controller, cap=cap)
                monitoring_until_clear = True
                waiting_for_release_key = False
                has_printed_waiting_message = False
                continue

            card_name = card_info.get('name', 'Unknown')
            identified_card_names.append(card_name)
            print(f"Card identified: {card_name}", flush=True)
            archive_identified_card_artifacts(capture_path, card_info, orientation, distance, prices)
            publish_latest_match_to_expo(capture_path, card_info, orientation, distance, prices)

            pending_post_release_route = build_post_release_route(prices, identified=True)
            envservorelease(controller, cap=cap)
            monitoring_until_clear = True
            waiting_for_release_key = False
            has_printed_waiting_message = False

        if mode_action == RUN_ACTION_EXIT:
            cv2.destroyWindow(window_name)
            return RUN_ACTION_EXIT
        if mode_action == RUN_ACTION_RESTART:
            continue
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
        controller = DryRunController() if dry_run_mode else StepperController(port=SERIAL_PORT)
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
    start_expo_visualization_server()

    # diagnostics_ok = run_startup_and_motor_diagnostics(controller, cap)
    # if not diagnostics_ok:
    #     print("Startup diagnostics reported a failure. Exiting before loop execution.", flush=True)

    # Gradio interface commented out - using OpenCV window only
    # print("Creating Gradio interface...", flush=True)
    # demo = create_gradio_interface()
    
    # Run requested perpetual release/detect/identify/sort loop
    identified_cards = []
    process_exit_requested = False
    try:
        run_result = continuous_release_identify_and_sort(controller, cap, card_hash_db, annoy_index, idx_to_card)
        if run_result == RUN_ACTION_EXIT:
            process_exit_requested = True
            identified_cards = []
        else:
            identified_cards = run_result
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
        if process_exit_requested:
            print("Escape exit requested: process terminated by operator.", flush=True)
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
    controller = StepperController(port=SERIAL_PORT)
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