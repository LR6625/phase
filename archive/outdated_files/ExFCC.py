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
import msvcrt
import sys
import serial
import threading

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
from annoy import AnnoyIndex


# ==============================================================================
# STEPPER CONTROLLER CLASS (from AllMotors.py)
# ==============================================================================
class StepperController:
    def __init__(self, port: str, baudrate: int = 9600, timeout: float = 120.0) -> None:
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self._ser: Optional[serial.Serial] = None

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
        if not self._ser or not self._ser.is_open:
            self.open()
        command = line.strip() + "\n"
        print(f"> Sending: {command.strip()}", flush=True)
        self._ser.write(command.encode("utf-8"))
        response = self._ser.readline().decode("utf-8").strip()
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

    def tt_motor_reverse(self, speed: Literal["SLOW", "MEDIUM", "FAST"] = "MEDIUM") -> str:
        """Run TT motor in reverse at specified speed"""
        return self._send(f"TT_REVERSE:{speed}")

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

    def tt_motor_forward(self, speed: Literal["SLOW", "MEDIUM", "FAST"] = "MEDIUM") -> str:
        return self._log(f"tt_motor_forward({speed})")

    def tt_motor_reverse(self, speed: Literal["SLOW", "MEDIUM", "FAST"] = "MEDIUM") -> str:
        return self._log(f"tt_motor_reverse({speed})")

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

# Scryfall "normal" image size
SCRYFALL_NORMAL_SIZE = (488, 680)  # (width, height)

# Annoy index configuration
ANNOY_INDEX_PATH = "phash_annoy_index.ann"
PHASH_IDX_TO_CARD_PATH = "phash_idx_to_card.json"
ANNOY_DIM = 64

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
    
    # Dilate edges to make them more connected
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges = cv2.dilate(edges, kernel, iterations=1)

    return image, gray_enhanced, edges

def detect_card(image, edges):
    """Detect the card in the image using contours with improved border detection"""
    # Additional morphological operations to close gaps in borders
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    edges_closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # Find contours
    contours, _ = cv2.findContours(edges_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the largest contour
    largest_contour = max(contours, key=cv2.contourArea)

    # Create a copy of the image to draw the rectangle for debugging
    debug_image = image.copy()
    x, y, w, h = cv2.boundingRect(largest_contour)
    cv2.rectangle(debug_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    return debug_image, largest_contour

def rotate_and_crop_image(image, contour):
    """
    Rotate and crop the image to align edges and ensure correct orientation
    Ensures the detected edges form a flat rectangle by rotating the image
    """
    # Find the minimum area rectangle around the contour
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)
    box = np.intp(box)

    # Order the box points: top-left, top-right, bottom-right, bottom-left
    def order_points(pts):
        pts = pts[np.argsort(pts[:, 1])]  # sort by y
        top = pts[:2][np.argsort(pts[:2, 0])]
        bottom = pts[2:][np.argsort(pts[2:, 0])]
        return np.array([top[0], top[1], bottom[1], bottom[0]], dtype="float32")

    ordered_box = order_points(box)

    # Compute width and height of the new image
    widthA = np.linalg.norm(ordered_box[2] - ordered_box[3])
    widthB = np.linalg.norm(ordered_box[1] - ordered_box[0])
    maxWidth = int(max(widthA, widthB))

    heightA = np.linalg.norm(ordered_box[1] - ordered_box[2])
    heightB = np.linalg.norm(ordered_box[0] - ordered_box[3])
    maxHeight = int(max(heightA, heightB))

    # Destination points for the upright rectangle
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]
    ], dtype="float32")

    # Perspective transform to get the upright, cropped card
    M = cv2.getPerspectiveTransform(ordered_box, dst)
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

def get_scryfall_prices(card_info):
    """Retrieve price information for the identified card from Scryfall"""
    scryfall_uri = card_info.get('scryfall_uri', '')

    m = re.search(r'/card/([^/]+)/([^/?]+)', scryfall_uri)
    if m:
        set_code, collector_number = m.group(1), m.group(2)
        api_url = f"https://api.scryfall.com/cards/{set_code}/{collector_number}"
    else:
        # As a last resort, try to use the full scryfall_uri (replace domain)
        api_url = scryfall_uri.replace('scryfall.com/card/', 'api.scryfall.com/cards/')
    response = requests.get(api_url)
    if response.status_code == 200:
        data = response.json()
        return data.get('prices', {})
    else:
        print(f"Failed to retrieve prices from {api_url} (status {response.status_code})", flush=True)
        return {}

def get_cad_price(prices, usd_to_cad=1.37, eur_to_cad=1.47):
    """Convert USD or EUR prices to CAD"""
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
    return None, None

def print_card_prices(prices, usd_to_cad=1.37, eur_to_cad=1.47):
    """Print card prices in requested order: CAD, USD, EUR, then foils"""
    cad_price, cad_source = get_cad_price(prices, usd_to_cad, eur_to_cad)

    print("Card prices:", flush=True)
    if cad_price is not None:
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

# ==============================================================================
# CARD RECOGNITION PIPELINE
# ==============================================================================
def recognize_card(image_path, card_hash_db, annoy_index, idx_to_card):
    """
    Complete card recognition pipeline
    Returns card_info, orientation, distance, and prices
    """
    # Clear old debug images
    clear_debug_images(DEBUG_DIR)

    # Step 1: Preprocess the image
    image, gray, edges = preprocess_image(image_path)

    # Step 2: Detect the card
    debug_image, largest_contour = detect_card(image, edges)

    # Save debug images
    cv2.imwrite(os.path.join(DEBUG_DIR, 'detected_card.png'), debug_image)
    cv2.imwrite(os.path.join(DEBUG_DIR, 'edge_detection.png'), edges)

    # Step 3: Rotate and crop
    rotated_cropped_image = rotate_and_crop_image(image, largest_contour)

    if rotated_cropped_image.shape[0] > 0 and rotated_cropped_image.shape[1] > 0:
        cv2.imwrite(os.path.join(DEBUG_DIR, 'rotated_cropped_image.png'), rotated_cropped_image)
    else:
        print('Error: Cropped image has invalid dimensions.', flush=True)
        return None, None, None, None

    # Step 4: Detect mana symbols and determine orientation
    top_crop, bottom_crop, debug_img = crop_physical_regions(rotated_cropped_image)

    # Save debug images
    cv2.imwrite(os.path.join(DEBUG_DIR, "debug_crops.png"), debug_img)
    cv2.imwrite(os.path.join(DEBUG_DIR, "top_crop.png"), top_crop)
    cv2.imwrite(os.path.join(DEBUG_DIR, "bottom_crop.png"), bottom_crop)

    symbols_top = segment_symbols(top_crop, debug_prefix="top_symbol")
    symbols_bottom = segment_symbols(bottom_crop, debug_prefix="bottom_symbol")

    # Decide which image to use and whether to check orientation
    if len(symbols_top) > 0 and len(symbols_bottom) == 0:
        print("Mana symbols detected in top crop (upright).", flush=True)
        image_for_hashing = rotated_cropped_image
        check_orientation = False
    elif len(symbols_bottom) > 0 and len(symbols_top) == 0:
        print("Mana symbols detected in bottom crop (inverted).", flush=True)
        image_for_hashing = cv2.rotate(rotated_cropped_image, cv2.ROTATE_180)
        check_orientation = False
    elif len(symbols_top) > 0 and len(symbols_bottom) > 0:
        print("Mana symbols detected in both crops.", flush=True)
        image_for_hashing = rotated_cropped_image
        check_orientation = True
    else:
        print("No mana symbols detected in either crop.", flush=True)
        image_for_hashing = rotated_cropped_image
        check_orientation = True

    # Step 5: Use Annoy for image hash matching
    if check_orientation:
        card_info, orientation, distance = annoy_nearest_phash_orientation(
            image_for_hashing, annoy_index, idx_to_card, n=3
        )
        if card_info:
            print(f"Matched card: {card_info.get('name', 'Unknown')} ({card_info.get('set', '')}), "
                  f"Orientation: {orientation}, Hash distance: {distance}", flush=True)
            print(f"Scryfall URI: {card_info.get('scryfall_uri', '')}", flush=True)
        else:
            print("No matching card found or orientation could not be determined.", flush=True)
    else:
        results = annoy_nearest_phash(image_for_hashing, annoy_index, idx_to_card, n=1)
        if results and results[0][0]:
            card_info, distance = results[0]
            orientation = 'upright'
            print(f"Matched card: {card_info.get('name', 'Unknown')} ({card_info.get('set', '')}), "
                  f"Hash distance: {distance}", flush=True)
            print(f"Scryfall URI: {card_info.get('scryfall_uri', '')}", flush=True)
        else:
            print("No matching card found.", flush=True)
            card_info = None
            orientation = None
            distance = None

    # Step 6: Get prices if card was identified
    prices = None
    if card_info:
        prices = get_scryfall_prices(card_info)
        print_card_prices(prices)

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
    Detect if a card is present in the center region of the frame.
    Uses CLAHE enhancement to detect black borders on dark backgrounds.
    Returns (has_card, debug_info_dict)
    """
    roi, roi_x1, roi_y1, roi_x2, roi_y2 = get_center_roi(frame)
    
    # Apply slight blur to reduce noise
    blurred = cv2.GaussianBlur(roi, (3, 3), 0)
    
    # Convert to grayscale
    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
    
    # Apply CLAHE to enhance contrast for black borders
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    gray_enhanced = clahe.apply(gray)
    
    # Apply edge detection with higher thresholds to reduce noise
    edges = cv2.Canny(gray_enhanced, 80, 200)
    
    # Morphological operations to close gaps and connect edges
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    debug_info = {'contours': len(contours), 'area': 0, 'aspect_ratio': 0, 'detected': False, 'bbox': None}
    
    if len(contours) == 0:
        return False, debug_info
    
    largest_contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest_contour)
    debug_info['area'] = int(area)
    
    if MIN_CARD_AREA <= area <= MAX_CARD_AREA:
        x, y, w, h = cv2.boundingRect(largest_contour)
        aspect_ratio = w / h if h > 0 else 0
        debug_info['aspect_ratio'] = round(aspect_ratio, 2)
        
        if 0.5 <= aspect_ratio <= 2.0:
            debug_info['detected'] = True
            # Convert ROI coordinates to full frame coordinates
            debug_info['bbox'] = (roi_x1 + x, roi_y1 + y, w, h)
            return True, debug_info
    
    return False, debug_info

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
        debug_frame = frame.copy()
        
        # Draw ROI rectangle
        roi, roi_x1, roi_y1, roi_x2, roi_y2 = get_center_roi(frame)
        cv2.rectangle(debug_frame, (roi_x1, roi_y1), (roi_x2, roi_y2), (255, 255, 0), 2)
        
        # Check for card detection
        card_detected = False
        if current_time - last_detection_time > detection_interval:
            if detect_card_in_frame(frame):
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
            
            # Set servo 2 to 90 degrees and keep it there
        print("Setting servo 2 to 90 degrees...", flush=True)        
        controller.servo2_move(90)
        time.sleep(0.5)
        print("Servo 2 positioned at 90 degrees", flush=True)
    except Exception as e:
        print(f"Error connecting to controller: {e}", flush=True)
        exit(1)

    print("opening camera...", flush=True)
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("ERROR: Yeah that camera aint opening chief try a different port", flush=True)
    exit(1)
    
    # Configure for full resolution with brightness and contrast adjustments
    print("Configuring camera for full 12MP resolution...", flush=True)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 3840)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 3032)
    
    # Adjust image quality settings to fix dark image
    cap.set(cv2.CAP_PROP_BRIGHTNESS, 130)  # Increase brightness
    cap.set(cv2.CAP_PROP_CONTRAST, 10)     # Set contrast to reasonable value
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)  # Auto exposure mode
    
    # Warm up camera and flush initial frames
    time.sleep(2)
    for _ in range(10):
        cap.read()
    
    actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    actual_brightness = cap.get(cv2.CAP_PROP_BRIGHTNESS)
    actual_contrast = cap.get(cv2.CAP_PROP_CONTRAST)
    print(f"Camera initialized at {int(actual_width)}x{int(actual_height)}", flush=True)
    print(f"Brightness: {int(actual_brightness)}, Contrast: {int(actual_contrast)}", flush=True)

    print("loading hash database...", flush=True)
    with open('scryfall_phash_db_sample.json', 'r') as f:
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
            debug_frame = frame.copy()
            
            # Draw ROI rectangle (bright cyan, thick line)
            roi, roi_x1, roi_y1, roi_x2, roi_y2 = get_center_roi(frame)
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
                    roi, roi_x1, roi_y1, roi_x2, roi_y2 = get_center_roi(capture_frame)
                    cropped_frame = capture_frame[roi_y1:roi_y2, roi_x1:roi_x2]
                    
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
    """Stop when switch is pressed or any keyboard key is pressed."""
    try:
        if controller.is_switch_pressed():
            print("Manual stop requested: hardware switch pressed.", flush=True)
            return True
    except Exception:
        pass

    try:
        if msvcrt.kbhit():
            key = msvcrt.getch().decode('utf-8', errors='ignore')
            if key:
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


def run_card_release_attempt(controller):
    """Attempt a single card release using TT motor sequence."""
    controller.spin_up_m1(240)
    controller.tt_motor_forward("SLOW")
    time.sleep(0.5)
    controller.tt_motor_stop()
    time.sleep(0.3)

    controller.spin_up_m1(240)
    controller.tt_motor_reverse("MEDIUM")
    time.sleep(0.3)
    controller.tt_motor_stop()
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
    """Show live feed and wait until user presses 's' to start process."""
    print("Waiting for start signal. Press 's' to begin, or 'q'/ESC to cancel.", flush=True)

    while True:
        ret, live_frame = cap.read()
        if not ret or live_frame is None:
            continue

        preview = live_frame.copy()
        _, roi_x1, roi_y1, roi_x2, roi_y2 = get_center_roi(live_frame)
        cv2.rectangle(preview, (roi_x1, roi_y1), (roi_x2, roi_y2), (0, 255, 255), 3)
        draw_text_with_background(preview, "Live Feed Ready", (20, 35), font_scale=0.8, thickness=2)
        draw_text_with_background(preview, "Press 's' to start", (20, 70), font_scale=0.8, thickness=2)
        draw_text_with_background(preview, "Press 'q' or ESC to cancel", (20, live_frame.shape[0] - 20), font_scale=0.6, thickness=2)
        cv2.imshow(window_name, preview)

        window_key = cv2.waitKey(1) & 0xFF
        if window_key in (ord('s'), ord('S')):
            clear_keyboard_buffer()
            print("Start signal received from camera window.", flush=True)
            return True
        if window_key == ord('q') or window_key == 27:
            print("Start canceled from camera window.", flush=True)
            return False

        try:
            if msvcrt.kbhit():
                key = msvcrt.getch().decode('utf-8', errors='ignore').lower()
                if key == 's':
                    clear_keyboard_buffer()
                    print("Start signal received from keyboard.", flush=True)
                    return True
                if key == 'q':
                    print("Start canceled from keyboard.", flush=True)
                    return False
        except Exception:
            pass


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

    _, roi_x1, roi_y1, roi_x2, roi_y2 = get_center_roi(capture_frame)
    cropped_frame = capture_frame[roi_y1:roi_y2, roi_x1:roi_x2]
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


def route_identified_card(controller, prices, distance, state):
    """Route identified card to storage logic requested by user."""
    cad_price, _ = get_cad_price(prices if prices else {})
    if cad_price is None:
        cad_price = 0.0

    if distance is not None and distance > 7.5:
        print("card failed to identify, hashing distance greater than 7.5", flush=True)
        state['isreadytorecieve_m2_container'] = False
        controller.spin_up_m2(2050)
        controller.servo1_rotate_90_and_back(delay_seconds=0.3)
        controller.spin_down_m2(2050)
        state['isreadytorecieve_m2_container'] = True
        return

    if cad_price < 0.50:
        state['isreadytorecieve_m2_container'] = False
        controller.spin_down_m2(2050)
        controller.servo1_rotate_90_and_back(delay_seconds=0.3)
        state['iscardpresent'] = False
        controller.spin_up_m2(2050)
        state['isreadytorecieve_m2_container'] = True
        return

    state['isreadytorecieve_m2_container'] = False
    controller.servo1_rotate_90_and_back(delay_seconds=0.3)
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

        print("Testing M2: down then back up...", flush=True)
        controller.spin_down_m2(2050)
        time.sleep(0.35)
        controller.spin_up_m2(2050)
        time.sleep(0.35)

        print("Testing TT motor: forward and reverse...", flush=True)
        controller.tt_motor_forward("SLOW")
        time.sleep(0.9)
        controller.tt_motor_stop()
        time.sleep(1)
        controller.tt_motor_reverse("SLOW")
        time.sleep(0.9)
        controller.tt_motor_stop()
        time.sleep(0.2)

        print("Testing servo 1 sweep and return to center...", flush=True)
        controller.servo1_rotate_90_and_back(delay_seconds=0.3)
        time.sleep(0.3)

        print("Testing servo 2 movement and return to expected position...", flush=True)
        controller.servo2_move(-50)
        time.sleep(0.3)
        controller.servo2_move(35)
        time.sleep(0.3)

        print("Diagnostics complete: all motors exercised and returned.", flush=True)
        return True
    except Exception as exc:
        print(f"Diagnostics failed while testing actuators: {exc}", flush=True)
        return False


def continuous_release_identify_and_sort(controller, cap, card_hash_db, annoy_index, idx_to_card):
    """
    Perpetual loop:
    - release one card attempt
    - wait for card entry
    - detect + identify
    - use servo2 retries until card leaves camera (max 5)
    - route by CAD price and hashing distance
    Stops when switch is pressed or keyboard letter is pressed.
    """
    state = {
        'isreadytorecieve_under_camera': True,
        'isreadytorecieve_m2_container': True,
        'iscardpresent': False,
    }
    first_card_identified = False
    identified_card_names = []
    stop_requested = False
    window_name = "Live Camera Feed"

    print("Starting perpetual release-identify-sort loop", flush=True)
    print("Stop with hardware switch or any keyboard key.", flush=True)
    print("Live preview enabled.", flush=True)

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1000, 700)

    if not wait_for_start_signal(cap, window_name):
        cv2.destroyWindow(window_name)
        return identified_card_names

    print("Startup sequence: smooth M1 down to switch, then smooth M1 up to switch.", flush=True)
    controller.spin_down_m1_until_switch()
    time.sleep(0.2)
    controller.spin_up_m1_until_switch()
    time.sleep(0.2)

    while True:
        ret_live, live_frame = cap.read()
        if ret_live and live_frame is not None:
            preview = live_frame.copy()
            _, roi_x1, roi_y1, roi_x2, roi_y2 = get_center_roi(live_frame)
            cv2.rectangle(preview, (roi_x1, roi_y1), (roi_x2, roi_y2), (0, 255, 255), 3)
            draw_text_with_background(preview, "Live Feed", (20, 35), font_scale=0.8, thickness=2)
            draw_text_with_background(preview, f"Identified: {len(identified_card_names)}", (20, 70), font_scale=0.7, thickness=2)
            draw_text_with_background(preview, "Press q or ESC in window to stop", (20, live_frame.shape[0] - 20), font_scale=0.6, thickness=2)
            cv2.imshow(window_name, preview)

            window_key = cv2.waitKey(1) & 0xFF
            if window_key == ord('q') or window_key == 27:
                print("Stopping process from camera window keypress.", flush=True)
                stop_requested = True
                break

        if manual_stop_requested(controller):
            print("Stopping process on manual request.", flush=True)
            stop_requested = True
            break

        run_card_release_attempt(controller)

        if manual_stop_requested(controller):
            print("Stopping process on manual request.", flush=True)
            stop_requested = True
            break

        card_entered = wait_for_card_entry_after_release(cap, controller)
        if card_entered is None:
            print("Stopping process on manual request.", flush=True)
            break
        if not card_entered:
            state['isreadytorecieve_under_camera'] = True
            print("No card detected after release window. Retrying release.", flush=True)
            continue

        state['iscardpresent'] = True
        state['isreadytorecieve_under_camera'] = False
        print("Card detected in imaging area. Capturing and identifying...", flush=True)

        capture_path, _ = capture_roi_image(cap)
        if not capture_path:
            print("Capture failed. Retrying release loop.", flush=True)
            continue

        card_info, orientation, distance, prices = recognize_card(
            capture_path, card_hash_db, annoy_index, idx_to_card
        )

        if not card_info:
            print("Card could not be identified. Restarting release loop.", flush=True)
            continue

        first_card_identified = True
        card_name = card_info.get('name', 'Unknown')
        identified_card_names.append(card_name)
        print(f"Card identified: {card_name}", flush=True)

        released_from_imaging = False
        for attempt in range(1, 6):
            if manual_stop_requested(controller):
                print("Stopping process on manual request.", flush=True)
                stop_requested = True
                break

            if not state['isreadytorecieve_m2_container']:
                print("Servo2 movement blocked: motor2 container isreadytorecieve is False.", flush=True)
                break

            controller.servo2_move_abs(-50)
            if sleep_with_manual_stop(0.4, controller):
                print("Stopping process on manual request.", flush=True)
                stop_requested = True
                break
            controller.servo2_move_abs(90)
            state['isreadytorecieve_m2_container'] = False

            if sleep_with_manual_stop(0.15, controller):
                print("Stopping process on manual request.", flush=True)
                stop_requested = True
                break

            has_card, _ = read_frame_and_detect_card(cap)
            state['iscardpresent'] = has_card
            state['isreadytorecieve_under_camera'] = not has_card
            print(f"Servo2 release attempt {attempt}/5 | card still present: {has_card}", flush=True)

            if not has_card:
                state['isreadytorecieve_under_camera'] = True
                route_identified_card(controller, prices, distance, state)
                released_from_imaging = True
                break
            state['isreadytorecieve_m2_container'] = True

        if stop_requested:
            break

        if not released_from_imaging and state['iscardpresent']:
            if first_card_identified:
                print("Card still detected after 5 servo2 attempts. Breaking all processes.", flush=True)
                break
            print("Card still detected after 5 attempts before first ID. Continuing release loop.", flush=True)

    print("Perpetual loop ended.", flush=True)
    cv2.destroyWindow(window_name)
    return identified_card_names
# ==============================================================================
# MAIN EXECUTION
# ==============================================================================
if __name__ == "__main__":
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
        
        # Set servo 2 to 50 degrees and keep it there
        print("Setting servo 2 to 50 degrees...", flush=True)
        controller.servo2_move(35)
        time.sleep(0.5)
        print("Servo 2 positioned at 50 degrees", flush=True)
    except Exception as e:
        print(f"Error connecting to controller: {e}", flush=True)
        exit(1)

    print("opening camera...", flush=True)
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("ERROR: Yeah that camera aint opening chief try a different port", flush=True)
        exit(1)
    
    # Configure for full resolution with brightness and contrast adjustments
    print("Configuring camera for full 12MP resolution...", flush=True)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 3840)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 3032)
    
    # Adjust image quality settings to fix dark image
    cap.set(cv2.CAP_PROP_BRIGHTNESS, 130)  # Increase brightness
    cap.set(cv2.CAP_PROP_CONTRAST, 10)     # Set contrast to reasonable value
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)  # Auto exposure mode
    
    # Warm up camera and flush initial frames
    time.sleep(2)
    for _ in range(10):
        cap.read()
    
    actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    actual_brightness = cap.get(cv2.CAP_PROP_BRIGHTNESS)
    actual_contrast = cap.get(cv2.CAP_PROP_CONTRAST)
    print(f"Camera initialized at {int(actual_width)}x{int(actual_height)}", flush=True)
    print(f"Brightness: {int(actual_brightness)}, Contrast: {int(actual_contrast)}", flush=True)

    print("loading hash database...", flush=True)
    with open('scryfall_phash_db_sample.json', 'r') as f:
        card_hash_db = json.load(f)
    annoy_index, idx_to_card = load_or_build_annoy_index(card_hash_db)
    print("camera and database ready", flush=True)
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
    with open('scryfall_phash_db_sample.json', 'r') as f:
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