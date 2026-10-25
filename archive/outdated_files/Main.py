# Magic: The Gathering Card Text Recognition System

# --- Imports and Constants ---
import cv2
import numpy as np
import os
import imagehash
from PIL import Image
import json
import requests
import re

DEBUG_DIR = 'Debugging Images'
os.makedirs(DEBUG_DIR, exist_ok=True)

SCRYFALL_NORMAL_SIZE = (488, 680)  # (width, height)
image_storage_path = 'Image_storage/'
image_path = image_storage_path + 'Lord_of_the_pit.png'

USD_TO_CAD = 1.36
EUR_TO_CAD = 1.47

# --- Function Definitions ---
#------------------------------------------------------------------------------------------------------
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
#------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------
def preprocess_image(image_path):
    # Loading the image
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found at path: {image_path}")

    # Resize to Scryfall normal size (preserving aspect ratio, no loss of resolution)
    image = resize_to_scryfall_normal(image)

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply edge detection
    edges = cv2.Canny(gray, 50, 150)

    return image, gray, edges
#------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------
def detect_card(image, edges):
    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the largest contour
    largest_contour = max(contours, key=cv2.contourArea)

    # Create a copy of the image to draw the rectangle for debugging
    debug_image = image.copy()
    x, y, w, h = cv2.boundingRect(largest_contour)
    cv2.rectangle(debug_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    return debug_image, largest_contour
#------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------
def rotate_and_crop_image(image, contour):
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
#------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------
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
#------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------
def segment_symbols(crop_img, debug_prefix="symbol"):
    """
    Segments individual mana symbols from a cropped mana cost region.
    Returns a list of symbol images and a debug image with bounding boxes.
    Bounding boxes are intended to detect circular shapes (mana symbols).
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
    # Saving the debug bounding boxes image
    cv2.imwrite(os.path.join(DEBUG_DIR, f"{debug_prefix}_bounding_boxes.png"), debug_img)
    return symbol_imgs
#------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------
def compute_phash(image):
    """Compute perceptual hash for a given image using PIL and imagehash."""
    pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    return str(imagehash.phash(pil_img))
#------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------
def match_card_orientation_serial(image, hash_db, threshold=5):
    """
    Serial version: Match the image to the hash database, checking both upright and 180-degree rotated.
    Returns (card_info, orientation, hash_distance).
    """
    img_hash = compute_phash(image)
    rotated_img = cv2.rotate(image, cv2.ROTATE_180)
    rotated_hash = compute_phash(rotated_img)

    best_match = None
    best_distance = float('inf')
    best_orientation = None
    for entry in hash_db:
        db_hash = entry.get('phash')
        if db_hash is None:
            continue
        dist_upright = imagehash.hex_to_hash(db_hash) - imagehash.hex_to_hash(img_hash)
        if dist_upright < best_distance:
            best_distance = dist_upright
            best_match = entry
            best_orientation = 'upright'
        dist_rotated = imagehash.hex_to_hash(db_hash) - imagehash.hex_to_hash(rotated_hash)
        if dist_rotated < best_distance:
            best_distance = dist_rotated
            best_match = entry
            best_orientation = 'rotated 180'
    if best_match is not None and best_distance <= threshold:
        return best_match, best_orientation, best_distance
    else:
        return None, None, None
#------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------
def match_card_upright_serial(image, hash_db, threshold=5):
    """
    Serial version: Match the image to the hash database (upright only).
    Returns (card_info, hash_distance).
    """
    img_hash = compute_phash(image)
    best_match = None
    best_distance = float('inf')
    for entry in hash_db:
        db_hash = entry.get('phash')
        if db_hash is None:
            continue
        dist = imagehash.hex_to_hash(db_hash) - imagehash.hex_to_hash(img_hash)
        if dist < best_distance:
            best_distance = dist
            best_match = entry
    if best_match is not None and best_distance <= threshold:
        return best_match, best_distance
    else:
        return None, None
#------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------
def find_closest_match(image, hash_db):
    img_hash = compute_phash(image)
    best_match = None
    best_distance = float('inf')
    for entry in hash_db:
        db_hash = entry.get('phash')
        if db_hash is None:
            continue
        dist = imagehash.hex_to_hash(db_hash) - imagehash.hex_to_hash(img_hash)
        if dist < best_distance:
            best_distance = dist
            best_match = entry
    return best_match, best_distance
#------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------
def get_scryfall_prices(card_info):
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
        print(f"Failed to retrieve prices from {api_url} (status {response.status_code})")
        return {}
#------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------
def get_cad_price(prices):
    if prices.get('usd'):
        try:
            return round(float(prices['usd']) * USD_TO_CAD, 2), 'usd'
        except Exception:
            pass
    if prices.get('eur'):
        try:
            return round(float(prices['eur']) * EUR_TO_CAD, 2), 'eur'
        except Exception:
            pass
    return None, None
#------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------


# --- Main Execution ---

if __name__ == "__main__":
    # Step 2: Load and Preprocess the Image
    try:
        image, gray, edges = preprocess_image(image_path)

        # Detect the card and draw a border on a debug image
        debug_image, largest_contour = detect_card(image, edges)

        # Save debug images instead of displaying them
        cv2.imwrite(os.path.join(DEBUG_DIR, 'detected_card.png'), debug_image)
        cv2.imwrite(os.path.join(DEBUG_DIR, 'edge_detection.png'), edges)
    except FileNotFoundError as e:
        print(e)
        exit(1)

    # Step 3: Rotate and Crop the Image to Align Edges
    rotated_cropped_image = rotate_and_crop_image(image, largest_contour)

    if rotated_cropped_image.shape[0] > 0 and rotated_cropped_image.shape[1] > 0:
        # Save the rotated and cropped image
        cv2.imwrite(os.path.join(DEBUG_DIR, 'rotated_cropped_image.png'), rotated_cropped_image)
    else:
        print('Error: Cropped image has invalid dimensions.')
        exit(1)

    # Step 4: Detect Mana Symbols from the Cropped Image
    top_crop, bottom_crop, debug_img = crop_physical_regions(rotated_cropped_image)

    # Save debug image and cropped regions in the debugging folder
    cv2.imwrite(os.path.join(DEBUG_DIR, "debug_crops.png"), debug_img)
    cv2.imwrite(os.path.join(DEBUG_DIR, "top_crop.png"), top_crop)
    cv2.imwrite(os.path.join(DEBUG_DIR, "bottom_crop.png"), bottom_crop)

    symbols_top = segment_symbols(top_crop, debug_prefix="top_symbol")
    symbols_bottom = segment_symbols(bottom_crop, debug_prefix="bottom_symbol")

    # Decide which image to use and whether to check orientation in step 5
    if len(symbols_top) > 0 and len(symbols_bottom) == 0:
        print("Mana symbols detected in top crop (upright).")
        image_for_hashing = rotated_cropped_image
        check_orientation = False
    elif len(symbols_bottom) > 0 and len(symbols_top) == 0:
        print("Mana symbols detected in bottom crop (inverted).")
        image_for_hashing = cv2.rotate(rotated_cropped_image, cv2.ROTATE_180)
        check_orientation = False
    elif len(symbols_top) > 0 and len(symbols_bottom) > 0:
        print("Mana symbols detected in both crops.")
        image_for_hashing = rotated_cropped_image
        check_orientation = True
    else:
        print("No mana symbols detected in either crop.")
        image_for_hashing = rotated_cropped_image
        check_orientation = True

    # Step 5: Use image hashing to identify the image
    with open('scryfall_phash_db_sample.json', 'r') as f:
        card_hash_db_raw = json.load(f)
    card_hash_db = card_hash_db_raw

    if check_orientation:
        card_info, orientation, distance = match_card_orientation_serial(image_for_hashing, card_hash_db)
        if card_info:
            print(f"Matched card: {card_info.get('name', 'Unknown')} ({card_info.get('set', '')}), Orientation: {orientation}, Hash distance: {distance}")
            print(f"Scryfall URI: {card_info.get('scryfall_uri', '')}")
        else:
            print("No matching card found or orientation could not be determined.")
    else:
        card_info, distance = match_card_upright_serial(image_for_hashing, card_hash_db)
        if card_info:
            print(f"Matched card: {card_info.get('name', 'Unknown')} ({card_info.get('set', '')}), Hash distance: {distance}")
            print(f"Scryfall URI: {card_info.get('scryfall_uri', '')}")
        else:
            print("No matching card found.")
        '''
        # Step 5.5: Print Closest Match Regardless of Threshold
        closest_match, closest_distance = find_closest_match(image_for_hashing, card_hash_db)
        if closest_match:
            print(f"Closest match: {closest_match.get('name', 'Unknown')} ({closest_match.get('set', '')}), Hash distance: {closest_distance}")
            print(f"Scryfall URI: {closest_match.get('scryfall_uri', '')}")
        else:
            print("No closest match found.")
        '''
    # Step 6: Retrieve price information for the identified image
    if card_info:
        prices = get_scryfall_prices(card_info)
        cad_price, cad_source = get_cad_price(prices)

        print("Card prices:")
        if cad_price is not None:
            print(f"CAD: ${cad_price} (converted from {cad_source.upper()})")
        if prices.get('usd'):
            print(f"USD: ${prices['usd']}")
        if prices.get('eur'):
            print(f"EUR: €{prices['eur']}")
        if prices.get('usd_foil'):
            print(f"USD Foil: ${prices['usd_foil']}")
        if prices.get('usd_etched'):
            print(f"USD Etched: ${prices['usd_etched']}")
        if prices.get('eur_foil'):
            print(f"EUR Foil: €{prices['eur_foil']}")
        if prices.get('tix'):
            print(f"MTGO Tix: {prices['tix']}")
