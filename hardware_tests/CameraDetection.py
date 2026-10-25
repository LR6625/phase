"""
Camera Detection System for Magic: The Gathering Cards
Connects to USB camera at full 12MP resolution (3840×3032)

TO ADJUST SETTINGS:
- Edit the settings at lines 47-61
- All set to None = camera defaults (auto mode)
- Set specific values to override (see inline comments)
"""

import cv2
import numpy as np
import os
from datetime import datetime
import time

# Configuration
CAMERA_INDEX = 1  # 0 for built-in webcam, 1 for USB camera, etc.
CAPTURE_DIR = 'Captured_Cards'
DEBUG_DIR = 'Debugging Images'
os.makedirs(CAPTURE_DIR, exist_ok=True)
os.makedirs(DEBUG_DIR, exist_ok=True)

# Camera Quality Settings - ADJUST THESE TO TUNE IMAGE QUALITY
FRAME_WIDTH = 3840  # Arducam 12MP full resolution
FRAME_HEIGHT = 3032  # Arducam 12MP full resolution
FRAME_RATE = 15  # 15fps for 4K
USE_MJPEG = True  # Better quality compression

# Camera Control Settings - TUNE THESE FOR BRIGHTNESS/SHARPNESS
# Set to None to use camera defaults
AUTOFOCUS = None  # None = use default, 0 = manual, 1 = auto
FOCUS = None  # None = use default, 0-255 when manual
AUTO_EXPOSURE = 0.75  # 0.75 = auto mode for Arducam, 0.25 = manual
EXPOSURE = None  # None = use default (only used in manual mode)
BRIGHTNESS = None  # None = use default, 0-255 to adjust
CONTRAST = None  # None = use default, 0-255 to adjust
SHARPNESS = None  # None = use default, 0-255 to adjust
GAIN = None  # None = use default, 0-255 to adjust

# Card detection parameters
MIN_CARD_AREA = 5000  # Minimum area to consider as a card
MAX_CARD_AREA = 1000000  # Maximum area to avoid false positives
EDGE_THRESHOLD_LOW = 30  # Lower = detects darker edges
EDGE_THRESHOLD_HIGH = 100  # Adjust for edge sensitivity
USE_CLAHE = True  # Contrast enhancement for dark borders


def get_center_roi(frame, roi_width_percent=0.28, roi_height_percent=0.4):
    """
    Extract center region of interest from frame, shifted left by 2.5% of frame width.
    Returns (roi, roi_x1, roi_y1, roi_x2, roi_y2)
    """
    h, w = frame.shape[:2]
    
    # Calculate ROI dimensions (center of frame)
    roi_w = int(w * roi_width_percent)
    roi_h = int(h * roi_height_percent)
    roi_x1 = (w - roi_w) // 2 - int(w * 0.025)  # Shift left by 2.5% of width
    roi_y1 = (h - roi_h) // 2
    roi_x2 = roi_x1 + roi_w
    roi_y2 = roi_y1 + roi_h
    
    roi = frame[roi_y1:roi_y2, roi_x1:roi_x2]
    return roi, roi_x1, roi_y1, roi_x2, roi_y2


def detect_card_in_frame(frame):
    """
    Detect if a card is present in the center region of the frame.
    Returns (has_card, largest_contour, debug_image)
    """
    debug_image = frame.copy()
    
    # Extract center ROI
    roi, roi_x1, roi_y1, roi_x2, roi_y2 = get_center_roi(frame)
    
    # Draw ROI rectangle on debug image
    cv2.rectangle(debug_image, (roi_x1, roi_y1), (roi_x2, roi_y2), (255, 255, 0), 2)
    
    # Apply slight blur to reduce noise
    blurred = cv2.GaussianBlur(roi, (3, 3), 0)
    
    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
    
    # Apply CLAHE if enabled (helps detect dark borders on dark backgrounds)
    if USE_CLAHE:
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
    
    edges = cv2.Canny(gray, EDGE_THRESHOLD_LOW, EDGE_THRESHOLD_HIGH)
    
    # Apply morphological operations to enhance edges
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) == 0:
        return False, None, debug_image
    
    # Find largest contour
    largest_contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest_contour)
    
    # Check if area is within valid range for a card
    if MIN_CARD_AREA <= area <= MAX_CARD_AREA:
        # Additionally check if contour is roughly rectangular
        x, y, w, h = cv2.boundingRect(largest_contour)
        aspect_ratio = w / h if h > 0 else 0
        
        # Card aspect ratio should be between 0.5 and 2.0 (portrait or landscape orientation)
        if 0.5 <= aspect_ratio <= 2.0:
            # Convert ROI coordinates back to full frame coordinates
            card_x = roi_x1 + x
            card_y = roi_y1 + y
            cv2.rectangle(debug_image, (card_x, card_y), (card_x + w, card_y + h), (0, 255, 0), 2)
            cv2.putText(debug_image, f"Card Detected! Area: {int(area)}", 
                       (card_x, card_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            return True, largest_contour, debug_image
    
    # Draw info for debugging (even if not detected as card)
    x, y, w, h = cv2.boundingRect(largest_contour)
    card_x = roi_x1 + x
    card_y = roi_y1 + y
    cv2.rectangle(debug_image, (card_x, card_y), (card_x + w, card_y + h), (0, 0, 255), 1)
    cv2.putText(debug_image, f"Area: {int(area)}", 
               (card_x, card_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    
    return False, largest_contour, debug_image


def capture_and_save_card(frame, frame_count):
    """
    Save the captured frame as an image file.
    Returns filename of saved image.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(CAPTURE_DIR, f"card_{timestamp}_{frame_count}.jpg")
    cv2.imwrite(filename, frame)
    print(f"Card captured and saved: {filename}", flush=True)
    return filename


def run_camera_detection():
    """
    Main function to run camera detection
    """
    print(f"Attempting to open camera (index {CAMERA_INDEX})...", flush=True)
    
    # Open camera with DirectShow backend for better control on Windows
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    
    if not cap.isOpened():
        print(f"ERROR: Could not open camera at index {CAMERA_INDEX}", flush=True)
        print("Available cameras: Try indices 0, 1, 2, etc.", flush=True)
        return
    
    # Configure camera for optimal quality
    print("Configuring camera settings...", flush=True)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, FRAME_RATE)
    
    if USE_MJPEG:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
    
    # Camera control settings - only set if not None
    if AUTOFOCUS is not None:
        cap.set(cv2.CAP_PROP_AUTOFOCUS, AUTOFOCUS)
    if FOCUS is not None:
        cap.set(cv2.CAP_PROP_FOCUS, FOCUS)
    if AUTO_EXPOSURE is not None:
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, AUTO_EXPOSURE)
    if EXPOSURE is not None:
        cap.set(cv2.CAP_PROP_EXPOSURE, EXPOSURE)
    if BRIGHTNESS is not None:
        cap.set(cv2.CAP_PROP_BRIGHTNESS, BRIGHTNESS)
    if CONTRAST is not None:
        cap.set(cv2.CAP_PROP_CONTRAST, CONTRAST)
    if SHARPNESS is not None:
        cap.set(cv2.CAP_PROP_SHARPNESS, SHARPNESS)
    if GAIN is not None:
        cap.set(cv2.CAP_PROP_GAIN, GAIN)
    
    # Warm up camera and flush initial frames
    time.sleep(2)
    for _ in range(10):
        cap.read()
    
    # Print actual settings
    actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    actual_focus = cap.get(cv2.CAP_PROP_FOCUS)
    actual_exposure = cap.get(cv2.CAP_PROP_EXPOSURE)
    actual_brightness = cap.get(cv2.CAP_PROP_BRIGHTNESS)
    actual_contrast = cap.get(cv2.CAP_PROP_CONTRAST)
    
    print("=" * 60, flush=True)
    print("CAMERA SETTINGS", flush=True)
    print("=" * 60, flush=True)
    print(f"Resolution: {int(actual_width)}x{int(actual_height)} @ {actual_fps}fps", flush=True)
    print(f"Exposure: Auto, Brightness: {int(actual_brightness)}, Contrast: {int(actual_contrast)}", flush=True)
    print("=" * 60, flush=True)
    
    print("Camera ready!", flush=True)
    print("Controls: SPACE = capture, 'c' = clear captures, 'q' = quit", flush=True)
    
    # Create window
    window_name = 'Card Detection - Press Q to quit'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    
    frame_count = 0
    last_capture_time = 0
    last_detection_time = 0
    capture_cooldown = 2.0  # Prevent multiple captures within 2 seconds
    detection_interval = 0.8  # Only check for card every 0.8 seconds
    card_in_frame = False
    
    try:
        while True:
            ret, frame = cap.read()
            
            if not ret:
                print("ERROR: Failed to read frame from camera", flush=True)
                break
            
            frame_count += 1
            current_time = time.time()
            
            # Only detect card every 0.8 seconds
            if current_time - last_detection_time > detection_interval:
                has_card, contour, debug_frame = detect_card_in_frame(frame)
                last_detection_time = current_time
            else:
                # Use previous detection or show full frame with ROI
                _, _, debug_frame = detect_card_in_frame(frame)
                has_card = False  # Don't trigger capture between detection intervals
            
            # Check if we should capture
            should_capture = False
            
            # Auto-capture when card is detected (with cooldown)
            if has_card and current_time - last_capture_time > capture_cooldown:
                should_capture = True
                card_in_frame = True
            elif not has_card:
                card_in_frame = False
            
            # Display frame (window will scale image to fit fixed size)
            cv2.imshow(window_name, debug_frame)
            
            # Capture if conditions met
            if should_capture:
                capture_and_save_card(frame, frame_count)
                last_capture_time = current_time
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q') or key == 27:  # 'q' or ESC
                print("Quitting...", flush=True)
                break
            elif key == ord(' '):  # SPACE - force capture
                capture_and_save_card(frame, frame_count)
                print("Manual capture triggered", flush=True)
            elif key == ord('c'):  # 'c' - clear directory
                files = [f for f in os.listdir(CAPTURE_DIR) if f.endswith('.jpg')]
                for f in files:
                    os.remove(os.path.join(CAPTURE_DIR, f))
                print(f"Cleared {len(files)} files from capture directory", flush=True)
            
            # Show FPS
            if frame_count % 30 == 0:
                print(f"Frame: {frame_count} | Card detected: {has_card}", flush=True)
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Camera closed", flush=True)


if __name__ == "__main__":
    print("Magic: The Gathering Card Camera Detection", flush=True)
    print("=" * 50, flush=True)
    run_camera_detection()
