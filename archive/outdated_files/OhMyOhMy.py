"""
MTG Card Recognition System - Simplified Version
Process card images and send to Flask visualization server.
Run this script directly to process test images or import the function for use in other scripts.
"""

import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import imagehash
from PIL import Image

# Flask server integration for live visualization (optional)
try:
	FLASK_DIR = os.path.join(os.path.dirname(__file__), "Visualization")
	if FLASK_DIR not in sys.path:
		sys.path.insert(0, FLASK_DIR)
	from server_integration import send_card_to_server  # type: ignore
	FLASK_AVAILABLE = True
except ImportError as e:
	print(f"[WARNING] Flask visualization not available: {e}")
	FLASK_AVAILABLE = False
	def send_card_to_server(data):
		"""Dummy function when Flask not available"""
		return False

from Excecutable_File_Final import (
	load_or_build_annoy_index,
	preprocess_image,
	resize_to_scryfall_normal,
	compute_phash,
)

# Default test image if you run this script directly
TEST_IMAGE_PATH = os.path.join("Captured_Cards", "card_20260224_232336.jpg")


def process_and_send_card(image_path, card_hash_db, annoy_index, idx_to_card):
	"""
	Process a card image and send results to Flask visualization server.
	
	Args:
		image_path: Path to card image file or numpy array
		card_hash_db: Card hash database
		annoy_index: Annoy index for fast matching
		idx_to_card: Index to card mapping dictionary
		
	Returns:
		bool: True if successful, False otherwise
	"""
	print(f"\n{'='*60}")
	print(f"Processing: {image_path if isinstance(image_path, str) else 'numpy array'}")
	print(f"{'='*60}")
	
	try:
		# Import here to avoid circular dependency
		from Excecutable_File_Final import recognize_card
		
		# Process the card
		card_info, orientation, distance, prices = recognize_card(
			image_path, card_hash_db, annoy_index, idx_to_card
		)
		
		if not card_info:
			print("[ERROR] Card not recognized")
			return False
		
		# Load the processed image
		if isinstance(image_path, str):
			processed_image = cv2.imread(image_path)
		else:
			processed_image = image_path
			
		# Calculate quality and similarity
		hamming_distance = distance if distance is not None else 64
		similarity = f"{((64 - hamming_distance) / 64 * 100):.1f}%"
		
		if hamming_distance <= 5:
			quality = "EXCELLENT [OK]"
		elif hamming_distance <= 10:
			quality = "GOOD [OK]"
		elif hamming_distance <= 15:
			quality = "FAIR [OK]"
		else:
			quality = "POOR [WARNING]"
		
		# Get hash strings
		input_hash = card_info.get('phash', '')
		
		# Print results
		print(f"\n[OK] CARD IDENTIFIED")
		print(f"Name: {card_info.get('name', 'Unknown')}")
		print(f"Set: {card_info.get('set', 'N/A')}")
		print(f"Rarity: {card_info.get('rarity', 'N/A')}")
		print(f"Orientation: {orientation}")
		print(f"Hamming Distance: {hamming_distance}/64")
		print(f"Similarity: {similarity}")
		print(f"Quality: {quality}")
		
		# Send to Flask server
		if FLASK_AVAILABLE:
			success = send_card_to_server({
				'input_image': processed_image,
				'detected_image': processed_image,
				'input_hash': input_hash,
				'matched_hash': card_info.get('phash', ''),
				'hamming_distance': hamming_distance,
				'similarity': similarity,
				'quality': quality,
				'orientation': orientation if orientation else 'unknown',
				'card_info': {
					'name': card_info.get('name', 'Unknown'),
					'set': card_info.get('set', 'N/A'),
					'rarity': card_info.get('rarity', 'N/A'),
					'scryfall_id': card_info.get('id', ''),
				},
			})
			
			if success:
				print("\n[OK] Card data sent to Flask visualization server!")
				print("  -> Open http://localhost:5000 and click 'Refresh Data' to view")
			else:
				print("\n[WARNING] Could not send to Flask server (is it running?)")
				print("  -> Start server: cd 'Visualisation With Flask' && python app.py")
		else:
			print("\n[WARNING] Flask visualization not configured")
			print("  -> To enable: Ensure 'Visualisation With Flask' folder exists")
		
		return FLASK_AVAILABLE and success
		
	except Exception as e:
		print(f"[ERROR] Error processing card: {e}")
		import traceback
		traceback.print_exc()
		return False


if __name__ == "__main__":
	"""
	Run this script directly to process a test card image.
	"""
	print("\n" + "="*60)
	print("MTG CARD RECOGNITION - FLASK VISUALIZATION")
	print("="*60)
	
	# Load database and index
	try:
		print("\nLoading card database...")
		with open("scryfall_phash_db_sample.json", "r", encoding="utf-8") as f:
			card_hash_db = json.load(f)
		
		print("Building/loading Annoy index...")
		annoy_index, idx_to_card = load_or_build_annoy_index(card_hash_db)
		
		print("[OK] Database and index loaded successfully\n")
	except Exception as e:
		print(f"[ERROR] Failed to load database: {e}")
		exit(1)
	
	# Process test image
	if os.path.exists(TEST_IMAGE_PATH):
		process_and_send_card(TEST_IMAGE_PATH, card_hash_db, annoy_index, idx_to_card)
	else:
		print(f"[ERROR] Test image not found: {TEST_IMAGE_PATH}")
		print("\nTo use this script:")
		print("1. Place a card image in the Captured_Cards folder")
		print("2. Or import process_and_send_card() from another script")
	
	print("\n" + "="*60)
	print("Done! Check Flask server at http://localhost:5000")
	print("="*60 + "\n")
