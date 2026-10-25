"""
Integration module for sending card analysis from OhMyOhMy.py to Flask visualization server

Add this to your OhMyOhMy.py to enable sending data to the web server
"""

import requests
import json
import base64
from io import BytesIO
from PIL import Image


# Configuration
FLASK_SERVER_URL = "http://localhost:5000"


def send_card_to_server(card_data):
    """
    Send processed card data to the Flask visualization server
    
    Parameters:
    -----------
    card_data : dict
        Dictionary containing:
        - input_image: PIL Image or numpy array
        - detected_image: PIL Image or numpy array  
        - input_hash: string (hex)
        - matched_hash: string (hex)
        - hamming_distance: int
        - similarity: string (e.g., "89.1%")
        - quality: string (e.g., "EXCELLENT ✓")
        - orientation: string
        - card_info: dict with 'name', 'set', 'rarity', etc.
        - hash_visualization: PIL Image (optional)
        - comparison_visualization: PIL Image (optional)
    
    Returns:
    --------
    bool: True if sent successfully, False otherwise
    """
    try:
        # Convert images to base64
        payload = {
            'input_hash': card_data.get('input_hash', ''),
            'matched_hash': card_data.get('matched_hash', ''),
            'hamming_distance': card_data.get('hamming_distance', 0),
            'similarity': card_data.get('similarity', ''),
            'quality': card_data.get('quality', ''),
            'orientation': card_data.get('orientation', ''),
            'card_info': card_data.get('card_info', {}),
        }
        
        # Convert images to base64
        if 'input_image' in card_data:
            payload['input_image'] = image_to_base64(card_data['input_image'])
        
        if 'detected_image' in card_data:
            payload['detected_image'] = image_to_base64(card_data['detected_image'])
        
        if 'hash_visualization' in card_data:
            payload['hash_visualization'] = image_to_base64(card_data['hash_visualization'])
        
        if 'comparison_visualization' in card_data:
            payload['comparison_visualization'] = image_to_base64(card_data['comparison_visualization'])
        
        # Send to server
        response = requests.post(
            f"{FLASK_SERVER_URL}/api/receive-card",
            json=payload,
            timeout=5
        )
        
        if response.status_code == 200:
            print(f"✓ Card data sent to server successfully")
            return True
        else:
            print(f"⚠ Server returned status {response.status_code}: {response.text}")
            return False
    
    except requests.exceptions.ConnectionError:
        print(f"✗ Cannot connect to Flask server at {FLASK_SERVER_URL}")
        print("  Make sure to start the visualization server first:")
        print("  python Visualisation\\ With\\ Flask/app.py")
        return False
    except Exception as e:
        print(f"✗ Error sending data to server: {str(e)}")
        return False


def image_to_base64(img):
    """
    Convert PIL Image or numpy array to base64 data URI
    """
    import numpy as np
    
    # Convert numpy array to PIL Image if needed
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img)
    
    # Convert PIL Image to base64
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/png;base64,{img_str}"


# ==============================================================================
# Usage Example
# ==============================================================================

"""
In your OhMyOhMy.py process_card_image() function, after computing results,
add this line to send to the visualization server:

    # After getting all the results...
    from server_integration import send_card_to_server
    
    send_card_to_server({
        'input_image': rotated_cropped,      # The card image
        'detected_image': detected_card,      # The detected region
        'input_hash': str(input_phash),
        'matched_hash': matched_phash,
        'hamming_distance': hamming_distance,
        'similarity': f"{((64 - hamming_distance) / 64 * 100):.1f}%",
        'quality': "EXCELLENT" if hamming_distance <= 5 else "GOOD",
        'orientation': orientation,
        'card_info': {
            'name': best_card.get('name', 'Unknown'),
            'set': best_card.get('set', 'N/A'),
            'rarity': best_card.get('rarity', 'N/A'),
            'scryfall_id': best_card.get('scryfall_id', 'N/A'),
        },
        'hash_visualization': hash_viz_pil,      # If available
        'comparison_visualization': comparison_pil  # If available
    })
"""
