"""
Flask web application for MTG Card Hash Visualization
Receives card data from OhMyOhMy.py and visualizes it
"""

from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import json
import os
import base64
from io import BytesIO
import matplotlib.pyplot as plt
import matplotlib
from datetime import datetime
from pathlib import Path
matplotlib.use('Agg')

# Get the directory of the current script
BASE_DIR = Path(__file__).parent
TEMPLATE_DIR = BASE_DIR / 'templates'
STATIC_DIR = BASE_DIR / 'static'

app = Flask(__name__, template_folder=str(TEMPLATE_DIR), static_folder=str(STATIC_DIR))
CORS(app)

# Store the latest card data globally
LATEST_CARD_DATA = {
    'success': False,
    'data': None,
    'timestamp': None,
    'card_name': 'None'
}


# ==============================================================================
# API ROUTES
# ==============================================================================

@app.route('/')
def index():
    """Serve the main page"""
    return render_template('index.html')


@app.route('/api/receive-card', methods=['POST'])
def receive_card():
    """
    Receive card data from OhMyOhMy.py
    
    Expected JSON format:
    {
        'input_image': 'base64_image',
        'detected_image': 'base64_image',
        'input_hash': 'hex_string',
        'matched_hash': 'hex_string',
        'hamming_distance': int,
        'similarity': 'percentage_string',
        'quality': 'quality_string',
        'orientation': 'orientation_string',
        'card_info': {
            'name': 'string',
            'set': 'string',
            'rarity': 'string',
            ...
        },
        'hash_visualization': 'base64_image',
        'comparison_visualization': 'base64_image'
    }
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Store the latest card data
        global LATEST_CARD_DATA
        LATEST_CARD_DATA = {
            'success': True,
            'data': data,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'card_name': data.get('card_info', {}).get('name', 'Unknown Card')
        }
        
        print(f"[LOG] Card data received: {LATEST_CARD_DATA['card_name']} at {LATEST_CARD_DATA['timestamp']}")
        
        return jsonify({
            'success': True,
            'message': 'Card data received and stored',
            'status': 'ready',
            'card_name': LATEST_CARD_DATA['card_name'],
            'timestamp': LATEST_CARD_DATA['timestamp']
        }), 200
    
    except Exception as e:
        return jsonify({'error': f'Failed to receive card data: {str(e)}'}), 500


@app.route('/api/get-latest', methods=['GET'])
def get_latest():
    """
    Get the latest card data
    Called by the web interface via button click
    """
    global LATEST_CARD_DATA
    
    if not LATEST_CARD_DATA['success']:
        return jsonify({
            'success': False,
            'message': 'No card data available yet. Run OhMyOhMy.py to send data.'
        }), 200
    
    return jsonify(LATEST_CARD_DATA['data']), 200


@app.route('/api/has-new-data', methods=['GET'])
def has_new_data():
    """
    Check if there's new data available
    (Optional: for showing a notification badge)
    """
    global LATEST_CARD_DATA
    return jsonify({
        'has_data': LATEST_CARD_DATA['success'],
        'ready': True
    }), 200


@app.route('/api/status', methods=['GET'])
def status():
    """Get current server and data status"""
    try:
        global LATEST_CARD_DATA
        return jsonify({
            'server_ready': True,
            'has_data': LATEST_CARD_DATA['success'],
            'current_card': LATEST_CARD_DATA.get('card_name', 'None'),
            'timestamp': LATEST_CARD_DATA.get('timestamp', 'Never'),
            'message': f"Currently showing: {LATEST_CARD_DATA.get('card_name', 'None')} (updated {LATEST_CARD_DATA.get('timestamp', 'Never')})" if LATEST_CARD_DATA['success'] else "Waiting for card data..."
        }), 200
    except Exception as e:
        print(f"[ERROR] Status endpoint error: {e}")
        return jsonify({
            'server_ready': True,
            'has_data': False,
            'current_card': 'Error',
            'timestamp': 'Error',
            'message': f'Status error: {str(e)}'
        }), 200


if __name__ == '__main__':
    # Create templates directory if needed
    os.makedirs('templates', exist_ok=True)
    os.makedirs('static', exist_ok=True)
    
    print("\n" + "="*60)
    print("  MTG Card Hash Visualization Server")
    print("="*60)
    print("\nServer running on: http://localhost:5000")
    print("\nWaiting for card data from OhMyOhMy.py...")
    print("="*60 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False)
