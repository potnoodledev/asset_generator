from flask import Flask, request, jsonify, send_file, Response
import os
import json
import time
from asset_generator import process_assets
import queue
import threading
import sys
import shutil
from io import StringIO
from pathlib import Path

app = Flask(__name__)

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'game/output'
for folder in [UPLOAD_FOLDER, OUTPUT_FOLDER]:
    os.makedirs(folder, exist_ok=True)

# Queue for progress messages
progress_queue = queue.Queue()

def clean_output_directory():
    """Remove all contents from the output directory"""
    for root, dirs, files in os.walk(OUTPUT_FOLDER):
        for f in files:
            os.unlink(os.path.join(root, f))
        for d in dirs:
            shutil.rmtree(os.path.join(root, d))

def progress_callback(message):
    """Callback function to handle progress messages"""
    progress_queue.put({"type": "progress", "message": str(message)})

class OutputCapture:
    def __init__(self):
        self.stdout = StringIO()
    
    def write(self, text):
        if text.strip():  # Only forward non-empty lines
            progress_queue.put({"type": "progress", "message": text.strip()})
    
    def flush(self):
        pass

@app.route('/')
def serve_index():
    return send_file('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not file.filename.endswith('.json'):
        return jsonify({'error': 'File must be JSON'}), 400
    
    # Save the uploaded file
    filepath = os.path.join(UPLOAD_FOLDER, 'config.json')
    file.save(filepath)

    # Save pixel art settings if provided
    pixel_art_settings = request.form.get('pixel_art_settings')
    if pixel_art_settings:
        settings_path = os.path.join(UPLOAD_FOLDER, 'pixel_settings.json')
        with open(settings_path, 'w') as f:
            f.write(pixel_art_settings)

    # Save art style if provided
    art_style = request.form.get('art_style', 'cartoon')
    style_path = os.path.join(UPLOAD_FOLDER, 'style_settings.json')
    with open(style_path, 'w') as f:
        json.dump({'style': art_style}, f)

    return jsonify({'message': 'File uploaded successfully'})

def generate_assets(config_path):
    try:
        # Clean up previous assets
        clean_output_directory()
        progress_queue.put({"type": "progress", "message": "Cleaned previous assets"})
        
        # Load pixel art settings if they exist
        settings_path = os.path.join(UPLOAD_FOLDER, 'pixel_settings.json')
        pixel_settings = {
            'enabled': True,
            'gridSize': 6,
            'colorCount': 8
        }
        if os.path.exists(settings_path):
            with open(settings_path, 'r') as f:
                pixel_settings = json.load(f)

        # Load art style settings if they exist
        style_path = os.path.join(UPLOAD_FOLDER, 'style_settings.json')
        art_style = "cartoon"  # default style
        if os.path.exists(style_path):
            with open(style_path, 'r') as f:
                style_data = json.load(f)
                art_style = style_data.get('style', 'cartoon')
        
        # Capture stdout to get progress messages
        output_capture = OutputCapture()
        old_stdout = sys.stdout
        sys.stdout = output_capture
        
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Process assets with pixel art settings and art style
        process_assets(
            config, 
            OUTPUT_FOLDER,
            enable_pixelation=pixel_settings.get('enabled', True),
            grid_size=pixel_settings.get('gridSize', 6),
            color_count=pixel_settings.get('colorCount', 8),
            art_style=art_style
        )
        
        # Restore stdout
        sys.stdout = old_stdout
        
        # Signal completion
        progress_queue.put({
            "type": "complete",
            "assets": get_generated_assets()
        })
    except Exception as e:
        # Restore stdout in case of error
        sys.stdout = old_stdout
        progress_queue.put({
            "type": "error",
            "message": str(e)
        })

def get_generated_assets():
    """Get list of generated assets with their URLs"""
    assets = []
    for root, dirs, files in os.walk(OUTPUT_FOLDER):
        for file in files:
            if file.endswith('.png'):
                relative_path = os.path.relpath(root, OUTPUT_FOLDER)
                asset_path = os.path.join(relative_path, file)
                
                # Skip the main directory copies (we'll use the organized ones)
                if relative_path == '.':
                    continue
                    
                # Determine if this is an original or pixelated version
                is_original = 'originals' in root
                base_name = file.replace('_original.png', '.png')
                
                # Find or create asset entry
                asset_entry = next(
                    (a for a in assets if a["name"] == base_name), 
                    None
                )
                
                if asset_entry is None:
                    asset_entry = {
                        "name": base_name,
                        "original_url": None,
                        "pixelated_url": None
                    }
                    assets.append(asset_entry)
                
                # Add URL to appropriate field
                if is_original:
                    asset_entry["original_url"] = f"/assets/{asset_path}"
                else:
                    asset_entry["pixelated_url"] = f"/assets/{asset_path}"
                    
    return assets

@app.route('/generate')
def generate():
    def generate_events():
        # Start generation in a separate thread
        config_path = os.path.join(UPLOAD_FOLDER, 'config.json')
        thread = threading.Thread(target=generate_assets, args=(config_path,))
        thread.start()
        
        # Stream progress events
        while True:
            try:
                message = progress_queue.get(timeout=60)  # 1 minute timeout
                yield f"data: {json.dumps(message)}\n\n"
                
                if message['type'] in ['complete', 'error']:
                    break
            except queue.Empty:
                yield f"data: {json.dumps({'type': 'error', 'message': 'Generation timeout'})}\n\n"
                break
    
    return Response(generate_events(), mimetype='text/event-stream')

@app.route('/assets/<path:filename>')
def serve_asset(filename):
    return send_file(os.path.join(OUTPUT_FOLDER, filename))

if __name__ == '__main__':
    # Use port 8080 by default since 5000 might be used by AirPlay on macOS
    try:
        app.run(debug=True, port=8080, host='0.0.0.0')
    except OSError:
        print("Port 8080 is in use, trying port 3000...")
        app.run(debug=True, port=3000, host='0.0.0.0') 