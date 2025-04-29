# Asset Generator

A tool that uses OpenAI's GPT-4 Vision (gpt-image-1) model to generate game assets from detailed visual descriptions, with optional pixel art conversion.

## Overview

This tool takes JSON descriptions of game assets and generates:
1. Character sprites
2. Item sprites
3. UI elements
4. Background elements

Each asset is generated with:
- Consistent style and perspective
- Clean, flat colors with bold outlines (cartoon style)
- Clear silhouettes
- Transparent backgrounds (using OpenAI's transparent background feature)
- Appropriate dimensions (1024x1024, 1024x1536, or 1536x1024)
- Optional pixel art conversion with advanced color harmonization

## Features

- Web-based interface for easy asset generation
- Drag-and-drop JSON file upload
- Real-time generation progress updates
- Multiple art style options (currently supporting cartoon style, more coming soon)
- Advanced pixel art conversion with:
  - Adjustable grid size (4-32 pixels)
  - Color palette optimization (2-32 colors)
  - Color harmonization with saturation and brightness adjustment
  - Edge cleaning and anti-aliasing removal
  - Smart color quantization using Lab color space
- Side-by-side comparison of original and pixelated assets
- Direct asset preview and download
- Organized asset categorization with separate directories for originals and pixelated versions

## Installation

1. Clone this repository
2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```
3. Install dependencies:
```bash
pip install -r requirements.txt
```
4. Copy `.env.example` to `.env` and add your OpenAI API key:
```bash
cp .env.example .env
# Edit .env and add your API key
```

## Usage

### Web Interface

1. Start the server:
```bash
python server.py
```
2. Open your browser and navigate to `http://127.0.0.1:8080`
3. Upload your JSON file using the web interface
4. Select your desired art style
5. Configure pixel art settings if desired:
   - Enable/disable pixel art conversion
   - Adjust grid size for pixel blocks
   - Set the maximum number of colors
6. Click "Generate Assets" to start the process
7. Monitor progress in real-time
8. Download both original and pixelated versions directly from the interface

Note: On macOS, if port 5000 is in use by AirPlay Receiver, the server will automatically use port 8080 instead.

### Command Line

```bash
python asset_generator.py input_file.json output_dir [--project_dir project_name] [--no-pixelate]
```

The script will:
1. Read the JSON input file containing asset descriptions
2. Generate images for each asset using OpenAI's image generation
3. Save them in appropriate subdirectories:
   - characters/
     - originals/
     - pixelated/
   - items/
     - originals/
     - pixelated/

### Input Format

The input JSON should follow this structure:
```json
{
    "game_description": "Description of the game",
    "game_theme": "Visual theme",
    "assets": [
        {
            "name": "Asset name",
            "visual_prompt": "Detailed visual description",
            "type": "character|item",
            "dimensions": [width, height],  // Optional: defaults to [1024, 1024]
            "pixelate": true  // Optional: override global pixelation setting
        }
    ]
}
```

## Environment Variables

Required environment variables:
- `OPENAI_API_KEY`: Your OpenAI API key

## Files

- `server.py`: Web server implementation
- `asset_generator.py`: Main generation script with OpenAI integration
- `pixelate.py`: Advanced pixel art conversion utilities
- `index.html`: Modern web interface with Tailwind CSS
- `.env`: Environment variables
- `requirements.txt`: Python dependencies

## Dependencies

- Flask (web server)
- openai
- python-dotenv
- Pillow (PIL)
- scikit-image (for color processing)
- scikit-learn (for color clustering)
- scipy (for image processing)
- numpy (for numerical operations)

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. 