# asset_generator.py
# Image generation script using getimg.ai's FLUX.1 [schnell] model for isolated character assets

import requests
import json
import os
import base64
import sys
import time
from pathlib import Path
from dotenv import load_dotenv
import argparse
import re
from PIL import Image
from io import BytesIO
from pixelate import Pixelator
from openai import OpenAI

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# Standard character asset size - hardcoded to 1024x1024 for GPT-4 Vision
CHARACTER_SIZE = (1024, 1024)

def sanitize_filename(name):
    name = re.sub(r'[^\w\s-]', '', name)
    name = re.sub(r'[-\s]+', '_', name).strip('_')
    return name.lower()

def generate_game_asset(prompt, negative_prompt="", width=1024, height=1024):
    """
    Generate a game asset using OpenAI's GPT-4 Vision model with gpt-image-1.
    
    Args:
        prompt (str): The prompt describing the image to generate
        negative_prompt (str): Elements to avoid in the image (incorporated into main prompt)
        width (int): The width of the generated image
        height (int): The height of the generated image
        
    Returns:
        PIL.Image: The generated image
    """
    # Clean the prompt and combine with negative prompt if provided
    sanitized_prompt = prompt.strip()
    if negative_prompt:
        sanitized_prompt += f". Please avoid: {negative_prompt}"
    
    try:
        # Determine the closest supported size
        supported_sizes = ["1024x1024", "1536x1024", "1024x1536"]
        aspect_ratio = width / height
        if aspect_ratio > 1.2:
            size = "1536x1024"
        elif aspect_ratio < 0.8:
            size = "1024x1536"
        else:
            size = "1024x1024"
            
        print(f"Generating image with prompt: {sanitized_prompt}")
        print(f"Using size: {size}")
        
        # Make the API request with transparent background
        response = client.images.generate(
            model="gpt-image-1",
            prompt=sanitized_prompt,
            n=1,
            size=size,
            background="transparent",
            output_format="png",
            quality="high"
        )
        
        # Process the response - gpt-image-1 always returns base64-encoded images
        if response.data and len(response.data) > 0:
            image_data = base64.b64decode(response.data[0].b64_json)
            image = Image.open(BytesIO(image_data))
            
            # Print usage information if available
            if hasattr(response, 'usage'):
                print(f"Token usage: {response.usage}")
                
            return image
        else:
            print("No image data found in the response")
            return None
            
    except Exception as e:
        print(f"Error generating image: {e}")
        print(f"Exception type: {type(e)}")
        import traceback
        traceback.print_exc()
        return None

def process_assets(game_data, output_dir, enable_pixelation=True, grid_size=6, color_count=8, art_style="cartoon"):
    """
    Process and generate character and item assets from game data.
    
    Args:
        game_data (dict): The game data containing asset definitions
        output_dir (str): Directory to save generated assets
        enable_pixelation (bool): Whether to apply pixelation effect
        grid_size (int): Size of the pixel grid
        color_count (int): Number of colors to use in pixelation
        art_style (str): The art style to use for generation
    """
    start_time = time.time()
    total_cost = 0

    # Define style-specific prompt modifiers
    style_modifiers = {
        "cartoon": {
            "prefix": "A clean, stylized digital art style with flat colors, bold outlines, and clear silhouettes. ",
            "suffix": " Flat colors only, no gradients, no shadows, no textures, no reflections, no background elements. Iconic, simplified shapes with bold outlines or clean edges. Pure white background. Designed for pixel-perfect clarity and easy silhouette recognition. No text or visual effects."
        }
        # More styles can be added here in the future
    }

    # Get the style modifier (default to cartoon if style not found)
    style = style_modifiers.get(art_style, style_modifiers["cartoon"])

    os.makedirs(output_dir, exist_ok=True)
    
    # Create separate directories for each asset type
    characters_dir = os.path.join(output_dir, "characters")
    items_dir = os.path.join(output_dir, "items")
    
    os.makedirs(characters_dir, exist_ok=True)
    os.makedirs(items_dir, exist_ok=True)

    if "assets" in game_data and game_data["assets"]:
        # Process character assets
        character_assets = [asset for asset in game_data["assets"] if asset.get("type") == "character"]
        if character_assets:
            print(f"Generating {len(character_assets)} character assets in {art_style} style...")
            cost = process_asset_type(
                character_assets, 
                characters_dir, 
                "character",
                enable_pixelation=enable_pixelation,
                grid_size=grid_size,
                color_count=color_count,
                style_modifier=style
            )
            total_cost += cost
            
        # Process item assets
        item_assets = [asset for asset in game_data["assets"] if asset.get("type") == "item"]
        if item_assets:
            print(f"Generating {len(item_assets)} item assets in {art_style} style...")
            cost = process_asset_type(
                item_assets, 
                items_dir, 
                "item",
                enable_pixelation=enable_pixelation,
                grid_size=grid_size,
                color_count=color_count,
                style_modifier=style
            )
            total_cost += cost

    end_time = time.time()
    print(f"Asset generation completed in {end_time - start_time:.2f} seconds")
    return total_cost

def process_asset_type(assets, output_dir, asset_type, enable_pixelation=True, grid_size=6, color_count=8, style_modifier=None):
    """
    Process and generate images for a specific type of asset.
    
    Args:
        assets (list): List of asset objects
        output_dir (str): Directory to save the generated images
        asset_type (str): Type of asset being processed (for logging)
        enable_pixelation (bool): Whether to apply pixelation effect
        grid_size (int): Size of the pixel grid
        color_count (int): Number of colors to use in pixelation
        style_modifier (dict): Style-specific prompt modifiers
        
    Returns:
        float: Total cost of generation
    """
    total_cost = 0
    
    # Initialize pixelator for post-processing if needed
    pixelator = None
    if enable_pixelation:
        print(f"Initializing pixelator with grid_size={grid_size}, color_count={color_count}")
        pixelator = Pixelator(grid_size=grid_size, scale_factor=1, num_colors=color_count)
    
    # Create subdirectories for original and pixelated images
    originals_dir = os.path.join(output_dir, "originals")
    pixelated_dir = os.path.join(output_dir, "pixelated")
    os.makedirs(originals_dir, exist_ok=True)
    os.makedirs(pixelated_dir, exist_ok=True)
    
    # Define default negative prompts based on asset type
    default_negative_prompts = {
        "character": "multiple objects, complex background, gradients, heavy shading, 3D rendering, detailed texturing, internal details, dithering, noise texture, text, labels, ui elements, blurry, low quality, low resolution, cropped image, deformed, disfigured, duplicate, error, extra arms, extra fingers, extra legs, extra limbs, fused fingers, gross proportions, jpeg artifacts, long neck, low contrast, malformed, morbid, mutated, mutation, mutilated, poorly drawn, poorly rendered, duplicate, unfocused, watermark, signature, text, grainy texture, internal lines, noisy details",
        "item": "multiple objects, complex background, gradients, heavy shading, 3D rendering, detailed texturing, internal details, dithering, noise texture, text, labels, ui elements, blurry, low quality, low resolution, cropped image, deformed, disfigured, duplicate, error, extra limbs, fused elements, gross proportions, jpeg artifacts, low contrast, malformed, morbid, mutated, mutation, mutilated, poorly drawn, poorly rendered, duplicate, unfocused, watermark, signature, text, grainy texture, internal lines, noisy details"
    }
    
    for asset in assets:
        asset_name = asset.get("name", "").lower().replace(" ", "_")
        visual_prompt = asset.get("visual_prompt", "")
        
        # Apply style modifiers to the prompt if available
        if style_modifier:
            visual_prompt = style_modifier["prefix"] + visual_prompt + style_modifier["suffix"]
        
        # Check if pixelation is enabled for this specific asset
        asset_pixelation = asset.get("pixelate", enable_pixelation)

        if not asset_name or not visual_prompt:
            print(f"Skipping {asset_type} with missing name or prompt: {asset}")
            continue

        print(f"Generating {asset_type}: {asset_name}")
        print(f"Prompt: {visual_prompt}")
        
        # Get dimensions from asset if available
        if "dimensions" in asset and isinstance(asset["dimensions"], list) and len(asset["dimensions"]) == 2:
            # Extract dimensions from the asset
            orig_width, orig_height = asset["dimensions"]
            
            # Calculate a scaling factor to ensure reasonable image size
            # Max dimension should be 1024px for OpenAI, and we'll preserve aspect ratio
            max_dim = 1024
            scale = min(max_dim / orig_width, max_dim / orig_height)
            
            # Apply scaling to get final dimensions
            width = max(256, min(1536, int(orig_width * scale)))
            height = max(256, min(1536, int(orig_height * scale)))
            
            # Ensure dimensions are even numbers
            width = width + (width % 2)
            height = height + (height % 2)
            
            print(f"Using dimensions: {width}x{height} (scaled from original {orig_width}x{orig_height})")
        else:
            # Default to square 1024x1024 if dimensions not provided
            width, height = 1024, 1024
            print(f"No dimensions found in asset data, using default: {width}x{height}")

        # Use the dimensions in the API call
        image = generate_game_asset(
            prompt=visual_prompt,
            negative_prompt=default_negative_prompts[asset_type],
            width=width,
            height=height
        )

        if image:
            try:
                # Save original image
                original_path = os.path.join(originals_dir, f"{asset_name}_original.png")
                image.save(original_path, "PNG")
                print(f"Saved original: {original_path}")
                
                final_image = image
                
                # Apply pixelation if enabled for this asset
                if asset_pixelation and pixelator:
                    print(f"Pixelating {asset_name} with grid_size={grid_size}, color_count={color_count}...")
                    final_image = pixelator.pixelate(image)
                    
                    # Scale if needed
                    if pixelator.scale_factor > 1:
                        final_image = pixelator.scale_image(final_image)
                
                # Save pixelated image
                pixelated_path = os.path.join(pixelated_dir, f"{asset_name}.png")
                final_image.save(pixelated_path, "PNG")
                print(f"Saved pixelated: {pixelated_path}")
                
                # Also save a copy in the main output directory for backward compatibility
                output_path = os.path.join(output_dir, f"{asset_name}.png")
                final_image.save(output_path, "PNG")
                
            except Exception as e:
                print(f"Error processing {asset_name}: {str(e)}")
                # If processing fails, save the original image as fallback
                output_path = os.path.join(output_dir, f"{asset_name}.png")
                image.save(output_path, "PNG")
                print(f"Saved fallback: {output_path}")
        else:
            print(f"Failed to generate image for {asset_type}: {asset_name}")
            
    return total_cost

def main():
    parser = argparse.ArgumentParser(description='Generate isolated game assets from JSON data')
    parser.add_argument("input_file", help="Input JSON file containing game data")
    parser.add_argument("output_dir", help="Directory to save generated assets")
    parser.add_argument("--project_dir", help="Project directory name (optional)")
    parser.add_argument("--no-pixelate", action="store_true", help="Disable pixelation effect globally")
    args = parser.parse_args()

    try:
        with open(args.input_file, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error loading input file: {e}")
        return

    project_dir = args.project_dir if args.project_dir else "game"
    output_dir = os.path.join(project_dir, args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    process_assets(data, output_dir, enable_pixelation=not args.no_pixelate)

if __name__ == "__main__":
    main()
