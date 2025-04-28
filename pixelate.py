import os
from PIL import Image
import numpy as np
import logging
import math
import colorsys

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Pixelator:
    def __init__(self, grid_size=16, scale_factor=3, num_colors=8, saturation_factor=1.1, value_factor=1.05):
        """
        Initialize the Pixelator.
        
        Args:
            grid_size: Size of each pixel block
            scale_factor: Final scaling factor for the pixelated image
            num_colors: Maximum number of colors in the output
            saturation_factor: Factor to adjust color saturation (1.0 = no change)
            value_factor: Factor to adjust brightness (1.0 = no change)
        """
        self.grid_size = max(1, grid_size)
        self.scale_factor = max(1, scale_factor)
        self.num_colors = num_colors
        self.saturation_factor = saturation_factor
        self.value_factor = value_factor
        
    def load_image(self, image_path):
        """Load and convert image to RGBA mode."""
        img = Image.open(image_path)
        if img.mode != 'RGBA':
            img = img.convert('RGBA')
        return img
    
    def get_dominant_colors(self, img):
        """Extract dominant colors from the image."""
        img_array = np.array(img)
        pixels = img_array.reshape(-1, 4)
        opaque_pixels = pixels[pixels[:, 3] > 128][:, :3]
        
        if opaque_pixels.shape[0] > 0:
            # Find unique colors and take the most frequent ones
            unique_colors, counts = np.unique(opaque_pixels, axis=0, return_counts=True)
            # Sort by frequency
            sorted_indices = np.argsort(-counts)
            # Take top N colors
            top_colors = unique_colors[sorted_indices[:self.num_colors]]
            return top_colors.astype(int)
        else:
            return np.array([[0,0,0],[255,255,255]])  # Default fallback

    def quantize_colors(self, img, colors):
        """Quantize image to the specified colors."""
        img_array = np.array(img)
        height, width = img_array.shape[:2]
        output = np.zeros_like(img_array)
        
        # Convert colors to numpy array for vectorized calculation
        colors_np = np.array(colors)
        
        for y in range(height):
            for x in range(width):
                if img_array[y, x, 3] > 128:  # Only process sufficiently opaque pixels
                    pixel = img_array[y, x, :3]
                    # Calculate distances to all colors
                    distances = np.sum((colors_np - pixel) ** 2, axis=1)
                    # Find the index of the closest color
                    closest_color_index = np.argmin(distances)
                    # Assign the closest color
                    output[y, x, :3] = colors_np[closest_color_index]
                    output[y, x, 3] = 255
                else:
                    output[y, x] = [0, 0, 0, 0]
        
        return Image.fromarray(output)

    def ensure_grid_alignment(self, img):
        """Ensure the image aligns to the grid by padding and adjusting dimensions."""
        width, height = img.size
        
        # Calculate new dimensions that are multiple of grid_size
        new_width = math.ceil(width / self.grid_size) * self.grid_size
        new_height = math.ceil(height / self.grid_size) * self.grid_size
        
        # Only proceed if resizing is needed
        if new_width == width and new_height == height:
            return img
            
        # Create a new transparent image with the adjusted dimensions
        new_img = Image.new('RGBA', (new_width, new_height), (0, 0, 0, 0))
        
        # Center the original image
        offset_x = (new_width - width) // 2
        offset_y = (new_height - height) // 2
        
        if img.mode == 'RGBA':
            new_img.paste(img, (offset_x, offset_y), img)
        else:
            img_rgba = img.convert('RGBA')
            new_img.paste(img_rgba, (offset_x, offset_y), img_rgba)
        
        return new_img

    def harmonize_colors(self, img):
        """
        Harmonize colors to create a more cohesive palette.
        Adjusts saturation and brightness while preserving transparency.
        
        Args:
            img: PIL Image
            
        Returns:
            PIL Image with harmonized colors
        """
        img_array = np.array(img)
        height, width = img_array.shape[:2]
        
        # Create output array
        output = np.zeros_like(img_array)
        
        # Adjust colors using HSV color space
        for y in range(height):
            for x in range(width):
                if img_array[y, x, 3] > 10:  # Only process visible pixels
                    r, g, b = img_array[y, x, :3]
                    
                    # Convert to HSV
                    h, s, v = colorsys.rgb_to_hsv(r/255, g/255, b/255)
                    
                    # Adjust saturation and value
                    s = min(1.0, s * self.saturation_factor)
                    v = min(1.0, v * self.value_factor)
                    
                    # Convert back to RGB
                    r, g, b = colorsys.hsv_to_rgb(h, s, v)
                    
                    # Scale back to 0-255 range
                    output[y, x, :3] = [int(r*255), int(g*255), int(b*255)]
                    output[y, x, 3] = img_array[y, x, 3]  # Preserve original alpha
                else:
                    output[y, x] = [0, 0, 0, 0]  # Fully transparent
        
        return Image.fromarray(output)

    def pixelate(self, img):
        """
        Pixelate the image to create visible pixel grid.
        
        Args:
            img: PIL Image
            
        Returns:
            PIL Image with visible pixel grid
        """
        # First ensure the image aligns to our grid
        img = self.ensure_grid_alignment(img)
        
        # Get image dimensions
        width, height = img.size
        
        # Calculate dimensions for downscaling
        target_w = max(1, round(width / self.grid_size))
        target_h = max(1, round(height / self.grid_size))

        # Downscale using NEAREST to pick a single color for the block
        small_img = img.resize((target_w, target_h), Image.NEAREST)
        
        # Get dominant colors and quantize
        colors = self.get_dominant_colors(small_img)
        small_img = self.quantize_colors(small_img, colors)
        
        # Harmonize colors for a more cohesive look
        small_img = self.harmonize_colors(small_img)
        
        # Upscale back to original size using NEAREST to create blocky pixels
        pixelated = small_img.resize((width, height), Image.NEAREST)
        
        # Clean up alpha channel
        img_array = np.array(pixelated)
        if len(img_array.shape) == 3 and img_array.shape[2] == 4:
            img_array[:, :, 3] = np.where(img_array[:, :, 3] > 128, 255, 0)
            pixelated = Image.fromarray(img_array)
        
        return pixelated
    
    def scale_image(self, img):
        """
        Scale the image by the configured scale factor.
        
        Args:
            img: PIL Image
            
        Returns:
            Scaled PIL Image
        """
        if self.scale_factor <= 1:
            return img
            
        width, height = img.size
        new_w = int(width * self.scale_factor)
        new_h = int(height * self.scale_factor)
        
        # Scale using nearest neighbor to preserve sharp pixels
        return img.resize((new_w, new_h), Image.NEAREST)
    
    def process_image(self, input_path, output_dir):
        """Process a single image through the pixelation pipeline."""
        try:
            # Load image
            img = self.load_image(input_path)
            
            # Ensure grid alignment
            img = self.ensure_grid_alignment(img)
            
            # Get dominant colors and quantize
            colors = self.get_dominant_colors(img)
            img = self.quantize_colors(img, colors)
            
            # Pixelate
            pixelated = self.pixelate(img)
            
            # Scale if needed
            if self.scale_factor > 1:
                pixelated = self.scale_image(pixelated)
            
            # Save result
            base_name = os.path.splitext(os.path.basename(input_path))[0]
            output_path = os.path.join(output_dir, f"{base_name}_pixelated.png")
            
            # Ensure output directory exists
            os.makedirs(output_dir, exist_ok=True)
            
            # Set any partially transparent pixels to fully transparent or fully opaque
            img_array = np.array(pixelated)
            if len(img_array.shape) == 3 and img_array.shape[2] == 4:
                img_array[:, :, 3] = np.where(img_array[:, :, 3] > 128, 255, 0)
                pixelated = Image.fromarray(img_array)
            
            # Save with maximum quality
            pixelated.save(output_path, format='PNG', compress_level=0)
            logger.info(f"Saved pixelated image to {output_path}")
            
            return pixelated
            
        except Exception as e:
            logger.error(f"Error processing {input_path}: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
    
    def process_directory(self, input_dir, output_dir):
        """Process all images in a directory."""
        os.makedirs(output_dir, exist_ok=True)
        
        image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff')
        found_images = False
        
        for filename in os.listdir(input_dir):
            if filename.lower().endswith(image_extensions):
                input_path = os.path.join(input_dir, filename)
                logger.info(f"Processing {filename}...")
                self.process_image(input_path, output_dir)
                found_images = True

        if not found_images:
            logger.warning(f"No image files found in directory: {input_dir}")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Pixelate images with customizable grid size and scaling.')
    parser.add_argument('input_dir', help='Directory containing input images')
    parser.add_argument('output_dir', help='Directory to save processed images')
    parser.add_argument('--grid-size', type=int, default=16, help='Size of pixel grid (e.g., 16 for 16x16 blocks)')
    parser.add_argument('--scale-factor', type=int, default=3, help='Final scaling factor (1 for no scaling)')
    parser.add_argument('--num-colors', type=int, default=8, help='Maximum number of colors in the output')
    
    args = parser.parse_args()
    
    processor = Pixelator(
        grid_size=args.grid_size,
        scale_factor=args.scale_factor,
        num_colors=args.num_colors
    )
    
    processor.process_directory(args.input_dir, args.output_dir)

if __name__ == "__main__":
    main() 