import os
from PIL import Image
import numpy as np
import logging
import math
import colorsys
from skimage import color
from sklearn.cluster import KMeans
from scipy import ndimage

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
        """Extract dominant colors from the image using k-means clustering."""
        img_array = np.array(img)
        pixels = img_array.reshape(-1, 4)
        
        # Only process pixels that are sufficiently opaque
        opaque_mask = pixels[:, 3] > 128
        opaque_pixels = pixels[opaque_mask][:, :3]
        
        if opaque_pixels.shape[0] == 0:
            return np.array([[0,0,0],[255,255,255]])
        
        # Convert to float32 for better precision
        pixels_float = opaque_pixels.astype('float32') / 255
        
        # Convert to Lab color space for better color clustering
        pixels_lab = color.rgb2lab(pixels_float.reshape(1, -1, 3)).reshape(-1, 3)
        
        # Use k-means clustering to find dominant colors
        kmeans = KMeans(n_clusters=self.num_colors, random_state=42, n_init=10)
        kmeans.fit(pixels_lab)
        
        # Convert cluster centers back to RGB
        centers_lab = kmeans.cluster_centers_
        centers_rgb = color.lab2rgb(centers_lab.reshape(1, -1, 3)).reshape(-1, 3)
        
        # Convert back to 0-255 range
        colors = (centers_rgb * 255).astype(int)
        
        # Sort colors by frequency
        labels = kmeans.labels_
        unique_labels, counts = np.unique(labels, return_counts=True)
        sorted_indices = np.argsort(-counts)
        colors = colors[sorted_indices]
        
        return colors

    def quantize_colors(self, img, colors):
        """Quantize image to the specified colors using Lab color space for better matching."""
        img_array = np.array(img)
        height, width = img_array.shape[:2]
        output = np.zeros_like(img_array)
        
        # Convert colors to Lab color space
        colors_rgb = colors.astype('float32') / 255
        colors_lab = color.rgb2lab(colors_rgb.reshape(1, -1, 3)).reshape(-1, 3)
        
        # Process the image in chunks to avoid memory issues
        chunk_size = 1000
        for y in range(0, height, chunk_size):
            y_end = min(y + chunk_size, height)
            for x in range(0, width, chunk_size):
                x_end = min(x + chunk_size, width)
                
                chunk = img_array[y:y_end, x:x_end]
                chunk_pixels = chunk.reshape(-1, 4)
                
                # Only process sufficiently opaque pixels
                opaque_mask = chunk_pixels[:, 3] > 200  # More aggressive threshold
                
                if np.any(opaque_mask):
                    # Convert opaque pixels to Lab color space
                    opaque_rgb = chunk_pixels[opaque_mask][:, :3].astype('float32') / 255
                    opaque_lab = color.rgb2lab(opaque_rgb.reshape(1, -1, 3)).reshape(-1, 3)
                    
                    # Calculate distances in Lab space
                    distances = np.sqrt(((opaque_lab[:, np.newaxis] - colors_lab) ** 2).sum(axis=2))
                    closest_color_indices = np.argmin(distances, axis=1)
                    
                    # Assign colors
                    chunk_pixels[opaque_mask, :3] = colors[closest_color_indices]
                    chunk_pixels[opaque_mask, 3] = 255
                    
                    # Assign fully transparent pixels
                    chunk_pixels[~opaque_mask] = [0, 0, 0, 0]
                    
                    output[y:y_end, x:x_end] = chunk_pixels.reshape(chunk.shape)
                else:
                    output[y:y_end, x:x_end] = 0
        
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
        """
        img_array = np.array(img)
        height, width = img_array.shape[:2]
        
        # Create output array
        output = np.zeros_like(img_array)
        
        # Convert to HSV for color adjustment
        rgb_pixels = img_array[:, :, :3].astype('float32') / 255
        hsv_pixels = color.rgb2hsv(rgb_pixels)
        
        # Adjust saturation and value
        hsv_pixels[:, :, 1] = np.clip(hsv_pixels[:, :, 1] * self.saturation_factor, 0, 1)
        hsv_pixels[:, :, 2] = np.clip(hsv_pixels[:, :, 2] * self.value_factor, 0, 1)
        
        # Convert back to RGB
        rgb_adjusted = color.hsv2rgb(hsv_pixels)
        
        # Scale back to 0-255 range and preserve alpha
        output[:, :, :3] = (rgb_adjusted * 255).astype(np.uint8)
        output[:, :, 3] = img_array[:, :, 3]
        
        return Image.fromarray(output)

    def clean_edges(self, img):
        """Clean up edges by removing anti-aliasing and partial transparency."""
        img_array = np.array(img)
        
        # Higher threshold for alpha to remove semi-transparent pixels
        alpha_threshold = 200  # More aggressive threshold
        
        # Create binary mask for alpha
        alpha_mask = img_array[:, :, 3] > alpha_threshold
        
        # Set partially transparent pixels to fully transparent
        img_array[:, :, 3] = np.where(alpha_mask, 255, 0)
        
        # Clean up isolated pixels
        structure = np.ones((3, 3), dtype=bool)  # 3x3 structural element
        alpha_cleaned = ndimage.binary_opening(alpha_mask, structure=structure)
        alpha_cleaned = ndimage.binary_closing(alpha_cleaned, structure=structure)
        
        # Apply cleaned alpha mask
        img_array[:, :, 3] = np.where(alpha_cleaned, 255, 0)
        
        return Image.fromarray(img_array)

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
        
        # Clean edges before processing
        img = self.clean_edges(img)
        
        # Get image dimensions
        width, height = img.size
        
        # Calculate dimensions for downscaling
        target_w = max(1, round(width / self.grid_size))
        target_h = max(1, round(height / self.grid_size))

        # Downscale using NEAREST to pick a single color for the block
        small_img = img.resize((target_w, target_h), Image.Resampling.NEAREST)
        
        # Get dominant colors and quantize
        colors = self.get_dominant_colors(small_img)
        small_img = self.quantize_colors(small_img, colors)
        
        # Harmonize colors for a more cohesive look
        small_img = self.harmonize_colors(small_img)
        
        # Upscale back to original size using NEAREST to create blocky pixels
        pixelated = small_img.resize((width, height), Image.Resampling.NEAREST)
        
        # Clean edges again after pixelation
        pixelated = self.clean_edges(pixelated)
        
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