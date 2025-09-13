#!/usr/bin/env python3
"""
Icon creation script for Argus installer
"""

import os
import sys
from pathlib import Path

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:
    print("PIL (Pillow) not found. Installing...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "Pillow"])
    from PIL import Image, ImageDraw, ImageFont

def create_argus_icon():
    """Create an Argus icon programmatically"""
    # Create a 256x256 image
    size = 256
    img = Image.new('RGBA', (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    
    # Background circle (light gray)
    bg_radius = 120
    bg_center = (size // 2, size // 2)
    draw.ellipse([
        bg_center[0] - bg_radius, bg_center[1] - bg_radius,
        bg_center[0] + bg_radius, bg_center[1] + bg_radius
    ], fill=(224, 224, 224, 255))
    
    # Main green circle
    main_radius = 60
    main_center = (size // 2, size // 2 - 28)  # Slightly up
    draw.ellipse([
        main_center[0] - main_radius, main_center[1] - main_radius,
        main_center[0] + main_radius, main_center[1] + main_radius
    ], fill=(34, 197, 94, 255), outline=(0, 0, 0, 255), width=6)
    
    # Small white circle
    small_radius = 20
    small_center = (size // 2 - 20, size // 2 + 12)  # Bottom left of main circle
    draw.ellipse([
        small_center[0] - small_radius, small_center[1] - small_radius,
        small_center[0] + small_radius, small_center[1] + small_radius
    ], fill=(255, 255, 255, 255), outline=(0, 0, 0, 255), width=4)
    
    # Try to add text
    try:
        font = ImageFont.truetype("arial.ttf", 18)
    except:
        try:
            font = ImageFont.truetype("Arial.ttf", 18)
        except:
            font = ImageFont.load_default()
    
    text = "ARGUS"
    text_bbox = draw.textbbox((0, 0), text, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_x = (size - text_width) // 2
    text_y = size - 56
    
    draw.text((text_x, text_y), text, fill=(34, 197, 94, 255), font=font)
    
    return img

def convert_to_ico(image, output_path):
    """Convert PIL Image to ICO format with multiple sizes"""
    # Create multiple sizes for the ICO file
    sizes = [16, 24, 32, 48, 64, 128, 256]
    images = []
    
    for size in sizes:
        resized = image.resize((size, size), Image.Resampling.LANCZOS)
        images.append(resized)
    
    # Save as ICO
    images[0].save(output_path, format='ICO', sizes=[(img.width, img.height) for img in images])
    print(f"Icon saved to: {output_path}")

def main():
    # Get the directory of this script
    script_dir = Path(__file__).parent
    assets_dir = script_dir / "assets"
    
    # Create assets directory if it doesn't exist
    assets_dir.mkdir(exist_ok=True)
    
    # Create the icon
    print("Creating Argus icon...")
    icon_image = create_argus_icon()
    
    # Save as ICO
    ico_path = assets_dir / "argus.ico"
    convert_to_ico(icon_image, ico_path)
    
    # Also save as PNG for reference
    png_path = assets_dir / "argus.png"
    icon_image.save(png_path, "PNG")
    print(f"PNG version saved to: {png_path}")
    
    print("Icon creation completed!")

if __name__ == "__main__":
    main()