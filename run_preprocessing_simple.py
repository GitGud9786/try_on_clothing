"""
Quick preprocessing script - Step 1: Human Parsing Only
This generates: parsing, agnostic masks, agnostic images

DensePose must be handled separately (see instructions below)
"""

import os
import sys
from pathlib import Path

print("Starting preprocessing...")
print("=" * 60)

# Check if running in correct directory
if not os.path.exists("preprocess/humanparsing/run_parsing.py"):
    print("ERROR: Please run this script from the PromptDresser-main directory")
    sys.exit(1)

# Add paths
sys.path.insert(0, str(Path(__file__).parent / "preprocess" / "humanparsing"))

try:
    from run_parsing import Parsing
    import numpy as np
    from PIL import Image
    print("✓ Imports successful")
except ImportError as e:
    print(f"✗ Import error: {e}")
    print("Make sure you're in the promptdresser conda environment")
    sys.exit(1)

def create_agnostic_files(parsing_img, original_img, img_name, output_dirs):
    """Create agnostic mask and image from parsing result and original image"""
    parse_array = np.array(parsing_img)
    original_array = np.array(original_img)
    
    # LIP parsing colormap - matches VITON-HD visualization
    lip_colormap = np.array([
        [0, 0, 0],        # 0: background
        [128, 0, 0],      # 1: hat
        [255, 0, 0],      # 2: hair
        [0, 85, 0],       # 3: glove
        [170, 0, 51],     # 4: sunglasses
        [255, 85, 0],     # 5: upper-clothes
        [0, 0, 85],       # 6: dress
        [0, 119, 221],    # 7: coat
        [85, 85, 0],      # 8: socks
        [0, 85, 85],      # 9: pants
        [85, 51, 0],      # 10: jumpsuits
        [52, 86, 128],    # 11: scarf
        [0, 128, 0],      # 12: skirt
        [0, 0, 255],      # 13: face
        [51, 170, 221],   # 14: left-arm
        [0, 255, 255],    # 15: right-arm
        [85, 255, 170],   # 16: left-leg
        [170, 255, 85],   # 17: right-leg
        [255, 255, 0],    # 18: left-shoe
        [255, 170, 0]     # 19: right-shoe
    ])
    
    # Create mask removing upper clothing and related areas
    # Classes to remove: 5 (upper-clothes), 6 (dress), 7 (coat/jacket)
    parse_mask = np.zeros_like(parse_array)
    parse_mask[parse_array == 5] = 1   # upper-clothes
    parse_mask[parse_array == 6] = 1   # dress
    parse_mask[parse_array == 7] = 1   # coat/jacket
    
    # Create agnostic image: Start with original image, fill clothing areas with gray
    # This preserves face, hair, hands, pants, etc. - only masks the upper clothing
    agnostic = original_array.copy()
    agnostic[parse_mask == 1] = [127, 127, 127]  # Gray in RGB
    
    # Save binary mask
    mask_img = Image.fromarray((parse_mask * 255).astype(np.uint8))
    mask_path = os.path.join(output_dirs['agnostic-mask'], f"{img_name}_mask.png")
    mask_img.save(mask_path)
    
    # Save agnostic image (original with masked clothing)
    agnostic_img = Image.fromarray(agnostic.astype(np.uint8))
    agnostic_path = os.path.join(output_dirs['agnostic-v3.2'], f"{img_name}.png")
    agnostic_img.save(agnostic_path)
    
    # Create color-coded agnostic parsing: Convert grayscale parsing to RGB using colormap
    # Then mask clothing areas with gray
    h, w = parse_array.shape
    parse_colored = np.zeros((h, w, 3), dtype=np.uint8)
    
    # Apply colormap to each parsing class
    for class_id in range(20):
        parse_colored[parse_array == class_id] = lip_colormap[class_id]
    
    # Mask clothing areas with BLACK in the colored parsing (not gray)
    parse_colored[parse_mask == 1] = [0, 0, 0]  # Black where clothing removed
    
    # Save colored agnostic parsing
    parse_agnostic_img = Image.fromarray(parse_colored)
    parse_agnostic_path = os.path.join(output_dirs['image-parse-agnostic-v3.2'], f"{img_name}.png")
    parse_agnostic_img.save(parse_agnostic_path)

def main():
    # Configuration
    input_dir = "./DATA/custom/test/image"
    base_output_dir = "./DATA/custom/test"
    
    # Output directories
    output_dirs = {
        'image-parse-v3': os.path.join(base_output_dir, "image-parse-v3"),
        'agnostic-mask': os.path.join(base_output_dir, "agnostic-mask"),
        'agnostic-v3.2': os.path.join(base_output_dir, "agnostic-v3.2"),
        'image-parse-agnostic-v3.2': os.path.join(base_output_dir, "image-parse-agnostic-v3.2"),
        'openpose_json': os.path.join(base_output_dir, "openpose_json"),
    }
    
    # Create directories
    for dir_path in output_dirs.values():
        os.makedirs(dir_path, exist_ok=True)
    
    print(f"\nInput directory: {input_dir}")
    print(f"Output directory: {base_output_dir}\n")
    
    # Initialize parser
    print("Initializing human parsing model...")
    parser = Parsing(gpu_id=0)
    print("✓ Parser initialized\n")
    
    # Get image files
    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if len(image_files) == 0:
        print(f"ERROR: No images found in {input_dir}")
        return
    
    print(f"Found {len(image_files)} images to process\n")
    print("=" * 60)
    
    # Process each image
    for i, img_file in enumerate(image_files, 1):
        print(f"[{i}/{len(image_files)}] Processing: {img_file}")
        
        img_path = os.path.join(input_dir, img_file)
        img_name = os.path.splitext(img_file)[0]
        
        try:
            # Load original image
            original_img = Image.open(img_path).convert('RGB')
            
            # Run parsing
            parsed_img, face_mask = parser(img_path)
            
            # Save parsing result - the parsed_img already contains class IDs as pixel values
            # Just save it directly as PNG (it's already in the correct format)
            parse_path = os.path.join(output_dirs['image-parse-v3'], f"{img_name}.png")
            parsed_img.save(parse_path)
            
            # Create and save agnostic files (pass both parsing and original image)
            create_agnostic_files(parsed_img, original_img, img_name, output_dirs)
            
            # Create dummy OpenPose JSON
            json_path = os.path.join(output_dirs['openpose_json'], f"{img_name}_keypoints.json")
            with open(json_path, 'w') as f:
                f.write('{"people": []}')
            
            print(f"    ✓ Parsing saved")
            print(f"    ✓ Agnostic mask saved")
            print(f"    ✓ Agnostic image saved")
            
        except Exception as e:
            print(f"    ✗ ERROR: {e}")
            continue
    
    print("\n" + "=" * 60)
    print("PREPROCESSING COMPLETE")
    print("=" * 60)
    print("✓ Human parsing: DONE")
    print("✓ Agnostic masks: DONE")
    print("✓ Agnostic images: DONE")
    print("✓ OpenPose JSON: Dummy files created")
    print("\n⚠️  STILL NEEDED: DensePose files")
    print("=" * 60)
    print("\nNext steps for DensePose:")
    print("1. Copy from VITON-HD dataset (quickest):")
    print("   Copy files from DATA/zalando-hd-resized/test_*/image-densepose/")
    print("   to DATA/custom/test/image-densepose/")
    print("\n2. OR install DensePose (advanced):")
    print("   pip install 'git+https://github.com/facebookresearch/detectron2.git'")
    print("   Then use Detectron2 to generate DensePose maps")
    print("\nWithout DensePose files, inference will fail!")
    print("=" * 60)

if __name__ == "__main__":
    main()
