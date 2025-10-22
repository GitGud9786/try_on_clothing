#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
Generate Agnostic Parsing Maps for HR-VITON
Removes clothing regions from human parsing segmentation maps
"""

import os
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm


# LIP (Look Into Person) dataset label definitions
LIP_LABELS = {
    0: 'Background',
    1: 'Hat',
    2: 'Hair',
    3: 'Glove',
    4: 'Sunglasses',
    5: 'Upper-clothes',
    6: 'Dress',
    7: 'Coat',
    8: 'Socks',
    9: 'Pants',
    10: 'Jumpsuits',
    11: 'Scarf',
    12: 'Skirt',
    13: 'Face',
    14: 'Left-arm',
    15: 'Right-arm',
    16: 'Left-leg',
    17: 'Right-leg',
    18: 'Left-shoe',
    19: 'Right-shoe'
}

# Define which labels to remove for agnostic parsing
# These are the clothing items that should be removed
CLOTHING_LABELS = {
    5,   # Upper-clothes (shirts, t-shirts, etc.)
    6,   # Dress
    7,   # Coat
    10,  # Jumpsuits
    # Optionally add these if needed:
    # 11,  # Scarf
    # 3,   # Glove
}

# For upper-body garments (most common in virtual try-on)
UPPER_BODY_LABELS = {
    5,   # Upper-clothes
    7,   # Coat
}

# For full-body garments
FULL_BODY_LABELS = {
    5,   # Upper-clothes
    6,   # Dress
    7,   # Coat
    9,   # Pants
    10,  # Jumpsuits
    12,  # Skirt
}


def create_agnostic_parsing(parsing_map, labels_to_remove):
    """
    Create agnostic parsing by removing specified labels
    
    Args:
        parsing_map: numpy array of parsing labels
        labels_to_remove: set of label indices to remove
    
    Returns:
        agnostic_map: numpy array with specified labels set to 0 (background)
    """
    agnostic_map = parsing_map.copy()
    
    # Set all clothing labels to 0 (background)
    for label in labels_to_remove:
        agnostic_map[parsing_map == label] = 0
    
    return agnostic_map


def visualize_parsing(parsing_map, num_classes=20):
    """
    Convert parsing map to colored visualization
    
    Args:
        parsing_map: numpy array of parsing labels
        num_classes: number of classes
    
    Returns:
        colored_map: PIL Image with colors applied
    """
    parsing_img = Image.fromarray(parsing_map.astype(np.uint8))
    parsing_img.putpalette(get_palette(num_classes))
    return parsing_img


def get_palette(num_cls):
    """Generate color palette for visualization"""
    palette = [0] * (num_cls * 3)
    for j in range(num_cls):
        lab = j
        palette[j*3+0] = 0
        palette[j*3+1] = 0
        palette[j*3+2] = 0
        i = 0
        while lab:
            palette[j*3+0] |= (((lab >> 0) & 1) << (7-i))
            palette[j*3+1] |= (((lab >> 1) & 1) << (7-i))
            palette[j*3+2] |= (((lab >> 2) & 1) << (7-i))
            i += 1
            lab >>= 3
    return palette


def process_batch(input_dir, output_dir, garment_type='upper', visualize=False):
    """
    Process all parsing maps in a directory
    
    Args:
        input_dir: directory containing parsing maps
        output_dir: directory to save agnostic parsing maps
        garment_type: 'upper', 'full', or 'custom'
        visualize: whether to save colored visualizations
    """
    print("\n" + "="*70)
    print("Creating Agnostic Parsing Maps")
    print("="*70)
    
    print(f"\nInput directory:  {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Garment type:     {garment_type}")
    
    # Check input directory
    if not os.path.exists(input_dir):
        print(f"\n✗ ERROR: Input directory does not exist!")
        return
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Select labels to remove based on garment type
    if garment_type == 'upper':
        labels_to_remove = UPPER_BODY_LABELS
        print(f"\nRemoving upper-body clothing:")
    elif garment_type == 'full':
        labels_to_remove = FULL_BODY_LABELS
        print(f"\nRemoving full-body clothing:")
    else:  # custom
        labels_to_remove = CLOTHING_LABELS
        print(f"\nRemoving clothing items:")
    
    for label in labels_to_remove:
        print(f"  - {LIP_LABELS[label]} (label {label})")
    
    # Get all parsing map files
    valid_extensions = ('.png', '.jpg', '.jpeg')
    parsing_files = [f for f in os.listdir(input_dir) 
                     if f.lower().endswith(valid_extensions) and not f.endswith('_vis.png')]
    
    print(f"\nFound {len(parsing_files)} parsing maps")
    
    if not parsing_files:
        print(f"✗ No parsing map files found!")
        return
    
    print("\nProcessing...")
    print("-" * 70)
    
    success_count = 0
    error_count = 0
    
    for filename in tqdm(parsing_files, desc="Creating agnostic maps"):
        try:
            # Load parsing map
            input_path = os.path.join(input_dir, filename)
            parsing_img = Image.open(input_path)
            parsing_map = np.array(parsing_img)
            
            # Create agnostic parsing
            agnostic_map = create_agnostic_parsing(parsing_map, labels_to_remove)
            
            # Save agnostic parsing
            base_name = os.path.splitext(filename)[0]
            output_path = os.path.join(output_dir, f"{base_name}.png")
            
            agnostic_img = Image.fromarray(agnostic_map.astype(np.uint8))
            agnostic_img.save(output_path)
            
            # Optionally save colored visualization
            if visualize:
                colored_img = visualize_parsing(agnostic_map)
                vis_path = os.path.join(output_dir, f"{base_name}_vis.png")
                colored_img.save(vis_path)
            
            success_count += 1
            
        except Exception as e:
            error_count += 1
            print(f"\n✗ Error processing {filename}: {e}")
            continue
    
    print("\n" + "="*70)
    print("Processing Complete!")
    print("="*70)
    print(f"\nResults:")
    print(f"  Total files:          {len(parsing_files)}")
    print(f"  Successfully processed: {success_count}")
    print(f"  Errors:               {error_count}")
    print(f"  Output directory:     {output_dir}")
    print("\n" + "="*70 + "\n")


def analyze_parsing_labels(input_dir):
    """
    Analyze parsing maps to see which labels are present
    Useful for debugging and understanding your data
    """
    print("\n" + "="*70)
    print("Analyzing Parsing Labels")
    print("="*70)
    
    print(f"\nInput directory: {input_dir}")
    
    # Get all parsing map files
    valid_extensions = ('.png', '.jpg', '.jpeg')
    parsing_files = [f for f in os.listdir(input_dir) 
                     if f.lower().endswith(valid_extensions) and not f.endswith('_vis.png')]
    
    if not parsing_files:
        print("✗ No parsing files found!")
        return
    
    print(f"Analyzing {len(parsing_files)} files...\n")
    
    # Collect label statistics
    label_counts = {}
    
    for filename in tqdm(parsing_files[:10], desc="Analyzing"):  # Analyze first 10 files
        input_path = os.path.join(input_dir, filename)
        parsing_img = Image.open(input_path)
        parsing_map = np.array(parsing_img)
        
        unique_labels = np.unique(parsing_map)
        for label in unique_labels:
            if label not in label_counts:
                label_counts[label] = 0
            label_counts[label] += 1
    
    print("\nLabels found in parsing maps:")
    print("-" * 70)
    for label in sorted(label_counts.keys()):
        label_name = LIP_LABELS.get(label, 'Unknown')
        count = label_counts[label]
        print(f"  Label {label:2d}: {label_name:20s} (in {count} files)")
    
    print("\n" + "="*70 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description='Generate agnostic parsing maps for HR-VITON',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create agnostic parsing for upper-body garments (default)
  python create_agnostic_parsing.py --input-dir "./test/image-parse" --output-dir "./test/image-parse-agnostic"
  
  # Create agnostic parsing for full-body garments
  python create_agnostic_parsing.py --input-dir "./test/image-parse" --output-dir "./test/image-parse-agnostic" --garment-type full
  
  # Analyze what labels are in your parsing maps
  python create_agnostic_parsing.py --input-dir "./test/image-parse" --analyze
        """
    )
    
    parser.add_argument('--input-dir', type=str, required=True,
                       help='Directory containing parsing maps')
    parser.add_argument('--output-dir', type=str,
                       help='Directory to save agnostic parsing maps')
    parser.add_argument('--garment-type', type=str, default='upper',
                       choices=['upper', 'full', 'custom'],
                       help='Type of garment to remove (upper/full/custom)')
    parser.add_argument('--visualize', action='store_true',
                       help='Save colored visualizations')
    parser.add_argument('--analyze', action='store_true',
                       help='Only analyze labels without creating agnostic maps')
    
    args = parser.parse_args()
    
    if args.analyze:
        # Just analyze the labels
        analyze_parsing_labels(args.input_dir)
    else:
        # Create agnostic parsing maps
        if not args.output_dir:
            print("✗ ERROR: --output-dir is required when not using --analyze")
            return 1
        
        process_batch(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            garment_type=args.garment_type,
            visualize=args.visualize
        )
    
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())