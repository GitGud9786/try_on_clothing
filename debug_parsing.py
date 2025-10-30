"""
Debug script to analyze parsing output and compare with VITON-HD
"""
import numpy as np
from PIL import Image
import os

# Define class names for reference
CLASS_NAMES = {
    0: 'background',
    1: 'hat',
    2: 'hair',
    3: 'glove',
    4: 'sunglasses',
    5: 'upper-clothes',
    6: 'dress',
    7: 'coat',
    8: 'socks',
    9: 'pants',
    10: 'jumpsuits',
    11: 'scarf',
    12: 'skirt',
    13: 'face',
    14: 'left-arm',
    15: 'right-arm',
    16: 'left-leg',
    17: 'right-leg',
    18: 'left-shoe',
    19: 'right-shoe'
}

def analyze_parsing(parse_path, name):
    """Analyze a parsing file and print class distribution"""
    parse_img = Image.open(parse_path)
    parse_array = np.array(parse_img)
    
    print(f"\n{'='*60}")
    print(f"Analyzing: {name}")
    print(f"{'='*60}")
    
    unique_classes = np.unique(parse_array)
    total_pixels = parse_array.size
    
    print(f"\nDetected classes:")
    for cls in unique_classes:
        count = np.sum(parse_array == cls)
        percentage = (count / total_pixels) * 100
        class_name = CLASS_NAMES.get(cls, f'unknown-{cls}')
        print(f"  Class {cls:2d} ({class_name:15s}): {count:7d} pixels ({percentage:5.2f}%)")
    
    # Check which clothing classes are present
    clothing_classes = [5, 6, 7, 10]  # upper-clothes, dress, coat, jumpsuits
    print(f"\nClothing classes present:")
    for cls in clothing_classes:
        if cls in unique_classes:
            count = np.sum(parse_array == cls)
            percentage = (count / total_pixels) * 100
            print(f"  ✓ Class {cls} ({CLASS_NAMES[cls]}): {percentage:.2f}%")
        else:
            print(f"  ✗ Class {cls} ({CLASS_NAMES[cls]}): Not detected")
    
    return parse_array, unique_classes

# Analyze custom dataset samples
print("\n" + "="*70)
print("CUSTOM DATASET ANALYSIS")
print("="*70)

custom_samples = ['0000', '0009', '0012', '0016', '0021', '0031', '0039', '0041']
custom_base = './DATA/custom/test'

for sample in custom_samples:
    parse_path = os.path.join(custom_base, 'image-parse-v3', f'{sample}.png')
    if os.path.exists(parse_path):
        analyze_parsing(parse_path, f"Custom {sample}")

# Analyze a few VITON-HD samples for comparison
print("\n\n" + "="*70)
print("VITON-HD DATASET ANALYSIS (for comparison)")
print("="*70)

viton_samples = ['00006_00', '00008_00', '00013_00']
viton_base = './DATA/zalando-hd-resized/test_fine'

for sample in viton_samples:
    parse_path = os.path.join(viton_base, 'image-parse-v3', f'{sample}.png')
    if os.path.exists(parse_path):
        analyze_parsing(parse_path, f"VITON-HD {sample}")

print("\n" + "="*70)
print("ANALYSIS COMPLETE")
print("="*70)
