"""
Compare VITON-HD agnostic-v3.2 with original parsing to see what gets removed
"""
import numpy as np
from PIL import Image

# Load a VITON-HD sample
sample = '00006_00'
parse_path = f'./DATA/zalando-hd-resized/test_fine/image-parse-v3/{sample}.png'
agnostic_path = f'./DATA/zalando-hd-resized/test_fine/agnostic-v3.2/{sample}.jpg'

parse_img = Image.open(parse_path)
agnostic_img = Image.open(agnostic_path).convert('L')  # Convert to grayscale

parse_array = np.array(parse_img)
agnostic_array = np.array(agnostic_img)

# Find where they differ (where clothing was removed)
difference = (parse_array != agnostic_array)

print(f"Analyzing VITON-HD sample: {sample}")
print(f"Total pixels: {parse_array.size}")
print(f"Pixels changed: {np.sum(difference)} ({np.sum(difference)/parse_array.size*100:.2f}%)")

# See what classes were in the original parsing where changes occurred
print(f"\nClasses that were removed (changed to gray):")
changed_classes = parse_array[difference]
unique_changed = np.unique(changed_classes)
for cls in unique_changed:
    count = np.sum(changed_classes == cls)
    print(f"  Class {cls:2d}: {count:6d} pixels")

# See what values they were changed to
print(f"\nValues in agnostic where changes occurred:")
agnostic_changed = agnostic_array[difference]
unique_agnostic = np.unique(agnostic_changed)
for val in unique_agnostic:
    count = np.sum(agnostic_changed == val)
    print(f"  Value {val:3d}: {count:6d} pixels")
