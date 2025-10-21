import os
import cv2
import numpy as np
from PIL import Image

print("="*60)
print("GENERATING AGNOSTIC MAPS (StableVITON Format)")
print("="*60)

test_dir = "./DATA/my_dataset/test"
image_dir = os.path.join(test_dir, "image")
cloth_mask_dir = os.path.join(test_dir, "cloth-mask")
densepose_dir = os.path.join(test_dir, "image-densepose")
agnostic_dir = os.path.join(test_dir, "agnostic-v3.2")
agnostic_mask_dir = os.path.join(test_dir, "agnostic-mask")

# Create output directories
os.makedirs(agnostic_dir, exist_ok=True)
os.makedirs(agnostic_mask_dir, exist_ok=True)

# Get image files
image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png'))])

print(f"Found {len(image_files)} images\n")

# Parameters for agnostic generation
GRAY_VALUE = 127  # Middle gray for removed clothing regions
BLUR_KERNEL = 15   # Kernel for mask edge smoothing
DILATION_ITER = 5  # Dilate mask to cover more clothing area

for idx, img_file in enumerate(image_files, 1):
    print(f"[{idx}/{len(image_files)}] Processing {img_file}...")
    
    # Read original RGB image
    img_path = os.path.join(image_dir, img_file)
    img = cv2.imread(img_path)
    
    if img is None:
        print(f"  ✗ Could not read {img_file}")
        continue
    
    h, w = img.shape[:2]
    
    # Initialize agnostic image (start with original image)
    agnostic = img.copy()
    
    # Read cloth mask
    cloth_mask_path = os.path.join(cloth_mask_dir, img_file)
    cloth_mask = cv2.imread(cloth_mask_path, cv2.IMREAD_GRAYSCALE)
    
    if cloth_mask is None:
        print(f"  ⚠ No cloth mask found, creating default torso mask")
        # Create default elliptical mask for torso region
        cloth_mask = np.zeros((h, w), dtype=np.uint8)
        center = (w // 2, int(h * 0.45))
        axes = (int(w * 0.25), int(h * 0.35))
        cv2.ellipse(cloth_mask, center, axes, 0, 0, 360, 255, -1)
    else:
        # Resize to match image dimensions
        if cloth_mask.shape[:2] != (h, w):
            cloth_mask = cv2.resize(cloth_mask, (w, h), interpolation=cv2.INTER_NEAREST)
    
    # Threshold to binary mask
    _, binary_mask = cv2.threshold(cloth_mask, 127, 255, cv2.THRESH_BINARY)
    
    # Dilate mask to ensure full clothing coverage
    kernel = np.ones((5, 5), np.uint8)
    dilated_mask = cv2.dilate(binary_mask, kernel, iterations=DILATION_ITER)
    
    # Blur mask edges for smooth transitions
    blurred_mask = cv2.GaussianBlur(dilated_mask, (BLUR_KERNEL, BLUR_KERNEL), 0)
    
    # Normalize mask to [0, 1] for blending
    mask_normalized = blurred_mask.astype(np.float32) / 255.0
    mask_3ch = np.stack([mask_normalized] * 3, axis=-1)
    
    # Create gray region (127, 127, 127) for clothing area
    gray_region = np.ones_like(agnostic) * GRAY_VALUE
    
    # Blend: Replace clothing region with gray
    agnostic_float = agnostic.astype(np.float32)
    gray_float = gray_region.astype(np.float32)
    
    # agnostic = original * (1 - mask) + gray * mask
    agnostic_final = agnostic_float * (1 - mask_3ch) + gray_float * mask_3ch
    agnostic_final = np.clip(agnostic_final, 0, 255).astype(np.uint8)
    
    # Optional: If DensePose exists, use it for better body structure
    densepose_path = os.path.join(densepose_dir, img_file)
    if os.path.exists(densepose_path):
        densepose = cv2.imread(densepose_path)
        if densepose is not None and densepose.shape[:2] == (h, w):
            # Blend densepose into clothing region for body structure
            # This helps preserve pose information
            densepose_float = densepose.astype(np.float32)
            # Use 70% gray, 30% densepose in clothing region
            blended = gray_float * 0.7 + densepose_float * 0.3
            agnostic_final = agnostic_float * (1 - mask_3ch) + blended * mask_3ch
            agnostic_final = np.clip(agnostic_final, 0, 255).astype(np.uint8)
            print(f"  → Used DensePose for body structure")
    
    # Save agnostic image (3-channel RGB)
    agnostic_path = os.path.join(agnostic_dir, img_file)
    cv2.imwrite(agnostic_path, agnostic_final)
    
    # Save agnostic mask (grayscale, blurred for smooth transitions)
    # The mask shows WHERE clothing was removed
    mask_path = os.path.join(agnostic_mask_dir, img_file)
    cv2.imwrite(mask_path, blurred_mask)
    
    print(f"  ✓ Saved 3-channel agnostic image: {img_file}")
    print(f"  ✓ Saved agnostic mask: {img_file}")

print("\n" + "="*60)
print("✓ AGNOSTIC MAP GENERATION COMPLETE!")
print("="*60)
print("\nGenerated:")
print(f"  - agnostic-v3.2/: 3-channel RGB images (clothing → gray)")
print(f"  - agnostic-mask/: Grayscale masks (clothing region)")
print("\nThe agnostic images preserve:")
print("  ✓ Face and skin")
print("  ✓ Body pose/structure") 
print("  ✓ 3-channel RGB format (as required)")
print("  ✗ Clothing removed (replaced with gray)")