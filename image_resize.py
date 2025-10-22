import cv2, os
from pathlib import Path

# Target HR-VITON resolution
TARGET_SIZE = (768, 1024)   # (width, height)

# Your dataset root
base = Path(r"D:\CSE\Thesis\try_on_clothing\test_set\my_dataset\test")

# Folders that must all match
folders = [
    "image",
    "image-parse-v3",
    "image-parse-agnostic-v3.2",
    "cloth",
    "cloth-mask",
    "openpose_img",
    "image-densepose"
]

for folder in folders:
    fdir = base / folder
    if not fdir.exists():
        continue
    for p in fdir.glob("*.*"):
        img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        if img is None: 
            print("skip", p)
            continue

        # Choose interpolation based on type
        if img.ndim == 2 or folder.startswith("image-parse"):
            interp = cv2.INTER_NEAREST  # for label maps
        else:
            interp = cv2.INTER_LINEAR   # for RGB images

        resized = cv2.resize(img, TARGET_SIZE, interpolation=interp)
        cv2.imwrite(str(p), resized)
        print("Resized:", p.name)
