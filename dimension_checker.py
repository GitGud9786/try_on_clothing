# check_image_sizes.py
from pathlib import Path
from PIL import Image
import numpy as np

base = Path(r"D:\CSE\Thesis\try_on_clothing\test_set\my_dataset\test")
folders = [
    "image",
    "cloth",
    "cloth-mask",
    "image-parse-v3",
    "image-parse-agnostic-v3.2",
    "openpose_img",
    "image-densepose",
]
ids = ["0000", "0001", "0007"]  # or all files in your dataset

for fid in ids:
    print(f"\nID {fid}")
    for f in folders:
        path = base / f / f"{fid}.png"
        if not path.exists():
            path = base / f / f"{fid}.jpg"
        if not path.exists():
            continue
        im = Image.open(path)
        print(f"  {f:28s} -> {im.size} {im.mode}")
