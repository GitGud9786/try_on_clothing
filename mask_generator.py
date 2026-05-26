# save as make_masks.py and run: python make_masks.py
import os
import cv2
import numpy as np

ROOT = r"g:\Code\Thesis\stable_train\StableVITON\DATA\new_dataset\train"
IN_DIRS = ["cloth_outer"]
OUT_DIRS = ["cloth-outer-mask"]

# background considered "white" if all channels >= this
WHITE_THR = 245

def make_mask(img_bgr):
    # background: near-white
    bg = (img_bgr[:, :, 0] >= WHITE_THR) & (img_bgr[:, :, 1] >= WHITE_THR) & (img_bgr[:, :, 2] >= WHITE_THR)
    mask = (~bg).astype(np.uint8)  # cloth=1, bg=0
    return mask * 255

for in_name, out_name in zip(IN_DIRS, OUT_DIRS):
    in_dir = os.path.join(ROOT, in_name)
    out_dir = os.path.join(ROOT, out_name)
    os.makedirs(out_dir, exist_ok=True)

    for fn in os.listdir(in_dir):
        if not fn.lower().endswith((".png", ".jpg", ".jpeg")):
            continue
        path = os.path.join(in_dir, fn)
        img = cv2.imread(path)
        if img is None:
            continue

        mask = make_mask(img)
        out_path = os.path.join(out_dir, os.path.splitext(fn)[0] + "_mask.png")
        cv2.imwrite(out_path, mask)

print("Done.")