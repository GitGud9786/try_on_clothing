# save as jpeg_to_png.py and run: python jpeg_to_png.py
import os
import cv2

ROOT = r"g:\Code\Thesis\stable_train\StableVITON\DATA\new_dataset"
RECURSIVE = True

for dirpath, dirnames, filenames in os.walk(ROOT):
    for fn in filenames:
        if not fn.lower().endswith((".jpg", ".jpeg")):
            continue
        src = os.path.join(dirpath, fn)
        img = cv2.imread(src)
        if img is None:
            print(f"Skip (unreadable): {src}")
            continue
        dst = os.path.splitext(src)[0] + ".png"
        cv2.imwrite(dst, img)
    if not RECURSIVE:
        break

print("Done.")