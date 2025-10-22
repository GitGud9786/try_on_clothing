# pose_generator.py
import json
from pathlib import Path
import cv2
import numpy as np

# ---- CONFIG ----
BASE = Path(r"D:\CSE\Thesis\try_on_clothing\test_set\my_dataset\test")
JSON_DIR = BASE / "openpose_json"
IMG_DIR  = BASE / "openpose_img"
PERSON_DIR = BASE / "image"  # to copy image size
IMG_DIR.mkdir(parents=True, exist_ok=True)

# OpenPose COCO pairs (18-keypoint format)
PAIRS = [
    (1, 2), (1, 5), (2, 3), (3, 4), (5, 6), (6, 7),
    (1, 8), (8, 9), (9, 10), (1, 11), (11, 12), (12, 13),
    (0, 1), (0, 14), (14, 16), (0, 15), (15, 17)
]

def find_person_image_size(stem: str, default=(768, 1024)):
    """
    Return (W,H) from person image if exists, else default.
    Tries .jpg then .png (match HR-VITON conventions).
    """
    for ext in (".jpg", ".png", ".jpeg"):
        p = PERSON_DIR / f"{stem}{ext}"
        if p.exists():
            img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
            if img is not None:
                h, w = img.shape[:2]
                return (w, h)
    return default  # (W,H)

def draw_from_json(json_path: Path):
    with open(json_path, "r") as f:
        data = json.load(f)

    # Determine canvas size from person image
    stem = json_path.stem.replace("_keypoints", "")
    W, H = find_person_image_size(stem, default=(768, 1024))
    canvas = np.zeros((H, W, 3), dtype=np.uint8)

    people = data.get("people", [])
    if not people:
        return canvas  # blank if no detections

    # Use first person
    kps = np.array(people[0].get("pose_keypoints_2d", []), dtype=float)
    if kps.size == 0:
        return canvas

    kps = kps.reshape(-1, 3)  # (18, 3) as x,y,confidence

    # Draw limbs
    for i, j in PAIRS:
        if i < len(kps) and j < len(kps):
            xi, yi, ci = kps[i]
            xj, yj, cj = kps[j]
            if ci > 0.1 and cj > 0.1:
                cv2.line(canvas, (int(xi), int(yi)), (int(xj), int(yj)), (0, 255, 0), 2)

    # Draw keypoints
    for x, y, c in kps:
        if c > 0.1:
            cv2.circle(canvas, (int(x), int(y)), 3, (0, 0, 255), -1)

    return canvas

def main():
    json_files = sorted(JSON_DIR.glob("*.json"))
    if not json_files:
        print(f"No JSON files found in {JSON_DIR}")
        return

    for jf in json_files:
        stem = jf.stem.replace("_keypoints", "")
        out_path = IMG_DIR / f"{stem}.png"  # <- FIX: Path / string
        img = draw_from_json(jf)
        # If drawing failed for any reason, ensure a placeholder exists
        if img is None or img.size == 0:
            W, H = find_person_image_size(stem, default=(768, 1024))
            img = np.zeros((H, W, 3), dtype=np.uint8)
        cv2.imwrite(str(out_path), img)
        print("Saved", out_path)

if __name__ == "__main__":
    main()
