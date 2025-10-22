import json, cv2, sys
from pathlib import Path
import mediapipe as mp

SRC = Path(r"D:/CSE/Thesis/try_on_clothing/test_set/my_dataset/test/image")
DST = Path(r"D:/CSE/Thesis/try_on_clothing/test_set/my_dataset/test/pose")
DST.mkdir(parents=True, exist_ok=True)

# gather images (jpg/jpeg/png, case-insensitive)
imgs = []
for pat in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"):
    imgs += list(SRC.glob(pat))
imgs = sorted(set(imgs))

print(f"[INFO] Images folder: {SRC}")
print(f"[INFO] Found {len(imgs)} image(s).")
if not imgs:
    sys.exit("[ERROR] No images found. Check the path or extensions.")

mp_pose = mp.solutions.pose
with mp_pose.Pose(static_image_mode=True) as pose:
    ok, fail = 0, 0
    for i, img_path in enumerate(imgs, 1):
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"[WARN] Could not read: {img_path.name}")
            fail += 1
            continue

        res = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        keypoints = []
        if res.pose_landmarks:
            h, w = image.shape[:2]
            for lm in res.pose_landmarks.landmark:
                keypoints += [lm.x * w, lm.y * h, lm.visibility]
        else:
            print(f"[WARN] No person detected in {img_path.name}")

        data = {"version": 1.3, "people": [{"pose_keypoints_2d": keypoints}]}
        out_path = DST / f"{img_path.stem}_keypoints.json"
        with open(out_path, "w") as f:
            json.dump(data, f)

        print(f"[OK] {i}/{len(imgs)} → {out_path.name} "
              f"({'keypoints' if keypoints else 'empty'})")
        ok += 1

print(f"[DONE] Wrote {ok} JSON(s), {fail} failed.")
print(f"[HINT] Output dir: {DST}")
