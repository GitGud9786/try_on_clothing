import os
from pathlib import Path
import numpy as np
import cv2

# --- INPUT / OUTPUT FOLDERS ---
SRC = Path(r"D:\CSE\Thesis\try_on_clothing\test_set\my_dataset\test\image-parse-v3")
DST = Path(r"D:\CSE\Thesis\try_on_clothing\test_set\my_dataset\test\image-parse-agnostic-v3.2")

# --- CIHP LABELS (0..19) ---
# 0 bg, 1 hat, 2 hair, 3 glove, 4 sunglasses, 5 upper, 6 dress, 7 coat,
# 8 socks, 9 pants, 10 jumpsuit, 11 scarf, 12 skirt, 13 face,
# 14 l-arm, 15 r-arm, 16 l-leg, 17 r-leg, 18 l-shoe, 19 r-shoe
CLOTH_IDS = {5, 6, 7, 10, 11, 12}  # set these to background (0)

def main():
    if not SRC.is_dir():
        raise SystemExit(f"[ERROR] Source folder not found: {SRC}")

    DST.mkdir(parents=True, exist_ok=True)

    # collect .png (case-insensitive)
    pngs = sorted([p for p in SRC.iterdir() if p.suffix.lower() == ".png"])
    print(f"[INFO] Source: {SRC}")
    print(f"[INFO] Output: {DST}")
    print(f"[INFO] Found {len(pngs)} PNG(s).")

    if len(pngs) == 0:
        raise SystemExit("[ERROR] No PNGs found. Check the folder path.")

    done, skipped = 0, 0
    for i, p in enumerate(pngs, 1):
        seg = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)  # indexed labels
        if seg is None:
            print(f"[WARN] Could not read: {p.name} (skipped)")
            skipped += 1
            continue

        # zero out clothing classes
        mask = np.isin(seg, list(CLOTH_IDS))
        seg_agn = seg.copy()
        seg_agn[mask] = 0

        out_path = DST / p.name
        ok = cv2.imwrite(str(out_path), seg_agn)
        if not ok:
            print(f"[WARN] Failed to write: {out_path.name}")
            skipped += 1
        else:
            print(f"[OK] {i}/{len(pngs)} -> {out_path.name}")
            done += 1

    print(f"[DONE] Wrote {done} file(s). Skipped {skipped}.")
    if done < 60:
        print("[HINT] You expected ~60. If the count is off, ensure filenames end with .png")

if __name__ == "__main__":
    main()
