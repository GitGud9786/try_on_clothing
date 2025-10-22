# sanitize_and_make_parse_agnostic.py
# - Sanitizes parsing label maps to valid IDs [0..19] (CIHP/LIP-style).
# - Builds image-parse-agnostic-v3.2 by removing garment labels.

from pathlib import Path
import numpy as np
import cv2

# === EDIT THESE PATHS IF NEEDED ===
SRC = Path(r"D:\CSE\Thesis\try_on_clothing\test_set\my_dataset\test\image-parse-v3")
SAN = Path(r"D:\CSE\Thesis\try_on_clothing\test_set\my_dataset\test\image-parse-v3-sanitized")  # optional output
OUT = Path(r"D:\CSE\Thesis\try_on_clothing\test_set\my_dataset\test\image-parse-agnostic-v3.2")

# CIHP/LIP both use 20 classes (0..19). We will force labels to this range.
VALID = set(range(20))

# Garment-like classes to zero out for parse-agnostic (CIHP conventions).
# 0 bg, 1 hat, 2 hair, 3 glove, 4 sunglasses, 5 upper, 6 dress, 7 coat,
# 8 socks, 9 pants, 10 jumpsuit, 11 scarf, 12 skirt, 13 face,
# 14 l-arm, 15 r-arm, 16 l-leg, 17 r-leg, 18 l-shoe, 19 r-shoe
GARMENTS = {5, 6, 7, 10, 11, 12}

def is_visualization(img_bgr):
    """Heuristic: visualization is 3-channel; label map is single-channel."""
    return len(img_bgr.shape) == 3 and img_bgr.shape[2] == 3

def main():
    if not SRC.is_dir():
        raise SystemExit(f"[ERROR] Source folder not found: {SRC}")

    SAN.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)

    pngs = sorted([p for p in SRC.iterdir() if p.suffix.lower() == ".png"])
    print(f"[INFO] Source: {SRC}")
    print(f"[INFO] Found {len(pngs)} PNG(s).")

    bad_rgb = 0
    for i, p in enumerate(pngs, 1):
        img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)

        if img is None:
            print(f"[WARN] Cannot read: {p.name} (skipped)")
            continue

        # If it's RGB visualization, stop & tell the user to export raw labels
        if is_visualization(img):
            bad_rgb += 1
            # try to get a single channel (this is NOT guaranteed to be correct)
            # we’ll bail with a clear message; better to regenerate proper labels
            print(f"[ERROR] {p.name} looks like a color visualization (3 channels).")
            print("        You must export 'label' maps (single-channel indexed PNG), not colorized images.")
            continue

        # Ensure we have a 2D label array
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        labels = img.astype(np.int32)

        # Sanitize: any value not in 0..19 -> set to 0 (background)
        uniq = np.unique(labels)
        invalid = [u for u in uniq if u not in VALID]
        if invalid:
            labels[np.isin(labels, invalid)] = 0

        # Save sanitized parse (optional but useful)
        san_path = SAN / p.name
        cv2.imwrite(str(san_path), labels.astype(np.uint8))

        # Build parse-agnostic by zeroing garment classes
        mask_garment = np.isin(labels, list(GARMENTS))
        labels_agn = labels.copy()
        labels_agn[mask_garment] = 0

        out_path = OUT / p.name
        cv2.imwrite(str(out_path), labels_agn.astype(np.uint8))

        print(f"[OK] {i}/{len(pngs)}  uniq(before)={list(map(int, uniq))[:8]}...  -> {p.name}")

    if bad_rgb:
        print("\n[FAIL] Some files are color visualizations. Re-run your parsing tool to export RAW label maps.")
        print("       (In SCHP/Graphonomy, choose the output that stores IDs 0..19, not the colored vis.)")

    print(f"\n[DONE] Sanitized saved to: {SAN}")
    print(f"[DONE] Parse-agnostic saved to: {OUT}")

if __name__ == "__main__":
    main()
