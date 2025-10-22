# convert_everything_to_jpg_inplace.py
from pathlib import Path
import cv2
import re

# --------- CONFIG ---------
ROOT = Path(r"D:\CSE\Thesis\try_on_clothing\test_set\my_dataset\test")  # your "test" folder
PAIRS_FILE = ROOT / "test_pairs.txt"
BACKUP_PAIRS = ROOT / "test_pairs.bak"
JPEG_QUALITY = 100
IMG_EXTS = {".png", ".bmp", ".tif", ".tiff", ".jpeg", ".webp", ".jpg"}  # any image exts we’ll handle
# --------------------------

def is_image(p: Path) -> bool:
    return p.suffix.lower() in IMG_EXTS

def convert_one(p: Path) -> bool:
    """Convert a single image to .jpg (in place). Returns True if converted."""
    if not is_image(p):
        return False
    if p.suffix.lower() == ".jpg":
        return False

    img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
    if img is None:
        print(f"skip (unreadable): {p}")
        return False

    # Drop alpha if present
    if img.ndim == 3 and img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    out = p.with_suffix(".jpg")
    ok = cv2.imwrite(str(out), img, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
    if not ok:
        print(f"failed to write: {out}")
        return False

    try:
        p.unlink()
    except Exception as e:
        print(f"warning: could not remove {p}: {e}")
    print(f"converted: {p.relative_to(ROOT)}  ->  {out.relative_to(ROOT)}")
    return True

def convert_tree(root: Path):
    total = 0
    converted = 0
    for p in root.rglob("*"):
        if p.is_file() and is_image(p):
            total += 1
            if convert_one(p):
                converted += 1
    print(f"\nDone converting.\n  images scanned:   {total}\n  images converted: {converted}")

def update_pairs_file():
    if not PAIRS_FILE.exists():
        print("No test_pairs.txt found; skipping update.")
        return
    # backup
    PAIRS_FILE.replace(BACKUP_PAIRS)
    print(f"Backed up pairs to {BACKUP_PAIRS.name}")

    def swap_ext_to_jpg(path_str: str) -> str:
        # convert ending image extension to .jpg if that .jpg now exists
        p_abs = (ROOT.parent / path_str).resolve()  # ROOT is .../test
        p = Path(p_abs)
        if p.suffix.lower() != ".jpg":
            candidate = p.with_suffix(".jpg")
            if candidate.exists():
                # write relative to dataroot (i.e., parent of 'test')
                rel = candidate.relative_to(ROOT.parent)
                return str(rel).replace("\\", "/")
        return path_str

    lines_out = []
    with open(BACKUP_PAIRS, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 2:
                print(f"skip malformed line in pairs: {line!r}")
                continue
            a2 = swap_ext_to_jpg(parts[0])
            b2 = swap_ext_to_jpg(parts[1])
            lines_out.append(f"{a2} {b2}\n")

    with open(PAIRS_FILE, "w", encoding="utf-8") as f:
        f.writelines(lines_out)
    print(f"Updated {PAIRS_FILE.name} (backup at {BACKUP_PAIRS.name}).")

def main():
    print(f"Converting ALL images under: {ROOT}")
    convert_tree(ROOT)
    update_pairs_file()
    print("\nWARNING: JPG conversion of label/mask maps is lossy and can break HR-VITON.")
    print("If inference fails afterwards, restore from backup and only convert 'image/' and 'cloth/'.")

if __name__ == "__main__":
    main()
