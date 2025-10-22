from pathlib import Path
import shutil

pose_dir = Path(r"D:\CSE\Thesis\try_on_clothing\test_set\my_dataset\test\openpose_img")
for p in pose_dir.glob("*.png"):
    stem = p.stem
    if stem.endswith("_rendered"):
        continue
    target = pose_dir / f"{stem}_rendered.png"
    if not target.exists():
        shutil.copy2(p, target)
        print("Created", target.name)
