#!/usr/bin/env python3
"""Strip the _mask suffix from cloth-mask filenames (00553_00_mask.png -> 00553_00.png)
so they match the loader's plain cloth_name lookup.
Dry-run by default; add --apply to actually rename.
Deliberately does NOT touch agnostic-mask (the loader expects its _mask suffix)."""
import os, argparse

TARGET_DIRS = [
    "cloth-inner-mask",
    "cloth-outer-mask",
    "gt_cloth_warped_inner_mask",
    "gt_cloth_warped_outer_mask",
]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", help="dataset root, e.g. my_dataset")
    ap.add_argument("--apply", action="store_true", help="actually rename (default: dry run)")
    args = ap.parse_args()

    renamed = skipped = 0
    for split in ("train", "test"):
        for d in TARGET_DIRS:
            folder = os.path.join(args.root, split, d)
            if not os.path.isdir(folder):
                continue
            for fn in os.listdir(folder):
                stem, ext = os.path.splitext(fn)
                if not stem.endswith("_mask"):
                    continue
                new_fn = stem[:-len("_mask")] + ext
                src, dst = os.path.join(folder, fn), os.path.join(folder, new_fn)
                if os.path.exists(dst):
                    print(f"SKIP (target exists): {src}")
                    skipped += 1
                    continue
                print(f"{src}  ->  {new_fn}")
                if args.apply:
                    os.rename(src, dst)
                renamed += 1
    mode = "RENAMED" if args.apply else "WOULD RENAME (dry run)"
    print(f"\n{mode}: {renamed} files, skipped {skipped}")
    if not args.apply:
        print("Re-run with --apply to actually do it.")

if __name__ == "__main__":
    main()