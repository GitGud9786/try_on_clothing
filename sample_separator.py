import os
import shutil
import random
from pathlib import Path

# Set random seed for reproducibility
random.seed(42)

# Define paths
train_dir = Path(r"g:\Code\Thesis\stable_train\StableVITON\DATA\my_dataset\train")
test_dir = Path(r"g:\Code\Thesis\stable_train\StableVITON\DATA\my_dataset\test")

# Get all subdirectories
subdirs = [d for d in train_dir.iterdir() if d.is_dir()]

# Get all unique file IDs from one folder (assuming all folders have matching files)
sample_folder = train_dir / "agnostic-mask"
all_files = list(sample_folder.glob("*"))
file_ids = sorted(set([f.stem.rsplit('_', 2)[0] for f in all_files]))  # Extract ID like "00000" from "00000_00_mask"

# Shuffle and split
random.shuffle(file_ids)
split_idx = int(len(file_ids) * 0.8)
train_ids = set(file_ids[:split_idx])
test_ids = set(file_ids[split_idx:])

print(f"Total samples: {len(file_ids)}")
print(f"Train samples: {len(train_ids)}")
print(f"Test samples: {len(test_ids)}")

# Create test directory structure
test_dir.mkdir(parents=True, exist_ok=True)

# Move files to test set
for subdir in subdirs:
    test_subdir = test_dir / subdir.name
    test_subdir.mkdir(exist_ok=True)
    
    # Get all files in this subdirectory
    files = list(subdir.glob("*"))
    
    moved_count = 0
    for file in files:
        # Extract file ID
        file_id = file.stem.rsplit('_', 2)[0]
        
        # Move to test if in test set
        if file_id in test_ids:
            dest = test_subdir / file.name
            shutil.move(str(file), str(dest))
            moved_count += 1
    
    print(f"Moved {moved_count} files from {subdir.name} to test set")

print("\nDataset split complete!")