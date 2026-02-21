import os
from tqdm import tqdm

def remove_mask_suffix(folder_path):
    """Remove _mask from filenames in agnostic-mask folder"""
    
    if not os.path.exists(folder_path):
        print(f"Folder not found: {folder_path}")
        return
    
    # Get all files with _mask.png
    mask_files = [f for f in os.listdir(folder_path) 
                  if f.endswith('_mask.png')]
    
    if not mask_files:
        print(f"No *_mask.png files found in {folder_path}")
        return
    
    print(f"Renaming {len(mask_files)} files...")
    
    renamed = 0
    for filename in tqdm(mask_files):
        old_path = os.path.join(folder_path, filename)
        # Remove '_mask' from filename
        new_filename = filename.replace('_mask.png', '.png')
        new_path = os.path.join(folder_path, new_filename)
        
        try:
            os.rename(old_path, new_path)
            renamed += 1
        except Exception as e:
            print(f"Error renaming {filename}: {e}")
    
    print(f"\n✅ Renamed {renamed} files")
    print(f"Example: 00612_00_mask.png → 00612_00.png")

if __name__ == "__main__":
    # Rename train folder
    print("Processing train/agnostic-mask...")
    remove_mask_suffix("./DATA/my_dataset/train/agnostic-mask")
    
    # Rename test folder
    print("\nProcessing test/agnostic-mask...")
    remove_mask_suffix("./DATA/my_dataset/test/agnostic-mask")
    
    print("\n✅ All done!")