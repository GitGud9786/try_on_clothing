import os
import random

def generate_paired_txt(image_dir, cloth_dir, output_file):
    """Generate paired combinations (same ID)"""
    images = sorted([f for f in os.listdir(image_dir) if f.endswith('.jpg') or f.endswith('.png')])
    cloths = sorted([f for f in os.listdir(cloth_dir) if f.endswith('.jpg') or f.endswith('.png')])
    
    # Create set for faster lookup
    cloth_set = set(cloths)
    
    pairs = []
    for img in images:
        # Try to find matching cloth with same ID
        if img in cloth_set:
            pairs.append(f"{img} {img}\n")
    
    with open(output_file, 'w') as f:
        f.writelines(pairs)
    
    print(f"Generated {len(pairs)} paired combinations in {output_file}")

def generate_unpaired_txt(image_dir, cloth_dir, output_file, num_pairs=None):
    """Generate unpaired (random) combinations"""
    images = sorted([f for f in os.listdir(image_dir) if f.endswith('.jpg') or f.endswith('.png')])
    cloths = sorted([f for f in os.listdir(cloth_dir) if f.endswith('.jpg') or f.endswith('.png')])
    
    if num_pairs is None:
        num_pairs = len(images)
    
    pairs = []
    for img in images[:num_pairs]:
        # Random cloth selection
        cloth = random.choice(cloths)
        pairs.append(f"{img} {cloth}\n")
    
    with open(output_file, 'w') as f:
        f.writelines(pairs)
    
    print(f"Generated {len(pairs)} unpaired combinations in {output_file}")

if __name__ == "__main__":
    # Set paths
    train_image_dir = "./DATA/my_dataset/train/image"
    train_cloth_dir = "./DATA/my_dataset/train/cloth"
    test_image_dir = "./DATA/my_dataset/test/image"
    test_cloth_dir = "./DATA/my_dataset/test/cloth"
    
    # Generate paired files
    print("Generating paired combinations...")
    generate_paired_txt(train_image_dir, train_cloth_dir, "train_pairs.txt")
    generate_paired_txt(test_image_dir, test_cloth_dir, "test_pairs.txt")
    
    # Generate unpaired files (optional - for testing unpaired try-on)
    print("\nGenerating unpaired combinations...")
    random.seed(42)  # For reproducibility
    generate_unpaired_txt(train_image_dir, train_cloth_dir, "train_pairs_unpaired.txt")
    generate_unpaired_txt(test_image_dir, test_cloth_dir, "test_pairs_unpaired.txt")
    
    print("\nDone! Generated files:")
    print("- train_pairs.txt (paired)")
    print("- test_pairs.txt (paired)")
    print("- train_pairs_unpaired.txt (unpaired)")
    print("- test_pairs_unpaired.txt (unpaired)")