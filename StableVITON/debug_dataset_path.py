import os
from os.path import join as opj

data_root_dir = "./DATA/my_dataset"
data_type = "test"

# What dataset.py ACTUALLY does now (line 156):
actual_path = opj(data_root_dir, data_type, f"{data_type}_pairs.txt")
print(f"What dataset.py constructs: {actual_path}")
print(f"Exists: {os.path.exists(actual_path)}")
print(f"Absolute path: {os.path.abspath(actual_path)}")

if os.path.exists(actual_path):
    print("\n✓ Path is CORRECT! The file should be found.")
    print("\nTry running inference now:")
    print("python inference.py --config_path ./configs/VITONHD.yaml --batch_size 1 --model_load_path ./ckpts/VITONHD.ckpt --data_root_dir ./DATA/my_dataset --save_dir ./results/test")
else:
    print("\n✗ File not found. Check:")
    print(f"  1. Does this exist? {os.path.exists(data_root_dir)}")
    print(f"  2. Does this exist? {os.path.exists(opj(data_root_dir, data_type))}")
    print(f"  3. Files in test folder:")
    test_dir = opj(data_root_dir, data_type)
    if os.path.exists(test_dir):
        for f in os.listdir(test_dir):
            print(f"     - {f}")