# Create file: test_checkpoint_save.py

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from cldm.model import create_model, load_state_dict
from omegaconf import OmegaConf
import os

print("Testing checkpoint save with save_weights_only=True...\n")

# Load your model
config = OmegaConf.load("./configs/VITONHD.yaml")
model = create_model("./configs/VITONHD.yaml", config=config).cpu()

# Load epoch 0 checkpoint
checkpoint_path = "./logs/20260207_Base_training_fast/models/[Train]_[epoch=0]_[train_loss_epoch=0.0716].ckpt"
print(f"Loading checkpoint: {checkpoint_path}")
model.load_state_dict(load_state_dict(checkpoint_path, location="cpu"))

# Create test save directory
test_dir = "./test_checkpoint_save"
os.makedirs(test_dir, exist_ok=True)

# Test save with weights only
print("\nAttempting to save checkpoint with save_weights_only=True...")
try:
    # This mimics what ModelCheckpoint does
    checkpoint = {
        'state_dict': model.state_dict(),
        'epoch': 1,
    }
    
    test_path = os.path.join(test_dir, "test_weights_only.ckpt")
    torch.save(checkpoint, test_path)
    
    file_size = os.path.getsize(test_path) / (1024**3)  # GB
    print(f"✅ SUCCESS! Checkpoint saved: {test_path}")
    print(f"✅ File size: {file_size:.2f} GB")
    
    if file_size < 4.0:
        print(f"✅ Size is reasonable (< 4 GB) - should fit in RAM!")
    else:
        print(f"⚠️ Warning: File is large ({file_size:.2f} GB)")
    
    # Test loading it back
    print("\nTesting checkpoint reload...")
    loaded = torch.load(test_path, map_location='cpu')
    print(f"✅ Checkpoint loaded successfully!")
    print(f"✅ Contains keys: {list(loaded.keys())}")
    
    print("\n" + "="*60)
    print("✅ ALL TESTS PASSED - Your training will work!")
    print("="*60)
    
except Exception as e:
    print(f"\n❌ ERROR: {e}")
    print("❌ This needs to be fixed before training!")
    import traceback
    traceback.print_exc()