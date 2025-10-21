import torch
from omegaconf import OmegaConf
from cldm.model import create_model

print("Checking VAE component...\n")

# Load config and model
config = OmegaConf.load("./configs/VITONHD.yaml")
config.model.params.img_H = 512
config.model.params.img_W = 384

print("Creating model...")
model = create_model(config_path=None, config=config)

print("\nLoading checkpoint...")
checkpoint_path = "./ckpts/VITONHD.ckpt"
load_cp = torch.load(checkpoint_path, map_location="cpu")
load_cp = load_cp["state_dict"] if "state_dict" in load_cp.keys() else load_cp

# Check if VAE keys exist in checkpoint
vae_keys = [k for k in load_cp.keys() if 'first_stage_model' in k]
print(f"\n{'='*60}")
print(f"VAE Keys in checkpoint: {len(vae_keys)}")
print(f"{'='*60}")

if len(vae_keys) > 0:
    print("✓ VAE component found in checkpoint")
    print("\nSample VAE keys:")
    for key in list(vae_keys)[:10]:
        print(f"  - {key}")
else:
    print("✗ NO VAE KEYS FOUND! This is the problem!")
    print("The checkpoint may be incomplete or corrupted.")

# Load the model
model.load_state_dict(load_cp)
model.eval()

# Check if first_stage_model exists
if hasattr(model, 'first_stage_model'):
    print("\n✓ model.first_stage_model exists")
    print(f"  Type: {type(model.first_stage_model)}")
    
    # Check encoder/decoder
    if hasattr(model.first_stage_model, 'encoder'):
        print("  ✓ Has encoder")
    else:
        print("  ✗ Missing encoder!")
        
    if hasattr(model.first_stage_model, 'decoder'):
        print("  ✓ Has decoder")
    else:
        print("  ✗ Missing decoder!")
else:
    print("\n✗ model.first_stage_model DOES NOT EXIST!")

# Test VAE encode/decode
print("\n" + "="*60)
print("Testing VAE encode-decode cycle")
print("="*60)

import cv2
import numpy as np

# Load a test image
test_img_path = "./DATA/my_dataset/test/image/0000.png"
img = cv2.imread(test_img_path)
if img is not None:
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (384, 512))
    
    # Normalize to [-1, 1]
    img_norm = (img_resized.astype(np.float32) / 127.5) - 1.0
    img_tensor = torch.from_numpy(img_norm).permute(2, 0, 1).unsqueeze(0)
    
    print(f"\nInput image:")
    print(f"  Shape: {img_tensor.shape}")
    print(f"  Range: [{img_tensor.min():.3f}, {img_tensor.max():.3f}]")
    print(f"  Channel means: R={img_tensor[0,0].mean():.3f}, G={img_tensor[0,1].mean():.3f}, B={img_tensor[0,2].mean():.3f}")
    
    # Encode
    with torch.no_grad():
        latent_dist = model.encode_first_stage(img_tensor)
        
        # Get the actual latent tensor (sample from distribution)
        if hasattr(latent_dist, 'sample'):
            latent = latent_dist.sample()
        elif hasattr(latent_dist, 'mode'):
            latent = latent_dist.mode()
        else:
            latent = latent_dist
        
        print(f"\nEncoded latent:")
        print(f"  Type: {type(latent_dist)}")
        print(f"  Shape: {latent.shape}")
        print(f"  Range: [{latent.min():.3f}, {latent.max():.3f}]")
        
        # Decode
        reconstructed = model.decode_first_stage(latent)
        print(f"\nDecoded (reconstructed):")
        print(f"  Shape: {reconstructed.shape}")
        print(f"  Range: [{reconstructed.min():.3f}, {reconstructed.max():.3f}]")
        print(f"  Channel means: R={reconstructed[0,0].mean():.3f}, G={reconstructed[0,1].mean():.3f}, B={reconstructed[0,2].mean():.3f}")
        
        # Check if color is preserved
        r_mean = reconstructed[0,0].mean().item()
        g_mean = reconstructed[0,1].mean().item()
        b_mean = reconstructed[0,2].mean().item()
        
        color_variance = np.std([r_mean, g_mean, b_mean])
        
        if color_variance < 0.01:
            print(f"\n✗ PROBLEM: VAE output has no color variance ({color_variance:.6f})")
            print("  The VAE is collapsing colors to grayscale!")
        else:
            print(f"\n✓ VAE preserves color (variance: {color_variance:.6f})")
        
        # Save reconstruction
        from utils import tensor2img
        reconstructed_img = tensor2img(reconstructed[0])
        cv2.imwrite("vae_test_input.png", img)
        cv2.imwrite("vae_test_output.png", reconstructed_img[:,:,::-1])
        print("\n✓ Saved vae_test_input.png and vae_test_output.png")
        print("  Compare these to see if VAE preserves color")

else:
    print(f"✗ Could not load test image: {test_img_path}")

print("\n" + "="*60)
print("VAE Check Complete")
print("="*60)