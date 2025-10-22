#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
Simple Human Parsing for HR-VITON (Windows - No Compilation Required)
Uses a pre-trained model without custom CUDA extensions.

Adds:
  --visualize  -> also saves a colorized *_vis.png preview next to each mask
"""

import os
import sys
import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torchvision import transforms


# ---------- LIP 20-class color palette (for previews only) ----------
LIP_PALETTE = np.array([
    [0, 0, 0],        # 0 background
    [128, 0, 0],      # 1 hat
    [255, 0, 0],      # 2 hair
    [0, 85, 0],       # 3 glove
    [170, 0, 51],     # 4 sunglasses
    [255, 85, 0],     # 5 upper-clothes
    [0, 0, 85],       # 6 dress
    [0, 119, 221],    # 7 coat
    [85, 85, 0],      # 8 socks
    [0, 85, 85],      # 9 pants
    [85, 51, 0],      # 10 jumpsuits
    [52, 86, 128],    # 11 scarf
    [0, 128, 0],      # 12 skirt
    [0, 0, 255],      # 13 face
    [51, 170, 221],   # 14 left-arm
    [0, 255, 255],    # 15 right-arm
    [85, 255, 170],   # 16 left-leg
    [170, 255, 85],   # 17 right-leg
    [255, 255, 0],    # 18 left-shoe
    [255, 170, 0],    # 19 right-shoe
], dtype=np.uint8)


def colorize_parsing(parsing_np: np.ndarray) -> Image.Image:
    """Convert a [H,W] uint8 label map (0-19) into an RGB preview (for humans)."""
    h, w = parsing_np.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)
    for idx in range(LIP_PALETTE.shape[0]):
        out[parsing_np == idx] = LIP_PALETTE[idx]
    return Image.fromarray(out)


class SimpleResNet101(torch.nn.Module):
    """Simplified ResNet101 for parsing without InPlaceABN"""
    def __init__(self, num_classes=20):
        super(SimpleResNet101, self).__init__()
        from torchvision.models import resnet101
        resnet = resnet101(pretrained=False)

        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        self.decoder = torch.nn.Sequential(
            torch.nn.Conv2d(2048, 512, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(512),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(512, num_classes, kernel_size=1),
        )

    def forward(self, x):
        # Encoder
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # Decoder
        x = self.decoder(x)

        # Upsample to fixed 512x512 (then we’ll resize back to original)
        x = F.interpolate(x, size=(512, 512), mode='bilinear', align_corners=True)
        return x


def load_checkpoint_compatible(model, checkpoint_path, device):
    """Load checkpoint while handling incompatible keys."""
    print(f"Loading checkpoint from: {checkpoint_path}")
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        print("✓ Checkpoint file loaded successfully")
    except Exception as e:
        print(f"✗ Error loading checkpoint file: {e}")
        return model

    state_dict = checkpoint.get('state_dict', checkpoint)
    print(f"  Using state_dict with {len(state_dict)} keys")

    from collections import OrderedDict
    new_state = OrderedDict()
    for k, v in state_dict.items():
        # strip "module." if present
        name = k[7:] if k.startswith('module.') else k
        new_state[name] = v

    try:
        missing, unexpected = model.load_state_dict(new_state, strict=False)
        print("✓ Checkpoint loaded (non-strict)")
        if missing:
            print(f"  Missing keys: {len(missing)} (random init)")
        if unexpected:
            print(f"  Unexpected keys: {len(unexpected)} (ignored)")
    except Exception as e:
        print(f"✗ Warning: Could not load checkpoint strictly: {e}")
        print("  Using randomly initialized model (results may be poor).")
    return model


class HumanParsingGenerator:
    def __init__(self, checkpoint_path=None, device=None, visualize=False):
        print("\n" + "="*70)
        print("Initializing Human Parsing Generator")
        print("="*70)

        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.visualize = visualize

        print(f"\n1. Device Setup")
        print(f"   Using device: {self.device}")
        if torch.cuda.is_available():
            print(f"   GPU: {torch.cuda.get_device_name(0)}")
            print(f"   CUDA Version: {torch.version.cuda}")

        print(f"\n2. Model Initialization")
        self.model = SimpleResNet101(num_classes=20)
        print("   ✓ Model created")

        print(f"\n3. Loading Checkpoint")
        if checkpoint_path and os.path.exists(checkpoint_path):
            print(f"   Checkpoint path: {checkpoint_path}")
            self.model = load_checkpoint_compatible(self.model, checkpoint_path, self.device)
        elif checkpoint_path:
            print(f"   ✗ Checkpoint not found, using random init: {checkpoint_path}")
        else:
            print("   No checkpoint provided (random init)")

        print(f"\n4. Moving Model to Device")
        self.model.to(self.device).eval()
        print(f"   ✓ Model ready on {self.device}")

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

        print("\n" + "="*70)
        print("Initialization Complete!")
        print("="*70 + "\n")

    def parse_image(self, img_path):
        """Return (mask_img_uint8, mask_np_uint8)."""
        img = Image.open(img_path).convert('RGB')
        orig_w, orig_h = img.size

        img_resized = img.resize((512, 512), Image.BILINEAR)
        img_tensor = self.transform(img_resized).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(img_tensor)
            parsing_np = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

        # Back to original size
        parsing_img = Image.fromarray(parsing_np)
        parsing_img = parsing_img.resize((orig_w, orig_h), Image.NEAREST)

        # Also resize the np map for visualization
        parsing_np = np.array(parsing_img, dtype=np.uint8)
        return parsing_img, parsing_np

    def generate_batch(self, input_dir, output_dir):
        print("\n" + "="*70)
        print("Starting Batch Processing")
        print("="*70)

        print(f"\nInput directory:  {input_dir}")
        print(f"Output directory: {output_dir}")

        if not os.path.exists(input_dir):
            print(f"\n✗ ERROR: Input directory does not exist: {input_dir}")
            return

        os.makedirs(output_dir, exist_ok=True)
        print("✓ Output directory ready")

        valid_exts = ('.jpg', '.jpeg', '.png', '.bmp')
        image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(valid_exts)]

        if not image_files:
            print(f"\n✗ ERROR: No image files found in {input_dir}")
            print(f"   Looking for: {valid_exts}")
            return

        print(f"\nProcessing {len(image_files)} images…")
        success = 0
        errors = 0

        for img_name in tqdm(image_files, desc="Generating parsing maps"):
            try:
                img_path = os.path.join(input_dir, img_name)
                base = os.path.splitext(img_name)[0]

                parsing_img, parsing_np = self.parse_image(img_path)

                # Save label-index mask (HR-VITON uses this)
                out_mask = os.path.join(output_dir, f"{base}.png")
                # ensure 8-bit single-channel (indices)
                parsing_img = parsing_img.convert("L")
                parsing_img.save(out_mask)

                # Optional: save human-friendly color preview
                if self.visualize:
                    vis = colorize_parsing(parsing_np)
                    vis_out = os.path.join(output_dir, f"{base}_vis.png")
                    vis.save(vis_out)

                success += 1
            except Exception as e:
                errors += 1
                print(f"\n✗ Error processing {img_name}: {e}")

        print("\n" + "="*70)
        print("Batch Processing Complete!")
        print("="*70)
        print(f"\nResults:")
        print(f"  Total images:           {len(image_files)}")
        print(f"  Successfully processed: {success}")
        print(f"  Errors:                 {errors}")
        print(f"  Output directory:       {output_dir}")
        print("\n" + "="*70 + "\n")


def main():
    print("\n" + "="*70)
    print("Human Parsing Generator for HR-VITON (Windows)")
    print("="*70 + "\n")

    parser = argparse.ArgumentParser(description="Generate human parsing maps (Windows compatible)")
    parser.add_argument('--input-dir', type=str, required=True, help='Path to input images directory')
    parser.add_argument('--output-dir', type=str, required=True, help='Path to save parsing results')
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to SCHP checkpoint (.pth)')
    parser.add_argument('--gpu', type=int, default=None, help='GPU device ID (default: auto)')
    parser.add_argument('--visualize', action='store_true',
                        help='Also save a colorized *_vis.png preview next to each mask')
    args = parser.parse_args()

    print("Configuration:")
    print(f"  Input directory:  {args.input_dir}")
    print(f"  Output directory: {args.output_dir}")
    print(f"  Checkpoint:       {args.checkpoint if args.checkpoint else 'None (random init)'}")
    print(f"  GPU:              {args.gpu if args.gpu is not None else 'Auto-detect'}")
    print(f"  Visualize:        {args.visualize}")
    print()

    # Device
    if args.gpu is not None:
        device = torch.device(f'cuda:{args.gpu}')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    try:
        generator = HumanParsingGenerator(args.checkpoint, device=device, visualize=args.visualize)
        generator.generate_batch(args.input_dir, args.output_dir)
        print("✓ Program completed successfully!")
        return 0
    except KeyboardInterrupt:
        print("\n\n✗ Interrupted by user")
        return 1
    except Exception as e:
        print(f"\n\n✗ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
