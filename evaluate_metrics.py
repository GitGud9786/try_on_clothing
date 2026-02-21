import os
from os.path import join as opj
import argparse
import numpy as np
import torch
import cv2
from glob import glob
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim
import lpips
from scipy import linalg
from torchvision import models, transforms
from PIL import Image

class MetricsCalculator:
    def __init__(self, device='cpu'):
        self.device = device
        # Initialize LPIPS model
        self.lpips_model = lpips.LPIPS(net='alex').to(device)
        self.lpips_model.eval()
        
        # Initialize Inception model for FID
        self.inception_model = models.inception_v3(pretrained=True, transform_input=False).to(device)
        self.inception_model.eval()
        self.inception_model.fc = torch.nn.Identity()
        
        self.transform = transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def calculate_ssim(self, img1, img2):
        """Calculate SSIM between two images"""
        # Convert to grayscale for SSIM calculation
        if len(img1.shape) == 3:
            img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
            img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        else:
            img1_gray = img1
            img2_gray = img2
        
        score, _ = ssim(img1_gray, img2_gray, full=True)
        return score
    
    def calculate_lpips(self, img1, img2):
        """Calculate LPIPS between two images"""
        # Convert BGR to RGB and normalize to [-1, 1]
        img1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        img2_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
        
        img1_tensor = torch.from_numpy(img1_rgb).permute(2, 0, 1).unsqueeze(0).float() / 127.5 - 1.0
        img2_tensor = torch.from_numpy(img2_rgb).permute(2, 0, 1).unsqueeze(0).float() / 127.5 - 1.0
        
        img1_tensor = img1_tensor.to(self.device)
        img2_tensor = img2_tensor.to(self.device)
        
        with torch.no_grad():
            distance = self.lpips_model(img1_tensor, img2_tensor)
        
        return distance.item()
    
    def get_inception_features(self, images):
        """Extract features from Inception network"""
        features = []
        
        for img in tqdm(images, desc="Extracting Inception features"):
            if isinstance(img, str):
                pil_img = Image.open(img).convert('RGB')
            else:
                pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            
            img_tensor = self.transform(pil_img).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                feat = self.inception_model(img_tensor)
                features.append(feat.cpu().numpy())
        
        return np.concatenate(features, axis=0)
    
    def calculate_fid(self, features1, features2):
        """Calculate Frechet Inception Distance"""
        mu1, sigma1 = features1.mean(axis=0), np.cov(features1, rowvar=False)
        mu2, sigma2 = features2.mean(axis=0), np.cov(features2, rowvar=False)
        
        ssdiff = np.sum((mu1 - mu2) ** 2.0)
        covmean = linalg.sqrtm(sigma1.dot(sigma2))
        
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        
        fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
        return fid

# Replace the load_image_pairs function (around line 104-130)

def load_image_pairs(generated_dir, ground_truth_dir):
    """Load generated and ground truth image pairs based on filename pattern"""
    generated_images = sorted(glob(opj(generated_dir, "*.jpg")) + 
                             glob(opj(generated_dir, "*.png")))
    
    pairs = []
    not_found = []
    
    for gen_path in generated_images:
        gen_name = os.path.basename(gen_path)
        # Extract person_id from filename
        # Format: 00003_00_00003_00.jpg -> 00003_00
        # or: 0001_0001.jpg -> 0001
        
        # Try to extract person ID (handle both formats)
        parts = gen_name.split('_')
        if len(parts) >= 2:
            # For format like: 00003_00_00003_00.jpg
            person_id = f"{parts[0]}_{parts[1]}"
        else:
            # Fallback for simple format: 0001.jpg
            person_id = parts[0]
        
        # Look for ground truth image
        gt_path_png = opj(ground_truth_dir, f"{person_id}.png")
        gt_path_jpg = opj(ground_truth_dir, f"{person_id}.jpg")
        
        if os.path.exists(gt_path_png):
            pairs.append((gen_path, gt_path_png))
        elif os.path.exists(gt_path_jpg):
            pairs.append((gen_path, gt_path_jpg))
        else:
            not_found.append(gen_name)
    
    if not_found:
        print(f"Warning: Ground truth not found for {len(not_found)} images:")
        for fn in not_found[:5]:  # Show first 5
            print(f"  - {fn}")
        if len(not_found) > 5:
            print(f"  ... and {len(not_found) - 5} more")
        print()
    
    return pairs

def evaluate_metrics(args):
    """Main evaluation function"""
    print("="*60)
    print("QUANTITATIVE EVALUATION - StableVITON")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    print(f"Using device: {device}\n")
    
    calculator = MetricsCalculator(device=device)
    
    # Set up directories
    generated_dir = opj(args.samples_dir, "unpair" if args.unpair else "pair")
    ground_truth_dir = opj(args.data_root_dir, "test", "image")
    
    print(f"Generated images dir: {generated_dir}")
    print(f"Ground truth dir: {ground_truth_dir}\n")
    
    # Load image pairs
    print("Loading image pairs...")
    pairs = load_image_pairs(generated_dir, ground_truth_dir)
    print(f"Found {len(pairs)} valid image pairs\n")
    
    if len(pairs) == 0:
        print("Error: No valid image pairs found!")
        print("Please check that:")
        print(f"  1. Generated images exist in: {generated_dir}")
        print(f"  2. Ground truth images exist in: {ground_truth_dir}")
        print(f"  3. Filenames match the pattern: XXXX_YYYY.jpg -> XXXX.png")
        return
    
    # Calculate SSIM and LPIPS
    ssim_scores = []
    lpips_scores = []
    
    print("Calculating SSIM and LPIPS...")
    for gen_path, gt_path in tqdm(pairs, desc="Processing pairs"):
        gen_img = cv2.imread(gen_path)
        gt_img = cv2.imread(gt_path)
        
        if gen_img is None or gt_img is None:
            print(f"Warning: Failed to load {gen_path} or {gt_path}")
            continue
        
        # Resize if needed
        if gen_img.shape != gt_img.shape:
            gt_img = cv2.resize(gt_img, (gen_img.shape[1], gen_img.shape[0]))
        
        # SSIM
        ssim_score = calculator.calculate_ssim(gen_img, gt_img)
        ssim_scores.append(ssim_score)
        
        # LPIPS
        lpips_score = calculator.calculate_lpips(gen_img, gt_img)
        lpips_scores.append(lpips_score)
    
    # Calculate FID
    print("\nCalculating FID...")
    gen_images = [pair[0] for pair in pairs]
    gt_images = [pair[1] for pair in pairs]
    
    gen_features = calculator.get_inception_features(gen_images)
    gt_features = calculator.get_inception_features(gt_images)
    
    fid_score = calculator.calculate_fid(gen_features, gt_features)
    
    # Print results in the format shown in the table
    print("\n" + "="*60)
    print(f"RESULTS - {'UNPAIRED' if args.unpair else 'PAIRED'}")
    print("="*60)
    print(f"Resolution: {args.img_H} × {args.img_W}")
    print(f"Number of images: {len(pairs)}")
    print("-"*60)
    print(f"SSIM ↑:  {np.mean(ssim_scores):.3f}")
    print(f"LPIPS ↓: {np.mean(lpips_scores):.3f}")
    print(f"FID ↓:   {fid_score:.2f}")
    print("="*60)
    
    # Save results to file
    if args.output_file:
        output_path = opj(args.samples_dir, args.output_file)
        with open(output_path, 'w') as f:
            f.write(f"StableVITON Evaluation Results\n")
            f.write(f"{'='*60}\n")
            f.write(f"Mode: {'UNPAIRED' if args.unpair else 'PAIRED'}\n")
            f.write(f"Resolution: {args.img_H} × {args.img_W}\n")
            f.write(f"Number of images: {len(pairs)}\n")
            f.write(f"{'-'*60}\n")
            f.write(f"SSIM:  {np.mean(ssim_scores):.3f}\n")
            f.write(f"LPIPS: {np.mean(lpips_scores):.3f}\n")
            f.write(f"FID:   {fid_score:.2f}\n")
            f.write(f"{'='*60}\n")
        print(f"\nResults saved to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Evaluate StableVITON metrics")
    parser.add_argument("--samples_dir", type=str, default="./samples",
                       help="Directory containing generated images (with pair/unpair subdirs)")
    parser.add_argument("--data_root_dir", type=str, default="./DATA/my_dataset",
                       help="Root directory of dataset")
    parser.add_argument("--unpair", action="store_true",
                       help="Evaluate unpaired results")
    parser.add_argument("--img_H", type=int, default=512,
                       help="Image height")
    parser.add_argument("--img_W", type=int, default=384,
                       help="Image width")
    parser.add_argument("--cpu", action="store_true",
                       help="Force CPU usage")
    parser.add_argument("--output_file", type=str, default="metrics_results.txt",
                       help="File to save results")
    
    args = parser.parse_args()
    evaluate_metrics(args)

if __name__ == "__main__":
    main()