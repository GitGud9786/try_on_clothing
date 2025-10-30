"""
Calculate evaluation metrics for virtual try-on results:
- SSIM (Structural Similarity Index)
- FID (Fréchet Inception Distance)
- LPIPS (Learned Perceptual Image Patch Similarity)
"""

import os
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from skimage.metrics import structural_similarity as ssim
from scipy import linalg
import lpips

def load_image(path):
    """Load and convert image to RGB numpy array"""
    img = Image.open(path).convert('RGB')
    return np.array(img)

def calculate_ssim(img1, img2):
    """Calculate SSIM between two images"""
    # SSIM requires grayscale or same channel images
    # We'll calculate per channel and average
    ssim_values = []
    for i in range(3):  # RGB channels
        ssim_val = ssim(img1[:, :, i], img2[:, :, i], data_range=255)
        ssim_values.append(ssim_val)
    return np.mean(ssim_values)

def calculate_fid(real_images, generated_images):
    """
    Calculate FID score between real and generated images
    Using a simplified version with InceptionV3 features
    """
    from torchvision.models import inception_v3
    
    # Load InceptionV3
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    inception = inception_v3(pretrained=True, transform_input=False).to(device)
    inception.eval()
    
    # Prepare transform
    transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    def get_features(images):
        features = []
        with torch.no_grad():
            for img_path in images:
                img = Image.open(img_path).convert('RGB')
                img_t = transform(img).unsqueeze(0).to(device)
                feat = inception(img_t)
                features.append(feat.cpu().numpy().flatten())
        return np.array(features)
    
    print("Extracting features for FID calculation...")
    real_features = get_features(real_images)
    gen_features = get_features(generated_images)
    
    # Calculate mean and covariance
    mu_real = np.mean(real_features, axis=0)
    sigma_real = np.cov(real_features, rowvar=False)
    mu_gen = np.mean(gen_features, axis=0)
    sigma_gen = np.cov(gen_features, rowvar=False)
    
    # Calculate FID
    diff = mu_real - mu_gen
    covmean, _ = linalg.sqrtm(sigma_real.dot(sigma_gen), disp=False)
    
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    
    fid = diff.dot(diff) + np.trace(sigma_real + sigma_gen - 2 * covmean)
    return fid

def calculate_lpips_batch(real_images, generated_images):
    """Calculate LPIPS scores"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize LPIPS model (using AlexNet backbone)
    lpips_model = lpips.LPIPS(net='alex').to(device)
    
    # Transform for LPIPS (expects [-1, 1] range)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    lpips_scores = []
    
    print("Calculating LPIPS scores...")
    with torch.no_grad():
        for real_path, gen_path in zip(real_images, generated_images):
            real_img = Image.open(real_path).convert('RGB')
            gen_img = Image.open(gen_path).convert('RGB')
            
            # Resize to same size if needed
            if real_img.size != gen_img.size:
                gen_img = gen_img.resize(real_img.size, Image.LANCZOS)
            
            real_t = transform(real_img).unsqueeze(0).to(device)
            gen_t = transform(gen_img).unsqueeze(0).to(device)
            
            score = lpips_model(real_t, gen_t)
            lpips_scores.append(score.item())
    
    return lpips_scores

def main():
    # Paths
    generated_dir = "./sampled_images/custom_test_8images/paired"
    ground_truth_dir = "./DATA/custom/test/image"
    
    # Image IDs from test_pairs.txt
    image_ids = ['0000', '0009', '0012', '0016', '0021', '0031', '0039', '0041']
    
    print("=" * 70)
    print("VIRTUAL TRY-ON EVALUATION METRICS")
    print("=" * 70)
    print(f"\nGenerated images: {generated_dir}")
    print(f"Ground truth images: {ground_truth_dir}")
    print(f"Number of test images: {len(image_ids)}\n")
    
    # Collect image paths
    generated_paths = []
    ground_truth_paths = []
    
    for img_id in image_ids:
        gen_path = os.path.join(generated_dir, f"{img_id}.jpg")
        gt_path = os.path.join(ground_truth_dir, f"{img_id}.png")
        
        if not os.path.exists(gen_path):
            print(f"Warning: Generated image not found: {gen_path}")
            continue
        if not os.path.exists(gt_path):
            print(f"Warning: Ground truth image not found: {gt_path}")
            continue
        
        generated_paths.append(gen_path)
        ground_truth_paths.append(gt_path)
    
    print(f"Found {len(generated_paths)} valid image pairs\n")
    
    if len(generated_paths) == 0:
        print("Error: No valid image pairs found!")
        return
    
    # Calculate SSIM
    print("Calculating SSIM...")
    ssim_scores = []
    for gen_path, gt_path in zip(generated_paths, ground_truth_paths):
        gen_img = load_image(gen_path)
        gt_img = load_image(gt_path)
        
        # Resize generated image to match ground truth if needed
        if gen_img.shape != gt_img.shape:
            gen_pil = Image.fromarray(gen_img)
            gen_pil = gen_pil.resize((gt_img.shape[1], gt_img.shape[0]), Image.LANCZOS)
            gen_img = np.array(gen_pil)
        
        ssim_score = calculate_ssim(gen_img, gt_img)
        ssim_scores.append(ssim_score)
        img_name = os.path.basename(gen_path)
        print(f"  {img_name}: {ssim_score:.4f}")
    
    avg_ssim = np.mean(ssim_scores)
    print(f"\n✓ Average SSIM: {avg_ssim:.4f}")
    
    # Calculate FID
    print("\n" + "=" * 70)
    try:
        fid_score = calculate_fid(ground_truth_paths, generated_paths)
        print(f"✓ FID Score: {fid_score:.4f}")
    except Exception as e:
        print(f"✗ FID calculation failed: {e}")
        fid_score = None
    
    # Calculate LPIPS
    print("\n" + "=" * 70)
    try:
        lpips_scores = calculate_lpips_batch(ground_truth_paths, generated_paths)
        for i, (img_id, score) in enumerate(zip(image_ids[:len(lpips_scores)], lpips_scores)):
            print(f"  {img_id}: {score:.4f}")
        avg_lpips = np.mean(lpips_scores)
        print(f"\n✓ Average LPIPS: {avg_lpips:.4f}")
    except Exception as e:
        print(f"✗ LPIPS calculation failed: {e}")
        avg_lpips = None
    
    # Summary
    print("\n" + "=" * 70)
    print("FINAL RESULTS SUMMARY")
    print("=" * 70)
    print(f"SSIM (Structural Similarity):  {avg_ssim:.4f}  (higher is better, max=1.0)")
    if fid_score is not None:
        print(f"FID (Fréchet Distance):        {fid_score:.4f}  (lower is better)")
    if avg_lpips is not None:
        print(f"LPIPS (Perceptual Similarity): {avg_lpips:.4f}  (lower is better)")
    print("=" * 70)
    
    # Save results to file
    with open("./sampled_images/custom_test_8images/metrics_results.txt", 'w') as f:
        f.write("VIRTUAL TRY-ON EVALUATION METRICS\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Generated images: {generated_dir}\n")
        f.write(f"Ground truth images: {ground_truth_dir}\n")
        f.write(f"Number of images: {len(generated_paths)}\n\n")
        f.write("RESULTS:\n")
        f.write(f"Average SSIM:  {avg_ssim:.4f}\n")
        if fid_score is not None:
            f.write(f"FID Score:     {fid_score:.4f}\n")
        if avg_lpips is not None:
            f.write(f"Average LPIPS: {avg_lpips:.4f}\n")
        f.write("\nPer-image SSIM scores:\n")
        for img_id, score in zip(image_ids[:len(ssim_scores)], ssim_scores):
            f.write(f"  {img_id}: {score:.4f}\n")
        if avg_lpips is not None:
            f.write("\nPer-image LPIPS scores:\n")
            for img_id, score in zip(image_ids[:len(lpips_scores)], lpips_scores):
                f.write(f"  {img_id}: {score:.4f}\n")
    
    print(f"\n✓ Results saved to: ./sampled_images/custom_test_8images/metrics_results.txt")

if __name__ == "__main__":
    main()
