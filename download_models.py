from huggingface_hub import snapshot_download

# Download SDXL Inpainting model
snapshot_download(
    repo_id="stabilityai/stable-diffusion-xl-inpainting-1.0",
    local_dir="./pretrained_models/stable-diffusion-xl-1.0-inpainting-0.1",
    repo_type="model"
)

# Download SDXL VAE
snapshot_download(
    repo_id="stabilityai/sdxl-vae-fp16-fix",
    local_dir="./pretrained_models/sdxl-vae-fp16-fix",
    repo_type="model"
)

print("✅ Download complete")
