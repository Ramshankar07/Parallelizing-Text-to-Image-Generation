import torch
import torch.nn.functional as F
from transformers import CLIPTokenizer, CLIPTextModel
from diffusers import AutoencoderKL
from tqdm import tqdm
import argparse
from pathlib import Path
from datetime import datetime
from PIL import Image
from torchvision import transforms
import torch.nn as nn
import numpy as np
from train3 import ImprovedUNet, EnhancedNoiseScheduler


"""
Text-to-Image Generation Pipeline Structure and Flow:

1. Text Processing:
   - Uses CLIP tokenizer and encoder for text understanding
   - Converts text prompts into embeddings that guide image generation
   - Function: get_text_embeddings()

2. Image Generation Process (generate_image):
   a. Initial Setup:
      - Creates random noise in latent space
      - Sets up timesteps for denoising
      - Prepares model for inference

   b. Denoising Loop:
      - Classifier-free guidance using text embeddings
      - Progressive denoising through timesteps
      - Noise prediction and removal at each step
      - Value clamping for stability

   c. Final Processing:
      - VAE decoding of latents to image
      - Image normalization and cleanup

3. Image Saving (save_images):
   - Converts tensor to numpy array
   - Handles color space normalization
   - Saves to specified directory

4. Main Execution Flow:
   a. Argument parsing and setup
   b. Model loading:
      - UNet for diffusion
      - CLIP for text processing
      - VAE for image decoding
   c. Generation pipeline execution
   d. Image saving

Key Features:
- Classifier-free guidance for better text adherence
- Stable diffusion through value clamping
- Progress tracking during generation
- Configurable generation parameters

Usage:
python generate.py --prompt "your text" --model-path "path/to/model" 
                  --guidance-scale 2.0 --num-inference-steps 50
"""

def get_text_embeddings(prompt, tokenizer, text_encoder, device):
    """Convert text prompt to CLIP text embeddings."""
    # Tokenize text
    text_input = tokenizer(
        prompt,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt"
    )

    # Move input to device
    text_input = {k: v.to(device) for k, v in text_input.items()}

    # Get text embeddings
    with torch.no_grad():
        text_embeddings = text_encoder(**text_input).last_hidden_state

    return text_embeddings


def generate_image(
    prompt,
    model,
    vae,
    noise_scheduler,
    tokenizer,
    text_encoder,
    device,
    num_inference_steps=50,
    guidance_scale=2.0,  # Reduced further
    latent_shape=(4, 64, 64),
    latent_scaling_factor=0.18215
):
    
    text_embeddings = get_text_embeddings(prompt, tokenizer, text_encoder, device)
    
    # random noise
    latents = torch.randn(
        (1, *latent_shape),
        device=device,
        dtype=torch.float32
    ).clamp(-10, 10)  # Clamp initial noise
    
    #timesteps
    timesteps = torch.linspace(999, 0, num_inference_steps, device=device).long()
    progress_bar = tqdm(total=len(timesteps))
    
    model.eval()
    with torch.no_grad():
        for i, t in enumerate(timesteps):
            latent_model_input = torch.cat([latents] * 2) if guidance_scale > 1.0 else latents
            latent_model_input = latent_model_input.clamp(-10, 10)  # Clamp inputs
            
            # Predict noise
            noise_pred = model(
                latent_model_input,
                t.repeat(latent_model_input.shape[0]),
                text_embeddings.repeat(2, 1, 1) if guidance_scale > 1.0 else text_embeddings
            )
            
            noise_pred = noise_pred.clamp(-10, 10)
            if guidance_scale > 1.0:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                noise_pred = noise_pred.clamp(-10, 10)  # Clamp after guidance
            alpha_prod_t = noise_scheduler.alphas_cumprod[t]
            alpha_prod_t_prev = noise_scheduler.alphas_cumprod[t-1] if t > 0 else torch.tensor(1.0, device=device)
            
            # coefficients
            sqrt_alpha_prod_t = torch.sqrt(alpha_prod_t)
            sqrt_one_minus_alpha_prod_t = torch.sqrt(1 - alpha_prod_t)
            pred_original_sample = (latents - sqrt_one_minus_alpha_prod_t * noise_pred) / sqrt_alpha_prod_t
            pred_original_sample = pred_original_sample.clamp(-10, 10)  # Clamp predictions
            prev_sample = torch.sqrt(alpha_prod_t_prev) * pred_original_sample + \
                         torch.sqrt(1 - alpha_prod_t_prev) * noise_pred
            
            latents = prev_sample.clamp(-10, 10)  
            progress_bar.update(1)
            
            
    
    progress_bar.close()
    
    
    
    latents = (1 / 0.18215) * latents.clamp(-10, 10)
    
    with torch.no_grad():
        image = vae.decode(latents).sample
        image = image.clamp(-1, 1)
    
    return image

def save_images(images, output_dir, prompt, seed=None):
    """Save generated images."""
    with torch.no_grad():
        images = images.detach().cpu().float()
        
        # Converting from [-1, 1] to [0, 255]
        images = ((images + 1) * 127.5).clamp(0, 255).round()
        images = images.numpy().astype(np.uint8)
        images = images.transpose(0, 2, 3, 1)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for i, image in enumerate(images):
            save_path = output_dir / f"{prompt[:]}.png"
            Image.fromarray(image).save(save_path)
            print(f"Saved image to {save_path}")



def main():
    parser = argparse.ArgumentParser(description="Generate images from text prompts")
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="generated_images")
    parser.add_argument("--num-inference-steps", type=int, default=50)
    parser.add_argument("--guidance-scale", type=float, default=3.0)  # Reduced default
    parser.add_argument("--seed", type=int, default=42)  # Added default seed
    args = parser.parse_args()

    # Set all seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")

    # Load and verify model
    model = ImprovedUNet(image_channels=4)
    model = model.to(device)

    print("\nLoading other models...")
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(device)

    noise_scheduler = EnhancedNoiseScheduler(device=device)

    print(f"\nGenerating image for prompt: {args.prompt}")
    images = generate_image(
        prompt=args.prompt,
        model=model,
        vae=vae,
        noise_scheduler=noise_scheduler,
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        device=device,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale
    )

    save_images(images, args.output_dir, args.prompt, args.seed)

if __name__ == "__main__":
    main()