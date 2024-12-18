import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
import math
import time
import numpy as np
from datetime import timedelta, datetime
import os
import argparse
import json
from pathlib import Path
import torch.multiprocessing as mp
import pickle
from PIL import Image
from torchvision import transforms
from transformers import CLIPModel, CLIPTokenizer
from diffusers import AutoencoderKL



"""
Code Structure and Flow:

1. Model Architecture:
   - ImprovedUNet: Main backbone for diffusion model
     - Uses U-Net architecture with downsampling/upsampling paths
     - Incorporates attention blocks at multiple scales
     - Processes both image and timestep embeddings
   
   - Component Classes:
     - UNetBlock: Basic building block with conv layers, normalization, and time embedding
     - AttentionBlock: Self-attention mechanism for better feature processing
     - SinusoidalPositionEmbeddings: Time step encoding
     - EMAModel: Exponential Moving Average for model weights

2. Training Pipeline:
   - LatentDiffusion: Wrapper class integrating UNet, VAE, and CLIP
     - Handles image/text encoding and decoding
     - Manages model freezing and forward passes
   
   - EnhancedNoiseScheduler: Controls the diffusion process
     - Manages noise addition and removal
     - Handles beta schedule and alpha calculations

3. Data Handling:
   - EnhancedCOCODataset: Custom dataset class
     - Loads and processes image embeddings and captions
     - Handles distributed data splitting
     - Provides image transformation pipeline
   
   - custom_collate_fn: Batches data for training
     - Combines image features, text embeddings, and captions

4. Training Implementation:
   - train_ddp: Main distributed training loop
     - Initializes distributed process groups
     - Sets up data loading and model distribution
     - Manages training iterations and checkpointing
   
   - train_step: Core training iteration
     - Handles noise addition and prediction
     - Computes loss and manages gradients

5. Utility Functions:
   - evaluate_model: Validation during training
   - main: Entry point and argument parsing

Flow:
1. Data Loading → 2. Model Initialization → 3. Distributed Setup →
4. Training Loop (Noise Addition → Prediction → Loss Calculation → Update) →
5. Evaluation and Checkpointing

Key Features:
- Multi-GPU training support
- Mixed precision training
- Gradient scaling
- Memory optimization
- Distributed data handling
"""

## Component Class that helps the structure 

class EMAModel: 
    """Exponential Moving Average of model weights"""
    def __init__(self, model, decay=0.9999, device=None):
        self.decay = decay
        self.device = device
        self.model = model
        self.shadow = {}
        self.register()

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone().detach()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.shadow[name]
                
## This block add the attention mechanisim in text and image embedding

class AttentionBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.norm = nn.GroupNorm(8, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)
        
    def forward(self, x):
        B, C, H, W = x.shape
        qkv = self.qkv(self.norm(x))
        q, k, v = qkv.chunk(3, dim=1)
        
        q = q.view(B, C, -1) # query
        k = k.view(B, C, -1) # Key
        v = v.view(B, C, -1) # value 
        
        attn = torch.einsum('bci,bcj->bij', q, k) * (self.channels ** -0.5)
        attn = F.softmax(attn, dim=2)
        
        out = torch.einsum('bij,bcj->bci', attn, v)
        out = out.view(B, C, H, W)
        return self.proj(out) + x
    

class UNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_channels, has_attn=False):
        super().__init__()
        self.time_mlp = nn.Linear(time_channels, out_channels)
        
        # First convolution
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_channels)
        
        # Second convolution
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_channels)
        
        # Attention if specified
        self.attn = AttentionBlock(out_channels) if has_attn else nn.Identity()
        
        # Skip connection if input and output channels differ
        self.skip_connection = (
            nn.Conv2d(in_channels, out_channels, 1) 
            if in_channels != out_channels 
            else nn.Identity()
        )
        
    def forward(self, x, t):
        # Skip connection
        skip = self.skip_connection(x)
        
        # First conv block
        h = self.conv1(x)
        h = self.norm1(h)
        h += self.time_mlp(t)[..., None, None]
        h = F.silu(h)
        
        # Second conv block
        h = self.conv2(h)
        h = self.norm2(h)
        h = F.silu(h)
        
        # Attention
        h = self.attn(h)
        
        # Add skip connection
        return h + skip
    
    
class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=time.device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

###########################################
# Main Architecture of text to image generation
##########################################################3

class ImprovedUNet(nn.Module):
    def __init__(self, image_channels=4, time_channels=256, base_channels=64):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_channels),
            nn.Linear(time_channels, time_channels),
            nn.GELU(),
            nn.Linear(time_channels, time_channels)
        )

        # Down sample
        self.down1 = UNetBlock(image_channels, base_channels, time_channels)
        self.down2 = UNetBlock(base_channels, base_channels*2, time_channels, has_attn=True)
        self.down3 = UNetBlock(base_channels*2, base_channels*4, time_channels, has_attn=True)
        
        # Bottleneck
        self.bottleneck = UNetBlock(base_channels*4, base_channels*4, time_channels, has_attn=True)
        
        # Up sampling path 
        self.up3 = UNetBlock(base_channels*8, base_channels*2, time_channels, has_attn=True)
        self.up2 = UNetBlock(base_channels*4, base_channels, time_channels, has_attn=True)
        self.up1 = UNetBlock(base_channels*2, base_channels, time_channels)
        
        self.final = nn.Conv2d(base_channels, image_channels, 1)

    def forward(self, x, t, context=None):

        t = self.time_mlp(t)
        
        # Downsampling
        d1 = self.down1(x, t)                              # base_channels
        d2 = self.down2(F.max_pool2d(d1, 2), t)           # base_channels*2
        d3 = self.down3(F.max_pool2d(d2, 2), t)           # base_channels*4
        
        # Bottleneck
        bottleneck = self.bottleneck(F.max_pool2d(d3, 2), t)  # base_channels*4
        
        # Upsampling with skip connections
        # bottleneck (base_channels*4) + d3 (base_channels*4) = base_channels*8
        up3 = self.up3(torch.cat([F.interpolate(bottleneck, size=d3.shape[2:]), d3], dim=1), t)
        
        # up3 (base_channels*2) + d2 (base_channels*2) = base_channels*4
        up2 = self.up2(torch.cat([F.interpolate(up3, size=d2.shape[2:]), d2], dim=1), t)
        
        # up2 (base_channels) + d1 (base_channels) = base_channels*2
        up1 = self.up1(torch.cat([F.interpolate(up2, size=d1.shape[2:]), d1], dim=1), t)
        
        return self.final(up1)


##################################################3
# The dataset class that merges two pkl file using captions 
#######################################################3

class EnhancedCOCODataset(Dataset):
    def __init__(self, embeddings_path, images_path, image_size=128, rank=None, world_size=None):
        print(f"Loading embeddings from {embeddings_path}")
        print(f"Loading image data from {images_path}")

        with open(embeddings_path, 'rb') as f:
            embeddings_data = pickle.load(f)
            if isinstance(embeddings_data, dict):
                self.image_embeddings = embeddings_data.get('image_embeddings')
                if self.image_embeddings is None and len(embeddings_data) > 0:
                    self.image_embeddings = list(embeddings_data.values())[0]
            else:
                self.image_embeddings = embeddings_data
            del embeddings_data

        with open(images_path, 'rb') as f:
            self.image_data = pickle.load(f)

        print(f"Embeddings shape: {self.image_embeddings.shape if hasattr(self.image_embeddings, 'shape') else 'unknown'}")
        print(f"Number of images: {len(self.image_data)}")

        if hasattr(self.image_embeddings, 'shape'):
            self.num_samples = min(len(self.image_data), self.image_embeddings.shape[0])
        else:
            self.num_samples = len(self.image_data)

        print(f"Number of matched samples: {self.num_samples}")

        self.image_size = image_size
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

        # Spliting data for distributed training
        if rank is not None and world_size is not None:
            per_rank = self.num_samples // world_size
            start_idx = rank * per_rank
            end_idx = start_idx + per_rank if rank != world_size - 1 else self.num_samples
            
            self.image_embeddings = self.image_embeddings[start_idx:end_idx]
            self.image_data = self.image_data[start_idx:end_idx]
            self.num_samples = end_idx - start_idx

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        image_embedding = torch.tensor(self.image_embeddings[idx], dtype=torch.float32)
        img_item = self.image_data[idx]
        image = img_item['image']
        if isinstance(image, str):
            try:
                image = Image.open(image).convert('RGB')
            except Exception as e:
                print(f"Error loading image: {e}")
                image = Image.new('RGB', (self.image_size, self.image_size))
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        image = self.transform(image)
        caption = img_item.get('caption', '')
        text_embedding = image_embedding.clone()

        return {
            'image_features': image_embedding,
            'text_embedding': text_embedding,
            'image': image,
            'caption': caption
        }
##################################################3
# It handles the batching data from the dataset since some were having missing values previously
#######################################################3
def custom_collate_fn(batch):
    """Custom collate function that handles tensor data"""
    if len(batch) == 0:
        return {}

    image_features = []
    text_embeddings = []
    images = []
    captions = []

    for sample in batch:
        image_features.append(sample['image_features'])
        text_embeddings.append(sample['text_embedding'])
        images.append(sample['image'])
        captions.append(sample['caption'])

    batch_dict = {
        'image_features': torch.stack(image_features),
        'text_embedding': torch.stack(text_embeddings),
        'image': torch.stack(images),
        'caption': captions
    }

    return batch_dict



###########################3
# Training Pipeline
##############################
class LatentDiffusion(nn.Module):
    def __init__(self, unet_model, vae_model=None, clip_model=None, latent_scaling_factor=0.18215):
        super().__init__()
        self.unet = unet_model
        self.vae = vae_model
        self.clip = clip_model
        self.latent_scaling_factor = latent_scaling_factor

        if self.clip is None:
            self.clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
            self.clip.eval()

        # Freeze CLIP and VAE
        if self.vae is not None:
            for param in self.vae.parameters():
                param.requires_grad = False
        for param in self.clip.parameters():
            param.requires_grad = False

    def encode_images(self, x):
        with torch.no_grad():
            latents = self.vae.encode(x)
            latents = latents * self.latent_scaling_factor
        return latents

    def decode_latents(self, latents):
        with torch.no_grad():
            latents = latents / self.latent_scaling_factor
            images = self.vae.decode(latents)
        return images

    def encode_text(self, text):
        with torch.no_grad():
            text_embeddings = self.clip(text)
        return text_embeddings

    def forward(self, x, timesteps, context):
        return self.unet(x, timesteps, context)
    
## utils -- that adds noise in images based on timesteps

class EnhancedNoiseScheduler:
    def __init__(self, num_timesteps=1000, device=None):
        self.num_timesteps = num_timesteps
        self.device = device
        self.betas = torch.linspace(0.0001, 0.02, num_timesteps).to(device)
        self.alphas = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - self.alphas_cumprod)

    def add_noise(self, x, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x)

        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)

        return sqrt_alphas_cumprod_t * x + sqrt_one_minus_alphas_cumprod_t * noise, noise

    def remove_noise(self, x, t, noise_pred):
        alpha_t = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        one_minus_alpha_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)

        return (x - one_minus_alpha_t * noise_pred) / alpha_t

##################3 
# Utils
#############
def train_step(batch, model, noise_scheduler, optimizer, scaler, device):

        latents = batch['image_features'].to(device)  #  VAE encodings
        text_embeddings = batch['text_embedding'].to(device)  # CLIP embeddings
        
        noise = torch.randn_like(latents)
        timesteps = torch.randint(0, noise_scheduler.num_timesteps, (latents.size(0),), device=device)
        
        noisy_latents, _ = noise_scheduler.add_noise(latents, timesteps, noise)
        
        with autocast():
            noise_pred = model(noisy_latents, timesteps, text_embeddings)
            loss = F.mse_loss(noise_pred, noise)
        
        scaler.scale(loss).backward()
        return loss.item()

def evaluate_model(model, eval_loader, noise_scheduler, device, num_samples=100):
    model.eval()
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(eval_loader):
            if batch_idx * eval_loader.batch_size >= num_samples:
                break

            latents = batch['image_features'].to(device)
            text_embeddings = batch['text_embedding'].to(device)
            
            # Generate noise
            noise = torch.randn_like(latents)
            timesteps = torch.randint(0, noise_scheduler.num_timesteps, (latents.size(0),), device=device)
            
            # Add noise
            noisy_latents, _ = noise_scheduler.add_noise(latents, timesteps, noise)
            
            # Predict noise
            noise_pred = model(noisy_latents, timesteps, text_embeddings)
            loss = F.mse_loss(noise_pred, noise)
            
            total_loss += loss.item()
            num_batches += 1

    return {
        'eval_loss': total_loss / num_batches if num_batches > 0 else float('inf')
    }



######################################
# MAIN Training function that intializes model and gpus to run on it
############################################3


def train_ddp(rank, world_size, args):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    torch.distributed.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=world_size,
        rank=rank
    )

    torch.cuda.set_device(rank)
    torch.cuda.empty_cache()
    device = torch.device(f"cuda:{rank}")

    dataset = EnhancedCOCODataset(
        embeddings_path=args.embeddings_path,
        images_path=args.images_path,
        image_size=128,
        rank=rank,
        world_size=world_size
    )

    train_sampler = torch.utils.data.distributed.DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        drop_last=True
    )

    train_dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=1,  
        pin_memory=True,
        persistent_workers=True,
        drop_last=True,
        collate_fn=custom_collate_fn
    )


    unet = ImprovedUNet(image_channels=4).to(device)
    diffusion_model = DDP(unet, device_ids=[rank], find_unused_parameters=False)
    optimizer = torch.optim.AdamW(diffusion_model.parameters(), lr=args.learning_rate)
    noise_scheduler = EnhancedNoiseScheduler(device=device)
    scaler = GradScaler()

    total_diff_loss = 0.0
    best_loss = float('inf')
    start_time = time.time()

    try:
        for epoch in range(args.num_epochs):
            train_sampler.set_epoch(epoch)
            diffusion_model.train()

            for batch_idx, batch in enumerate(train_dataloader):
                optimizer.zero_grad(set_to_none=True)
                latents = batch['image_features'].to(device, non_blocking=True)
                text_embeddings = batch['text_embedding'].to(device, non_blocking=True)
                
                # Training step
                noise = torch.randn_like(latents)
                timesteps = torch.randint(0, noise_scheduler.num_timesteps, (latents.size(0),), device=device)
                noisy_latents, _ = noise_scheduler.add_noise(latents, timesteps, noise)
                
                with autocast():
                    noise_pred = diffusion_model(noisy_latents, timesteps, text_embeddings)
                    loss = F.mse_loss(noise_pred, noise)
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                
                total_diff_loss += loss.item()

                if rank == 0 and batch_idx % args.log_interval == 0:
                    avg_loss = total_diff_loss / args.log_interval
                    elapsed = time.time() - start_time
                    print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {avg_loss:.4f}, Time: {elapsed:.2f}s")
                    total_diff_loss = 0.0

                if batch_idx % 100 == 0:
                    torch.cuda.empty_cache()

            if rank == 0 and (epoch + 1) % args.save_interval == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': diffusion_model.module.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': total_diff_loss / (batch_idx + 1),
                }, f"{args.checkpoint_dir}/checkpoint_epoch_{epoch+1}.pt")

    except Exception as e:
        print(f"Rank {rank} encountered error: {str(e)}")
        raise e
    finally:
        torch.cuda.empty_cache()
        torch.distributed.destroy_process_group()

def main():
    parser = argparse.ArgumentParser(description='Distributed Text-to-Image Diffusion Training')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size per GPU')
    parser.add_argument('--num-epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--learning-rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--num-workers', type=int, default=2, help='Number of data loading workers per GPU')
    parser.add_argument('--log-interval', type=int, default=100, help='Logging interval')
    parser.add_argument('--save-interval', type=int, default=5, help='Checkpoint saving interval')
    parser.add_argument('--eval-interval', type=int, default=5, help='Evaluation interval')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints', help='Checkpoint directory')
    parser.add_argument('--num-gpus', type=int, default=torch.cuda.device_count(), help='Number of GPUs')
    parser.add_argument('--embeddings-path', type=str, required=True, help='Path to embeddings')
    parser.add_argument('--images-path', type=str, required=True, help='Path to images')

    args = parser.parse_args()
    
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    try:
        mp.spawn(
            train_ddp,
            args=(args.num_gpus, args),
            nprocs=args.num_gpus,
            join=True
        )
    except Exception as e:
        print(f"Training failed: {str(e)}")
        raise e
if __name__ == '__main__':


    main()