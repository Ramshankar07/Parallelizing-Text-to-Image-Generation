import pickle
import torch
from diffusers import AutoencoderKL
from PIL import Image
import numpy as np
import time
from datetime import timedelta
import warnings
import argparse
import gc
import dask
import dask.array as da
from dask.distributed import Client, LocalCluster, wait
from dask import delayed
warnings.filterwarnings('ignore')
##################
# Loading the VAE model and we use cache that it is loaded only once
#############################
def load_vae():
    vae = AutoencoderKL.from_pretrained(
        "stabilityai/sd-vae-ft-mse",
        torch_dtype=torch.float32
    )
    vae.eval()
    return vae


##################
# Processing image to convert into best format that is given as input to VAE using mixed precision
#############################

def process_image(image, target_size=(512, 512)):
    """Process image to match Stable Diffusion VAE input requirements"""
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    
    width, height = image.size
    scale = min(target_size[0] / width, target_size[1] / height)
    new_width = int(width * scale)
    new_height = int(height * scale)
    image = image.resize((new_width, new_height), Image.LANCZOS)
    
    new_image = Image.new('RGB', target_size, (0, 0, 0))
    offset_x = (target_size[0] - new_width) // 2
    offset_y = (target_size[1] - new_height) // 2
    new_image.paste(image, (offset_x, offset_y))
    
    return np.array(new_image)

    
##################
# Batch processing in CPU
#############################

@delayed
def process_batch(images, captions):
    processed_images = [process_image(img) for img in images]
    return np.stack(processed_images), captions

def batch_to_embeddings(batch_data, vae):
    images, captions = batch_data
    
    try:
        
        batch_tensor = torch.from_numpy(images).float() / 127.5 - 1
        batch_tensor = batch_tensor.permute(0, 3, 1, 2)  # NHWC to NCHW
        
        with torch.no_grad():
            latents = vae.encode(batch_tensor).latent_dist.sample()
            embeddings = latents.cpu().numpy()
        
        return [
            {'embedding': emb, 'caption': cap, 'status': 'success'}
            for emb, cap in zip(embeddings, captions)
        ]
    
    except Exception as e:
        print(f"Error processing batch: {str(e)}")
        return [
            {'caption': cap, 'status': 'error', 'error_message': str(e)}
            for cap in captions
        ]

    
    

##################
# Main function that handles the multi cpu and implements the process using pool 
#############################   

def encode(input_pkl_path, output_pkl_path, n_workers=4, batch_size=8, num_images=None):
    start_time = time.time()
    threads_per_worker = max(1,  n_workers)
    
    cluster = LocalCluster(
        n_workers=n_workers,
        threads_per_worker=threads_per_worker,
        memory_limit='40GB',  
        memory_target_fraction=0.8,
        memory_spill_fraction=0.9,
        memory_pause_fraction=0.95,
        processes=True
    )
    client = Client(cluster)
    print(f"Dashboard link: {client.dashboard_link}")
    print(f"Using {n_workers} workers with {threads_per_worker} threads each")
    
    print("Loading data")
    with open(input_pkl_path, 'rb') as f:
        data = pickle.load(f)
    
    if num_images:
        data = data[:num_images]
    
    vae = load_vae()
    
    batches = []
    for i in range(0, len(data), batch_size):
        batch_data = data[i:i + batch_size]
        batch_images = [item['image'] for item in batch_data]
        batch_captions = [item['caption'] for item in batch_data]
        batches.append((batch_images, batch_captions))
    
    print(f"Processing {len(data)} images in batches of {batch_size}...")
    results = []
    processed_count = 0
    delayed_batches = [process_batch(images, captions) for images, captions in batches]
    
    for i, delayed_batch in enumerate(delayed_batches):
        processed_batch = delayed_batch.compute()
        batch_results = batch_to_embeddings(processed_batch, vae)
        results.extend(batch_results)
        
        processed_count += len(batch_results)
        elapsed_time = time.time() - start_time
        images_per_second = processed_count / elapsed_time
        remaining_images = len(data) - processed_count
        eta = remaining_images / images_per_second if images_per_second > 0 else 0
        
        print(f"\nProcessed batch {i+1}/{len(batches)}")
        print(f"Progress: {processed_count}/{len(data)} images")
        print(f"Speed: {images_per_second:.2f} images/second")
        print(f"ETA: {timedelta(seconds=int(eta))}")
    
    total_time = time.time() - start_time
    
    print("\nSaving results...")
    with open(output_pkl_path, 'wb') as f:
        pickle.dump(results, f)
    
    successful_count = sum(1 for r in results if r['status'] == 'success')
    error_count = sum(1 for r in results if r['status'] == 'error')
    
    client.close()
    cluster.close()
    
    print("\nProcessing Summary:")
    print(f"Total time: {timedelta(seconds=int(total_time))}")
    print(f"Successfully processed: {successful_count} images")
    print(f"Failed to process: {error_count} images")
    print(f"Average time per image: {total_time/max(successful_count, 1):.3f} seconds")
    print(f"Overall speed: {successful_count/total_time:.2f} images/second")

def main():
    parser = argparse.ArgumentParser(description='Process images with Stable Diffusion VAE using Dask Multi-threading')
    parser.add_argument('--input', required=True, help='Input pickle file path')
    parser.add_argument('--output', required=True, help='Output pickle file path')
    parser.add_argument('--n-workers', type=int, default=4, help='Number of Dask workers')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size for processing')
    parser.add_argument('--num-images', type=int, help='Number of images to process (optional)')
    
    args = parser.parse_args()
    
    encode(
        args.input,
        args.output,
        n_workers=args.n_workers,
        batch_size=args.batch_size,
        num_images=args.num_images
    )

if __name__ == "__main__":
    main()