import pickle
import torch
from diffusers import AutoencoderKL
from PIL import Image
import numpy as np
import time
import psutil
from datetime import timedelta
import warnings
import argparse
import gc
from joblib import Parallel, delayed
warnings.filterwarnings('ignore')


##################
# Loading the VAE model and we use cache that it is loaded only once
#############################
def init_vae():
    vae = AutoencoderKL.from_pretrained(
        "stabilityai/sd-vae-ft-mse",
        torch_dtype=torch.float32
    )
    vae.eval()
    if torch.cuda.is_available():
        vae = vae.cuda()
    return vae


##################
# Processing image to convert into best format that is given as input to VAE using mixed precision
#############################

def process_image(image, target_size=(512, 512)):
    width, height = image.size
    scale = min(target_size[0] / width, target_size[1] / height)
    new_width = int(width * scale)
    new_height = int(height * scale)
    image = image.resize((new_width, new_height), Image.LANCZOS)
    
    new_image = Image.new('RGB', target_size, (0, 0, 0))
    offset_x = (target_size[0] - new_width) // 2
    offset_y = (target_size[1] - new_height) // 2
    new_image.paste(image, (offset_x, offset_y))
    
    image = np.array(new_image)
    image = torch.from_numpy(image).float() / 127.5 - 1
    image = image.permute(2, 0, 1).unsqueeze(0)
    
    return image


    
##################
# Batch processing in CPU
#############################

def process_chunk(chunk_data, device='cpu'):
    try:
        vae = init_vae()
        results = []
        
        for item in chunk_data:
            try:
                image = process_image(item['image'])
                if device == 'cuda':
                    image = image.cuda()
                
                with torch.no_grad():
                    latents = vae.encode(image).latent_dist.sample()
                    embedding = latents.cpu().numpy()
                
                results.append({
                    'embedding': embedding,
                    'caption': item['caption'],
                    'status': 'success'
                })
                
                del image, latents, embedding
                torch.cuda.empty_cache() if device == 'cuda' else gc.collect()
                
            except Exception as e:
                results.append({
                    'caption': item.get('caption', 'Unknown'),
                    'status': 'error',
                    'error_message': str(e)
                })
                
        del vae
        torch.cuda.empty_cache() if device == 'cuda' else gc.collect()
        return results
        
    except Exception as e:
        return [{
            'caption': 'Chunk processing failed',
            'status': 'error',
            'error_message': str(e)
        }]


##################
# chunking process
#############################

def chunk_data(data, num_chunks):
    chunk_size = len(data) // num_chunks + (1 if len(data) % num_chunks else 0)
    return [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]

##################
# Main function that handles the multi cpu and implements the process using pool 
#############################   

def encode_parallel(input_pkl_path, output_pkl_path, num_cpus=1, num_images=None):
    start_time = time.time()
    
    print(f"Loading data from {input_pkl_path}")
    with open(input_pkl_path, 'rb') as f:
        data = pickle.load(f)
    
    if num_images:
        data = data[:num_images]
    
    print(f"Processing {len(data)} images using {num_cpus} CPUs")
    chunks = chunk_data(data, num_cpus)
    
    print(f"Using {num_cpus} CPU(s)")
    
    results = []
    parallel = Parallel(n_jobs=num_cpus, verbose=1, backend='loky')
    chunk_results = parallel(delayed(process_chunk)(chunk) for chunk in chunks)
    results = [item for sublist in chunk_results for item in sublist]
    
    with open(output_pkl_path, 'wb') as f:
        pickle.dump(results, f)
            
    total_time = time.time() - start_time
    
    print("Saving results...")
    with open(output_pkl_path, 'wb') as f:
        pickle.dump(results, f)
    
    successful = sum(1 for r in results if r['status'] == 'success')
    failed = sum(1 for r in results if r['status'] == 'error')
    
    print("\nProcessing Summary:")
    print(f"Total time: {timedelta(seconds=int(total_time))}")
    print(f"Successfully processed: {successful} images")
    print(f"Failed to process: {failed} images")
    print(f"Average time per image: {total_time/max(successful, 1):.3f} seconds")
    print(f"Overall speed: {successful/total_time:.2f} images/second")

def main():
    parser = argparse.ArgumentParser(description='Process images with VAE using multiple CPUs')
    parser.add_argument('--input', required=True, help='Input pickle file path')
    parser.add_argument('--output', required=True, help='Output pickle file path')
    parser.add_argument('--num-cpus', type=int, default=1, help='Number of CPU processes')
    parser.add_argument('--num-images', type=int, help='Number of images to process (optional)')
    
    args = parser.parse_args()
    
    encode_parallel(
        args.input,
        args.output,
        num_cpus=args.num_cpus,
        num_images=args.num_images
    )

if __name__ == "__main__":
    main()