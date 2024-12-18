import pickle
import torch
import torch.multiprocessing as mp
from diffusers import AutoencoderKL
from PIL import Image
import numpy as np
import time
from datetime import timedelta
import warnings
import argparse
import gc
from tqdm import tqdm
warnings.filterwarnings('ignore')
##################
# Loading the VAE model
#############################
def load_vae(gpu_id):
    device = torch.device(f'cuda:{gpu_id}')
    print(f"Loading VAE model")
    model = AutoencoderKL.from_pretrained(
        "stabilityai/sd-vae-ft-mse",
        torch_dtype=torch.float16
    )
    model = model.to(device)
    model.eval()
    return model
##################
# Processing image to convert into best format that is given as input to VAE using mixed precision
#############################
@torch.cuda.amp.autocast()
def process_image(image, target_size=(512, 512), device=torch.device('cpu')):
    width, height = image.size
    scale = min(target_size[0] / width, target_size[1] / height)
    new_width = int(width * scale)
    new_height = int(height * scale)
    image = image.resize((new_width, new_height), Image.LANCZOS)
    
    new_image = Image.new('RGB', target_size, (0, 0, 0))
    offset_x = (target_size[0] - new_width) // 2
    offset_y = (target_size[1] - new_height) // 2
    new_image.paste(image, (offset_x, offset_y))
    
    image = np.array(new_image, dtype=np.float32)
    image = (image / 127.5) - 1.0
    image = torch.from_numpy(image).to(device)
    image = image.permute(2, 0, 1).unsqueeze(0)
    
    return image

##################
# Clearing GPU memory
#############################

def clear_gpu_memory(gpu_id=None):
    if torch.cuda.is_available():
        if gpu_id is not None:
            with torch.cuda.device(f'cuda:{gpu_id}'):
                torch.cuda.empty_cache()
        else:
            torch.cuda.empty_cache()
    gc.collect()
    
    
    
##################
# Batch processing in GPU
#############################
def process_gpu_batch(gpu_id, data_queue, result_queue):
    try:
        device = torch.device(f'cuda:{gpu_id}')
        vae = load_vae(gpu_id)
        print(f"GPU {gpu_id} worker started")
        
        while True:
            batch_data = data_queue.get()
            if batch_data is None:  # End signal
                break
                
            try:
                images = [item['image'] for item in batch_data]
                captions = [item['caption'] for item in batch_data]
                processed_images = []
                
                # Process images one at a time
                for img in images:
                    try:
                        processed = process_image(img, device=device)
                        processed_images.append(processed)
                    except Exception as e:
                        print(f"Error processing single image on GPU {gpu_id}: {str(e)}")
                    finally:
                        del img
                
                if processed_images:
                    try:
                        batch_tensor = torch.cat(processed_images, dim=0)
                        
                        with torch.no_grad(), torch.cuda.amp.autocast():
                            latents = vae.encode(batch_tensor).latent_dist.sample()
                            embeddings = latents.detach().cpu().numpy()
                        
                        batch_results = [
                            {'embedding': emb, 'caption': cap, 'status': 'success'}
                            for emb, cap in zip(embeddings, captions)
                        ]
                        
                        result_queue.put(batch_results)
                        
                    except Exception as e:
                        print(f"Error processing batch tensor on GPU {gpu_id}: {str(e)}")
                        result_queue.put([
                            {'caption': cap, 'status': 'error', 'error_message': str(e)}
                            for cap in captions
                        ])
                    
                    finally:
                        del batch_tensor, latents, embeddings
                        clear_gpu_memory(gpu_id)
                
                del processed_images
                clear_gpu_memory(gpu_id)
                
            except Exception as e:
                print(f"Error processing batch on GPU {gpu_id}: {str(e)}")
                result_queue.put([
                    {'caption': item['caption'], 'status': 'error', 'error_message': str(e)}
                    for item in batch_data
                ])
    
    except Exception as e:
        print(f"GPU {gpu_id} worker failed: {str(e)}")
    
    finally:
        print(f"GPU {gpu_id} worker finished")

##################
# Main function that handles the multi gpu and implements the process in Queue
#############################        
        
        
def encode(input_pkl_path, output_pkl_path, num_gpus=1, num_images=None):
    start_time = time.time()
    
    # number of available GPUs
    num_gpus = min(num_gpus, torch.cuda.device_count())
    print(f"Using {num_gpus} GPUs")
    
    print("Loading data")
    with open(input_pkl_path, 'rb') as f:
        data = pickle.load(f)
    
    if num_images:
        data = data[:num_images]
    
    # queues for multi-GPU processing
    data_queues = [mp.Queue() for _ in range(num_gpus)]
    result_queue = mp.Queue()
    
    # GPU processes
    processes = []
    for gpu_id in range(num_gpus):
        p = mp.Process(target=process_gpu_batch, args=(gpu_id, data_queues[gpu_id], result_queue))
        p.start()
        processes.append(p)
    mini_batch_size = 8
    total_batches = len(data) // mini_batch_size + (1 if len(data) % mini_batch_size else 0)
    
    print(f"Processing {len(data)} images using {num_gpus} GPUs...")
    results = []
    try:
        pbar = tqdm(total=len(data))
        current_gpu = 0
        
        for i in range(0, len(data), mini_batch_size):
            batch = data[i:i + mini_batch_size]
            data_queues[current_gpu].put(batch)
            current_gpu = (current_gpu + 1) % num_gpus
        
        # end signals
        for q in data_queues:
            q.put(None)
        
        completed = 0
        while completed < total_batches:
            batch_results = result_queue.get()
            results.extend(batch_results)
            completed += 1
            pbar.update(len(batch_results))
            success = sum(1 for r in results if r['status'] == 'success')
            if len(results) > 0:
                success_rate = success/len(results)*100
                pbar.set_description(f"Success rate: {success_rate:.1f}%")
            
            
    
    except KeyboardInterrupt:
        print("\nProcessing interrupted. Saving partial results...")
    
    finally:
        for p in processes:
            p.join()
        
        total_time = time.time() - start_time
        
        print("\nSaving")
        with open(output_pkl_path, 'wb') as f:
            pickle.dump(results, f)

        successful_count = sum(1 for r in results if r['status'] == 'success')
        error_count = sum(1 for r in results if r['status'] == 'error')
        
        print("\nProcessing Summary:")
        print(f"Total time: {timedelta(seconds=int(total_time))}")
        print(f"Successfully processed: {successful_count} images")
        print(f"Failed to process: {error_count} images")
        if successful_count > 0:
            print(f"Average time per image: {total_time/successful_count:.3f} seconds")
            print(f"Overall speed: {successful_count/total_time:.2f} images/second")

def main():
    parser = argparse.ArgumentParser(description='Process images with Stable Diffusion VAE')
    parser.add_argument('--input', required=True, help='Input pickle file path')
    parser.add_argument('--output', required=True, help='Output pickle file path')
    parser.add_argument('--num-gpus', type=int, default=1, help='Number of GPUs to use')
    parser.add_argument('--num-images', type=int, help='Number of images to process (optional)')
    
    args = parser.parse_args()
    mp.set_start_method('spawn')
    
    encode(
        args.input,
        args.output,
        num_gpus=args.num_gpus,
        num_images=args.num_images
    )

if __name__ == "__main__":
    main()