import torch
import torch.multiprocessing as mp
from transformers import CLIPTokenizer, CLIPModel
import pickle
import numpy as np
from tqdm import tqdm
import argparse
import os
import time
from datetime import datetime, timedelta
######################################3
# Parsing the arguments
############################
def parse_arguments():
    parser = argparse.ArgumentParser(description='Process dataset with CLIP model using GPUs')
    parser.add_argument('--input', type=str, required=True, help='Input pickle file path')
    parser.add_argument('--output', type=str, required=True, help='Output pickle file path')
    parser.add_argument('--batch-size', type=int, default=256, 
                       help='Batch size for processing')
    parser.add_argument('--num-gpus', type=int, default=None,
                       help='Number of GPUs to use (default: all available)')
    parser.add_argument('--num-captions', type=int, default=None,
                       help='Number of captions to process (default: all)')
    return parser.parse_args()
#############################33333
# Getting number of gpus to check
#########################333333
def get_available_gpus():
    if torch.cuda.is_available():
        return torch.cuda.device_count()
    return 0
#####################

# Class that initializes the CLIP model and tokenizer and also process the batches and concatinates the result

#############################3
class GPUWorker:
    def __init__(self, gpu_id):
        self.gpu_id = gpu_id
        self.device = torch.device(f'cuda:{gpu_id}')
        self.model = None
        self.tokenizer = None
        
    def initialize(self):
        if self.model is None:
            self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
            self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
            self.model.eval()
    
    @torch.cuda.amp.autocast()
    def process_batch(self, captions, batch_size):
        self.initialize()
        all_embeddings = []
        all_masks = []
        
        for i in range(0, len(captions), batch_size):
            sub_batch = captions[i:min(i + batch_size, len(captions))]
            
            inputs = self.tokenizer(
                sub_batch,
                padding='max_length',
                truncation=True,
                max_length=77,
                return_tensors="pt"
            )
            
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model.get_text_features(**inputs)
                embeddings = outputs.cpu().numpy()
                masks = inputs['attention_mask'].cpu().numpy()
                
                all_embeddings.append(embeddings)
                all_masks.append(masks)
                
                # GPU cache
                torch.cuda.empty_cache()
        
        return np.concatenate(all_embeddings, axis=0), np.concatenate(all_masks, axis=0)

##### function to parse the args and initialize in worker 

def process_on_gpu(args):
    """Handle processing for a single GPU"""
    captions, gpu_id, batch_size = args
    worker = GPUWorker(gpu_id)
    return worker.process_batch(captions, batch_size)

# ------------------------------------
# Main Function
# ---------------------------------
def process_and_save_dataset(args):
    print("\nInitialization Phase:")
    num_available_gpus = get_available_gpus()
    if num_available_gpus == 0:
        raise RuntimeError("No GPUs available!")
    
    if args.num_gpus is None:
        args.num_gpus = num_available_gpus
    elif args.num_gpus > num_available_gpus:
        print(f"Warning: Requested {args.num_gpus} GPUs but only {num_available_gpus} available")
        args.num_gpus = num_available_gpus
    
    print(f"Using {args.num_gpus} GPUs")
    print("Loading dataset")
    with open(args.input, 'rb') as f:
        data = pickle.load(f)
    if args.num_captions:
        data = data[:args.num_captions]
        print(f"Processing subset of {args.num_captions} captions")
    
    image_embeddings = np.stack([sample['embedding'] for sample in data])
    captions = [sample['caption'] for sample in data]
    print(f"Total captions to process: {len(captions)}")
    
    # Spliting data across GPUs 
    num_samples = len(captions)
    chunk_size = (num_samples + args.num_gpus - 1) // args.num_gpus  # Ceiling division
    caption_chunks = []
    
    for i in range(0, num_samples, chunk_size):
        chunk = captions[i:min(i + chunk_size, num_samples)]
        if chunk:  # adding non-empty chunks
            caption_chunks.append(chunk)
    
    actual_num_gpus = len(caption_chunks)  # This might be less than args.num_gpus
    print(f"Actual number of GPU workers being used: {actual_num_gpus}")

    gpu_data = [(chunk, gpu_id % actual_num_gpus, args.batch_size) 
                for gpu_id, chunk in enumerate(caption_chunks)]
    
    # chunks in parallel across GPUs
    print("\nProcessing Phase:")
    
    mp.set_start_method('spawn', force=True)
    
    with mp.Pool(actual_num_gpus) as pool:
        # Initialize models (dummy run)
        for gpu_id in range(actual_num_gpus):
            worker = GPUWorker(gpu_id)
            worker.initialize()
            del worker
            torch.cuda.empty_cache()
        
        # Start timing after initialization
        process_start = time.perf_counter()
        
        results = list(tqdm(
            pool.imap(process_on_gpu, gpu_data),
            total=len(gpu_data)
        ))
        
        process_time = time.perf_counter() - process_start
    
    print(f"\nActual processing time: {process_time:.2f} seconds")
    print(f"Average time per caption: {(process_time/len(captions)):.3f} seconds")

    print("\nFinalization Phase:")
    text_embeddings = np.concatenate([r[0] for r in results], axis=0)
    attention_masks = np.concatenate([r[1] for r in results], axis=0)
    
    processed_data = {
        'image_embeddings': image_embeddings,
        'text_embeddings': text_embeddings,
        'attention_masks': attention_masks,
        'captions': captions
    }
    
    print("\nSaving")
    with open(args.output, 'wb') as f:
        pickle.dump(processed_data, f)
    
    print("\nDataset Information:")
    print(f"Image embeddings shape: {image_embeddings.shape}")
    print(f"Text embeddings shape: {text_embeddings.shape}")
    print(f"Attention masks shape: {attention_masks.shape}")
    print(f"Number of captions processed: {len(captions)}")

if __name__ == "__main__":
    args = parse_arguments()
    process_and_save_dataset(args)