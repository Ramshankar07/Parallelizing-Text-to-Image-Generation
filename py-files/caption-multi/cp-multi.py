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
    parser = argparse.ArgumentParser(description='Process dataset with CLIP model using CPUs')
    parser.add_argument('--input', type=str, required=True, help='Input pickle file path')
    parser.add_argument('--output', type=str, required=True, help='Output pickle file path')
    parser.add_argument('--batch-size', type=int, default=32, 
                       help='Batch size for processing')
    parser.add_argument('--num-workers', type=int, default=None,
                       help='Number of CPU workers to use (default: number of CPU cores)')
    parser.add_argument('--num-captions', type=int, default=None,
                       help='Number of captions to process (default: all)')
    return parser.parse_args()
#############################33333
# Getting number of  cores to check
#########################333333
def get_num_cores():
    """Get the number of CPU cores"""
    return os.cpu_count()
#####################

# Class that initializes the CLIP model and tokenizer and also process the batches and concatinates the result

#############################3
class CPUWorker:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        
    def initialize(self):
        if self.model is None:
            self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
            self.model.eval()
    
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
            
            with torch.no_grad():
                outputs = self.model.get_text_features(**inputs)
                embeddings = outputs.numpy()
                masks = inputs['attention_mask'].numpy()
                
                all_embeddings.append(embeddings)
                all_masks.append(masks)
        
        return np.concatenate(all_embeddings, axis=0), np.concatenate(all_masks, axis=0)

    
##### function to parse the args and initialize in worker 


def process_on_cpu(args):
    captions, batch_size = args
    worker = CPUWorker()
    return worker.process_batch(captions, batch_size)


# ------------------------------------
# Main Function
# ---------------------------------
def process_and_save_dataset(args):
    print("\nInitialization Phase:")
    num_cores = get_num_cores()
    if args.num_workers is None:
        args.num_workers = num_cores
    elif args.num_workers > num_cores:
        print(f"Warning: Requested {args.num_workers} workers but only {num_cores} CPU cores available")
        args.num_workers = num_cores
    
    print(f"Using {args.num_workers} CPU workers")
    print("Loading dataset")
    with open(args.input, 'rb') as f:
        data = pickle.load(f)
    if args.num_captions:
        data = data[:args.num_captions]
        print(f"Processing subset of {args.num_captions} captions")
    
    image_embeddings = np.stack([sample['embedding'] for sample in data])
    captions = [sample['caption'] for sample in data]
    print(f"Total captions to process: {len(captions)}")

    chunk_size = len(captions) // args.num_workers # Chunking the captions
    if chunk_size == 0:
        chunk_size = len(captions)
        args.num_workers = 1
    
    caption_chunks = [captions[i:i + chunk_size] for i in range(0, len(captions), chunk_size)]
    worker_data = [(chunk, args.batch_size) for chunk in caption_chunks]
    print("\nProcessing Phase:")
    print("Processing captions...")
    
    mp.set_start_method('spawn', force=True)
    
    with mp.Pool(args.num_workers) as pool:
        
        process_start = time.perf_counter()
        
        results = list(tqdm(
            pool.imap(process_on_cpu, worker_data),
            total=len(worker_data)
        ))
        
        process_time = time.perf_counter() - process_start
    
    print(f"\nActual processing time: {process_time:.2f} seconds")
    print(f"Average time per caption: {(process_time/len(captions)):.3f} seconds")
    
    print("\nFinalization Phase:")
    print("Combining results...")
    text_embeddings = np.concatenate([r[0] for r in results], axis=0)
    attention_masks = np.concatenate([r[1] for r in results], axis=0)
    
    processed_data = {
        'image_embeddings': image_embeddings,
        'text_embeddings': text_embeddings,
        'attention_masks': attention_masks,
        'captions': captions
    }
    
    print("\nSaving results...")
    with open(args.output, 'wb') as f:
        pickle.dump(processed_data, f)
    
    print("\nDataset Information:")
    print("-" * 50)
    print(f"Image embeddings shape: {image_embeddings.shape}")
    print(f"Text embeddings shape: {text_embeddings.shape}")
    print(f"Attention masks shape: {attention_masks.shape}")
    print(f"Number of captions processed: {len(captions)}")

if __name__ == "__main__":
    args = parse_arguments()
    process_and_save_dataset(args)