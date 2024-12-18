import torch
from transformers import CLIPTokenizer, CLIPModel
import pickle
import numpy as np
from tqdm import tqdm
import argparse
import os
import time
from datetime import datetime, timedelta
import dask
import dask.array as da
from dask.distributed import Client, LocalCluster, wait
from dask import delayed

######################################3
# Parsing the arguments
############################
def parse_arguments():
    parser = argparse.ArgumentParser(description='Process dataset with CLIP model using Dask')
    parser.add_argument('--input', type=str, required=True, help='Input pickle file path')
    parser.add_argument('--output', type=str, required=True, help='Output pickle file path')
    parser.add_argument('--batch-size', type=int, default=32, 
                       help='Batch size for processing')
    parser.add_argument('--num-workers', type=int, default=None,
                       help='Number of Dask workers to use (default: number of CPU cores)')
    parser.add_argument('--num-captions', type=int, default=None,
                       help='Number of captions to process (default: all)')
    parser.add_argument('--threads-per-worker', type=int, default=None,
                       help='Number of threads per worker (default: auto)')
    return parser.parse_args()


#############################33333
# Getting number of  cores to check
#########################333333

def get_num_cores():
    return os.cpu_count()
#####################

# Class that initializes the CLIP model and tokenizer and also process the batches and concatinates the result

#############################3
class DaskCLIPProcessor:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        
    def initialize(self):
        if self.model is None:
            self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
            self.model.eval()
    
    @delayed
    def process_batch(self, captions):
        """Process a batch of captions using CLIP"""
        self.initialize()
        
        inputs = self.tokenizer(
            captions,
            padding='max_length',
            truncation=True,
            max_length=77,
            return_tensors="pt"
        )
        
        with torch.no_grad():
            outputs = self.model.get_text_features(**inputs)
            embeddings = outputs.numpy()
            masks = inputs['attention_mask'].numpy()
            
        return embeddings, masks
    
####################################    
##### function to initialize dask cluster
#########################################
def setup_dask_client(num_workers, threads_per_worker):
    """Setup and configure Dask client"""
    if threads_per_worker is None:
        threads_per_worker = max(1, get_num_cores() // num_workers)
    
    cluster = LocalCluster(
        n_workers=num_workers,
        threads_per_worker=threads_per_worker,
        memory_limit='20GB',
        memory_target_fraction=0.8,
        memory_spill_fraction=0.9,
        memory_pause_fraction=0.95,
        processes=True
    )
    client = Client(cluster)
    return client, cluster

def create_batches(captions, batch_size):
    """Create batches of captions"""
    for i in range(0, len(captions), batch_size):
        yield captions[i:i + batch_size]
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
    
    print(f"Using {args.num_workers} Dask workers")
    client, cluster = setup_dask_client(args.num_workers, args.threads_per_worker)
    print(f"Dask dashboard available at: {client.dashboard_link}")
    print("Loading dataset")
    with open(args.input, 'rb') as f:
        data = pickle.load(f)
    
    if args.num_captions:
        data = data[:args.num_captions]
        print(f"Processing subset of {args.num_captions} captions")
    captions = [sample['caption'] for sample in data]
    print(f"Total captions to process: {len(captions)}")
    processor = DaskCLIPProcessor()
    

    print("\nProcessing Phase:")
    delayed_results = []
    batch_count = 0
    
    process_start = time.perf_counter()
    for batch in create_batches(captions, args.batch_size):
        delayed_result = processor.process_batch(batch)
        delayed_results.append(delayed_result)
        batch_count += 1
    
    print(f"Created {batch_count} processing tasks")
    print("Processing batches...")
    results = []
    
    with tqdm(total=len(delayed_results)) as pbar:
        for delayed_result in delayed_results:
            result = delayed_result.compute()
            results.append(result)
            pbar.update(1)
    
    process_time = time.perf_counter() - process_start
    
    print(f"\nActual processing time: {process_time:.2f} seconds")
    print(f"Average time per caption: {(process_time/len(captions)):.3f} seconds")
    
    print("\nFinalization Phase:")
    text_embeddings = np.concatenate([r[0] for r in results], axis=0)
    attention_masks = np.concatenate([r[1] for r in results], axis=0)
    
    processed_data = {
        'text_embeddings': text_embeddings,
        'attention_masks': attention_masks,
        'captions': captions
    }
    print("\nSaving")
    with open(args.output, 'wb') as f:
        pickle.dump(processed_data, f)
    print("\nDataset Information:")
    print(f"Text embeddings shape: {text_embeddings.shape}")
    print(f"Attention masks shape: {attention_masks.shape}")
    print(f"Number of captions processed: {len(captions)}")
    

    client.close()
    cluster.close()

if __name__ == "__main__":
    args = parse_arguments()
    process_and_save_dataset(args)