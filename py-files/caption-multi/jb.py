import torch
from transformers import CLIPTokenizer, CLIPModel
from joblib import Parallel, delayed
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
    parser = argparse.ArgumentParser(description='Process dataset with CLIP model using joblib')
    parser.add_argument('--input', type=str, required=True, help='Input pickle file path')
    parser.add_argument('--output', type=str, required=True, help='Output pickle file path')
    parser.add_argument('--batch-size', type=int, default=32, 
                       help='Batch size for processing')
    parser.add_argument('--n-jobs', type=int, default=-1,
                       help='Number of CPU jobs to use (default: all cores)')
    parser.add_argument('--num-captions', type=int, default=None,
                       help='Number of captions to process (default: all)')
    return parser.parse_args()

## initialization of the model 

def init_model():
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    model.eval()
    return model, tokenizer
# -----------------------#
# Processing of every batch which it needs to go through
##-----------------------------
def process_batch(captions, model, tokenizer):
    inputs = tokenizer(
        captions,
        padding='max_length',
        truncation=True,
        max_length=77,
        return_tensors="pt"
    )
    
    with torch.no_grad():
        outputs = model.get_text_features(**inputs)
        embeddings = outputs.numpy()
        masks = inputs['attention_mask'].numpy()
    
    return embeddings, masks

### ------------------------------------------
# Chunking of the whole captions inside the dataset
###--------------------------------------------

def process_chunk(captions, batch_size):
    model, tokenizer = init_model()
    all_embeddings = []
    all_masks = []
    
    for i in range(0, len(captions), batch_size):
        batch = captions[i:min(i + batch_size, len(captions))]
        embeddings, masks = process_batch(batch, model, tokenizer)
        all_embeddings.append(embeddings)
        all_masks.append(masks)
    
    return (np.concatenate(all_embeddings, axis=0),
            np.concatenate(all_masks, axis=0))

# ------------------------------------
# Main Function
# ---------------------------------

def process_and_save_dataset(args):
    print("\nInitialization Phase:")
    print("Loading dataset")
    with open(args.input, 'rb') as f:
        data = pickle.load(f)
    if args.num_captions:
        data = data[:args.num_captions]
        print(f"Processing subset of {args.num_captions} captions")
    
    image_embeddings = np.stack([sample['embedding'] for sample in data])
    captions = [sample['caption'] for sample in data]
    print(f"Total captions to process: {len(captions)}")
    
    n_jobs = args.n_jobs if args.n_jobs > 0 else os.cpu_count()
    chunk_size = max(1, len(captions) // n_jobs)
    caption_chunks = [captions[i:i + chunk_size] 
                     for i in range(0, len(captions), chunk_size)]
    
    print(f"Using {n_jobs} workers with {len(caption_chunks)} chunks")
    print(f"Chunk size: {chunk_size} captions")
    

    print("\nProcessing Phase:")
    print("Processing captions")
    
    process_start = time.perf_counter()
    
    results = Parallel(n_jobs=args.n_jobs, verbose=1)(
        delayed(process_chunk)(chunk, args.batch_size)
        for chunk in caption_chunks
    )
    
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