# Parallelizing Text-to-Image Generation Using Diffusion

## Team Members:
- Abdul Azeem Syed 
- Ramshankar Bhuvaneswaran 

**Instructor:** Prof. Handan Liu

## Project Overview
This project explores distributed data parallelization strategies for text-to-image generation pipelines. By leveraging multiple CPUs and GPUs, we evaluate the feasibility of reducing reliance on GPUs for preprocessing tasks, enabling cost-effective solutions for large-scale image and text embedding generation.

### Key Goals:
1. Evaluate CPU-based parallel preprocessing against GPU-based preprocessing.
2. Identify trade-offs in terms of speedup, efficiency, and cost.
3. Assess scalability and economic viability in text-to-image pipelines.

---

## Motivation
While GPUs are powerful, their high cost and limited availability often hinder accessibility. This project investigates whether CPU clusters, using distributed frameworks, can serve as viable alternatives for preprocessing stages in text-to-image generation.

---

## Dataset
- **Dataset Used:** MS COCO 2014 (Filtered Subset)
- **Number of Images:** ~83,000 (filtered for people-centric captions)
- **Resolution:** ~640×480 pixels
- **Data Size:** ~13 GB

---

## Methodology
### Preprocessing Tasks
1. **VAE Embedding (Image Preprocessing):** Converts images into latent embeddings.
2. **CLIP Embedding (Text Preprocessing):** Generates text embeddings and attention masks from captions.

### Parallelization Strategies
1. **CPU-Based Parallelization:**
   - Native Multiprocessing
   - Joblib
   - Dask
2. **GPU-Based Parallelization:**
   - Multithreading
   - Distributed data loading

### Full Training Pipeline
- Implements Distributed Data Parallelism (DDP) with U-Net-based architectures.
- Explores mixed precision training for optimization.

---

## Project Structure
### Preprocessing Notebooks
- **`Propress.ipynb`**: Filters the MS COCO dataset for people-centric imagery.
- **`Main.ipynb`**: Prepares image and text embeddings using VAE and CLIP models.

### Caption-Multi Directory
Scripts for converting captions into embeddings using various parallelization methods:
- `cp-multi.py`: Native multiprocessing.
- `jb.py`: Joblib parallelization.
- `multi.py`: GPU-based acceleration.
- `dasky.py`: Dask-based distributed processing.

### Img-Multi Directory
Scripts for converting images into embeddings with similar methods:
- `multi.py`: Native multiprocessing.
- `job.py`: Joblib parallelization.
- `g.py`: GPU-based acceleration.
- `dasky.py`: Dask-based distributed processing.

### Training Scripts
- `train3.py`: Full training pipeline without mixed precision.
- `train-mixed-p.py`: Training with mixed precision for faster execution and resource efficiency.

### Generation Script
- **`generate.py`**: Generates images based on user-provided prompts using the trained model.
  
#### Usage Example:
```bash
python generate.py --prompt "your text" --model-path "path/to/model" \
                   --guidance-scale 2.0 --num-inference-steps 50
```

---

## Results and Analysis
### Metrics Evaluated:
1. Time (Wall-clock processing time)
2. Speedup (Single CPU/GPU vs. multiple CPUs/GPUs)
3. Efficiency (Scaling performance relative to workers)

### Key Findings:
- **CPU Parallelization:** Limited scalability and efficiency, with diminishing returns beyond 2-4 CPUs.
- **GPU Parallelization:** Significant speedups with reasonable efficiency, especially for training and CLIP embedding tasks.
- **Mixed Precision Training:** Improved final loss and reduced training time by ~10-15%.

---

## References
1. Ramesh, A., et al. (2021). Zero-Shot Text-to-Image Generation. ICML.
2. Ho, J., et al. (2020). Denoising Diffusion Probabilistic Models. NeurIPS.
3. Radford, A., et al. (2021). Learning Transferable Visual Models From Natural Language Supervision. ICML.
4. Paszke, A., et al. (2019). PyTorch: An Imperative Style, High-Performance Deep Learning Library. NeurIPS.
5. Micikevicius, P., et al. (2017). Mixed Precision Training. arXiv:1710.03740

---

## Setup and Installation
### Prerequisites
- Python 3.8+
- PyTorch 1.12+
- CUDA Toolkit (for GPU acceleration)
- Required Libraries: `joblib`, `dask`, `transformers`, `torchvision`, `scikit-learn`

---

## Contribution
Contributions are welcome! Feel free to open issues or submit pull requests for improvements or bug fixes.

---

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.
