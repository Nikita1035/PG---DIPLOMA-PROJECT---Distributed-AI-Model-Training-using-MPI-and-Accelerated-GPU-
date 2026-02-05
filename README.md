# Distributed AI Model Training Using MPI & GPU Acceleration

This repository contains an academic project developed as part of the CDAC curriculum, focused on training and comparing deep learning models using serial CPU, single-GPU, and distributed multi-GPU approaches. The project demonstrates how distributed training improves performance while maintaining comparable accuracy.

---

## Project Overview

The objective of this project is to analyze the performance impact of GPU acceleration and distributed training on deep learning workloads. A pretrained MobileNetV2 model is fine-tuned on the CIFAR-10 dataset using transfer learning, and experiments are conducted across different training configurations.

---

## Dataset

- **CIFAR-10**
- 60,000 color images
- 10 image classes
- Images resized from **32×32 to 224×224** to match ImageNet-pretrained model requirements

---

## Model & Approach

- **Model:** MobileNetV2 (pretrained on ImageNet)
- **Technique:** Transfer Learning
- **Modification:** Final classification layer replaced to output 10 CIFAR-10 classes
- **Loss Function:** CrossEntropyLoss
- **Optimizer:** Adam
- **Training Framework:** PyTorch

---

## Training Configurations

### 1. Serial CPU Training
- Baseline implementation
- Single process running on CPU
- Used for performance comparison

**Files:**
- `cifar10_serial_mobilenet_224.py`
- `serial.slurm`
- `logs_cifar10_cpu_*.out / .err`

---

### 2. Single-GPU Training
- CUDA-enabled GPU execution
- Faster training compared to CPU
- Same model and hyperparameters as CPU version

**Files:**
- `cifar10_128batch.py`
- `run_gpu128.sh`
- `cifar10_gpu_*.out`

---

### 3. Distributed Multi-GPU Training (MPI + DDP)
- Data-parallel distributed training
- One MPI process per GPU
- Dataset split across GPUs using `DistributedSampler`
- Gradients synchronized and averaged after backpropagation using PyTorch Distributed Data Parallel (DDP)

**Files:**
- `cifar10_mpi_mobilenet_224.py`
- `cifar10_gpu_parallel.sh`
- `cifar_mpi_gpu_*.out`

---

## Distributed Training Workflow

1. CIFAR-10 dataset is partitioned across GPUs
2. Each GPU trains on its own data subset
3. Gradients are synchronized across processes after backpropagation
4. Model parameters remain consistent across all GPUs
5. Final model and metrics are collected from the master process

---

## Performance Summary

| Training Mode | Hardware | Training Time | Accuracy |
|-------------|---------|---------------|----------|
| Serial | CPU | High | ~96% |
| Parallel | Single GPU | Medium | ~96% |
| Distributed | Multi-GPU (MPI + DDP) | Low | ~95–96% |

> Distributed training achieved significant speedup compared to CPU and single-GPU runs while maintaining comparable accuracy.

---

## Inference

- A simple inference interface was created using **Gradio**
- Allows image upload and class prediction
- Demonstrates end-to-end workflow from training to inference

---

## Technologies Used

- Python
- PyTorch
- Torchvision
- CUDA
- MPI / PyTorch Distributed Data Parallel (DDP)
- SLURM (HPC Job Scheduling)
- Gradio

---

## Key Learnings

- Understanding of transfer learning and fine-tuning
- Practical experience with GPU and distributed training
- Performance comparison across training architectures
- Exposure to HPC environments and job scheduling
- End-to-end deep learning workflow implementation

---

## Author

**Nikita Nasare**

---

## Note

This is an academic project developed for learning and experimentation purposes. The focus is on understanding performance scalability and distributed training concepts rather than production deployment.
