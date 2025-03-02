# Seaweed-Wan Implementation Guide

This document outlines the key considerations and steps for implementing the Seaweed APT approach using Wan as the base model.

## Project Overview

The goal is to adapt the Adversarial Post-Training (APT) methodology from the Seaweed paper to the Wan architecture, enabling one-step high-resolution video generation. The original Seaweed APT paper describes a technique for transforming a text-to-video diffusion model into a one-step generator using adversarial training.

## Architectural Comparison

### Seaweed (Original)
- **Base Model**: 36-layer MMDiT with 8 billion parameters
- **Training Process**: Deterministic distillation → Adversarial post-training
- **Discriminator Design**: Modified diffusion model with cross-attention blocks at layers 16, 26, and 36
- **Performance**: Generates 1280×720, 24fps videos in a single step

### Wan
- **Base Model**: 36-layer transformer with 8 billion parameters (T2V-14B variant)
- **Architecture**: Transformer-based with attention blocks, compatible with Seaweed's approach
- **Advantages**: Pre-trained on diverse data, supports both text-to-video and image-to-video generation

## Implementation Steps

### 1. Consistency Distillation Phase

The first step is to create a one-step generator from the original Wan model using consistency distillation:

**Key Components:**
- Target: Train a model to predict the velocity field at the final timestep T
- Loss: Mean squared error between student and teacher velocity predictions
- Teacher: Original Wan model with classifier-free guidance (CFG scale 7.5)
- Student: Initialize from the original Wan model weights

**Implementation Notes:**
- This phase takes the original Wan pre-trained model and distills it for one-step prediction
- The distilled model serves as initialization for the generator in the adversarial phase
- This produces blurry but structurally coherent one-step samples

### 2. Adversarial Post-Training Phase

This phase improves the quality of the one-step generation through adversarial training:

**Key Components:**
- Generator: Initialized from the distilled model
- Discriminator: Modified from original Wan model
- Training Protocol: Two phases (images, then videos)
- Discriminator Architecture: Cross-attention blocks at specific layers
- Stabilization: Approximated R1 regularization

**Implementation Notes:**
- The discriminator architecture must be adapted to extract features from Wan's transformer blocks
- Timestep ensemble technique is crucial for stability
- Approximated R1 regularization is essential to prevent training collapse

### 3. Training Protocol

The training follows a two-phase approach:

**Image Phase:**
- Resolution: 1024px
- Batch Size: 9062 (distributed across GPUs)
- Learning Rate: 5e-6
- Updates: 350
- EMA Decay: 0.995
- R1 Sigma: 0.01

**Video Phase:**
- Resolution: 1280×720, 24fps, 2 seconds
- Batch Size: 2048
- Learning Rate: 3e-6
- Updates: 300
- Generator: Initialized from image EMA checkpoint
- Discriminator: Re-initialized from original weights
- R1 Sigma: 0.1

## Challenges and Solutions

### 1. Training Stability

The approximated R1 regularization is critical for preventing training collapse. The original R1 regularization requires second-order gradients, which is computationally expensive and may not be well-supported with large models. Our approximation perturbs real samples with Gaussian noise and penalizes the discriminator for producing different predictions.

### 2. Discriminator Architecture

Adapting the discriminator to Wan's architecture requires careful consideration of layer placement. We've positioned cross-attention blocks at layers 16, 26, and 36, which extract the multi-level features needed for effective discrimination.

### 3. Large Batch Sizes

The paper emphasizes the importance of large batch sizes to prevent mode collapse. You'll need to use gradient accumulation and distributed training across multiple GPUs to achieve the target batch sizes (9062 for images, 2048 for videos).

## Performance Expectations

When successfully implemented, the Seaweed-Wan model should be able to:

1. Generate 1024px images in a single forward pass
2. Generate 1280×720, 24fps videos (2 seconds) in a single step
3. Run in real-time with 8 GPUs (less than 2 seconds for a 2-second video)

## Recommended Resources

- **GPUs**: At least 8 H100 GPUs for optimal performance
- **Memory**: 80GB+ per GPU
- **Storage**: 1TB+ for model checkpoints and training data
- **Frameworks**: PyTorch with FSDP support for distributed training

## Next Steps

1. Implement the consistency distillation phase
2. Validate the quality of the distilled model
3. Implement the adversarial training phase with the modified discriminator
4. Train on images first, then videos
5. Evaluate one-step generation quality and performance

By following this guide, you should be able to successfully adapt the Seaweed APT methodology to the Wan architecture, enabling efficient one-step video generation.
