

# Status
I focus here - it's using wan2.1 architecture as base model to train seaweed.
https://seaweed-apt.com/ - this work would yield a distilled teacher / student diffusion in 1 step.


https://github.com/johndpope/OmniHuman-1-hack/blob/main/SeaweedAPT/Wan2.1/wan.py

I believe the seaweed derived from sd3.5 but that model is slightly off only 32 layers - not 36.

```shell
pip install -r requirements.txt
# you need to agree to ts and cs
git clone https://huggingface.co/stabilityai/stable-diffusion-3.5-medium models
python seaweed.py
```


UPDATE 1. 
I throw hunyuan code at claude 3.7 and ask it to rearchictect things - 


UPDATE 2. 
It seems Wan2.1 is more promising match.



Understanding the Key Components
First, let's look at the key components we need to adapt:

Base Model Architecture: Seaweed APT uses a 36-layer MMDiT with 8B parameters, while Wan appears to be a transformer-based model with similar scale (using a 36-layer transformer with 8B parameters for T2V-14B variant).
Adversarial Post-Training (APT) Process:

Deterministic distillation to initialize the one-step generator
Adversarial training with a discriminator derived from the original model
Custom discriminator architecture with cross-attention blocks at specific layers
Approximated R1 regularization for stability


Training Protocol:

Two-phase training: first on images, then on videos
Specific batch sizes, learning rates, and update counts



Adapting Seaweed's APT to Wan
Here's my proposed approach for rearchitecting Seaweed using Wan:
1. Base Model Preparation
Wan seems to be an ideal candidate for adaptation because:

It operates on a similar scale (8B parameters)
It's a transformer-based model (using WanModel)
It supports both text-to-video and image-to-video generation
It has similar transformer block structures that can be modified for discriminator use

Use the Wan T2V-14B variant as the base model, which has the full 36-layer transformer and 8B parameters matching Seaweed's scale.





### Functional Specification: OmniHuman Recreation



#### Overview
The OmniHuman system is a Diffusion Transformer (DiT)-based framework designed for scalable, end-to-end human video generation conditioned on multiple modalities (text, audio, pose, and reference images). It aims to produce highly realistic human animations supporting various input types (e.g., portrait, half-body, full-body), aspect ratios, and styles, while excelling in gesture generation and object interaction. The system leverages a novel "omni-conditions" training strategy to maximize data utilization and scalability.

This spec outlines the components required to recreate OmniHuman, focusing on modularity and functionality without implementation details.

---

### Component Breakdown

#### 1. Core Model Architecture
**Purpose:** Defines the foundational structure for video generation using a Diffusion Transformer (DiT) backbone.

- **Subcomponent 1.1: Pretrained Base Model Adapter**
  - **Functionality:** Adapts a pretrained video diffusion model (e.g., Seaweed with MMDiT) as the starting point.
  - **Inputs:** Pretrained weights from a general text-to-video model.
  - **Outputs:** Initialized DiT backbone for further conditioning and training.
  - **Requirements:** Supports integration of multiple modalities and retains text-to-video capabilities.

- **Subcomponent 1.2: Latent Space Encoder (3D Causal VAE)**
  - **Functionality:** Compresses input videos and reference images into a latent space for efficient processing.
  - **Inputs:** Raw video frames or reference images at native resolutions.
  - **Outputs:** Latent representations of video and image data.
  - **Requirements:** Preserves spatial and temporal features; supports variable aspect ratios.

- **Subcomponent 1.3: Denoising Network (MMDiT)**
  - **Functionality:** Performs video denoising using a Multi-Modal Diffusion Transformer (MMDiT) architecture.
  - **Inputs:** Noisy latent representations, condition tokens (text, audio, pose, reference).
  - **Outputs:** Denoised video latent representations.
  - **Requirements:** Integrates multi-modal conditions via cross-attention and self-attention; uses flow matching as the training objective.

---

#### 2. Omni-Conditions Module
**Purpose:** Handles the integration of multiple driving and appearance conditions into the DiT backbone.

- **Subcomponent 2.1: Audio Condition Processor**
  - **Functionality:** Extracts and processes audio features for conditioning.
  - **Inputs:** Raw audio waveforms.
  - **Outputs:** Audio tokens aligned with MMDiT hidden size.
  - **Requirements:**
    - Uses wav2vec for feature extraction.
    - Compresses features via an MLP.
    - Concatenates adjacent frame audio features for temporal context.
    - Injects tokens into MMDiT via cross-attention.

- **Subcomponent 2.2: Pose Condition Processor**
  - **Functionality:** Encodes pose heatmap sequences for conditioning.
  - **Inputs:** Pose heatmap sequences (e.g., 2D keypoints).
  - **Outputs:** Pose tokens for motion guidance.
  - **Requirements:**
    - Employs a pose guider for encoding.
    - Concatenates adjacent frame features for temporal continuity.
    - Stacks tokens with noisy latents along the channel dimension.

- **Subcomponent 2.3: Text Condition Processor**
  - **Functionality:** Processes text inputs as weak motion conditions.
  - **Inputs:** Text descriptions of events or scenes.
  - **Outputs:** Text tokens compatible with MMDiT’s text branch.
  - **Requirements:** Retains original MMDiT text conditioning pipeline.

- **Subcomponent 2.4: Appearance Condition Processor**
  - **Functionality:** Encodes reference images to preserve identity and background details.
  - **Inputs:** Single reference image.
  - **Outputs:** Reference tokens for self-attention interaction.
  - **Requirements:**
    - Reuses the DiT backbone to encode the reference image into latent space.
    - Flattens and packs reference tokens with video tokens.
    - Modifies 3D Rotational Position Embeddings (RoPE) to distinguish reference tokens (zero temporal component) from video tokens.
    - Supports motion frames for long video generation by concatenating features.

---

#### 3. Omni-Conditions Training Strategy
**Purpose:** Manages the multi-stage training process to scale up data utilization and balance condition strengths.

- **Subcomponent 3.1: Training Stage Manager**
  - **Functionality:** Orchestrates the three-stage post-training process.
  - **Inputs:** Training configuration (stages, ratios), mixed-condition dataset.
  - **Outputs:** Trained OmniHuman model weights.
  - **Requirements:**
    - **Stage 1:** Trains on text and reference image conditions only (weakest conditions).
    - **Stage 2:** Adds audio conditions, drops pose conditions.
    - **Stage 3:** Includes all conditions (text, reference, audio, pose).
    - Adjusts training ratios progressively (e.g., halves ratios for stronger conditions).

- **Subcomponent 3.2: Condition Ratio Balancer**
  - **Functionality:** Implements training principles for condition weighting.
  - **Inputs:** Condition type (text, audio, pose, reference), training data subsets.
  - **Outputs:** Adjusted training data ratios per stage.
  - **Requirements:**
    - Principle 1: Leverages weaker condition data (e.g., text) to include data filtered out by stronger conditions (e.g., audio, pose).
    - Principle 2: Assigns lower training ratios to stronger conditions (e.g., pose < audio < text) to prevent overfitting and ensure balanced learning.

- **Subcomponent 3.3: Data Scaler**
  - **Functionality:** Expands usable training data by mixing conditions.
  - **Inputs:** Raw video dataset (e.g., 18.7K hours of human-related data).
  - **Outputs:** Filtered and conditioned data subsets for each stage.
  - **Requirements:**
    - Filters data based on aesthetics, quality, motion amplitude.
    - Selects subsets for audio/pose conditions (e.g., 13% of total data) based on lip-sync and pose visibility criteria.

---

#### 4. Data Management Module
**Purpose:** Prepares and organizes training and testing datasets.

- **Subcomponent 4.1: Dataset Preprocessor**
  - **Functionality:** Cleans and formats raw video data for training.
  - **Inputs:** Raw video clips, associated audio, poses, and text annotations.
  - **Outputs:** Processed video latents, condition data (text, audio features, pose heatmaps).
  - **Requirements:** Supports variable resolutions and aspect ratios; aligns data with omni-conditions.

- **Subcomponent 4.2: Test Dataset Generator**
  - **Functionality:** Prepares evaluation datasets for benchmarking.
  - **Inputs:** Public datasets (e.g., CelebV-HQ, RAVDESS, CyberHost test set).
  - **Outputs:** Test samples with reference images, audio, and optional poses.
  - **Requirements:** Matches evaluation setups from Loopy (portrait) and CyberHost (half-body).

---

#### 5. Inference Engine
**Purpose:** Generates human videos from conditioned inputs during deployment.

- **Subcomponent 5.1: Condition Activator**
  - **Functionality:** Selects and activates driving conditions for inference.
  - **Inputs:** User-specified conditions (e.g., audio, pose, text, reference image).
  - **Outputs:** Activated condition tokens for the denoising process.
  - **Requirements:**
    - Audio-driven: Activates text, reference, and audio conditions.
    - Pose-driven: Activates all conditions or disables audio for pose-only.
    - Ensures weaker conditions are activated with stronger ones unless specified otherwise.

- **Subcomponent 5.2: Video Generator**
  - **Functionality:** Produces video segments from conditioned inputs.
  - **Inputs:** Noisy latents, condition tokens, inference parameters (e.g., CFG scale).
  - **Outputs:** Generated video latents (decoded to frames via 3D VAE).
  - **Requirements:**
    - Applies classifier-free guidance (CFG) to audio and text conditions.
    - Uses CFG annealing to reduce wrinkles while maintaining expressiveness.
    - Supports arbitrary video lengths within memory limits.

- **Subcomponent 5.3: Temporal Coherence Handler**
  - **Functionality:** Ensures consistency in long video generation.
  - **Inputs:** Previous video segment’s last frames (e.g., 5 frames).
  - **Outputs:** Motion frames concatenated with current noisy latents.
  - **Requirements:** Maintains identity and temporal continuity across segments.

---

#### 6. Evaluation Suite
**Purpose:** Assesses model performance against baselines and ablation studies.

- **Subcomponent 6.1: Metric Calculator**
  - **Functionality:** Computes quantitative metrics for generated videos.
  - **Inputs:** Generated videos, ground truth data (if available).
  - **Outputs:** Metrics (FID, FVD, IQA, ASE, Sync-C, HKC, HKV).
  - **Requirements:**
    - Visual quality: FID, FVD, IQA (q-align), ASE (q-align).
    - Lip sync: Sync-C.
    - Hand quality: HKC, HKV.

- **Subcomponent 6.2: Baseline Comparator**
  - **Functionality:** Compares OmniHuman against existing methods.
  - **Inputs:** Generated videos from OmniHuman and baselines (e.g., SadTalker, CyberHost).
  - **Outputs:** Comparative performance tables (e.g., Tables 1, 2).
  - **Requirements:** Supports portrait (CelebV-HQ, RAVDESS) and body (CyberHost) datasets.

- **Subcomponent 6.3: Ablation Study Runner**
  - **Functionality:** Tests impact of training ratios and conditions.
  - **Inputs:** Training configurations (e.g., audio ratios: 10%, 50%, 100%).
  - **Outputs:** Visualizations and subjective evaluation results (e.g., Table 3).
  - **Requirements:** Evaluates identity consistency, lip-sync accuracy, visual quality, action diversity.

---

### System Interactions
- **Training Flow:**
  1. Data Management Module prepares mixed-condition dataset.
  2. Omni-Conditions Training Strategy configures stages and ratios.
  3. Core Model Architecture trains with condition tokens from Omni-Conditions Module.
  4. Evaluation Suite assesses intermediate and final model performance.

- **Inference Flow:**
  1. User provides input conditions (e.g., reference image, audio).
  2. Omni-Conditions Module processes conditions into tokens.
  3. Inference Engine generates video latents, ensuring temporal coherence.
  4. Latent Space Encoder decodes latents into final video frames.

---

### Requirements Summary
- **Scalability:** Handles large-scale datasets (e.g., 18.7K hours) and diverse conditions.
- **Flexibility:** Supports multiple modalities, aspect ratios, and body proportions.
- **Performance:** Outperforms baselines in realism, gesture generation, and object interaction.
- **Efficiency:** Balances expressiveness and computational cost via CFG annealing.

---

This functional specification provides a modular blueprint for recreating OmniHuman, aligning with the paper’s innovations in data scaling, multi-modal conditioning, and realistic human animation. Each component is designed to be independently developed and integrated, facilitating collaboration and iterative refinement.