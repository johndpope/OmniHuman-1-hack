# SEAWEED

### Key Points
- The functional specification for recreating the code from the paper "Diffusion Adversarial Post-Training for One-Step Video Generation" involves breaking down the process into components like a pre-trained diffusion model, deterministic distillation, and adversarial training with a modified discriminator and approximated R1 regularization.
- It enables one-step video generation at high resolution (1280×720, 24fps) in real time, a significant improvement over previous methods requiring multiple steps.
- **Surprising detail:** The method can generate 2-second videos in one step on a single H100 GPU in 6.03 seconds, and with 8 GPUs, it runs in real time, showcasing impressive efficiency.

---

### Overview
This specification outlines the process to recreate the code for the paper "Diffusion Adversarial Post-Training for One-Step Video Generation" by Shanchuan Lin et al., published by ByteDance Seed and available at [seaweed-apt.com](https://seaweed-apt.com). The paper introduces Adversarial Post-Training (APT) to convert a text-to-video diffusion model into a one-step generator, significantly reducing generation time while maintaining high quality, especially for high-resolution videos.

### Components and Process
The process involves several key components, each designed to transform a multi-step diffusion model into a fast, one-step generator capable of producing 1280×720, 24fps videos in real time.

- **Pre-trained Diffusion Model:** This is a text-to-video diffusion transformer (MMDiT) with 36 layers and 8 billion parameters, capable of generating videos through multiple diffusion steps.
- **Deterministic Distillation:** Initializes a one-step generator using consistency distillation with mean squared error loss, producing blurry but structured samples.
- **Adversarial Training Setup:** Involves training a generator (initialized from the distilled model) and a discriminator (modified from the diffusion model) using standard GAN losses, enhanced with approximated R1 regularization for stability.
- **Discriminator Modifications:** Includes adding cross-attention blocks at specific layers and using a timestep ensemble for input stability.
- **Training Details:** Specifies batch sizes, learning rates, and update counts for both image and video training phases.

This approach allows for efficient one-step generation, with detailed training procedures ensuring stability and quality, particularly for high-resolution video outputs.

---

### Survey Note: Detailed Functional Specification for Code Recreation

#### Introduction
The paper "Diffusion Adversarial Post-Training for One-Step Video Generation" by Shanchuan Lin et al. ([seaweed-apt.com](https://seaweed-apt.com)) presents a novel method, Adversarial Post-Training (APT), to accelerate text-to-video diffusion models for one-step generation. This survey note provides a comprehensive functional specification for recreating the code, detailing each component and process to ensure a complete implementation. The method is particularly notable for achieving real-time generation of high-resolution videos (1280×720, 24fps) in a single forward evaluation, surpassing previous state-of-the-art methods that required multiple steps.

#### Background and Motivation
Diffusion models, widely used for image and video generation, rely on iterative denoising steps, making them computationally expensive and slow. While distillation approaches have enabled one-step image generation, video generation has seen limited progress, especially at high resolutions. The paper addresses this by proposing APT, which uses a pre-trained diffusion model as initialization and trains it adversarially against real data, introducing architectural and training improvements for stability and quality.

#### Functional Specification Components

##### Pre-trained Diffusion Model
The foundation is a pre-trained text-to-video diffusion transformer (MMDiT) with the following specifications:
- **Architecture:** 36-layer transformer, 8 billion parameters.
- **Function:** Generates videos through \( T \) diffusion steps, predicting a velocity field \( v \) to denoise noise samples \( z \) to samples \( x \) using \( x = z - v \).
- **Inputs:** Noise sample \( z \in \mathbb{R}^{t' \times h' \times w' \times c'} \), text condition \( c \), timestep \( t \).
- **Outputs:** Predicted velocity field \( v \).
- This model is trained with a flow-matching objective over a mixture of images and videos at native resolutions in the latent space, enabling both image and video generation.

##### Deterministic Distillation
This step initializes the generator for adversarial training by distilling the diffusion model into a one-step generator:
- **Method:** Uses consistency distillation with mean squared error loss, focusing on predicting the velocity field at the final timestep \( T \).
- **Process:** The distilled model \( \hat{G} \) predicts \( \hat{v} = \hat{G}(z, c, T) \), and the sample is computed as \( \hat{x} = z - \hat{v} \).
- **Output:** A blurry but structurally coherent one-step generator \( G(z, c) = z - \hat{G}(z, c, T) \), used to initialize the generator for subsequent adversarial training.
- **Purpose:** Provides a stable starting point, mitigating the instability of direct adversarial training on diffusion models.

##### Adversarial Training Setup
The adversarial training involves a generator and discriminator, trained in a min-max game to improve one-step generation quality:
- **Generator:**
  - **Initialization:** From the distilled model \( \hat{G} \).
  - **Function:** \( G(z, c) = z - \hat{G}(z, c, T) \), generating samples in one step.
  - **Loss:** Standard GAN generator loss \( L_G = \mathbb{E}_{z \sim N, c \sim T} [\log \sigma(D(G(z, c), c))] \), where \( \sigma \) is the sigmoid function.
- **Discriminator:**
  - **Initialization:** From the pre-trained diffusion model, with modifications for logit production.
  - **Function:** Produces a scalar logit to classify real samples \( x \) vs. generated samples \( G(z, c) \), with loss \( L_D = \mathbb{E}_{x, c \sim T} [\log \sigma(D(x, c))] + \mathbb{E}_{z \sim N, c \sim T} [\log(1 - \sigma(D(G(z, c), c)))] \).
  - **Training:** Alternates with the generator, incorporating approximated R1 regularization for stability.
- **Approximated R1 Regularization:**
  - **Method:** Instead of computing higher-order gradients, perturbs real data \( x \) with Gaussian noise \( N(0, \sigma I) \) to get \( x_{\text{perturbed}} \), and computes loss \( L_{\text{aR1}} = ||D(x, c) - D(x_{\text{perturbed}}, c)||^2_2 \).
  - **Parameters:** Uses \( \lambda = 100 \), \( \sigma = 0.01 \) for images, \( \sigma = 0.1 \) for videos.
  - **Purpose:** Stabilizes training by reducing discriminator gradient on real data, preventing collapse in large-scale transformer models.

##### Discriminator Modifications
To ensure stability and quality, the discriminator is modified as follows:
- **Architecture:** Adds new cross-attention-only transformer blocks at layers 16, 26, and 36. Each block uses a learnable token to cross-attend to all visual tokens, producing a single token output. These are concatenated, normalized, and projected to a scalar logit.
- **Timestep Ensemble:** Instead of using \( t = 0 \), samples \( t \) uniformly from [0, T] and shifts it using \( \text{shift}(t, s) = s \times t / (1 + (s - 1) \times t) \), with \( s = 1 \) for images and \( s = 12 \) for videos. For efficiency, uses a single \( t \) per training sample.
- **Purpose:** Enhances learning capacity and stability, mitigating collapse by leveraging multi-layer features and diverse timestep inputs.

##### Training Details
The training is split into image and video phases, with specific hyperparameters:
- **Image Training:**
  - **Resolution:** 1024px images.
  - **Batch Size:** 9062, achieved with 128–256 H100 GPUs and gradient accumulation.
  - **Learning Rate:** 5e-6 for both generator and discriminator.
  - **EMA Decay Rate:** 0.995, with checkpoint taken after 350 updates to avoid quality degradation.
  - **Optimizer:** RMSProp with \( \alpha = 0.9 \), equivalent to Adam with \( \beta_1 = 0, \beta_2 = 0.9 \), no weight decay or gradient clipping, BF16 mixed precision.
- **Video Training:**
  - **Resolution:** 1280×720, 2 seconds at 24fps.
  - **Batch Size:** 2048, with 1024 H100 GPUs and gradient accumulation.
  - **Learning Rate:** 3e-6 for stability.
  - **Updates:** 300, with generator initialized from image EMA checkpoint, discriminator re-initialized from diffusion weights.
  - **Optimizer:** Same as image training, BF16 mixed precision.
- **Dataset:** Uses the same datasets as the original diffusion model, with re-captions for improved text alignment.

#### Performance and Evaluation
The method achieves significant performance, generating 2-second, 1280×720, 24fps videos in one step, with inference times detailed in Table 1 below. On a single H100 GPU, it takes 6.03 seconds, and with 8 GPUs, it runs in real time (1.97 seconds total).

| # of H100 | Component        | Seconds |
|-----------|------------------|---------|
| 1         | Text Encoder     | 0.28    |
|           | DiT              | 2.65    |
|           | VAE              | 3.10    |
|           | Total            | 6.03    |
| 4         | Text Encoder     | 0.28    |
|           | DiT              | 0.73    |
|           | VAE              | 1.19    |
|           | Total            | 2.20    |
| 8         | Text Encoder     | 0.28    |
|           | DiT              | 0.50    |
|           | VAE              | 1.19    |
|           | Total            | 1.97    |

User studies show improvements in visual fidelity compared to 25-step diffusion models, though with some degradation in structural integrity and text alignment, particularly for videos due to motion complexity.

#### Ablation Studies and Insights
Ablation studies highlight the importance of approximated R1 regularization, with training collapsing without it (discriminator loss reaching zero, leading to colored plate outputs). Deeper discriminators and multi-layer features improve quality, and large batch sizes prevent mode collapse, especially for videos.

#### Conclusion
This functional specification provides a detailed roadmap for implementing the APT method, ensuring all components are clearly defined for code recreation. It supports the generation of high-resolution videos in one step, with training details and performance metrics ensuring reproducibility and scalability.

#### Key Citations
- [seaweed-apt comprehensive video generation study](https://seaweed-apt.com)
