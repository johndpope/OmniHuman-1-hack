# Understanding the "Velocity Field" in Diffusion Models

## Intuitive Understanding

The "velocity field" in flow-matching diffusion models is like a map of directions showing how noise should transform into an image or video.

### Analogy: Rivers to Oceans

Imagine a satellite view of rivers flowing to the ocean:

1. **Noisy Image = Scattered Raindrops**: When it rains, water droplets land randomly across the landscape (like random noise in your latent space)

2. **Target Image = Ocean**: The final destination where all water should flow (your desired clean image or video)

3. **Velocity Field = Terrain Contours**: A map showing which way and how fast water should flow at each point on the landscape to reach the ocean efficiently

Multi-step diffusion is like tracking each raindrop as it follows small streams, then larger rivers, and finally reaches the ocean - taking many small steps along the terrain.

One-step generation is like having a "super map" that shows each raindrop exactly how to jump directly to the ocean in a single move - skipping all the intermediate streams and rivers.

## Technical Perspective

The velocity field in this context is a tensor with the same shape as your latent representation. For Wan's implementation with a 14B model:

```
Velocity Field Shape = [B, C, T, H, W]
```

Where:
- B = Batch size
- C = Number of channels in the latent space (16 for Wan's VAE)
- T = Number of frames (e.g., 21 for 2 seconds at 24fps after VAE compression)
- H = Height in latent space (e.g., 90 for 720p after VAE compression)
- W = Width in latent space (e.g., 160 for 1280p after VAE compression)

For a single 1280×720 video with 81 frames, the velocity field would be approximately:
```
[1, 16, 21, 90, 160] = 48,384,000 values
```

This is a substantial tensor, but manageable by modern GPUs.

## What's Actually Happening in Training

During consistency distillation:

1. The original diffusion model predicts a velocity field at the final timestep using multiple steps
2. Your one-step model learns to predict this same velocity field directly from noise
3. The training minimizes the difference between these two predictions

During APT (adversarial post-training):
1. The generator produces sample latents using its one-step prediction
2. The discriminator checks if these look like real images/videos
3. The competition helps the generator improve its velocity predictions

The "magic" of Seaweed is teaching the model to predict this entire complex field in a single forward pass, rather than requiring 40-50 iterations.

## Visual Representation

```
Noise (Z) → [One-Step Generator] → Velocity Field (v) → Sample (x = z - v)
```

In mathematical terms, the flow-matching diffusion model is learning to predict:

v = dx/dt

Where:
- v is the velocity field
- x is the sample at timestep t
- dx/dt represents the rate of change of x with respect to time

The consistency distillation teaches the model to predict the final velocity field directly, while the adversarial training improves the quality and realism of this prediction.
