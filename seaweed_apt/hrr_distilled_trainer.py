from logger import logger,log_tensor_sizes, log_gpu_memory_usage,debug_memory
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import wandb
from omegaconf import OmegaConf
from accelerate import Accelerator

from video_hidtr import HiDETRWanModel
from wan.modules.model import WanModel
from wan.utils.utils import str2bool
import gc
import sys
import torch.random
from torch.cuda.amp import GradScaler, autocast
torch.cuda.set_per_process_memory_fraction(0.8)

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:64"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
class HRRDistiller(nn.Module):
    def __init__(self, dim=16, hrr_dim=4096):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv3d(dim, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(64, 32, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.hrr_predictor = nn.Linear(32 * 60 * 104, hrr_dim)
        self.context_proj = nn.Linear(4096 * 512, hrr_dim)  # Project prompt context

    def forward(self, noise, context, proj_weight):
        # noise: [B, 16, 1, 60, 104], context: [B, 512, 4096]
        enc = self.encoder(noise)  # [B, 32, 1, 60, 104]
        enc_flat = enc.flatten(1)  # [B, 32*60*104]
        
        # Predict HRR fingerprint
        hrr_pred = self.hrr_predictor(enc_flat)  # [B, 4096]
        
        # Incorporate prompt context as key
        context_flat = context.flatten(1)  # [B, 512*4096]
        key = self.context_proj(context_flat)  # [B, 4096]
        
        # Bind noise-derived HRR with context key
        hrr_combined = circular_convolution(hrr_pred, key)  # [B, 4096]
        
        # Decode back to latent
        v_pred = hrr_decode(hrr_combined, key, proj_weight)  # [B, 16, 1, 60, 104]
        return hrr_combined, v_pred

def train_hrr_distillation(config, train_dataloader, output_dir, device, accelerator, num_epochs=10, learning_rate=5e-6):
    from torch.optim import AdamW
    from torch.cuda.amp import GradScaler, autocast
    
    model = HRRDistiller().to(device)
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    model, optimizer, train_dataloader = accelerator.prepare(model, optimizer, train_dataloader)
    scaler = GradScaler()

    for epoch in range(num_epochs):
        for batch in tqdm(train_dataloader):
            noise, pos_ctx, v_teacher, hrr_target, hrr_key, proj_weight = batch
            noise, pos_ctx, v_teacher, hrr_target, hrr_key, proj_weight = (
                noise.to(device), pos_ctx.to(device), v_teacher.to(device),
                hrr_target.to(device), hrr_key.to(device), proj_weight.to(device)
            )
            
            with autocast():
                hrr_pred, v_pred = model(noise, pos_ctx, proj_weight)
                hrr_loss = F.mse_loss(hrr_pred, hrr_target)  # Fingerprint similarity
                v_loss = F.mse_loss(v_pred, v_teacher)  # Latent reconstruction
                loss = 0.5 * hrr_loss + 0.5 * v_loss
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            
            logger.debug(f"Epoch {epoch+1}, Loss: {loss.item()}")
        
        torch.save(model.state_dict(), f"{output_dir}/hrr_distiller_epoch_{epoch+1}.pt")
    
    return model

if __name__ == "__main__":
    from wan.configs import t2v_1_3B
    args = argparse.Namespace(checkpoint_dir="../models/Wan2.1-T2V-1.3B", output_dir="./output", device_id=0, t5_fsdp=False, dit_fsdp=False, ulysses_size=1, ring_size=1, t5_cpu=False, batch_size=1)
    data_dict = generate_batch(t2v_1_3B, args, num_samples=100)
    dataloader, _ = create_dataloader("dummy_data_480x832.pt", batch_size=args.batch_size)
    accelerator = Accelerator(mixed_precision="bf16")
    train_hrr_distillation(t2v_1_3B, dataloader, args.output_dir, accelerator.device, accelerator)