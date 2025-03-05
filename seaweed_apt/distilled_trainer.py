from logger import logger, debug_memory
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

from wan.modules.model import WanModel
from wan.utils.utils import str2bool
import gc
import sys

torch.cuda.set_per_process_memory_fraction(0.8)

def train_consistency_distillation(
    config,
    train_dataloader,
    checkpoint_dir,
    output_dir,
    device,
    accelerator,
    num_epochs=10,
    learning_rate=5e-6,
    cfg_scale=7.5,
    save_interval=350,
    use_wandb=False,
    project_name="wan-consistency-distillation",
    run_name=None,
    use_gradient_checkpointing=True,
    gradient_accumulation_steps=8
):
    logger.debug("Initializing consistency distillation training...")
    os.makedirs(output_dir, exist_ok=True)

    if use_wandb and accelerator.is_main_process:
        wandb.init(project=project_name, name=run_name, config={
            "learning_rate": learning_rate, "num_epochs": num_epochs, "save_interval": save_interval,
            "method": "consistency_distillation", "use_gradient_checkpointing": use_gradient_checkpointing,
            "gradient_accumulation_steps": gradient_accumulation_steps
        })

    # Load and prepare distilled model
    distilled_model = WanModel.from_pretrained(checkpoint_dir, use_checkpoint=use_gradient_checkpointing)
    optimizer = optim.RMSprop(distilled_model.parameters(), lr=learning_rate, alpha=0.9)
    distilled_model, optimizer, train_dataloader = accelerator.prepare(distilled_model, optimizer, train_dataloader)

    distilled_model.train()
    ema_model = WanModel.from_pretrained(checkpoint_dir, use_checkpoint=use_gradient_checkpointing)
    ema_model.eval()
    ema_decay = 0.995

    def update_ema(target_model, source_model, decay):
        with torch.no_grad():
            for target_param, source_param in zip(target_model.parameters(), source_model.parameters()):
                target_param.data.mul_(decay).add_(source_param.data, alpha=1 - decay)

    total_loss = 0.0
    step = 0

    for epoch in range(num_epochs):
        logger.debug(f"Starting epoch {epoch+1}/{num_epochs}")
        debug_memory(f"Before epoch {epoch+1}")
        for batch_idx, batch in enumerate(tqdm(train_dataloader)):
            noise, positive_contexts, v_teacher = batch
            logger.debug(f"Batch {batch_idx}: noise shape={noise.shape}, "
                         f"positive_contexts shape={positive_contexts.shape}, "
                         f"v_teacher shape={v_teacher.shape}")
            debug_memory(f"After batch unpack - Batch {batch_idx}")

            with accelerator.accumulate(distilled_model):
                noise = noise.to(device)  # [B, 16, 1, 60, 104]
                context = positive_contexts.to(device)  # [B, 512, 4096]
                v_teacher = v_teacher.to(device)  # [B, 16, 1, 60, 104]
                debug_memory(f"After moving tensors to device - Batch {batch_idx}")

                patch_size = distilled_model.patch_size  # (1, 2, 2)
                seq_len = (noise.shape[2] // patch_size[0]) * \
                          (noise.shape[3] // patch_size[1]) * \
                          (noise.shape[4] // patch_size[2])  # e.g., 1560
                logger.debug(f"Computed seq_len for batch: {seq_len}")

                timestep = torch.ones(noise.shape[0], device=device) * config.num_train_timesteps
                logger.debug(f"Timestep shape: {timestep.shape}, value: {timestep[0].item()}")

                # Student prediction
                logger.debug("Running student model prediction")
                v_student_output = distilled_model(noise, t=timestep, context=context, seq_len=seq_len)
                v_student = v_student_output[0]  # Extract the first element (prediction tensor)
                logger.debug(f"Student prediction shape: {v_student.shape}")
                debug_memory(f"After student prediction - Batch {batch_idx}")

                # MSE loss
                logger.debug("Calculating MSE loss")
                loss = F.mse_loss(v_student, v_teacher)
                logger.debug(f"Loss value: {loss.item()}")
                debug_memory(f"After loss calculation - Batch {batch_idx}")

                # Backpropagation
                logger.debug("Starting backpropagation")
                accelerator.backward(loss)
                logger.debug("Completed backpropagation")
                debug_memory(f"After backpropagation - Batch {batch_idx}")

                if (batch_idx + 1) % gradient_accumulation_steps == 0 or (batch_idx + 1) == len(train_dataloader):
                    logger.debug("Updating optimizer")
                    optimizer.step()
                    optimizer.zero_grad()
                    debug_memory(f"After optimizer step - Batch {batch_idx}")

                    logger.debug("Updating EMA model")
                    ema_model.to(device)
                    update_ema(ema_model, distilled_model, ema_decay)
                    ema_model.to('cpu')
                    debug_memory(f"After EMA update - Batch {batch_idx}")

                # Cleanup
                logger.debug("Cleaning up GPU memory")
                del v_student, v_student_output, noise, context, v_teacher
                torch.cuda.empty_cache()
                debug_memory(f"After cleanup - Batch {batch_idx}")

            total_loss += loss.item()
            step += 1

            if use_wandb and accelerator.is_main_process and batch_idx % 5 == 0:
                wandb.log({"step": step, "batch_loss": loss.item(), "avg_loss": total_loss / (batch_idx + 1), "epoch": epoch + 1})

            if accelerator.is_main_process and step % save_interval == 0:
                checkpoint_path = f"{output_dir}/consistency_model_step_{step}.pt"
                logger.debug(f"Saving checkpoint to {checkpoint_path}")
                unwrapped_ema = accelerator.unwrap_model(ema_model)
                torch.save(unwrapped_ema.state_dict(), checkpoint_path)
                logger.debug(f"Saved EMA checkpoint to {checkpoint_path}")

        if accelerator.is_main_process:
            checkpoint_path = f"{output_dir}/consistency_model_epoch_{epoch+1}.pt"
            logger.debug(f"Saving epoch checkpoint to {checkpoint_path}")
            unwrapped_ema = accelerator.unwrap_model(ema_model)
            torch.save(unwrapped_ema.state_dict(), checkpoint_path)
            logger.debug(f"Saved EMA epoch checkpoint to {checkpoint_path}")

        logger.debug(f"Completed epoch {epoch+1}/{num_epochs}")
        total_loss = 0.0

    if accelerator.is_main_process:
        final_path = f"{output_dir}/consistency_model_final.pt"
        logger.debug(f"Saving final EMA model to {final_path}")
        unwrapped_ema = accelerator.unwrap_model(ema_model)
        torch.save(unwrapped_ema.state_dict(), final_path)
        logger.debug(f"Saved final EMA consistency model to {final_path}")

    if use_wandb and accelerator.is_main_process:
        wandb.finish()

    return accelerator.unwrap_model(ema_model)

# Dataset (unchanged)
class TextVideoDataset(torch.utils.data.Dataset):
    def __init__(self, data_path):
        data_dict = torch.load(data_path, map_location='cpu')
        self.noise = data_dict['noise']  # [100, 16, 1, 60, 104]
        self.positive_contexts = data_dict['positive_contexts']  # List of [512, 4096]
        self.v_teacher = data_dict['v_teacher']  # [100, 16, 1, 60, 104]
        self.num_samples = len(self.noise)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.noise[idx], self.positive_contexts[idx], self.v_teacher[idx]

if __name__ == "__main__":
    import argparse
    from wan.configs import t2v_14B, t2v_1_3B

    parser = argparse.ArgumentParser(description="Train consistency distillation for Seaweed-APT")
    parser.add_argument("--checkpoint_dir", type=str, required=True, help="Path to Wan T2V model checkpoints")
    parser.add_argument("--output_dir", type=str, default="./output", help="Output directory for checkpoints")
    parser.add_argument("--device_id", type=int, default=0, help="CUDA device ID")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=5e-6, help="Learning rate (paper: 5e-6)")
    parser.add_argument("--cfg_scale", type=float, default=7.5, help="CFG scale (paper: 7.5)")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--use_wandb", type=str2bool, default=True, help="Use Weights & Biases")
    parser.add_argument("--wandb_project", type=str, default="seaweed-apt-distillation", help="WandB project name")
    parser.add_argument("--wandb_run_name", type=str, default=None, help="WandB run name")
    parser.add_argument("--config_file", type=str, default="./config.yaml", help="Path to config file")
    parser.add_argument("--save_interval", type=int, default=350, help="Save interval (paper: 350 updates)")
    parser.add_argument("--t5_cpu", action="store_true", default=False, help="Whether to place T5 model on CPU.")
    parser.add_argument("--dit_fsdp", action="store_true", default=False, help="Whether to use FSDP for DiT.")
    parser.add_argument("--ulysses_size", type=int, default=1, help="The size of the ulysses parallelism in DiT.")
    parser.add_argument("--ring_size", type=int, default=1, help="The size of the ring attention parallelism in DiT.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Steps for gradient accumulation")
    parser.add_argument("--t5_fsdp", action="store_true", default=False, help="Whether to use FSDP for T5.")
    args = parser.parse_args()

    if args.config_file and os.path.exists(args.config_file):
        config = OmegaConf.load(args.config_file)
        args_dict = vars(args)
        for key, value in config.items():
            if key not in args_dict or args_dict[key] is None:
                args_dict[key] = value
        args = argparse.Namespace(**args_dict)

    accelerator = Accelerator(mixed_precision="bf16")
    device = accelerator.device

    config = t2v_1_3B
    if not hasattr(config, 'seq_len'):
        config.seq_len = 1560  # Computed based on 480x832 resolution

    data_file = "dummy_data_480x832.pt"
    if not os.path.exists(data_file):
        logger.error(f"Required file {data_file} not found.")
        logger.info("Please run generate_batch to create the dummy data with contexts first.")
        sys.exit(1)

    train_dataset = TextVideoDataset(data_file)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=1, pin_memory=True)

    distilled_model = train_consistency_distillation(
        config=config,
        train_dataloader=train_dataloader,
        checkpoint_dir=args.checkpoint_dir,
        output_dir=args.output_dir,
        device=device,
        accelerator=accelerator,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        cfg_scale=args.cfg_scale,
        save_interval=args.save_interval,
        use_wandb=args.use_wandb,
        project_name=args.wandb_project,
        run_name=args.wandb_run_name,
        gradient_accumulation_steps=args.gradient_accumulation_steps
    )