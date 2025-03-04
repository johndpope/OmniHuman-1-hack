import torch
t5_path = "models_t5_umt5-xxl-enc-bf16.pth"
checkpoint = torch.load(t5_path, map_location='cpu')
total_size = 0
for key, tensor in checkpoint.items():
    shape = tensor.shape
    dtype = tensor.dtype
    num_elements = tensor.numel()
    size_gb = tensor.element_size() * num_elements / (1024**3)
    total_size += size_gb
    print(f"{key}: shape={shape}, dtype={dtype}, elements={num_elements}, size={size_gb:.3f} GB")
print(f"Total size: {total_size:.3f} GB")

from transformers import T5EncoderModel
import torch

# Load UMT5-Small encoder-only
model = T5EncoderModel.from_pretrained("google/umt5-small")
model.to(torch.bfloat16)

# Save state dict
torch.save(model.state_dict(), "./models/Wan2.1-T2V-1.3B/umt5_small_enc_bf16.pth")

# Verify size
state_dict = torch.load("./models/Wan2.1-T2V-1.3B/umt5_small_enc_bf16.pth", map_location='cpu', weights_only=True)
total_size = sum(t.element_size() * t.numel() for t in state_dict.values()) / (1024**3)
print(f"UMT5-Small size: {total_size:.3f} GB")  # Expect ~0.44 GB