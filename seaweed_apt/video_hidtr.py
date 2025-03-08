import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from logger import logger, TorchDebugger
from logger import logger, TorchDebugger
from graph import VisualizationHook
from wan.modules.model import WanModel
import json
from typing import *

class LSHTable(nn.Module):
    def __init__(self, dim: int, num_tables: int = 10, num_hash_functions: int = 4):
        super().__init__()
        self.dim = dim
        self.num_tables = num_tables
        self.num_hash_functions = num_hash_functions
        self.projection_matrices = nn.Parameter(
            torch.randn(num_tables * num_hash_functions, dim)
        )
        
    def hash(self, points: torch.Tensor, viz_hook: Optional[VisualizationHook] = None, 
             iteration: Optional[int] = None) -> torch.Tensor:
        if points.dim() == 2:
            points = points.unsqueeze(1)
            
        B, N, D = points.shape
        
        P = self.projection_matrices.view(self.num_tables, self.num_hash_functions, -1)
        projections = torch.matmul(points, P.reshape(-1, D).t())
        projections = projections.view(B, N, self.num_tables, self.num_hash_functions)
        binary_hash = (projections > 0).float()
        
        if viz_hook and iteration is not None:
            viz_hook.log_lsh_hash(points, binary_hash, iteration)
            
        return binary_hash
    
class HiDimQuery(nn.Module):
    def __init__(self, dim: int, hidden_dim: int = 256):
        super().__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.reference = nn.Parameter(torch.randn(dim))
        
        # Projections
        self.key_proj = nn.Linear(dim, hidden_dim)
        self.query_proj = nn.Linear(dim, hidden_dim)
        self.value_proj = nn.Linear(dim, hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, dim)
        
        # Adaptive sampling
        self.sampler = AdaptiveDirectionSampling(dim)
        
    def forward(self, points: torch.Tensor, lsh: LSHTable, viz_hook: Optional[VisualizationHook] = None,
                iteration: Optional[int] = None, query_idx: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        if points.dim() == 2:
            points = points.unsqueeze(1)
        
        sample_points = self.sampler(self.reference)
        sample_points = sample_points.unsqueeze(0).expand(points.shape[0], -1, -1)
        
        indices, _ = lsh.query(sample_points, points, k=5)
        
        # Gather neighbors
        B, N_samples, k = indices.shape
        points_expanded = points.view(B, 1, points.shape[1], self.dim).expand(B, N_samples, -1, self.dim)
        indices_expanded = indices.view(B, N_samples, k, 1).expand(B, N_samples, k, self.dim)
        neighbors = torch.gather(points_expanded, 2, indices_expanded)
        
        # Attention computation
        queries = self.query_proj(sample_points)
        keys = self.key_proj(neighbors)
        values = self.value_proj(neighbors)
        
        attention = torch.matmul(queries.unsqueeze(2), keys.transpose(2, 3))
        attention = attention / math.sqrt(self.hidden_dim)
        attention_weights = F.softmax(attention, dim=-1)
        
        if viz_hook and iteration is not None and query_idx is not None:
            viz_hook.log_query(sample_points, neighbors, attention_weights, query_idx, iteration)
        
        attended = torch.matmul(attention_weights, values).squeeze(2)
        output = self.output_proj(attended)
        output = output.mean(dim=1)
        
        attention_weights = attention_weights.squeeze(2).unsqueeze(1)
        return output, attention_weights


class LSHTable(nn.Module):
    def __init__(self, dim: int, num_tables: int = 10, num_hash_functions: int = 4):
        super().__init__()
        self.dim = dim
        self.num_tables = num_tables
        self.num_hash_functions = num_hash_functions
        self.projection_matrices = nn.Parameter(torch.randn(num_tables * num_hash_functions, dim))
        
   
        
    def hash(self, points: torch.Tensor, viz_hook: Optional[VisualizationHook] = None, 
             iteration: Optional[int] = None) -> torch.Tensor:
        if points.dim() == 2:
            points = points.unsqueeze(1)
            
        B, N, D = points.shape
        
        P = self.projection_matrices.view(self.num_tables, self.num_hash_functions, -1)
        projections = torch.matmul(points, P.reshape(-1, D).t())
        projections = projections.view(B, N, self.num_tables, self.num_hash_functions)
        binary_hash = (projections > 0).float()
        
        if viz_hook and iteration is not None:
            viz_hook.log_lsh_hash(points, binary_hash, iteration)
            
        return binary_hash
    
    def query(self, query_points: torch.Tensor, points: torch.Tensor, k: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Find k nearest neighbors
        Args:
            query_points: [batch_size, num_queries, dim] or [batch_size, dim]
            points: [batch_size, num_points, dim] or [batch_size, dim]
        Returns:
            indices: [batch_size, num_queries, k]
            distances: [batch_size, num_queries, k]
        """
        # Ensure 3D inputs
        if query_points.dim() == 2:
            query_points = query_points.unsqueeze(1)
        if points.dim() == 2:
            points = points.unsqueeze(1)
            
        logger.debug(f"LSH Query - query_points shape: {query_points.shape}, points shape: {points.shape}")
        
        # Hash both query and points
        query_hash = self.hash(query_points)  # [B, Nq, T, F]
        points_hash = self.hash(points)       # [B, Np, T, F]
        
        # Flatten hash codes
        B, Nq, T, F = query_hash.shape
        Np = points_hash.shape[1]
        
        query_hash = query_hash.reshape(B, Nq, -1)    # [B, Nq, T*F]
        points_hash = points_hash.reshape(B, Np, -1)  # [B, Np, T*F]
        
        # Compute Hamming distances
        distances = torch.cdist(query_hash, points_hash, p=1)  # [B, Nq, Np]
        
        # Get nearest neighbors
        k = min(k, Np)
        values, indices = torch.topk(distances, k=k, dim=-1, largest=False)
        
        logger.debug(f"Query results - indices shape: {indices.shape}, values shape: {values.shape}")
        return indices, values

class HiDETR(nn.Module):
    def __init__(self, dim: int, num_queries: int = 10, num_layers: int = 6):
        super().__init__()
        self.dim = dim
        self.num_queries = num_queries
                # Visualization hook
        self.viz_hook = VisualizationHook()
        # Components
        self.lsh = LSHTable(dim)
        self.queries = nn.ModuleList([HiDimQuery(dim) for _ in range(num_queries)])
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=dim, nhead=8)
            for _ in range(num_layers)
        ])
        self.output_head = nn.Sequential(
            nn.Linear(dim * num_queries, dim * 2),
            nn.LayerNorm(dim * 2),
            nn.ReLU(),
            nn.Linear(dim * 2, dim)
        )
        

        
    def forward(self, points: torch.Tensor, iteration: Optional[int] = None) -> torch.Tensor:
        # Process queries
        query_outputs = []
        
        for i, query in enumerate(self.queries):
            output, attention = query(points, self.lsh, self.viz_hook, iteration, i)
            query_outputs.append(output)
        
        # Combine and process through transformer
        combined = torch.stack(query_outputs, dim=1)
        
        for i, layer in enumerate(self.layers):
            layer_output = layer(combined)
            if self.viz_hook and iteration is not None:
                self.viz_hook.log_transformer(combined, layer_output, i, iteration)
            combined = layer_output
        
        # Final output processing
        B, Q, D = combined.shape
        flattened = combined.reshape(B, Q * D)
        output = self.output_head(flattened)
        
        if self.viz_hook and iteration is not None:
            self.viz_hook.log_final_output(points, output, iteration)
        
        return output

class AdaptiveDirectionSampling(nn.Module):
    def __init__(self, dim: int, num_directions: int = 8, num_samples_per_direction: int = 4):
        super().__init__()
        self.dim = dim
        self.num_directions = num_directions
        self.num_samples = num_samples_per_direction
        
        # Initialize learnable parameters
        self.direction_vectors = nn.Parameter(torch.randn(num_directions, dim))
        self.sampling_offsets = nn.Parameter(torch.randn(num_directions, num_samples_per_direction))
        self.direction_scales = nn.Parameter(torch.ones(num_directions))
        
        # Normalize initial direction vectors
        with torch.no_grad():
            self.direction_vectors.data = F.normalize(self.direction_vectors.data, dim=1)
            
    def forward(self, reference_point: torch.Tensor) -> torch.Tensor:
        """Generate sampling points around reference_point"""
        logger.debug(f"Generating samples around reference point shape: {reference_point.shape}")
        
        # Scale directions
        scaled_directions = self.direction_vectors * self.direction_scales.unsqueeze(1)
        
        # Generate samples
        samples = []
        for direction, offsets in zip(scaled_directions, self.sampling_offsets):
            samples.extend([reference_point + direction * offset for offset in offsets])
            
        samples = torch.stack(samples)
        logger.debug(f"Generated samples shape: {samples.shape}")
        return samples


from wan.modules.model import WanModel

class HiDETRWanModel(WanModel):
    def __init__(self, *args, dim=16*60*104, num_queries=10, **kwargs):
        super().__init__(*args, **kwargs)
        self.hidetr = HiDETR(dim=dim, num_queries=num_queries)
 
    def usp_attn_forward(self, x, seq_lens, grid_sizes, freqs, dtype=torch.bfloat16):
        # Original qkv computation
        q, k, v = qkv_fn(x)
        q = rope_apply(q, grid_sizes, freqs)
        k = rope_apply(k, grid_sizes, freqs)

        # HiDETR processing
        points = torch.cat([q, k], dim=-1).flatten(2)  # [B, S, D]
        hidetr_output = self.hidetr(points)  # [B, D]
        
        # Reshape and project back
        x = hidetr_output.view_as(q).flatten(2)
        x = self.o(x)
        return x

    def forward(self, x, t, context, seq_len, *args, **kwargs):
        # Override forward to use HiDETR-enhanced attention
        return super().forward(x, t, context, seq_len, *args, **kwargs)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
        # Load base WanModel
        base_model = WanModel.from_pretrained(pretrained_model_name_or_path, *args, **kwargs)
        
        # Initialize HiDETRWanModel
        model = cls(*args, **kwargs)
        
        # Transfer WanModel weights
        state_dict = base_model.state_dict()
        model.load_state_dict(state_dict, strict=False)  # Ignore missing HiDETR keys
        
        return model