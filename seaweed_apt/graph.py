import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional
import math
from logger import logger, TorchDebugger
import matplotlib.pyplot as plt
from helper import *

import torch
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Optional

class VisualizationHook:
    def __init__(self, save_dir: str = 'visualizations/hidetr', viz_interval: int = 10):
        """Initialize visualization hook with a save directory and visualization interval"""
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.viz_interval = viz_interval
        print(f"Initialized VisualizationHook. Saving visualizations every {viz_interval} iterations to: {self.save_dir}")
        
    def _should_visualize(self, iteration: int) -> bool:
        """Check if we should visualize this iteration"""
        return iteration % self.viz_interval == 0

    def log_lsh_hash(self, points: torch.Tensor, hash_codes: torch.Tensor, iteration: int):
        """Visualize LSH hashing results"""
        if not self._should_visualize(iteration):
            return
            
        points_np = points.detach().cpu().numpy()
        hash_codes_np = hash_codes.detach().cpu().numpy()
        
        # Take first batch item
        points_sample = points_np[0]
        hash_codes_sample = hash_codes_np[0]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot input points distribution
        im1 = ax1.imshow(points_sample, aspect='auto', cmap='RdBu_r')
        ax1.set_title('Input Points')
        plt.colorbar(im1, ax=ax1)
        
        # Plot hash codes - reshape to 2D
        T, F = hash_codes_sample.shape[-2:]
        im2 = ax2.imshow(hash_codes_sample.reshape(-1, T*F), aspect='auto', cmap='binary')
        ax2.set_title('LSH Hash Codes')
        plt.colorbar(im2, ax=ax2)
        
        plt.suptitle(f'LSH Hashing - Iteration {iteration}')
        plt.tight_layout()
        
        save_path = self.save_dir / f'lsh_hash_{iteration}.png'
        plt.savefig(save_path)
        plt.close()

    def log_query(self, sample_points: torch.Tensor, neighbors: torch.Tensor, 
                 attention_weights: torch.Tensor, query_idx: int, iteration: int):
        """Visualize query processing and attention"""
        if not self._should_visualize(iteration):
            return
            
        # Convert to numpy and move to CPU
        samples_np = sample_points.detach().cpu().numpy()     # [B, N_samples, D]
        neighbors_np = neighbors.detach().cpu().numpy()       # [B, N_samples, k, D]
        attention_np = attention_weights.detach().cpu().numpy()  # [B, 1, N_samples, k]
        
        # Take first batch item
        samples_sample = samples_np[0]      # [N_samples, D]
        neighbors_sample = neighbors_np[0]   # [N_samples, k, D]
        attention_sample = attention_np[0, 0]  # [N_samples, k]
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot sample points
        im1 = axes[0, 0].imshow(samples_sample, aspect='auto', cmap='RdBu_r')
        plt.colorbar(im1, ax=axes[0, 0])
        axes[0, 0].set_title('Sample Points')
        axes[0, 0].set_xlabel('Feature Dimension')
        axes[0, 0].set_ylabel('Sample Index')
        
        # Plot neighbors - reshape to 2D
        N_samples, k, D = neighbors_sample.shape
        im2 = axes[0, 1].imshow(neighbors_sample.reshape(N_samples*k, D), 
                               aspect='auto', cmap='RdBu_r')
        plt.colorbar(im2, ax=axes[0, 1])
        axes[0, 1].set_title('Selected Neighbors')
        axes[0, 1].set_xlabel('Feature Dimension')
        axes[0, 1].set_ylabel('(Sample, Neighbor) Pair')
        
        # Plot attention pattern as N_samples x k heatmap
        im3 = axes[1, 0].imshow(attention_sample, aspect='auto', cmap='viridis')
        plt.colorbar(im3, ax=axes[1, 0])
        axes[1, 0].set_title('Attention Pattern')
        axes[1, 0].set_xlabel('Neighbor Index')
        axes[1, 0].set_ylabel('Sample Index')
        
        # Plot attention distribution
        axes[1, 1].hist(attention_sample.flatten(), bins=50, density=True)
        axes[1, 1].set_title('Attention Distribution')
        axes[1, 1].set_xlabel('Attention Weight')
        axes[1, 1].set_ylabel('Density')
        
        plt.suptitle(f'Query {query_idx} Processing - Iteration {iteration}')
        plt.tight_layout()
        
        save_path = self.save_dir / f'query_{query_idx}_iter_{iteration}.png'
        plt.savefig(save_path)
        plt.close()

    def log_transformer(self, input_features: torch.Tensor, output_features: torch.Tensor, 
                       layer_idx: int, iteration: int):
        """Visualize transformer layer processing"""
        if not self._should_visualize(iteration):
            return
            
        input_np = input_features.detach().cpu().numpy()
        output_np = output_features.detach().cpu().numpy()
        
        # Take first batch item
        input_sample = input_np[0]
        output_sample = output_np[0]
        
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 5))
        
        # Plot input features
        im1 = ax1.imshow(input_sample, aspect='auto', cmap='RdBu_r')
        plt.colorbar(im1, ax=ax1)
        ax1.set_title('Layer Input')
        
        # Plot output features
        im2 = ax2.imshow(output_sample, aspect='auto', cmap='RdBu_r')
        plt.colorbar(im2, ax=ax2)
        ax2.set_title('Layer Output')
        
        # Plot difference
        diff = output_sample - input_sample
        im3 = ax3.imshow(diff, aspect='auto', cmap='RdBu_r')
        plt.colorbar(im3, ax=ax3)
        ax3.set_title('Feature Difference')
        
        plt.suptitle(f'Transformer Layer {layer_idx} - Iteration {iteration}')
        plt.tight_layout()
        
        save_path = self.save_dir / f'transformer_layer_{layer_idx}_iter_{iteration}.png'
        plt.savefig(save_path)
        plt.close()

    def log_final_output(self, input_points: torch.Tensor, final_output: torch.Tensor, 
                        iteration: int):
        """Visualize final model output compared to input"""
        if not self._should_visualize(iteration):
            return
            
        from helper import detailed_model_output_visualization
        
        detailed_model_output_visualization(
            input_points,
            final_output,
            save_dir=str(self.save_dir),
            iteration=iteration
        )

# class VisualizationHook:
#     def __init__(self,save_dir='visualizations/hidetr'):
#         self.save_dir = Path(save_dir)
#         self.save_dir.mkdir(parents=True, exist_ok=True)
#         self.attention_patterns = []
#         self.lsh_buckets = []
#         self.training_stats = []
#         self.query_stats = []

#     def log_lsh_hash(self, points: torch.Tensor, hash_codes: torch.Tensor, iteration: int):
#         """Visualize LSH hashing process"""
#         points_np = points.detach().cpu().numpy()
#         hash_codes_np = hash_codes.detach().cpu().numpy()
        
#         # Take first batch item
#         points_sample = points_np[0]
#         hash_codes_sample = hash_codes_np[0]
        
#         plt.figure(figsize=(15, 5))
        
#         # Plot input point pattern
#         plt.subplot(121)
#         plt.imshow(points_sample.reshape(1, -1), aspect='auto', cmap='RdBu_r')
#         plt.colorbar()
#         plt.title('Input Point Pattern')
        
#         # Plot hash codes
#         plt.subplot(122)
#         plt.imshow(hash_codes_sample.reshape(-1, hash_codes_sample.shape[-1]), 
#                   aspect='auto', cmap='binary')
#         plt.colorbar()
#         plt.title('LSH Hash Codes')
        
#         plt.suptitle(f'LSH Hashing - Iteration {iteration}')
#         plt.tight_layout()
#         plt.savefig(self.save_dir / f'lsh_hash_{iteration}.png')
#         plt.close()

#     def log_query(self, sample_points: torch.Tensor, neighbors: torch.Tensor, 
#                  attention_weights: torch.Tensor, query_idx: int, iteration: int):
#         """Visualize query processing"""
#         samples_np = sample_points.detach().cpu().numpy()
#         neighbors_np = neighbors.detach().cpu().numpy()
#         attention_np = attention_weights.detach().cpu().numpy()
        
#         # Take first batch item
#         samples_sample = samples_np[0]
#         neighbors_sample = neighbors_np[0]
#         attention_sample = attention_np[0]
        
#         fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
#         # Plot sample points
#         im1 = axes[0, 0].imshow(samples_sample.reshape(1, -1), aspect='auto', cmap='RdBu_r')
#         plt.colorbar(im1, ax=axes[0, 0])
#         axes[0, 0].set_title('Sample Points')
        
#         # Plot neighbors
#         im2 = axes[0, 1].imshow(neighbors_sample.reshape(-1, neighbors_sample.shape[-1]), 
#                                aspect='auto', cmap='RdBu_r')
#         plt.colorbar(im2, ax=axes[0, 1])
#         axes[0, 1].set_title('Neighbor Points')
        
#         # Plot attention pattern
#         im3 = axes[1, 0].imshow(attention_sample.squeeze(), aspect='auto', cmap='viridis')
#         plt.colorbar(im3, ax=axes[1, 0])
#         axes[1, 0].set_title('Attention Pattern')
        
#         # Plot attention distribution
#         axes[1, 1].hist(attention_sample.flatten(), bins=50)
#         axes[1, 1].set_title('Attention Distribution')
        
#         plt.suptitle(f'Query {query_idx} Processing - Iteration {iteration}')
#         plt.tight_layout()
#         plt.savefig(self.save_dir / f'query_{query_idx}_iter_{iteration}.png')
#         plt.close()

#     def log_transformer(self, input_features: torch.Tensor, output_features: torch.Tensor, 
#                        layer_idx: int, iteration: int):
#         """Visualize transformer layer processing"""
#         input_np = input_features.detach().cpu().numpy()
#         output_np = output_features.detach().cpu().numpy()
        
#         # Take first batch item
#         input_sample = input_np[0]
#         output_sample = output_np[0]
        
#         fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 5))
        
#         # Plot input features
#         im1 = ax1.imshow(input_sample, aspect='auto', cmap='RdBu_r')
#         plt.colorbar(im1, ax=ax1)
#         ax1.set_title('Layer Input')
        
#         # Plot output features
#         im2 = ax2.imshow(output_sample, aspect='auto', cmap='RdBu_r')
#         plt.colorbar(im2, ax=ax2)
#         ax2.set_title('Layer Output')
        
#         # Plot difference
#         diff = output_sample - input_sample
#         im3 = ax3.imshow(diff, aspect='auto', cmap='RdBu_r')
#         plt.colorbar(im3, ax=ax3)
#         ax3.set_title('Feature Difference')
        
#         plt.suptitle(f'Transformer Layer {layer_idx} - Iteration {iteration}')
#         plt.tight_layout()
#         plt.savefig(self.save_dir / f'transformer_layer_{layer_idx}_iter_{iteration}.png')
#         plt.close()

#     def log_final_output(self, input_points: torch.Tensor, final_output: torch.Tensor, 
#                         iteration: int):
#         """Visualize final model output compared to input"""
#         detailed_model_output_visualization(
#             input_points,
#             final_output,
#             save_dir=str(self.save_dir),
#             iteration=iteration
#         )

#     def log_attention(self, attention_weights: torch.Tensor, query_idx: int):
#         """Log attention pattern for visualization
#         Args:
#             attention_weights: [batch_size, num_heads, query_len, key_len]
#             query_idx: current query index
#         """
#         with torch.no_grad():
#             # Average across heads and batch
#             avg_attention = attention_weights.mean(dim=(0, 1)).cpu()
            
#             # Convert to list of records
#             records = [
#                 {
#                     "queryId": query_idx,
#                     "targetId": target_idx,
#                     "strength": float(strength)
#                 }
#                 for target_idx, strength in enumerate(avg_attention.flatten())
#             ]
#             self.attention_patterns.extend(records)
            
#     def log_lsh_buckets(self, hash_codes: torch.Tensor):
#         """Log LSH bucket distribution
#         Args:
#             hash_codes: [batch_size, num_points, num_tables, num_hash_functions]
#         """
#         with torch.no_grad():
#             # Convert binary codes to bucket indices
#             B, N, T, F = hash_codes.shape
#             flat_codes = hash_codes.reshape(B*N, T*F)
            
#             # Count points per bucket
#             unique_codes, counts = torch.unique(flat_codes, return_counts=True, dim=0)
            
#             # Convert to records
#             records = [
#                 {
#                     "bucket": i,
#                     "count": int(count),
#                 }
#                 for i, count in enumerate(counts)
#             ]
#             self.lsh_buckets.extend(records)
            
#     def log_training(self, epoch: int, loss: float, accuracy: float):
#         """Log training progress"""
#         self.training_stats.append({
#             "epoch": epoch,
#             "loss": float(loss),
#             "accuracy": float(accuracy)
#         })
        
#     def log_query_stats(self, query_outputs: torch.Tensor):
#         """Analyze query behavior
#         Args:
#             query_outputs: [batch_size, num_queries, dim]
#         """
#         with torch.no_grad():
#             B, Q, D = query_outputs.shape
            
#             # Compute influence (average L2 norm of query outputs)
#             influence = torch.norm(query_outputs, dim=-1).mean(0)
            
#             # Compute diversity (average cosine distance between queries)
#             normalized = F.normalize(query_outputs, dim=-1)
#             similarity = torch.matmul(normalized, normalized.transpose(-2, -1))
#             diversity = 1 - similarity.mean(dim=(0,2))
            
#             # Add records
#             records = [
#                 {
#                     "queryId": i,
#                     "influence": float(inf),
#                     "diversity": float(div)
#                 }
#                 for i, (inf, div) in enumerate(zip(influence, diversity))
#             ]
#             self.query_stats.extend(records)
            
#     def get_visualization_data(self):
#         """Return all visualization data"""
#         return {
#             "attentionData": self.attention_patterns,
#             "bucketData": self.lsh_buckets,
#             "trainingStats": self.training_stats,
#             "queryStats": self.query_stats
#         }
    
#     def visualize_lsh_hashing(self, points: torch.Tensor, hash_codes: torch.Tensor, 
#                             iteration: int, name: str = 'lsh_hashing'):
#         """Visualize LSH hashing process"""
#         points = points.detach().cpu().numpy()
#         hash_codes = hash_codes.detach().cpu().numpy()
        
#         # Take first batch item
#         points_sample = points[0]
#         hash_codes_sample = hash_codes[0]
        
#         fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
#         # Visualize input points distribution
#         im1 = ax1.imshow(points_sample.reshape(1, -1), aspect='auto', cmap='RdBu_r')
#         ax1.set_title('Input Points Distribution')
#         plt.colorbar(im1, ax=ax1)
        
#         # Visualize hash codes
#         im2 = ax2.imshow(hash_codes_sample.reshape(-1, hash_codes_sample.shape[-1]), 
#                         aspect='auto', cmap='binary')
#         ax2.set_title('LSH Hash Codes')
#         plt.colorbar(im2, ax=ax2)
        
#         plt.suptitle(f'LSH Hashing Visualization - Iteration {iteration}')
#         plt.tight_layout()
        
#         plt.savefig(self.save_dir / f'{name}_{iteration}.png')
#         plt.close()

#     def visualize_adaptive_sampling(self, reference_point: torch.Tensor, samples: torch.Tensor,
#                                   iteration: int, name: str = 'adaptive_sampling'):
#         """Visualize adaptive sampling process"""
#         ref_point = reference_point.detach().cpu().numpy()
#         samples = samples.detach().cpu().numpy()
        
#         fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
#         # Visualize reference point
#         im1 = ax1.imshow(ref_point.reshape(1, -1), aspect='auto', cmap='RdBu_r')
#         ax1.set_title('Reference Point')
#         plt.colorbar(im1, ax=ax1)
        
#         # Visualize sampled points
#         im2 = ax2.imshow(samples.reshape(-1, samples.shape[-1]), aspect='auto', cmap='RdBu_r')
#         ax2.set_title('Adaptive Samples')
#         plt.colorbar(im2, ax=ax2)
        
#         plt.suptitle(f'Adaptive Sampling Visualization - Iteration {iteration}')
#         plt.tight_layout()
        
#         plt.savefig(self.save_dir / f'{name}_{iteration}.png')
#         plt.close()

#     def visualize_attention(self, attention_weights: torch.Tensor, query_idx: int,
#                           iteration: int, name: str = 'attention'):
#         """Visualize attention patterns"""
#         attn = attention_weights.detach().cpu().numpy()
        
#         # Take first batch item
#         attn_sample = attn[0]
        
#         plt.figure(figsize=(10, 8))
#         plt.imshow(attn_sample, aspect='auto', cmap='viridis')
#         plt.colorbar()
#         plt.title(f'Attention Pattern - Query {query_idx}, Iteration {iteration}')
#         plt.xlabel('Key Index')
#         plt.ylabel('Query Index')
        
#         plt.tight_layout()
#         plt.savefig(self.save_dir / f'{name}_q{query_idx}_{iteration}.png')
#         plt.close()

#     def visualize_query_stats(self, query_outputs: torch.Tensor, iteration: int,
#                             name: str = 'query_stats'):
#         """Visualize query output statistics"""
#         outputs = query_outputs.detach().cpu().numpy()
        
#         fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
#         # Distribution of query output values
#         ax1.hist(outputs.flatten(), bins=50, density=True)
#         ax1.set_title('Query Output Distribution')
#         ax1.set_xlabel('Output Value')
#         ax1.set_ylabel('Density')
        
#         # Mean activation per query
#         mean_activations = outputs.mean(axis=0)
#         ax2.plot(mean_activations)
#         ax2.set_title('Mean Query Activations')
#         ax2.set_xlabel('Query Index')
#         ax2.set_ylabel('Mean Activation')
        
#         plt.suptitle(f'Query Statistics - Iteration {iteration}')
#         plt.tight_layout()
        
#         plt.savefig(self.save_dir / f'{name}_{iteration}.png')
#         plt.close()