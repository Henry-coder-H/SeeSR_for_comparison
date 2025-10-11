"""
Custom ScaleTimeEmbedding implementation for SeeSR
This replaces the dependency on diffusers library for ScaleTimeEmbedding
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def rope_scalar_embedding(scalar, dim, base=10000.0):
    """
    RoPE (Rotary Position Embedding) for scalar values
    """
    device = scalar.device
    dtype = scalar.dtype
    
    # Create frequency tensor
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, device=device, dtype=dtype) / dim))
    
    # Compute angles
    t = scalar.unsqueeze(-1) * inv_freq.unsqueeze(0)
    
    # Create embeddings
    emb = torch.cat([torch.cos(t), torch.sin(t)], dim=-1)
    
    return emb


class ScaleTimeEmbedding(nn.Module):
    """
    Custom ScaleTimeEmbedding that works with log-scale values
    """
    
    def __init__(self, time_embed_dim: int, rope_dim: int = 128, rope_base: float = 10000.0, use_norm: bool = True):
        super().__init__()
        self.time_embed_dim = time_embed_dim
        self.rope_dim = rope_dim
        self.rope_base = rope_base
        self.use_norm = use_norm
        
        # Linear layers for processing the RoPE embeddings
        self.linear1 = nn.Linear(rope_dim, time_embed_dim)
        self.linear2 = nn.Linear(time_embed_dim, time_embed_dim)
        
        if use_norm:
            self.norm = nn.LayerNorm(time_embed_dim)
        else:
            self.norm = None
            
        self.activation = nn.SiLU()
    
    def forward(self, scale_log: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for scale embedding
        
        Args:
            scale_log: Log-scale values of shape (batch_size,) or (batch_size, 1)
            
        Returns:
            Scale embeddings of shape (batch_size, time_embed_dim)
        """
        # Ensure scale_log is 2D for RoPE processing
        if scale_log.dim() == 1:
            scale_log = scale_log.unsqueeze(-1)
        
        # Generate RoPE embeddings
        rope_emb = rope_scalar_embedding(scale_log, self.rope_dim, self.rope_base)
        
        # Process through linear layers
        emb = self.linear1(rope_emb)
        emb = self.activation(emb)
        emb = self.linear2(emb)
        
        # Apply normalization if enabled
        if self.norm is not None:
            emb = self.norm(emb)
        
        # Squeeze the extra dimension to get (batch_size, time_embed_dim)
        if emb.dim() == 3 and emb.shape[1] == 1:
            emb = emb.squeeze(1)
            
        return emb


class LogScaleSampler:
    """
    Utility class for sampling scale values in log space
    """
    
    def __init__(self, min_scale: float = 1/16, max_scale: float = 1.0):
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.log_min_scale = math.log(min_scale)
        self.log_max_scale = math.log(max_scale)
    
    def sample(self, batch_size: int, device: torch.device = None) -> torch.Tensor:
        """
        Sample scale values in log space
        
        Args:
            batch_size: Number of samples to generate
            device: Device to place tensors on
            
        Returns:
            Log-scale values of shape (batch_size,)
        """
        # Sample uniformly in log space
        log_scale = torch.rand(batch_size, device=device) * (self.log_max_scale - self.log_min_scale) + self.log_min_scale
        return log_scale
    
    def log_to_scale(self, log_scale: torch.Tensor) -> torch.Tensor:
        """
        Convert log-scale values back to original scale values
        
        Args:
            log_scale: Log-scale values
            
        Returns:
            Original scale values
        """
        return torch.exp(log_scale)
    
    def scale_to_log(self, scale: torch.Tensor) -> torch.Tensor:
        """
        Convert original scale values to log-scale values
        
        Args:
            scale: Original scale values
            
        Returns:
            Log-scale values
        """
        return torch.log(scale)
