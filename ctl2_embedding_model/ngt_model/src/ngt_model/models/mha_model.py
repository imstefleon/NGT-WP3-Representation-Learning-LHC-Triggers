"""
Standard multi-head self-attention encoder for Collide-2V L1 candidate tensors.

This is the architecture ablation of `LinformerAttention`/`LinformerEmbeddingFullPrecision`
(see `NGT_Linformer_NodesAligned_FullPrecision_Collide2V_L1features.ipynb`): the low-rank
Linformer sequence projection (`proj_k`, `E`/`F`) is replaced by standard (full, non-low-rank)
multi-head self-attention. Everything else -- residual attention blocks, mean or attention
pooling, the post-pool FFN, and the {'latent','logits'} output -- is unchanged, so the two
architectures are directly comparable at matched `latent_dim`/`ff_dim`/`num_heads`.

Shared by `NGT_MHA_NodesAligned_FullPrecision_Collide2V_L1features.ipynb`, `scale_test.py`,
and `eval_quality_collide2v.py` (via the `LatentOnlyWrapper` export) so the model definition
lives in exactly one place.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class StandardMHA(nn.Module):
    def __init__(self, dim: int, num_heads: int = 2, seq_len: int = 100, dropout: float = 0.0):
        super().__init__()
        assert dim % num_heads == 0, 'dim must be divisible by num_heads'
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.seq_len = seq_len  # unused, kept for call-site parity with the Linformer variant

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: [B, N, D]
        B, N, D = x.shape
        q = self.q_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # [B,H,N,d]
        k = self.k_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # [B,H,N,d]
        v = self.v_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # [B,H,N,d]

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # [B,H,N,N]
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)  # [B,H,N,d]
        out = out.transpose(1, 2).contiguous().view(B, N, D)
        return self.out_proj(out)


class MHAEmbeddingFullPrecision(nn.Module):
    def __init__(
        self,
        n_constituents: int,
        n_features: int,
        latent_dim: int,
        n_classes: int,
        ff_dim: int = 32,
        num_heads: int = 2,
        pooling_mode: str = 'mean',
        classifier_head: str = 'mlp',
        num_attention_blocks: int = 2,
    ):
        super().__init__()
        assert pooling_mode in {'mean', 'attention'}
        assert classifier_head in {'mlp', 'linear'}
        assert num_attention_blocks >= 1
        self.n_constituents = n_constituents
        self.pooling_mode = pooling_mode
        self.classifier_head = classifier_head
        self.num_attention_blocks = num_attention_blocks
        self.input_proj = nn.Linear(n_features, ff_dim)
        self.attention_blocks = nn.ModuleList(
            StandardMHA(ff_dim, num_heads=num_heads, seq_len=n_constituents)
            for _ in range(num_attention_blocks)
        )
        if pooling_mode == 'attention':
            pool_hidden = max(8, ff_dim // 2)
            self.pool_score = nn.Sequential(
                nn.Linear(ff_dim, pool_hidden),
                nn.ReLU(),
                nn.Linear(pool_hidden, 1),
            )
        else:
            self.pool_score = None
        self.post_pool = nn.Sequential(
            nn.Linear(ff_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, ff_dim),
            nn.ReLU(),
        )
        self.latent = nn.Linear(ff_dim, latent_dim)
        if classifier_head == 'mlp':
            self.cls_hidden = nn.Sequential(
                nn.Linear(latent_dim, ff_dim),
                nn.ReLU(),
            )
            self.class_logits = nn.Linear(ff_dim, n_classes)
        else:
            self.cls_hidden = nn.Identity()
            self.class_logits = nn.Linear(latent_dim, n_classes)

    def pool(self, res, valid_mask):
        valid_float = valid_mask.unsqueeze(-1).to(res.dtype)
        if self.pooling_mode == 'mean':
            denom = valid_float.sum(dim=1).clamp_min(1.0)
            return (res * valid_float).sum(dim=1) / denom

        scores = self.pool_score(res).squeeze(-1)
        scores = scores.masked_fill(~valid_mask, -1e9)
        weights = torch.softmax(scores, dim=1) * valid_mask.to(res.dtype)
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-6)
        return (res * weights.unsqueeze(-1)).sum(dim=1)

    def forward(self, x):
        valid_mask = torch.any(x != 0, dim=-1)
        emb = F.relu(self.input_proj(x))
        res = emb
        attention_blocks = getattr(self, 'attention_blocks', None)
        if attention_blocks is None:
            attention_blocks = [self.attention1, self.attention2]
        for attention in attention_blocks:
            res = attention(res) + res
        pooled = self.pool(res, valid_mask)
        h = self.post_pool(pooled)
        z = self.latent(h)
        logits = self.class_logits(self.cls_hidden(z))
        return {'latent': z, 'logits': logits}


class LatentOnlyWrapper(nn.Module):
    """Wraps an encoder returning {'latent','logits'} to expose only the latent tensor.

    eval_quality.py's `_load_model` expects `torch.load(path)` to return a callable
    `nn.Module` whose forward returns the latent tensor directly (not a dict).
    """

    def __init__(self, encoder: nn.Module):
        super().__init__()
        self.encoder = encoder

    def forward(self, x):
        return self.encoder(x)['latent']
