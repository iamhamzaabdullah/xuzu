"""
xuzu.layers
===========
Proprietary building blocks for XUZU nucleotide language model.

  FEBI  — Fundamental Encoding Block Intelligence
           Core transformer block with Rotary Positional Encoding.

  RJ    — Relational Junction
           Graph attention layer for structural contact encoding.

Authors : Hamza A
Lab     : terminalBio
Version : 1.0.0
License : terminalBio Proprietary — All Rights Reserved
"""
from __future__ import annotations
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Rotary Positional Encoding ───────────────────────────────────────────────
class RotaryEncoding(nn.Module):
    """RoPE — handles variable-length sequences without fixed-length assumptions."""
    def __init__(self, dim: int) -> None:
        super().__init__()
        assert dim % 2 == 0, "dim must be even for RoPE"
        inv = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        t    = torch.arange(L, device=x.device).float()
        freq = torch.outer(t, self.inv_freq)
        emb  = torch.cat([freq, freq], dim=-1)          # (L, D)
        cos_ = emb.cos().unsqueeze(0)
        sin_ = emb.sin().unsqueeze(0)
        half = D // 2
        x_rot = torch.cat([-x[..., half:], x[..., :half]], dim=-1)
        return x * cos_ + x_rot * sin_


# ── Multi-Head Self-Attention with RoPE ─────────────────────────────────────
class RoPEAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head  = d_model // n_heads
        self.scale   = self.d_head ** -0.5
        self.qkv     = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj    = nn.Linear(d_model, d_model, bias=False)
        self.rope    = RotaryEncoding(self.d_head)
        self.drop    = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor, key_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        B, L, D = x.shape
        H, Dh   = self.n_heads, self.d_head
        qkv = self.qkv(x).reshape(B, L, 3, H, Dh).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]                  # (B,H,L,Dh)
        # apply RoPE per head
        q = self.rope(q.reshape(B * H, L, Dh)).reshape(B, H, L, Dh)
        k = self.rope(k.reshape(B * H, L, Dh)).reshape(B, H, L, Dh)
        sc = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (B,H,L,L)
        if key_mask is not None:
            # key_mask: (B, L), 1=keep 0=pad
            sc = sc.masked_fill(key_mask[:, None, None, :] == 0, float("-inf"))
        w  = self.drop(F.softmax(sc, dim=-1))
        out = torch.matmul(w, v).transpose(1, 2).reshape(B, L, D)
        return self.proj(out)


# ── Transformer Block ────────────────────────────────────────────────────────
class FEBI(nn.Module):
    """FEBI — Fundamental Encoding Block Intelligence (Hamza A, terminalBio)
    Proprietary transformer block with RoPE attention + gated feed-forward.
    """
    def __init__(
        self, d_model: int, n_heads: int,
        ff_mult: int = 4, dropout: float = 0.1
    ) -> None:
        super().__init__()
        self.attn  = RoPEAttention(d_model, n_heads, dropout)
        self.ff    = nn.Sequential(
            nn.Linear(d_model, d_model * ff_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * ff_mult, d_model),
            nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), mask)
        x = x + self.ff(self.norm2(x))
        return x


# ── Graph Attention (for structure tower) ───────────────────────────────────
class RJ(nn.Module):
    """RJ — Relational Junction (Hamza A, terminalBio)
    Proprietary graph attention layer for base-pair contact graph encoding.
    Single-layer graph attention over base-pair contact graph.
    """
    def __init__(self, d_in: int, d_out: int, n_heads: int = 4,
                 dropout: float = 0.1) -> None:
        super().__init__()
        assert d_out % n_heads == 0
        self.n_heads = n_heads
        self.d_head  = d_out // n_heads
        self.d_out   = d_out          # stored for correct attention scale
        self.Wq = nn.Linear(d_in, d_out, bias=False)
        self.Wk = nn.Linear(d_in, d_out, bias=False)
        self.Wv = nn.Linear(d_in, d_out, bias=False)
        self.Wo = nn.Linear(d_out, d_out)
        self.norm = nn.LayerNorm(d_out)
        self.drop = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor, adj: torch.Tensor
    ) -> torch.Tensor:
        # x: (B,L,D_in)  adj: (B,L,L) binary contact map
        B, L, _ = x.shape
        Q = self.Wq(x)  # (B,L,D_out)
        K = self.Wk(x)
        V = self.Wv(x)
        # Scale by sqrt(d_out): Q/K are d_out-dimensional, NOT d_head-dimensional.
        # Using d_head here would over-scale by sqrt(n_heads), sharpening attention
        # by a factor of n_heads and destabilising training.
        sc = torch.bmm(Q, K.transpose(1, 2)) / math.sqrt(self.d_out)
        sc = sc.masked_fill(adj == 0, float("-inf"))
        # Replace full -inf rows (isolated nodes) with uniform attention
        all_inf = sc.isinf().all(dim=-1, keepdim=True)
        sc = sc.masked_fill(all_inf, 0.0)
        w  = self.drop(F.softmax(sc, dim=-1))
        out = torch.bmm(w, V)
        return self.norm(x + self.Wo(out))
