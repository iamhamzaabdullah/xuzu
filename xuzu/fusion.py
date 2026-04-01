"""
xuzu.fusion
===========
Proprietary cross-modal gated fusion of three XUZU encoder towers.

Authors : Hamza A
Lab     : terminalBio
License : terminalBio Proprietary — All Rights Reserved
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossModalFusion(nn.Module):
    """
    Fuses NLE (sequence), SGE (structure), TPE (target protein) outputs.

    Mechanism
    ---------
    1. Cross-attention: sequence positions attend to structure embeddings.
    2. Cross-attention: sequence positions attend to target protein context.
    3. Learned 3-way softmax gate per position blends the three streams.
    4. LayerNorm + residual projection.
    """
    def __init__(
        self, d_model: int = 256, n_heads: int = 8, dropout: float = 0.1
    ) -> None:
        super().__init__()
        self.ca_struct = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True)
        self.ca_target = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True)
        self.gate  = nn.Linear(d_model * 3, 3)
        self.proj  = nn.Linear(d_model, d_model)
        self.norm  = nn.LayerNorm(d_model)
        self.drop  = nn.Dropout(dropout)

    def forward(
        self,
        z_nle: torch.Tensor,    # (B, L, D)  sequence
        z_sge: torch.Tensor,    # (B, L, D)  structure
        z_tpe: torch.Tensor,    # (B, D)     protein pocket
    ) -> torch.Tensor:
        B, L, D = z_nle.shape
        z_tpe_exp = z_tpe.unsqueeze(1).expand(B, L, D)

        s_attn, _ = self.ca_struct(z_nle, z_sge, z_sge)
        t_attn, _ = self.ca_target(z_nle, z_tpe_exp, z_tpe_exp)

        cat   = torch.cat([z_nle, s_attn, t_attn], dim=-1)   # (B,L,3D)
        gates = F.softmax(self.gate(cat), dim=-1)              # (B,L,3)

        fused = (gates[..., 0:1] * z_nle  +
                 gates[..., 1:2] * s_attn +
                 gates[..., 2:3] * t_attn)
        return self.norm(self.proj(self.drop(fused)))
