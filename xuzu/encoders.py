"""
xuzu.encoders
=============
Proprietary three-tower encoder stack for XUZU.

  NucleotideLanguageEncoder  — Tower 1: sequence context
  StructureGraphEncoder      — Tower 2: base-pair graph (uses RJ blocks)
  TargetProteinEncoder       — Tower 3: protein pocket context (uses FEBI blocks)

Authors : Hamza A
Lab     : terminalBio
License : terminalBio Proprietary — All Rights Reserved
"""
from __future__ import annotations
from typing import Optional, Tuple
import torch
import torch.nn as nn
from .layers import FEBI, RJ


# ── Tower 1: Nucleotide Language Encoder ────────────────────────────────────
class NucleotideLanguageEncoder(nn.Module):
    """
    Bidirectional transformer with RoPE over nucleotide sequences.
    Learns base-composition patterns, motif grammar, and long-range covariation.
    """
    def __init__(
        self, vocab_size: int, d_model: int = 256,
        n_layers: int = 6, n_heads: int = 8, dropout: float = 0.1
    ) -> None:
        super().__init__()
        self.embed  = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.drop   = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [FEBI(d_model, n_heads, dropout=dropout)
             for _ in range(n_layers)]
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self, ids: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        x = self.drop(self.embed(ids))
        for blk in self.blocks:
            x = blk(x, mask)
        return self.norm(x)          # (B, L, D)


# ── Tower 2: Structure Graph Encoder ────────────────────────────────────────
class StructureGraphEncoder(nn.Module):
    """
    Multi-layer GAT over base-pair contact graphs derived from dot-bracket.
    Stacks n_layers GraphAttentionLayer to propagate structural context.
    """
    def __init__(
        self, d_model: int = 256, n_layers: int = 3,
        n_heads: int = 4, dropout: float = 0.1
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [RJ(d_model, d_model, n_heads, dropout)
             for _ in range(n_layers)]
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self, x: torch.Tensor, adj: torch.Tensor
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, adj)
        return self.norm(x)

    @staticmethod
    def dot_bracket_to_adj(db: str, device: torch.device = None) -> torch.Tensor:
        """Convert dot-bracket string → symmetric binary adjacency matrix."""
        n   = len(db)
        adj = torch.eye(n)
        stack: list = []
        for i, c in enumerate(db):
            if c == "(":
                stack.append(i)
            elif c == ")" and stack:
                j = stack.pop()
                adj[i, j] = adj[j, i] = 1.0
        return adj if device is None else adj.to(device)


# ── Tower 3: Target Protein Encoder ─────────────────────────────────────────
class TargetProteinEncoder(nn.Module):
    """
    Self-attention encoder over binding-pocket amino-acid sequence.
    No external protein LM dependency — learned from scratch.
    Output: single context vector per sample via mean pooling.
    """
    AA_VOCAB = list("ACDEFGHIKLMNPQRSTVWY") + ["[PAD]", "[UNK]"]

    def __init__(
        self, d_model: int = 256, n_layers: int = 3,
        n_heads: int = 8, dropout: float = 0.1
    ) -> None:
        super().__init__()
        self.aa2id = {a: i for i, a in enumerate(self.AA_VOCAB)}
        self.pad_id = self.aa2id["[PAD]"]
        self.embed  = nn.Embedding(len(self.AA_VOCAB), d_model,
                                   padding_idx=self.pad_id)
        self.blocks = nn.ModuleList(
            [FEBI(d_model, n_heads, dropout=dropout)
             for _ in range(n_layers)]
        )
        self.norm = nn.LayerNorm(d_model)
        self.proj = nn.Linear(d_model, d_model)

    def tokenize(
        self, seq: str, device: torch.device, max_len: int = 64
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        ids  = [self.aa2id.get(aa.upper(), 21) for aa in seq[:max_len]]
        pad  = [self.pad_id] * (max_len - len(ids))
        ids  += pad
        t    = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
        mask = (t != self.pad_id).long()
        return t, mask

    def forward(
        self, aa_ids: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        x = self.embed(aa_ids)
        for blk in self.blocks:
            x = blk(x, mask)
        x   = self.norm(x)
        ctx = (x * mask.unsqueeze(-1).float()).sum(1) / mask.float().sum(1, keepdim=True).clamp(1)
        return self.proj(ctx)                # (B, D)
