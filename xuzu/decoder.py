"""
xuzu.decoder
============
Proprietary absorbing-state discrete diffusion decoder for XUZU.
Forward : tokens → [MASK] (probabilistic absorbing noise)
Reverse : [MASK] → tokens (iterative denoising on multi-modal context)

Authors : Hamza A
Lab     : terminalBio
License : terminalBio Proprietary — All Rights Reserved
"""
from __future__ import annotations
from typing import Optional, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import FEBI
from .tokenizer import NucleotideTokenizer


class SinusoidalTimeEmbed(nn.Module):
    """Scalar diffusion timestep → fixed sinusoidal embedding → MLP."""
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.SiLU(),
            nn.Linear(d_model * 2, d_model),
        )
        half = d_model // 2
        freq = torch.exp(-torch.arange(half).float() * (8.0 / half))
        self.register_buffer("freq", freq)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: (B,) float in [0,1]
        arg  = t.unsqueeze(1) * self.freq.unsqueeze(0) * 1000.0
        emb  = torch.cat([arg.sin(), arg.cos()], dim=-1)
        return self.mlp(emb)


class DiscreteDiffusionDecoder(nn.Module):
    """
    Absorbing-state D3PM decoder.
    Trained with masked-nucleotide language modelling (MNLM) loss.
    Generation: iterative unmasking from fully masked sequence.
    """
    def __init__(
        self, vocab_size: int, d_model: int = 256,
        n_layers: int = 4, n_heads: int = 8,
        T: int = 100, dropout: float = 0.1
    ) -> None:
        super().__init__()
        self.T          = T
        self.vocab_size = vocab_size
        self.embed      = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.time_emb   = SinusoidalTimeEmbed(d_model)
        self.blocks     = nn.ModuleList(
            [FEBI(d_model, n_heads, dropout=dropout)
             for _ in range(n_layers)]
        )
        self.norm     = nn.LayerNorm(d_model)
        self.out_head = nn.Linear(d_model, vocab_size)

    # ── forward process ─────────────────────────────────────────────────────
    def corrupt(
        self, ids: torch.Tensor, t: float, mask_id: int
    ) -> torch.Tensor:
        """Replace fraction t of non-special tokens with [MASK]."""
        noisy = ids.clone()
        rand  = torch.rand_like(ids.float())
        noisy[rand < t] = mask_id
        return noisy

    # ── denoising forward ────────────────────────────────────────────────────
    def forward(
        self,
        noisy_ids: torch.Tensor,   # (B, L)
        context:   torch.Tensor,   # (B, L, D) fused multi-modal context
        t_frac:    float,
        pad_mask:  Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, L = noisy_ids.shape
        t_vec = torch.full((B,), t_frac, device=noisy_ids.device)

        x  = self.embed(noisy_ids)                        # (B,L,D)
        te = self.time_emb(t_vec).unsqueeze(1)            # (B,1,D)
        x  = x + te + context

        for blk in self.blocks:
            x = blk(x, pad_mask)
        return self.out_head(self.norm(x))                # (B,L,V)

    # ── generation ───────────────────────────────────────────────────────────
    @torch.no_grad()
    def generate(
        self,
        context:     torch.Tensor,         # (1, L, D)
        seq_len:     int,
        tokenizer:   NucleotideTokenizer,
        steps:       int = 25,
        temperature: float = 0.9,
        top_k:       int  = 0,
    ) -> str:
        device = context.device
        ids    = torch.full((1, seq_len), tokenizer.mask_id,
                            dtype=torch.long, device=device)

        for step in reversed(range(1, steps + 1)):
            t_frac  = step / steps
            logits  = self.forward(ids, context, t_frac)  # (1,L,V)

            if top_k > 0:
                topk_vals, _ = torch.topk(logits, top_k, dim=-1)
                thresh = topk_vals[..., -1:].expand_as(logits)
                logits = logits.masked_fill(logits < thresh, float("-inf"))

            probs   = F.softmax(logits / temperature, dim=-1)
            sampled = torch.multinomial(
                probs.view(-1, self.vocab_size), num_samples=1
            ).view(1, seq_len)

            still_masked    = (ids == tokenizer.mask_id)
            ids[still_masked] = sampled[still_masked]

        return tokenizer.decode(ids[0].tolist())
