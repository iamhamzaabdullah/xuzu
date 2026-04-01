"""
xuzu.model
==========
XUZU — X-modal Unified Zero-shot Universal Aptamer Language Model.
Full proprietary model assembly, configuration, and inference.

Authors  : Hamza A
Lab      : terminalBio
Version  : 1.0.0
License  : terminalBio Proprietary — All Rights Reserved
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import List, Optional

import torch
import torch.nn as nn

from .tokenizer import NucleotideTokenizer
from .encoders  import (NucleotideLanguageEncoder,
                         StructureGraphEncoder,
                         TargetProteinEncoder)
from .fusion    import CrossModalFusion
from .decoder   import DiscreteDiffusionDecoder


@dataclass
class XUZUConfig:
    # ── dimensions ──────────────────────────────────────────────────────────
    d_model:      int   = 256
    n_heads:      int   = 8
    # ── encoder depths ──────────────────────────────────────────────────────
    nle_layers:   int   = 6
    sge_layers:   int   = 3
    sge_heads:    int   = 4
    tpe_layers:   int   = 3
    # ── decoder ─────────────────────────────────────────────────────────────
    dec_layers:   int   = 4
    diffusion_T:  int   = 100
    # ── training ────────────────────────────────────────────────────────────
    max_seq_len:  int   = 128
    max_poc_len:  int   = 64
    dropout:      float = 0.1
    lr:           float = 1e-4
    weight_decay: float = 0.01
    batch_size:   int   = 32
    warmup_steps: int   = 500
    # ── vocab (auto-set by tokenizer) ────────────────────────────────────────
    vocab_size:   int   = 14

    def to_dict(self): return asdict(self)


class XUZU(nn.Module):
    """
    XUZU — Built by Hamza A @ terminalBio
    X-modal Unified Zero-shot Universal aptamer language model.
    Proprietary architecture: FEBI encoders + RJ structure tower +
    cross-modal fusion + absorbing discrete diffusion decoder.

    Inputs
    ------
    seq_ids   : (B, L)    nucleotide token ids
    seq_mask  : (B, L)    1=real 0=pad
    adj       : (B, L, L) base-pair adjacency matrix
    poc_ids   : (B, P)    pocket amino-acid token ids
    poc_mask  : (B, P)    1=real 0=pad
    t_frac    : float     diffusion noise fraction in [0,1]

    Output
    ------
    logits    : (B, L, V) token logits for MNLM loss
    """

    def __init__(self, cfg: XUZUConfig) -> None:
        super().__init__()
        self.cfg       = cfg
        self.tokenizer = NucleotideTokenizer()

        self.nle = NucleotideLanguageEncoder(
            vocab_size=cfg.vocab_size, d_model=cfg.d_model,
            n_layers=cfg.nle_layers, n_heads=cfg.n_heads, dropout=cfg.dropout)

        self.sge = StructureGraphEncoder(
            d_model=cfg.d_model, n_layers=cfg.sge_layers,
            n_heads=cfg.sge_heads, dropout=cfg.dropout)

        self.tpe = TargetProteinEncoder(
            d_model=cfg.d_model, n_layers=cfg.tpe_layers,
            n_heads=cfg.n_heads, dropout=cfg.dropout)

        self.fusion = CrossModalFusion(
            d_model=cfg.d_model, n_heads=cfg.n_heads, dropout=cfg.dropout)

        self.decoder = DiscreteDiffusionDecoder(
            vocab_size=cfg.vocab_size, d_model=cfg.d_model,
            n_layers=cfg.dec_layers, n_heads=cfg.n_heads,
            T=cfg.diffusion_T, dropout=cfg.dropout)

    # ── context encoding (shared by train + generate) ───────────────────────
    def encode(
        self,
        seq_ids:  torch.Tensor,
        seq_mask: torch.Tensor,
        adj:      torch.Tensor,
        poc_ids:  torch.Tensor,
        poc_mask: torch.Tensor,
    ) -> torch.Tensor:
        z_nle = self.nle(seq_ids, seq_mask)
        z_sge = self.sge(z_nle, adj)
        z_tpe = self.tpe(poc_ids, poc_mask)
        return self.fusion(z_nle, z_sge, z_tpe)

    def forward(
        self,
        seq_ids:  torch.Tensor,
        seq_mask: torch.Tensor,
        adj:      torch.Tensor,
        poc_ids:  torch.Tensor,
        poc_mask: torch.Tensor,
        t_frac:   float,
    ) -> torch.Tensor:
        context = self.encode(seq_ids, seq_mask, adj, poc_ids, poc_mask)
        noisy   = self.decoder.corrupt(
            seq_ids, t_frac, self.tokenizer.mask_id)
        return self.decoder(noisy, context, t_frac, seq_mask)

    @torch.no_grad()
    def design(
        self,
        pocket_seq:   str,
        template_seq: Optional[str]  = None,
        dot_bracket:  Optional[str]  = None,
        seq_len:      int            = 40,
        temperature:  float          = 0.85,
        top_k:        int            = 0,
        n_candidates: int            = 5,
        as_rna:       bool           = False,
    ) -> List[str]:
        """
        De novo aptamer design conditioned on a target protein pocket.

        Parameters
        ----------
        pocket_seq   : amino-acid sequence of binding pocket residues
        template_seq : optional seed sequence (partially masked and refined)
        dot_bracket  : optional secondary structure constraint
        seq_len      : desired aptamer length in nucleotides
        temperature  : sampling temperature (lower = more conservative)
        top_k        : top-k nucleus filtering (0 = disabled)
        n_candidates : number of independent designs to generate
        as_rna       : if True, replace T with U in output
        """
        self.eval()
        device = next(self.parameters()).device
        results: List[str] = []

        poc_ids, poc_mask = self.tpe.tokenize(pocket_seq, device,
                                               self.cfg.max_poc_len)

        db = (dot_bracket or "." * seq_len)[:seq_len].ljust(seq_len, ".")
        adj = StructureGraphEncoder.dot_bracket_to_adj(db, device).unsqueeze(0)

        for _ in range(n_candidates):
            if template_seq:
                seq_ids, seq_mask = self.tokenizer.batch_encode(
                    [template_seq], max_len=seq_len)
                seq_ids  = seq_ids.to(device)
                seq_mask = seq_mask.to(device)
            else:
                seq_ids  = torch.full((1, seq_len), self.tokenizer.mask_id,
                                      dtype=torch.long, device=device)
                seq_mask = torch.ones_like(seq_ids)

            z_nle   = self.nle(seq_ids, seq_mask)
            z_sge   = self.sge(z_nle, adj)
            z_tpe   = self.tpe(poc_ids, poc_mask)
            context = self.fusion(z_nle, z_sge, z_tpe)

            apt = self.decoder.generate(
                context, seq_len, self.tokenizer,
                temperature=temperature, top_k=top_k)
            if as_rna:
                apt = apt.replace("T", "U")
            results.append(apt)

        return results

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def save(self, path: str) -> None:
        torch.save({"config": self.cfg.to_dict(),
                    "model_state": self.state_dict()}, path)

    @classmethod
    def load(cls, path: str, device: str = "cpu") -> "XUZU":
        ckpt   = torch.load(path, map_location=device)
        cfg    = XUZUConfig(**ckpt["config"])
        model  = cls(cfg).to(device)
        model.load_state_dict(ckpt["model_state"])
        model.eval()
        return model
