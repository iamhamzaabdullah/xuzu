"""
╔══════════════════════════════════════════════════════════════╗
║         XUZU — Multi-Modal Nucleotide Language Model         ║
║         Built by Hamza A  ·  terminalBio                     ║
║         Version 1.0.0  ·  All Rights Reserved                ║
╚══════════════════════════════════════════════════════════════╝

XUZU: X-modal Unified Zero-shot Universal Aptamer Language Model

A de novo, multi-modal nucleotide language model for DNA/RNA aptamer
design. Proprietary architecture featuring:

  FEBI — Fundamental Encoding Block Intelligence
         Core transformer block with Rotary Positional Encoding (RoPE).
         Powers NucleotideLanguageEncoder and TargetProteinEncoder.

  RJ   — Relational Junction
         Graph attention layer for base-pair contact graphs.
         Powers StructureGraphEncoder.

Quick start
-----------
>>> from xuzu import XUZU, XUZUConfig
>>> model = XUZU(XUZUConfig())
>>> apts  = model.design(
...     pocket_seq   = "ARNDCQEGHILKMFPSTWYV",
...     seq_len      = 40,
...     n_candidates = 5,
...     as_rna       = True,
... )
>>> print(apts)

Authors : Hamza A
Lab     : terminalBio
License : terminalBio Proprietary — All Rights Reserved
"""

from .model     import XUZU, XUZUConfig
from .tokenizer import NucleotideTokenizer
from .layers    import FEBI, RJ, RotaryEncoding, RoPEAttention
from .encoders  import (NucleotideLanguageEncoder,
                         StructureGraphEncoder,
                         TargetProteinEncoder)
from .fusion    import CrossModalFusion
from .decoder   import DiscreteDiffusionDecoder, SinusoidalTimeEmbed
from .reward    import BindingAffinitySurrogate, GERLoop
from .data      import AptamerDataset, build_dataloaders, load_jsonl
from .metrics   import (evaluate_batch, gc_content, mfe_proxy,
                         novelty_score, diversity_score,
                         shannon_entropy, fid_nucleotide, levenshtein)
from .trainer   import XUZUTrainer

__version__  = "1.0.0"
__author__   = "Hamza A"
__lab__      = "terminalBio"
__license__  = "terminalBio Proprietary"

__all__ = [
    # ── Core model ───────────────────────────────────────────
    "XUZU", "XUZUConfig",
    # ── Proprietary blocks ───────────────────────────────────
    "FEBI", "RJ",
    # ── Tokenizer ────────────────────────────────────────────
    "NucleotideTokenizer",
    # ── Encoders ─────────────────────────────────────────────
    "NucleotideLanguageEncoder",
    "StructureGraphEncoder",
    "TargetProteinEncoder",
    # ── Fusion & Decoder ─────────────────────────────────────
    "CrossModalFusion",
    "DiscreteDiffusionDecoder",
    "SinusoidalTimeEmbed",
    # ── Reward & GER ─────────────────────────────────────────
    "BindingAffinitySurrogate",
    "GERLoop",
    # ── Data ─────────────────────────────────────────────────
    "AptamerDataset",
    "build_dataloaders",
    "load_jsonl",
    # ── Metrics ──────────────────────────────────────────────
    "evaluate_batch",
    "gc_content",
    "mfe_proxy",
    "novelty_score",
    "diversity_score",
    "shannon_entropy",
    "fid_nucleotide",
    "levenshtein",
    # ── Trainer ──────────────────────────────────────────────
    "XUZUTrainer",
]
