"""
xuzu.data
=========
Proprietary dataset, collation, and loading utilities for XUZU.

Authors : Hamza A
Lab     : terminalBio
License : terminalBio Proprietary — All Rights Reserved
"""
from __future__ import annotations
import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset, DataLoader

from .tokenizer import NucleotideTokenizer
from .encoders  import StructureGraphEncoder


# ── Record schema ────────────────────────────────────────────────────────────
# {
#   "seq"       : "GCGAUAGCUAGCUA",          # required
#   "structure" : "(((...)))",                # optional dot-bracket
#   "pocket"    : "ARNDCQEGHILK",             # optional pocket AA seq
#   "kd_nm"     : 12.5,                       # optional Kd in nM (for GER)
# }


class AptamerDataset(Dataset):
    def __init__(
        self,
        records:    List[Dict],
        tokenizer:  NucleotideTokenizer,
        max_seq_len: int = 128,
        max_poc_len: int = 64,
        augment:     bool = True,
    ) -> None:
        self.records     = records
        self.tokenizer   = tokenizer
        self.max_seq     = max_seq_len
        self.max_poc     = max_poc_len
        self.augment     = augment

    def __len__(self) -> int:
        return len(self.records)

    def _augment(self, seq: str) -> str:
        """Randomly mutate 1–3 positions to simulate sequence diversity."""
        if not self.augment or len(seq) < 5:
            return seq
        bases = list("ATGC") if "U" not in seq else list("AUGC")
        seq   = list(seq)
        n_mut = random.randint(0, min(3, len(seq) // 10 + 1))
        for _ in range(n_mut):
            pos       = random.randrange(len(seq))
            seq[pos]  = random.choice(bases)
        return "".join(seq)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        r   = self.records[idx]
        seq = self._augment(r.get("seq", "A" * 30))
        db  = r.get("structure", "." * len(seq))
        poc = r.get("pocket",   "ARNDCQ")
        kd  = float(r.get("kd_nm", -1.0))

        # ── nucleotide encoding ──────────────────────────────────────────────
        seq_ids, seq_mask = self.tokenizer.batch_encode([seq], self.max_seq)
        L   = seq_ids.shape[1]
        # seq_ids layout: [BOS, n1, n2, ..., nN, EOS, PAD…]
        # The dot-bracket covers the raw nucleotides only.  We insert one '.'
        # for the BOS token, then pad or truncate the dot-bracket to fill the
        # remaining L-1 positions (including EOS / PAD slots).
        n_nuc    = min(len(seq), self.max_seq - 2)    # nucleotide slots used
        db_nuc   = db[:n_nuc].ljust(n_nuc, ".")       # always exactly n_nuc chars
        db_rest  = "." * (L - 1 - n_nuc)              # EOS + PAD slots
        db_aligned = "." + db_nuc + db_rest            # always exactly L chars
        assert len(db_aligned) == L, f"db_aligned len {len(db_aligned)} != L {L}"
        adj = StructureGraphEncoder.dot_bracket_to_adj(db_aligned)  # (L, L)

        # ── pocket encoding ──────────────────────────────────────────────────
        AA_MAP = {a: i for i, a in enumerate(
            list("ACDEFGHIKLMNPQRSTVWY") + ["[PAD]", "[UNK]"])}
        poc_ids = [AA_MAP.get(aa.upper(), 21) for aa in poc[:self.max_poc]]
        poc_ids += [20] * (self.max_poc - len(poc_ids))    # pad
        poc_t   = torch.tensor(poc_ids, dtype=torch.long)
        poc_mask = (poc_t != 20).long()

        return {
            "seq_ids":  seq_ids.squeeze(0),    # (L,)
            "seq_mask": seq_mask.squeeze(0),   # (L,)
            "adj":      adj,                   # (L, L)
            "poc_ids":  poc_t,                 # (P,)
            "poc_mask": poc_mask,              # (P,)
            "kd":       torch.tensor(kd, dtype=torch.float32),
        }


def load_jsonl(path: str) -> List[Dict]:
    """Load aptamer records from a .jsonl file."""
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def build_dataloaders(
    path:       str,
    tokenizer:  NucleotideTokenizer,
    val_frac:   float = 0.1,
    batch_size: int   = 32,
    max_seq_len: int  = 128,
    max_poc_len: int  = 64,
    num_workers: int  = 0,
) -> Tuple[DataLoader, DataLoader]:
    records = load_jsonl(path)
    random.shuffle(records)
    n_val   = max(1, int(len(records) * val_frac))
    val_rec = records[:n_val]
    trn_rec = records[n_val:]

    trn_ds = AptamerDataset(trn_rec, tokenizer, max_seq_len, max_poc_len, augment=True)
    val_ds = AptamerDataset(val_rec, tokenizer, max_seq_len, max_poc_len, augment=False)

    trn_loader = DataLoader(trn_ds, batch_size=batch_size,
                            shuffle=True,  num_workers=num_workers,
                            pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size,
                            shuffle=False, num_workers=num_workers,
                            pin_memory=True)
    return trn_loader, val_loader
