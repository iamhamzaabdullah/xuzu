"""
xuzu.tokenizer
==============
Proprietary nucleotide tokenizer for DNA, RNA, and chemically modified bases.

Authors : Hamza A
Lab     : terminalBio
License : terminalBio Proprietary — All Rights Reserved
"""
from __future__ import annotations
from typing import Dict, List, Tuple
import torch


class NucleotideTokenizer:
    """
    Single-character tokenizer covering:
      DNA  : A T G C
      RNA  : A U G C
      Mod  : F (2\'F-C)  M (2\'-OMe)  L (LNA)  P (phosphorothioate)
      Spec : [PAD] [MASK] [BOS] [EOS] [UNK]
    T and U are kept distinct; use .to_rna() / .to_dna() for inter-conversion.
    """
    BASE_TOKENS: List[str] = list("ATUGCFMLP")
    SPECIAL:     List[str] = ["[PAD]", "[MASK]", "[BOS]", "[EOS]", "[UNK]"]

    def __init__(self) -> None:
        vocab = self.SPECIAL + self.BASE_TOKENS
        self.token2id: Dict[str, int] = {t: i for i, t in enumerate(vocab)}
        self.id2token: Dict[int, str] = {i: t for t, i in self.token2id.items()}
        self.pad_id   = self.token2id["[PAD]"]
        self.mask_id  = self.token2id["[MASK]"]
        self.bos_id   = self.token2id["[BOS]"]
        self.eos_id   = self.token2id["[EOS]"]
        self.unk_id   = self.token2id["[UNK]"]
        self.vocab_size: int = len(vocab)

    # ── encoding ────────────────────────────────────────────────────────────
    def encode(self, seq: str, add_special: bool = True) -> List[int]:
        ids = [self.token2id.get(c.upper(), self.unk_id) for c in seq]
        return ([self.bos_id] + ids + [self.eos_id]) if add_special else ids

    def decode(self, ids: List[int], skip_special: bool = True) -> str:
        skip = {self.pad_id, self.bos_id, self.eos_id, self.unk_id}
        out  = []
        for i in ids:
            if skip_special and i in skip:
                continue
            out.append("?" if i == self.mask_id else self.id2token.get(i, "?"))
        return "".join(out)

    def batch_encode(
        self, seqs: List[str], max_len: int = 128
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Truncate raw sequence BEFORE adding specials so BOS + EOS always fit.
        # max_len - 2 reserves one slot each for [BOS] and [EOS].
        encoded = [self.encode(s[:max_len - 2]) for s in seqs]
        padded  = [e + [self.pad_id] * (max_len - len(e)) for e in encoded]
        ids  = torch.tensor(padded, dtype=torch.long)
        mask = (ids != self.pad_id).long()
        return ids, mask

    # ── helpers ─────────────────────────────────────────────────────────────
    @staticmethod
    def to_rna(seq: str) -> str:
        return seq.upper().replace("T", "U")

    @staticmethod
    def to_dna(seq: str) -> str:
        return seq.upper().replace("U", "T")

    def gc_content(self, seq: str) -> float:
        s = seq.upper()
        gc = sum(1 for c in s if c in "GC")
        return gc / len(s) if s else 0.0

    def is_valid(self, seq: str) -> bool:
        return all(c.upper() in self.token2id for c in seq)
