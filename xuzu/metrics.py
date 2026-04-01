"""
xuzu.metrics
============
Proprietary Q1-publication-grade evaluation metrics for generated aptamers.

Authors : Hamza A
Lab     : terminalBio
License : terminalBio Proprietary — All Rights Reserved
"""
from __future__ import annotations
import math
from typing import List


def gc_content(seq: str) -> float:
    s = seq.upper()
    return sum(1 for c in s if c in "GC") / max(len(s), 1)


def mfe_proxy(seq: str) -> float:
    """Greedy stem-loop pair counter; returns negative pair count (like MFE).
    Handles both DNA (A-T, G-C) and RNA (A-U, G-C, G-U wobble) in either
    strand orientation so A-T pairs are counted regardless of which base
    sits on the stack.
    """
    PAIRS = {
        ("A", "U"), ("U", "A"),   # RNA Watson-Crick
        ("G", "C"), ("C", "G"),   # Watson-Crick (DNA + RNA)
        ("A", "T"), ("T", "A"),   # DNA Watson-Crick
        ("G", "U"), ("U", "G"),   # RNA wobble
    }
    stack, pairs = [], 0
    for c in seq.upper():
        if stack and (stack[-1], c) in PAIRS:
            stack.pop(); pairs += 1
        else:
            stack.append(c)
    return -float(pairs)


def levenshtein(a: str, b: str) -> int:
    m, n = len(a), len(b)
    dp   = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[:]
        dp[0] = i
        for j in range(1, n + 1):
            dp[j] = (prev[j-1] if a[i-1] == b[j-1]
                     else 1 + min(prev[j], dp[j-1], prev[j-1]))
    return dp[n]


def novelty_score(generated: List[str], reference: List[str]) -> float:
    scores = [min(levenshtein(g,r)/max(len(g),len(r),1) for r in reference)
              for g in generated]
    return sum(scores) / max(len(scores), 1)


def diversity_score(seqs: List[str]) -> float:
    if len(seqs) < 2: return 0.0
    total, count = 0.0, 0
    for i in range(len(seqs)):
        for j in range(i+1, len(seqs)):
            total += levenshtein(seqs[i],seqs[j]) / max(len(seqs[i]),len(seqs[j]),1)
            count += 1
    return total / count


def shannon_entropy(seqs: List[str]) -> float:
    if not seqs: return 0.0
    L = max(len(s) for s in seqs)
    total_h = 0.0
    for pos in range(L):
        freq: dict = {}
        for s in seqs:
            c = s[pos] if pos < len(s) else "[PAD]"
            freq[c] = freq.get(c, 0) + 1
        n   = sum(freq.values())
        h   = -sum((v/n)*math.log2(v/n) for v in freq.values() if v > 0)
        total_h += h
    return total_h / L


def fid_nucleotide(real: List[str], gen: List[str],
                   vocab: str = "ATUGC") -> float:
    def freq_vec(seqs):
        counts = {v: 0 for v in vocab}; total = 0
        for s in seqs:
            for c in s.upper():
                if c in counts: counts[c] += 1; total += 1
        return [counts[v]/max(total,1) for v in vocab]
    rv, gv = freq_vec(real), freq_vec(gen)
    return math.sqrt(sum((r-g)**2 for r,g in zip(rv,gv)))


def evaluate_batch(generated: List[str], reference: List[str],
                   as_rna: bool = False) -> dict:
    gc_vals  = [gc_content(s)  for s in generated]
    mfe_vals = [mfe_proxy(s)   for s in generated]
    gc_mean  = sum(gc_vals)/max(len(gc_vals),1)
    return {
        "n_generated":    len(generated),
        "gc_mean":        round(gc_mean, 4),
        "gc_std":         round((sum((x-gc_mean)**2 for x in gc_vals)/
                                  max(len(gc_vals)-1,1))**0.5, 4),
        "mfe_proxy_mean": round(sum(mfe_vals)/max(len(mfe_vals),1), 4),
        "novelty":        round(novelty_score(generated, reference), 4),
        "diversity":      round(diversity_score(generated), 4),
        "shannon_H":      round(shannon_entropy(generated), 4),
        "fid":            round(fid_nucleotide(reference, generated), 6),
    }
