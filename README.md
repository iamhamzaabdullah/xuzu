# XUZU — Multi-Modal Nucleotide Language Model
### Built by **Hamza A** · [terminalBio](https://terminalbio.io)
> Version 1.0.0 · terminalBio Proprietary · All Rights Reserved

---

## What is XUZU?

**XUZU** (X-modal Unified Zero-shot Universal) is a de novo, multi-modal
nucleotide language model for DNA and RNA aptamer design, developed entirely
at **terminalBio** by **Hamza A**.

It unifies sequence, structure, and protein-target context through two
proprietary blocks — **FEBI** and **RJ** — and generates novel aptamers
without requiring SELEX data.

---

## Proprietary Architecture

```
[Nucleotide Sequence] ──► NucleotideLanguageEncoder  [FEBI blocks]  ──┐
[Dot-bracket Structure] ─► StructureGraphEncoder      [RJ   blocks]  ──► CrossModalFusion ──► DiscreteDiffusionDecoder ──► Aptamer
[Protein Pocket AA seq] ─► TargetProteinEncoder       [FEBI blocks]  ──┘
                                                                             ▲
                                              GER Loop (BindingAffinitySurrogate + REINFORCE)
```

### FEBI — Fundamental Encoding Block Intelligence
> *Proprietary, Hamza A @ terminalBio*

The primary transformer block powering XUZU's sequence and protein encoders.
FEBI combines:
- **RoPEAttention** — multi-head self-attention with Rotary Positional Encoding
- **Gated feed-forward** (GELU activation, 4× expansion)
- Pre-norm residual connections (LayerNorm before each sublayer)

### RJ — Relational Junction
> *Proprietary, Hamza A @ terminalBio*

The graph attention block powering XUZU's structure encoder.
RJ operates over the base-pair contact graph derived from dot-bracket notation,
propagating structural context across bonded nucleotide positions.

---

## Module Map

| File | Contains | Key Exports |
|------|----------|-------------|
| `xuzu/layers.py` | **FEBI**, **RJ**, RoPE, Attention | `FEBI`, `RJ` |
| `xuzu/encoders.py` | Three encoder towers | `NucleotideLanguageEncoder`, `StructureGraphEncoder`, `TargetProteinEncoder` |
| `xuzu/fusion.py` | Cross-modal gated fusion | `CrossModalFusion` |
| `xuzu/decoder.py` | Absorbing discrete diffusion | `DiscreteDiffusionDecoder` |
| `xuzu/reward.py` | Affinity surrogate + GER loop | `BindingAffinitySurrogate`, `GERLoop` |
| `xuzu/tokenizer.py` | DNA/RNA/modified nucleotide tokenizer | `NucleotideTokenizer` |
| `xuzu/data.py` | Dataset, augmentation, dataloaders | `AptamerDataset`, `build_dataloaders` |
| `xuzu/metrics.py` | Q1-grade evaluation metrics | `evaluate_batch` |
| `xuzu/trainer.py` | Production trainer | `XUZUTrainer` |
| `xuzu/model.py` | Full XUZU model + config | `XUZU`, `XUZUConfig` |

---

## Installation

```bash
pip install torch>=2.0.0 numpy>=1.24.0
cd XUZU_PROJECT
pip install -e .
```

---

## Quick Start

```python
from xuzu import XUZU, XUZUConfig

model = XUZU(XUZUConfig())
print(f"XUZU | {model.num_parameters():,} parameters")
print(f"Built by: {model.__class__.__module__}")

# Design RNA aptamers for KPC-2 beta-lactamase pocket
aptamers = model.design(
    pocket_seq   = "SMKDYSFLTQFPGFVKHFNSLGGDGVQ",
    seq_len      = 45,
    n_candidates = 10,
    temperature  = 0.85,
    as_rna       = True,
)
for i, apt in enumerate(aptamers, 1):
    print(f"[{i}] {apt}")
```

---

## CLI Usage

### Training
```bash
python train.py \
  --data kpc2_aptamers.jsonl \
  --epochs 100 \
  --d_model 256 \
  --device cuda \
  --save xuzu_kpc2.pt \
  --log train.log
```

### Design
```bash
python design.py \
  --pocket "SMKDYSFLTQFPGFVKHFNSLGGDGVQ" \
  --model xuzu_kpc2.pt \
  --len 45 --n 50 \
  --rna --evaluate
```

---

## Training Data Format (JSONL)

```json
{"seq": "GCGAUAGCUAGCUAGCU", "structure": "(((...)))", "pocket": "ARNDCQEGHILK"}
{"seq": "ATCGATCGATCG",       "structure": "............", "pocket": "MFPSTWYV"}
```

| Field | Required | Description |
|-------|----------|-------------|
| `seq` | ✅ | Aptamer nucleotide sequence (DNA or RNA) |
| `structure` | optional | Dot-bracket secondary structure |
| `pocket` | optional | Binding pocket amino-acid residues |
| `kd_nm` | optional | Measured Kd in nM (for GER training) |

---

## Nucleotide Vocabulary

| Token | Base | Notes |
|-------|------|-------|
| A, T, G, C | DNA standard | |
| A, U, G, C | RNA standard | |
| F | 2'-Fluoro-C | Modified — serum stability |
| M | 2'-O-Methyl | Modified — nuclease resistance |
| L | LNA | Locked Nucleic Acid |
| P | Phosphorothioate | Backbone modification |
| [MASK] | — | Diffusion noise token |
| [BOS]/[EOS] | — | Sequence boundaries |

---

## Evaluation Metrics

| Metric | Description | Higher = Better? |
|--------|-------------|-----------------|
| `gc_mean` | Mean GC content | Target 40–60% |
| `mfe_proxy_mean` | Structural stability proxy | Lower (more negative) |
| `novelty` | Normalised edit distance vs. training set | ✅ Yes |
| `diversity` | Pairwise edit distance within ensemble | ✅ Yes |
| `shannon_H` | Per-position entropy | ✅ Yes |
| `fid` | Nucleotide frequency FID vs. training data | ❌ Lower |

---

## Citation

If you use XUZU in your research, please cite:

```
Hamza A. (2026). XUZU: A Multi-Modal Nucleotide Language Model for
De Novo Aptamer Design. terminalBio Technical Report v1.0.0.
```

---

## License

terminalBio Proprietary Software.
Copyright © 2026 Hamza A / terminalBio. All rights reserved.
Unauthorised copying, distribution, or modification is prohibited.
