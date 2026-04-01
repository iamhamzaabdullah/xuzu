<div align="center">

<!-- Logo / Banner -->
<img src="https://capsule-render.vercel.app/api?type=waving&color=0:00d4aa,100:5b8fff&height=200&section=header&text=XUZU&fontSize=80&fontColor=ffffff&fontAlignY=38&desc=X-modal%20Unified%20Zero-shot%20Universal%20Aptamer%20Language%20Model&descAlignY=58&descSize=18&animation=fadeIn" width="100%"/>

<br/>

[![Version](https://img.shields.io/badge/version-1.0.0-00d4aa?style=for-the-badge&logo=github&logoColor=white)](https://github.com/iamhamzaabdullah/xuzu)
[![Python](https://img.shields.io/badge/python-3.9%2B-5b8fff?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![Lab](https://img.shields.io/badge/terminalBio-Hamza%20A-a78bfa?style=for-the-badge&logo=atom&logoColor=white)](https://terminalbio.io)
[![License](https://img.shields.io/badge/license-Proprietary-ff6b6b?style=for-the-badge)](LICENSE)

<br/>

> **XUZU** is a de novo, multi-modal nucleotide language model for DNA and RNA aptamer design.  
> It unifies sequence, secondary structure, and protein-target context through two proprietary blocks —  
> **FEBI** and **RJ** — generating novel aptamers without requiring SELEX data.

<br/>

**Built by [Hamza A](https://hamzaabdullah.medium.com/about) · [terminalBio](https://terminalbio.io)**

<br/>

</div>

---

## ✦ What is XUZU?

**XUZU** *(X-modal Unified Zero-shot Universal)* is a general-purpose aptamer language model designed to work across **any protein target** — not a single-target tool. It is built as a launchable research platform that will evolve into a Hugging Face-hosted backbone with a fine-tuning API for research teams worldwide.

XUZU takes three parallel inputs — a nucleotide sequence, its secondary structure, and a target protein pocket — fuses them through proprietary cross-modal attention, and generates candidate aptamers through an absorbing discrete diffusion decoder. A built-in **GER (Generation–Evaluation–Refinement)** loop further tunes the decoder using a binding affinity reward signal.

---

## ✦ Architecture

```
                    ┌─────────────────────────────────────────────────────────┐
                    │                    XUZU v1.0.0                          │
                    │               Built by Hamza A @ terminalBio            │
                    └─────────────────────────────────────────────────────────┘

  [Nucleotide Sequence]  ──►  NucleotideLanguageEncoder  ──[ FEBI × 6 ]──┐
                                                                           │
  [Dot-bracket Structure] ──►  StructureGraphEncoder     ──[  RJ  × 3 ]──►  CrossModalFusion  ──►  DiscreteDiffusionDecoder  ──►  Aptamer
                                                                           │         ▲
  [Protein Pocket AA seq] ──►  TargetProteinEncoder      ──[ FEBI × 3 ]──┘         │
                                                                               GER Loop
                                                                        (BindingAffinitySurrogate
                                                                         + REINFORCE Policy Grad)
```

<br/>

### 🟣 FEBI — Fundamental Encoding Block Intelligence
*Proprietary · Hamza A @ terminalBio*

The primary transformer block powering XUZU's sequence and protein encoders. FEBI combines **RoPEAttention** (multi-head self-attention with Rotary Positional Encoding), a **gated feed-forward network** (GELU, 4× expansion), and pre-norm residual connections — enabling deep context encoding without positional bias.

### 🔵 RJ — Relational Junction
*Proprietary · Hamza A @ terminalBio*

The graph attention block powering XUZU's structure encoder. RJ operates over the **base-pair contact graph** derived from dot-bracket notation, propagating structural context across bonded nucleotide positions — capturing stem-loops, hairpins, and kissing-loop motifs that define aptamer binding geometry.

---

## ✦ Module Map

| Module | Proprietary Block | Role |
|--------|:-----------------:|------|
| `xuzu/layers.py` | **FEBI · RJ** | Core transformer + graph attention building blocks |
| `xuzu/encoders.py` | FEBI · RJ | Three independent encoder towers |
| `xuzu/fusion.py` | — | Cross-modal gated fusion (3-way softmax gate) |
| `xuzu/decoder.py` | FEBI | Absorbing-state D3PM diffusion decoder |
| `xuzu/reward.py` | — | Binding affinity surrogate + GER closed-loop refinement |
| `xuzu/tokenizer.py` | — | DNA / RNA / modified-base nucleotide tokenizer |
| `xuzu/data.py` | — | Dataset, augmentation, dataloaders |
| `xuzu/metrics.py` | — | Q1-publication-grade evaluation metrics |
| `xuzu/trainer.py` | — | Production trainer: AMP, warmup/cosine LR, early stopping |
| `xuzu/model.py` | — | Full XUZU model assembly + `XUZUConfig` |

---

## ✦ Installation

```bash
# Python 3.9+ required
pip install torch>=2.0.0 numpy>=1.24.0

# Install XUZU in editable mode
git clone https://github.com/iamhamzaabdullah/xuzu.git
cd xuzu
pip install -e .

# Verify
python -c "import xuzu; print(xuzu.__version__, '|', xuzu.__author__, '@', xuzu.__lab__)"
# → 1.0.0 | Hamza A @ terminalBio

# Optional: structural evaluation via RNAfold
pip install RNA>=2.6.0
```

---

## ✦ Quick Start

```python
from xuzu import XUZU, XUZUConfig

# Initialise model (~7.5M parameters, default config)
model = XUZU(XUZUConfig())
print(f"XUZU v{xuzu.__version__} | {model.num_parameters():,} parameters")
print(f"Author : {xuzu.__author__} @ {xuzu.__lab__}")

# Design RNA aptamers for any target pocket
aptamers = model.design(
    pocket_seq   = "SMKDYSFLTQFPGFVKHFNSLGGDGVQ",   # target binding pocket (AA)
    seq_len      = 45,
    n_candidates = 10,
    temperature  = 0.85,
    as_rna       = True,
)

for i, apt in enumerate(aptamers, 1):
    print(f"[{i:02d}] {apt}")
```

---

## ✦ CLI Usage

### Training
```bash
python train.py \
  --data     aptamers.jsonl \
  --epochs   200            \
  --d_model  256            \
  --device   cuda           \
  --save     xuzu_best.pt   \
  --log      train.log
```

### Aptamer Design
```bash
python design.py \
  --pocket   "SMKDYSFLTQFPGFVKHFNSLGGDGVQ" \
  --model    xuzu_best.pt                   \
  --len      45                             \
  --n        50                             \
  --rna                                     \
  --evaluate
```

---

## ✦ Training Data Format

One JSON record per line (`.jsonl`). Only `seq` is required.

```jsonc
// Minimal
{"seq": "GCGAUAGCUAGCUAGCU"}

// Full record
{
  "seq":       "GCGAUAGCUAGCUAGCUAGCUA",
  "structure": "((((.......))))",
  "pocket":    "SMKDYSFLTQFPGFVKHFNSLGG",
  "kd_nm":     12.5
}
```

| Field | Required | Description |
|-------|:--------:|-------------|
| `seq` | ✅ | Aptamer nucleotide sequence — DNA or RNA |
| `structure` | optional | Dot-bracket secondary structure (same length as `seq`) |
| `pocket` | optional | Target binding pocket amino-acid residues |
| `kd_nm` | optional | Measured Kd in nM — required for GER loop training |

---

## ✦ Nucleotide Vocabulary

| Token | Base | Notes |
|:-----:|------|-------|
| `A T G C` | DNA standard | |
| `A U G C` | RNA standard | |
| `F` | 2′-Fluoro-C | Modified — serum stability ↑ |
| `M` | 2′-O-Methyl | Modified — nuclease resistance ↑ |
| `L` | LNA | Locked Nucleic Acid — affinity ↑ |
| `P` | Phosphorothioate | Backbone modification |
| `[MASK]` | — | Absorbing diffusion noise token |
| `[BOS] [EOS]` | — | Sequence boundary tokens |

---

## ✦ Evaluation Metrics

| Metric | Function | Target |
|--------|----------|--------|
| `gc_mean` | `gc_content()` | 40–60% |
| `mfe_proxy_mean` | `mfe_proxy()` | Lower = more stable |
| `novelty` | `novelty_score()` | > 0.6 vs. training set |
| `diversity` | `diversity_score()` | Higher = broader ensemble |
| `shannon_H` | `shannon_entropy()` | Higher = less repetitive |
| `fid` | `fid_nucleotide()` | Lower = distribution match |

---

## ✦ Roadmap

| Version | Target | Status |
|---------|--------|:------:|
| **v1.0.0** — Proof of concept, FEBI + RJ architecture | Apr 2026 | ✅ Released |
| **v1.5.0** — ESM-2 pocket encoder + extended XNA tokenizer | Jun 2026 | 🔧 Planned |
| **v2.0.0** — Pre-trained backbone on RNAcentral + AptaBase (50K+ records) | Aug 2026 | 🔧 Planned |
| **v2.5.0** — Benchmark paper + Hugging Face Hub release | Oct 2026 | 🔧 Planned |
| **v3.0.0** — terminalBio web platform + Design/Fine-tune API | Jan 2027 | 🔧 Planned |

→ See [`ROADMAP.md`](ROADMAP.md) for full details.

---

## ✦ Citation

If you use XUZU in your research, please cite:

```bibtex
@article{hamza2026xuzu,
  title   = {XUZU: A Multi-Modal Nucleotide Language Model for De Novo Aptamer Design},
  author  = {Hamza A},
  journal = {bioRxiv},
  year    = {2026},
  note    = {terminalBio Technical Report v1.0.0}
}
```

---

## ✦ License

**terminalBio Proprietary Software.**  
Copyright © 2026 Hamza A / terminalBio. All rights reserved.  
Unauthorised copying, distribution, or modification is strictly prohibited.

---

<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:5b8fff,100:00d4aa&height=100&section=footer" width="100%"/>

**[terminalBio](https://terminalbio.io)** · Built with ♦ by **Hamza A**

</div>
