# BigBird-Enformer: Sparse Attention for Long-Sequence Genomics

This project modifies the original **Enformer** architecture by replacing its standard **full self-attention (O(N²))** with **custom sparse attention** variants designed for long genomic sequences.

The main goal is to **reduce memory usage and wall-clock training time** while preserving Enformer’s ability to model long-range regulatory interactions.

---

## Project Goals

This repository is built to answer the following questions:

- Can Enformer be trained efficiently with **true sparse attention** at Enformer-scale sequence lengths?
- How does sparse attention affect:
  - training throughput
  - GPU memory usage
  - final prediction performance (Pearson R)
- Can hierarchical sparse attention preserve long-range information while remaining computationally efficient?

---

## Implemented Attention Variants

This project compares three attention modes under the same training pipeline and dataset:

### 1. Full Attention (Baseline)

- Original Enformer-style full attention.
- Complexity: **O(N²)**
- Best accuracy baseline, but expensive at long sequence lengths.

---

### 2. BigBird-Style Block Sparse Attention

BigBird attention replaces full attention with a structured sparse pattern consisting of:

- **Local block attention**
  - Each token attends to tokens within a small local block window.
- **Global tokens**
  - A small number of special tokens attend to all tokens.
  - All tokens attend to the global tokens.

This preserves long-range communication while keeping computation near linear.

### 3. Hierarchical Sparse Attention

This project introduces a custom **hierarchical sparse attention** pattern.

The sequence is divided into a configurable number of chunks