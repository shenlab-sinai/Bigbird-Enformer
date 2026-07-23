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

## Repository layout

```text
src/bigbird_enformer/  # importable model, layer, training, and utility code
tests/                 # pytest tests
scripts/               # evaluation and fine-tuning entry points
```

Install the package and its test dependencies into the active environment:

```bash
python -m pip install -e ".[test,gtex]"
```

Run the test suite from the repository root:

```bash
python -m pytest
```

### Test suites

The tests use synthetic data by default and are divided into three groups:

- Fast CPU tests cover configuration, sequence encoding, attention layers,
  preprocessing, temporary-file datasets, model behavior, and training helpers.
- Slow tests exercise checkpointing and a full 196,608-base sparse forward pass.
- GPU and integration tests are opt-in because they require CUDA hardware or
  external Atlas data and checkpoints.

Run the fast local suite:

```bash
python -m pytest -m "not slow and not gpu and not integration"
```

Run all CPU tests, including the full-length smoke tests:

```bash
python -m pytest -m "not gpu and not integration"
```

Run only the slow tests:

```bash
python -m pytest -m slow
```

Run CUDA tests on Atlas or another GPU host:

```bash
python -m pytest -m gpu
```

Run an integration check against an external NPZ dataset:

```bash
ATLAS_TEST_DATA=/path/to/enformer_flat_npy/human \
ATLAS_TEST_SPLIT=test \
python -m pytest -m integration tests/test_datasets.py
```

Validate the structure of an external Lightning checkpoint:

```bash
ATLAS_TEST_CHECKPOINT=/path/to/model.ckpt \
python -m pytest -m integration tests/test_gpu_smoke.py
```

Run a single file or test during development:

```bash
python -m pytest tests/test_attention.py
python -m pytest tests/test_model.py::test_model_accepts_indices_and_one_hot
```

The cCRE sparse-attention tests use fixed top-k masks, matching the production
design: every sample has the same number of global tokens while their positions
can differ, allowing efficient batched GPU execution.

Run the standalone evaluation and GTEx fine-tuning workflows with:

```bash
python -m bigbird_enformer.train_lightning
python scripts/evaluate.py
GTEX_FOLD=0 python scripts/gtex_finetune.py
```
