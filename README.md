# SMishReLU: A Continuous C¹-Smooth Hybrid Activation Function for Deep Neural Networks

This repository accompanies the manuscript **"SMishReLU: A Continuous C¹-Smooth Hybrid
Activation Function Eliminating First-Derivative Discontinuity in Deep Neural Networks"**
and provides the experimental code used to evaluate the proposed activation function
against MishReLU and standard baseline activations.

## 1. Background

MishReLU combines the identity mapping of ReLU (for `x > 0`) with the smooth
non-monotonic Mish function (for `x <= 0`). However, MishReLU has a first-derivative
jump discontinuity at `x = 0`:

- Right-hand derivative: `1.0000`
- Left-hand derivative: `tanh(ln 2) ≈ 0.6000`

This mismatch (`Δ ≈ 0.4`) can cause gradient shocks during backpropagation.

**SMishReLU** fixes this by scaling the negative branch with
`α = 1 / tanh(ln 2) ≈ 1.6667`, so both derivatives at `x = 0` equal exactly `1.0000`:

```
SMishReLU(x) = x                                   if x > 0
             = α * x * tanh(softplus(x))            if x <= 0
```

## 2. Datasets

All datasets are public benchmark datasets. Four load automatically via TensorFlow/Keras
APIs; one (IMDB) must be attached manually on Kaggle.

| Dataset | Loaded via | Manual step needed |
|---|---|---|
| MNIST | `tensorflow.keras.datasets.mnist` | None (auto-downloads) |
| Fashion-MNIST | `tensorflow.keras.datasets.fashion_mnist` | None (auto-downloads) |
| CIFAR-100 | `tensorflow.keras.datasets.cifar100` | None (auto-downloads) |
| Reuters Newswire | `tensorflow.keras.datasets.reuters` | None (auto-downloads) |
| IMDB (50K Movie Reviews) | `pandas.read_csv` from a Kaggle input path | **Yes** — attach the Kaggle dataset "IMDB Dataset of 50K Movie Reviews" (see Section 5) |

## 3. Models

The following architectures are implemented, matching the original MishReLU study
for a fair, like-for-like comparison:

- **MLP** — Fashion-MNIST
- **CNN (3-block)** — MNIST
- **ResNet-18** — CIFAR-100
- **LSTM** — IMDB (binary sentiment)
- **BiLSTM** — Reuters (46-class topic classification)

All architectures accept a swappable `activation` argument so every activation
function is tested under identical conditions.

## 4. Activation Functions Compared

- **SMishReLU** (proposed)
- MishReLU (baseline hybrid from the prior paper)
- ReLU
- Mish
- ELU
- LeakyReLU
- SELU

Each is evaluated with:
- Accuracy
- Precision (macro)
- Recall (macro)
- F1-score (macro)

## 5. How to Run

### Recommended platform: **Kaggle Notebooks**

The code reads the IMDB dataset from `/kaggle/input/...`, which is Kaggle's dataset
mount path. Kaggle also provides free GPU (T4 / P100) with generous weekly hours,
which is needed for the CNN/ResNet-18/LSTM/BiLSTM training runs.

Steps:

1. Go to [kaggle.com](https://www.kaggle.com) → **New Notebook**.
2. Enable a GPU: **Settings (right panel) → Accelerator → GPU T4 x2 or P100**.
3. Attach the IMDB dataset:
   - Click **Add Input** in the right sidebar.
   - Search for **"IMDB Dataset of 50K Movie Reviews"**.
   - Click **Add**. It will mount at:
     `/kaggle/input/imdb-dataset-of-50k-movie-reviews/IMDB Dataset.csv`
   - Confirm this path matches what's used in the code (case-sensitive).
4. Upload/paste `smishrelu_experiments.py` (or the `.ipynb` version) into the notebook.
5. Run all cells.

> **Colab is not recommended as-is** — the IMDB loading path is Kaggle-specific. To use
> Colab, either upload the CSV manually and change the path, or use
> `tensorflow.keras.datasets.imdb` (a different tokenized format that would require
> changing the preprocessing code).

### Before running the full experiment

The full loop trains **7 activation functions × 3 learning rates × 5 random seeds**
per architecture — this is computationally expensive (many hours per architecture,
especially ResNet-18 on CIFAR-100). Before committing to a full run:

- Temporarily reduce `SEEDS` to 1–2 values and `epochs` to a small number (e.g. 5) to
  confirm the pipeline runs end-to-end without errors.
- Once verified, restore `SEEDS = [164, 343, 865, 599, 251]` and the full epoch count
  for final results.
- Kaggle sessions have a runtime limit (~9–12 hours) — long architectures (ResNet-18)
  may need to be split across sessions or run with fewer seeds if time-constrained.

## 6. Statistical Analysis

Statistical significance between activation functions is assessed using:

- **Friedman test** — overall comparison across all activation functions
- **Post-hoc Wilcoxon signed-rank test** — pairwise comparisons between activations

All statistical analyses are performed on results from repeated runs (5 seeds) at each
learning rate.

## 7. Reproducibility

All experiments fix random seeds across Python, NumPy, and TensorFlow via a shared
`set_seed()` function, and report results as mean ± standard deviation across the
5 seeds used.

## 8. Known Notes on the Code

- The CNN training/evaluation loop (Friedman + Wilcoxon tests) is fully implemented
  for **MNIST**. The `build_*` functions for Fashion-MNIST (MLP), CIFAR-100
  (ResNet-18), IMDB (LSTM), and Reuters (BiLSTM) are provided but their full
  train-and-evaluate loops (mirroring the MNIST loop) still need to be written for a
  complete multi-dataset comparison, matching Tables 2–6 of the manuscript.
