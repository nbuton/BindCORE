# BindCORE: Biophysical Ensemble Learning for Predicting Interaction Sites in Intrinsically Disordered Regions

This repository contains the official implementation of **BindCORE**, an ensemble-aware deep learning framework designed to predict interaction sites (LIPs and MoRFs) within Intrinsically Disordered Regions (IDRs) by integrating global, local, and pairwise biophysical descriptors.

### Citation & Paper Metadata

> **Preprint / Under Review** (2026)
> *Briefings in Bioinformatics*, 2026, pp. 1–14
> **Authors:** Nicolas Buton, Luiz Felipe Piochi, and Hamed Khakzad
> **DOI:** *Added during production*

---

## 🚀 Quick Start: Run BindCORE Immediately

To make BindCORE accessible to everyone, you can run predictions directly from an amino acid sequence without any manual biophysical feature preparation. The tools below automatically generate the required conformational ensembles for you.

### Option A: Google Colab (Zero Setup — Recommended)

The absolute fastest way to test BindCORE using a free cloud GPU. Environment setup, dependencies, and model weights are handled completely automatically in your browser.

* **[Open in Colab](https://colab.research.google.com/github/nbuton/BindCORE/blob/main/colab/BindCORE_Colab.ipynb)**: Just input your protein sequence and click **Runtime > Run All**.

### Option B: Local Interactive Notebook

Run the same interactive prediction studio locally on your workstation or cluster (CPU or CUDA supported).

* **Automated Setup:** The first cell auto-locates the repository, handles isolation dependencies with `uv`, downloads the IDPFold2 weights, and builds `cg2all` (initial setup takes ~10–15 min; subsequent runs are instant).
* **Execution:**
```bash
# From inside your cloned BindCORE repository
jupyter lab colab/BindCORE_Local.ipynb
```



---

## 📖 Abstract

Intrinsically disordered proteins and regions (IDPs/IDRs) mediate diverse cellular functions through binding segments whose functional properties are encoded in dynamic conformational ensembles rather than a single static state. Existing predictors of linear interacting peptides (LIPs) and molecular recognition features (MoRFs) rely primarily on sequence-derived features, leaving ensemble-level biophysical properties largely unexplored.

**BindCORE** bridges this gap by integrating global, local, and pairwise biophysical descriptors processed through a multi-scale architecture to predict interaction sites within IDRs. Across established benchmarks, BindCORE consistently improves average precision, ROC-AUC, and Matthews correlation over sequence-based baselines. Feature-attribution analyses reveal that pairwise descriptors are the dominant contributors to prediction, alongside complementary signals from solvent accessibility, backbone dihedral entropy, and global geometric properties.

---

## ✨ Key Features

* **Ensemble-Aware Architecture:** Integrates dynamic structural properties rather than relying solely on static or sequence-only features.
* **Multi-Scale Descriptors:** Leverages global geometric properties, local residues, and pairwise spatial relationships.
* **Interpretable Predictions:** Built-in support for feature attribution mapping via DeepLIFT-SHAP exposes the exact biophysical signals driving interaction-site propensity.

---

## 🛠️ Repository Setup

To install the framework locally for custom workflows or replication, clone the repository from GitLab:

```bash
git clone https://gitlab.inria.fr/nbuton/BindCORE
cd BindCORE
```

Core environment dependencies are listed in `requirements.txt` and primarily include `torch`, `h5py`, `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `scipy`, `mdtraj`, and `tqdm`.

---

## ⚙️ High-Throughput Production Pipeline

Use this option for large-scale datasets or custom workflows where you provide your own pre-generated structural trajectories.

### 1. Generate & Organize Conformational Ensembles

BindCORE predicts based on structural dynamics, requiring full-atom ensembles:

* **Fold:** Generate coarse-grained ensembles using [IDPFold2](https://github.com/Junjie-Zhu/IDPFold2).
* **Backmap:** Convert them to full-atom resolution using `cg2all`.
* **Path Structure:** Place outputs into `data/conformational_ensemble/IDPFold2/[Protein_ID]/` containing both the topology (`top_AA.pdb`) and trajectory (`traj_AA.xtc`) files.

### 2. Run Production Inference

```bash
python scripts/train.py \
    --config data/models/bindcore_IDPFold2/config.yaml \
    --device cuda
```

---

## 🧬 Model Architecture & Features

The `ProteinMultiScaleTransformer` fuses four distinct input streams into a single embedding space, processed via Transformer blocks with a **PairwiseCNN** and **BiasedMultiHeadAttention**.

| Input Stream | Shape | Encoding Strategy |
| --- | --- | --- |
| **Amino-acid sequence** | `[B, L]` | Learned embedding + sinusoidal positional encoding |
| **Local (per-residue)** | `[B, nb_local, L]` | 2-layer MLP projection |
| **Scalar (per-protein)** | `[B, nb_scalar]` | 2-layer MLP → broadcast to sequence length |
| **Pairwise** | `[B, nb_pairwise, L, L]` | Windowed row extraction + MLP |

### Biophysical Feature Space

* **Scalar (Global):** Asphericity, radius of gyration, end-to-end distance, shape anisotropy metrics, gyration eigenvalues, and scaling exponent.
* **Local:** $\phi/\psi$ dihedral entropies, absolute/relative SASA (mean & std), and secondary structure propensities.
* **Pairwise:** Dynamic cross-correlation matrix (DCCM), ensemble-averaged contact map, and distance fluctuation matrix.

---

## 📊 Replicating Paper Experiments & Figures

Execute the bash pipeline sequentially to fully replicate the published experimental findings:

1. **Hyperparameter Tuning:** Run Tree-structured Parzen Estimators (TPE) optimization across 100 runs.
```bash
bash bash_pipeline/hyperparam_tuning.sh
```


2. **Model Training:** Move the optimal configuration file generated by TPE into `data/models/[model_flavor]/` and train the models.
```bash
bash bash_pipeline/training.sh
```


3. **Inference & Evaluation:** Compute metrics on the test dataset (saved to `data/predictions/`).
```bash
bash bash_pipeline/predictions.sh
```


4. **Feature Attribution:** Generate standardized DeepLIFT-SHAP importance scores (saved to `data/interpretability/`).
```bash
bash bash_pipeline/interpretability.sh
```


5. **Replicate Figures:** Run the individual scripts inside `figures_code/` (e.g., curves, importance plots, and graphical abstracts) utilizing the outputs generated in steps 3 and 4.

---

## 📁 Project Layout

```text
BindCORE/
├── analysis/                         # Downstream analysis notebooks & scripts
├── bindcore/                         # Core Library (Data, Engine, Eval, Modeling)
│   ├── config.py                     # Pydantic configuration schemas
│   ├── data/                         # Data handling & Pre-processing
│   ├── engine/                       # Execution Logic (trainer, predictor, attribution)
│   ├── eval/                         # Evaluation & Metrics
│   └── modeling/                     # Neural Network Architecture
├── data/                             # Raw data, datasets, predictions, and properties
├── figures_code/                     # Scripts to generate publication-ready plots
├── results/                          # Generated figures, curves, and summary CSVs
├── scripts/                          # End-to-end execution pipeline (train, evaluate, etc.)
├── tests/                            # Unit tests
├── README.md                         # Project documentation
├── requirements.txt                  # Environment dependencies
└── setup.py                          # Package installation configuration

```