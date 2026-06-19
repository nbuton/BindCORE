# BindCORE Studio

Interactive notebooks for **BindCORE** LIP & MoRF prediction straight from a
protein sequence — no precomputed ensembles required.

| Notebook | Where it runs |
|---|---|
| [`BindCORE_Local.ipynb`](BindCORE_Local.ipynb) | Jupyter on your own machine (CPU or CUDA) |
| [`BindCORE_Colab.ipynb`](BindCORE_Colab.ipynb) | Google Colab (GPU runtime recommended) |

## What it does

A single "studio" cell sets up the environment once, then opens an interactive
panel. For each queued sequence it runs:

```
sequence
  → IDPFold2                     Cα conformational ensemble (flow-matching MoE)
  → cg2all (isolated env)        all-atom topology + trajectory
  → EnsembleMDP                  scalar / local / pairwise features → HDF5
  → BindCORE Transformer         per-residue LIP & MoRF probabilities
```

Outputs are interactive Plotly figures: per-residue LIP/MoRF tracks, a
conformational-ensemble fingerprint (Flory ν, Rg, shape metrics), a report card,
and cross-protein comparison plots.

## Running locally

```bash
# from inside your BindCORE checkout
jupyter lab colab/BindCORE_Local.ipynb
```

Run the first cell. It auto-locates the repo, installs dependencies with `uv`,
downloads the IDPFold2 weights, and builds `cg2all` in an isolated environment.
First-run setup takes ~10–15 min; subsequent runs reuse it. A CUDA PyTorch build
is strongly recommended (IDPFold2 is slow on CPU; cg2all backmapping is CPU-only
by design).

