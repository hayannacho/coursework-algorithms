# Coursework algorithms

Small implementations from my Georgia Tech MS coursework (Bioinformatics, Computational Science and Engineering), collected in one place. Each folder holds one module, its tests, and any example data it needs.

These are learning exercises, not research projects. Inputs are small generated examples; course datasets and assignment handouts are not included.

## Sequence analysis

| Folder | What it implements | Course |
|---|---|---|
| [`affine_global_alignment`](affine_global_alignment) | Global DNA alignment with affine gap penalties: three-state dynamic programming and traceback | CSE 6301, Fall 2025 |
| [`six_frame_translation`](six_frame_translation) | Six-frame translation with the standard genetic code, reverse complement, input validation | BIOL 6505, Fall 2025 (team) |
| [`primer_design`](primer_design) | Primer3 primer pair around a genomic position, 600–800 bp product | BIOL 6505, Fall 2025 (team) |
| [`local_alignment_paths`](local_alignment_paths) | Optimal local alignment score and the count of all tied traceback paths | CSE 6301, Fall 2025 |
| [`rad_seq_coverage`](rad_seq_coverage) | Recognition-window coverage for single RAD, ddRAD and 2bRAD, intersected with variant positions | BIOL 6505, Fall 2025 (team) |

## Machine learning

| Folder | What it implements | Course |
|---|---|---|
| [`protein_attention_classifier`](protein_attention_classifier) | Amino-acid one-hot encoding, multi-head self-attention, padding-masked mean pooling (PyTorch) | CX 4803, Spring 2025 |
| [`graph_learning_baselines`](graph_learning_baselines) | MLP and GCN node classifiers, pooled GCN graph classifier (PyTorch Geometric) | CX 4803, Spring 2025 |
| [`hmm_state_inference`](hmm_state_inference) | Scaled forward–backward smoothing for a two-state hidden Markov model | CSE 6740, Spring 2026 |
| [`kmeans_kmedoids`](kmeans_kmedoids) | K-means and K-medoids from scratch, compared with and without outliers | CSE 6740, Spring 2026 |
| [`matrix_factorization`](matrix_factorization) | Biased low-rank rating model trained with SGD, optional L2 regularization | CSE 6740, Spring 2026 |
| [`gmm_em_clustering`](gmm_em_clustering) | Full-covariance Gaussian mixtures and expectation–maximization in NumPy | CSE 6740, Spring 2026 |

## Simulation and data analysis

| Folder | What it implements | Course |
|---|---|---|
| [`bacterial_competition`](bacterial_competition) | Two-species killing and reproduction on a square lattice | BIOL 6505, Fall 2025 |
| [`climate_finance`](climate_finance) | Join emissions with mitigation-finance needs, normalize country names, plot coal share | CS 6730, Spring 2026 (team) |

Homework scaffolds supplied the function interfaces for the CX 4803, CSE 6740 and BIOL 6505 items; the implementations inside them are mine, or my team's where marked.

## Run

Python 3.11.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pytest -q
```

49 tests cover all thirteen modules. Most modules also run directly:

```bash
python affine_global_alignment/affine_alignment.py ACGT ACT
python six_frame_translation/six_frame.py ATGGCCTAA
python hmm_state_inference/forward_backward.py 1 -1 1 1 --q 0.7
```

PyTorch and PyTorch Geometric are needed only for the two CX 4803 folders, Primer3 only for `primer_design`, and Biopython only for `rad_seq_coverage`.
