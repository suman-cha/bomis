# BOMiS: Bounded Oversampling with Majority-informed Support

Code for **"Certifying Structural Zeros for Categorical Minority Oversampling"**.

BOMiS is an oversampler for imbalanced categorical data. A small minority sample misses configurations that are possible but unobserved. These are **sampling zeros**. The same sample also cannot show which configurations are impossible for the minority class. These are **structural zeros**. BOMiS generates new minority configurations and avoids the impossible ones in three steps.

1. **Certify structural zeros.** Some patterns of at most two features are frequent in the majority class and never occur in the minority class. An exact two-sample test certifies these as minority structural zeros, with family-wise error control at level α. The certified patterns are removed from the support estimate Ŝ.
2. **Allocate the novel mass.** The Good-Turing estimate of the missing mass sets the share of generated rows that are new configurations. The Chao1 estimate sets their number of distinct cells.
3. **Decode inside the support.** A class-conditional autoregressive Transformer is trained on both classes. It draws the new configurations under the constraint of Ŝ.

## Installation

The code requires Python 3.11.

```bash
pip install -r requirements.txt
```

The experiments run on CPU.

## Quickstart

```python
from bomis import BOMiS

# X0: majority rows, X1: minority rows, integer-coded arrays of shape (n, d) with levels 0..K[j]-1
model = BOMiS(seed=0).fit(X0, X1, K)
X_new = model.sample(len(X0) - len(X1))   # synthetic minority rows
```

A runnable example is in `python examples/quickstart.py`.

## Reproducing the experiments

```bash
bash run_all.sh
```

This one command does three things.

1. It downloads the twelve datasets into `data/raw` and checks their hashes.
2. It runs BOMiS on every dataset for seeds 0 to 19, writing `results/<dataset>_<seed>.csv`.
3. It writes the mean and standard deviation per dataset to `results/summary.csv`.

To run a subset:

```bash
SEEDS="0 1 2" DATASETS="car vote" bash run_all.sh
python scripts/run.py --dataset car --seeds 0     # a single dataset
```

### Evaluation protocol

Each dataset defines a population, a pair of class laws p₀ and p₁ whose supports S₀ and S₁ are known exactly. Leakage and recovered mass are therefore measured against the truth. There are two kinds of population.

- **Enumerated domains:** nursery2, krk, monks1, car and tictactoe. The data list every legal configuration, and pᵧ is uniform on the class-y configurations.
- **Sampled tables:** phish, naticus, moto, krkp, mushroom, splice and vote. Sᵧ is the order-2 closure of the class-y rows, and pᵧ is a Chow-Liu tree law restricted to Sᵧ.

For each seed:

- Independent training samples D₀ ~ p₀ and D₁ ~ p₁ are drawn. D₀ is 80% of the majority size of the table. D₁ is 8% of the minority size, with at least 50 rows.
- A balanced test sample of 2,000 rows per class is drawn.
- BOMiS generates |D₀| − |D₁| minority rows.

| Metric | Column | Better | Definition |
|---|---|---|---|
| Sampling-zero recovery | `recovery` | higher | p₁ mass of the distinct generated cells in S₁ \ O, divided by the missing mass p₁(S₁ \ O). O is the set of observed minority rows. |
| Structural-zero leakage | `leak` | lower | Share of generated rows outside S₁. |
| Structural-zero control | `control` | higher | 1 − leak / novel, the share of novel generated rows that are feasible. |
| Marginal fidelity | `uni_tv` | lower | Mean one-way total variation against 50,000 reference rows of p₁. |
| Cross-column dependency | `pair_tv` | lower | Mean two-way total variation against the same reference rows. |
| ML utility | `auroc` | higher | Test AUROC of a linear SVM on one-hot features, trained on D₀, D₁ and the generated rows. |

Bit-identical rows need the package versions in `requirements.txt` and an x86-64 CPU. Other platforms can differ in the last floating-point digits.

## Repository layout

```
bomis/model.py         Support certification, autoregressive Transformer, BOMiS sampler
bomis/data.py          dataset loaders
bomis/populations.py   populations with known supports and the training/test split
bomis/metrics.py       evaluation metrics
scripts/               download, run and summarize
examples/quickstart.py minimal usage example
run_all.sh             full reproduction
```

## Main hyperparameters

All experiments use the defaults of `BOMiS()`, with no per-dataset tuning.

| Argument | Default | Meaning |
|---|---|---|
| `k` | 2 | Largest order of the certified patterns. |
| `r` | 0.5 | Representation floor of the certificate, π₁ ≥ r π₀. |
| `alpha` | 0.05 | Family-wise error level of the certificates. |
| `steps` | (250, 500, 1000, 2000) | Training lengths, chosen by 5-fold held-out log loss. |
| `d_model`, `n_layers`, `n_heads` | 64, 2, 4 | Transformer size. |
| `batch`, `lr` | 256, 1e-3 | AdamW with a one-cycle schedule. |

## Data

The datasets are not redistributed. `scripts/download_data.py` fetches them from their original sources.

| Dataset | Source |
|---|---|
| car | UCI Car Evaluation (id 19) |
| nursery2 | UCI Nursery (id 76) |
| tictactoe | UCI Tic-Tac-Toe Endgame (id 101) |
| krk | UCI Chess (King-Rook vs. King) (id 23) |
| monks1 | UCI MONK's Problems (id 70) |
| krkp | UCI Chess (King-Rook vs. King-Pawn) (id 22) |
| mushroom | UCI Mushroom (id 73) |
| phish | UCI Phishing Websites (id 327) |
| naticus | UCI NATICUSdroid (id 722) |
| splice | UCI Molecular Biology (Splice-junction Gene Sequences) (id 69) |
| vote | UCI Congressional Voting Records (id 105) |
| moto | MotoDroid 22K, from the Malware-Hunter/Dataset-Matters repository |

The UCI datasets are under CC BY 4.0. Please follow the terms of each original source.

## License

MIT
