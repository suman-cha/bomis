"""Fit BOMiS on a small synthetic categorical table and oversample the minority class.

  python examples/quickstart.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from bomis import BOMiS  # noqa: E402

rng = np.random.default_rng(0)
K = [3, 3, 4, 2, 5]  # number of levels of each feature
X0 = np.stack([rng.integers(k, size=1000) for k in K], 1)  # majority rows
X1 = np.stack([rng.integers(k, size=60) for k in K], 1)    # minority rows
X1[:, 1] = np.minimum(X1[:, 1], 1)  # level 2 of feature 1 never occurs in the minority class

model = BOMiS(steps=250, seed=0).fit(X0, X1, K)
G = model.sample(len(X0) - len(X1))

observed = {r.tobytes() for r in X1}
print("generated rows:", G.shape)
print("certified structural zeros:", model.support.n_certified)
print("share of novel rows:", np.mean([r.tobytes() not in observed for r in G]))
print("rows with feature 1 = 2:", int((G[:, 1] == 2).sum()))
