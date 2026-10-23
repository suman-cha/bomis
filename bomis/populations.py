"""Populations with known class supports, built from the twelve datasets.

Each population is a pair of class laws p_0, p_1 whose supports S_0, S_1 are known exactly, so leakage
p(U minus S_1) and the missing mass p_1(S_1 minus O) are computed against the truth.

  enum     For a domain that lists every legal configuration once with its label (nursery2, krk, monks1,
           car, tictactoe), S_y is the set of class-y configurations and p_y is uniform on S_y.
  closure  For a sampled table (phish, naticus, moto, krkp, mushroom, splice, vote), S_y is the set of
           configurations all of whose levels and level pairs occur among the class-y rows, and p_y is the
           Chow-Liu tree law of these rows with add-1/2 smoothing, restricted to S_y.

The training samples D_0 and D_1 are independent samples of p_0 and p_1.
"""
import numpy as np

from .data import load

ENUM = ["nursery2", "krk", "monks1", "car", "tictactoe"]
CLOSURE = ["phish", "naticus", "moto", "krkp", "mushroom", "splice", "vote"]
SIZES = dict(frac0=0.8, frac1=0.08, min1=50, n_test=2000)  # sample sizes relative to the class-y rows of the table


def keys(X):
    return [r.tobytes() for r in np.ascontiguousarray(X, dtype=np.int64)]


class Tree:
    """Chow-Liu tree law of the rows X with add-a smoothing, rooted at feature 0."""

    def __init__(self, X, K, a=0.5):
        d, self.K = X.shape[1], K
        mi = np.zeros((d, d))
        for j in range(d):
            for l in range(j + 1, d):
                c = np.bincount(X[:, j] * K[l] + X[:, l], minlength=K[j] * K[l]).reshape(K[j], K[l]) + a
                p = c / c.sum()
                mi[j, l] = mi[l, j] = (p * np.log(p / p.sum(1, keepdims=True) / p.sum(0, keepdims=True))).sum()
        self.parent, self.order, done = [-1] * d, [0], {0}
        best, arg = mi[0].copy(), np.zeros(d, int)
        while len(done) < d:  # Prim's algorithm for the maximum spanning tree
            j = max((l for l in range(d) if l not in done), key=lambda l: best[l])
            self.parent[j] = int(arg[j])
            self.order.append(j)
            done.add(j)
            up = mi[j] > best
            best[up], arg[up] = mi[j][up], j
        root = np.bincount(X[:, 0], minlength=K[0]) + a
        self.cond = {0: np.log(root / root.sum())[None, :]}
        for j in self.order[1:]:
            pa = self.parent[j]
            c = np.bincount(X[:, pa] * K[j] + X[:, j], minlength=K[pa] * K[j]).reshape(K[pa], K[j]) + a
            self.cond[j] = np.log(c / c.sum(1, keepdims=True))

    def sample(self, n, rng):
        X = np.zeros((n, len(self.K)), np.int64)
        for j in self.order:
            P = np.exp(self.cond[j][np.zeros(n, int) if j == 0 else X[:, self.parent[j]]])
            X[:, j] = (rng.random((n, 1)) > np.cumsum(P, 1)[:, :-1]).sum(1)
        return X

    def log_prob(self, X):
        return sum(self.cond[j][np.zeros(len(X), int) if j == 0 else X[:, self.parent[j]], X[:, j]] for j in self.order)


class Population:
    """Class laws with known supports for one dataset. ref holds n_ref reference rows of each class."""

    def __init__(self, name, n_ref=50000):
        X, y, K = load(name)
        self.name, self.K, self.kind = name, K, "enum" if name in ENUM else "closure"
        self.n_table = [int((y == c).sum()) for c in (0, 1)]
        rng = np.random.default_rng(0)
        d = len(K)
        # Level and level-pair tables of each class, keyed by (j, l) with j <= l.
        self.pairs = [{(j, l): np.zeros((K[j], K[l]), bool) for j in range(d) for l in range(j, d)} for _ in (0, 1)]
        for c in (0, 1):
            for (j, l), t in self.pairs[c].items():
                t[X[y == c][:, j], X[y == c][:, l]] = True
        if self.kind == "enum":
            self.cells = [np.unique(X[y == c], axis=0) for c in (0, 1)]
            self.members = [set(keys(C)) for C in self.cells]
        else:
            self.tree = [Tree(X[y == c], K) for c in (0, 1)]
            # Normalizer p_tree(S_y), estimated once from n_ref tree draws.
            self.log_z = [np.log(self.contains(self.tree[c].sample(n_ref, rng), c).mean()) for c in (0, 1)]
        self.ref = [self.sample(n_ref, c, rng) for c in (0, 1)]

    def contains(self, X, c):
        """Membership of the rows X in S_c."""
        if self.kind == "enum":
            return np.array([k in self.members[c] for k in keys(X)], bool)
        ok = np.ones(len(X), bool)
        for (j, l), t in self.pairs[c].items():
            ok &= t[X[:, j], X[:, l]]
        return ok

    def sample(self, n, c, rng):
        if self.kind == "enum":
            return self.cells[c][rng.integers(len(self.cells[c]), size=n)]
        out, got = [], 0
        while got < n:  # rejection from the tree law to S_c
            Z = self.tree[c].sample(max(2 * (n - got), 1000), rng)
            Z = Z[self.contains(Z, c)]
            out.append(Z)
            got += len(Z)
        return np.vstack(out)[:n]

    def log_prob(self, X, c):
        """Exact log p_c for enum, and up to the Monte Carlo normalizer for closure."""
        inside = self.contains(X, c)
        if self.kind == "enum":
            lp = np.full(len(X), -np.log(len(self.cells[c])))
        else:
            lp = self.tree[c].log_prob(X) - self.log_z[c]
        return np.where(inside, lp, -np.inf)


def known_split(pop, seed, frac0, frac1, min1, n_test):
    """Independent training samples D_0 ~ p_0 and D_1 ~ p_1 and a balanced test sample."""
    rng = np.random.default_rng(seed)
    n0, n1 = round(frac0 * pop.n_table[0]), max(round(frac1 * pop.n_table[1]), min1)
    D0, D1 = pop.sample(n0, 0, rng), pop.sample(n1, 1, rng)
    T0, T1 = pop.sample(n_test, 0, rng), pop.sample(n_test, 1, rng)
    return dict(D0=D0, D1=D1, test_X=np.vstack([T1, T0]),
                test_y=np.r_[np.ones(n_test), np.zeros(n_test)].astype(int))
