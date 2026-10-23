"""BOMiS: minority oversampling of categorical data beyond the observed support.

The method has three parts.
  Support        certifies minority structural zeros at the pattern level (patterns of order at most k)
                 and returns the admitted pattern tables that define the support estimate S_hat.
  ARTransformer  class-conditional autoregressive Transformer trained on both classes and decoded
                 under the support constraint.
  BOMiS          Good-Turing stratified sampler. A Binomial(n, G) number of rows is novel, with G the
                 Good-Turing missing mass of the minority sample, and the rest are observed minority rows.
                 Novel rows are decoder draws outside the observed rows.
Data are integer-coded arrays of shape (n, d) with levels 0..K_j-1. Class 0 is the majority, class 1 the minority.
"""
import itertools
import math

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import binom


def row_keys(X):
    """Hashable key per row, used for set membership."""
    X = np.ascontiguousarray(X, dtype=np.int64)
    return [r.tobytes() for r in X]


def good_turing(X):
    """Good-Turing missing-mass estimate (number of rows seen once) / n."""
    _, counts = np.unique(X, axis=0, return_counts=True)
    return float((counts == 1).sum()) / len(X)


def chao1(X):
    """Bias-corrected Chao1 estimate f1 (f1 - 1) / {2 (f2 + 1)} of the number of unseen configurations."""
    _, counts = np.unique(X, axis=0, return_counts=True)
    f1, f2 = int((counts == 1).sum()), int((counts == 2).sum())
    return f1 * (f1 - 1) / (2 * (f2 + 1))


def shell_law(X):
    """Shares of the Hamming distances from each row seen once to the other distinct rows of X.

    A row seen once is the one row outside the rest of the sample, so the share at distance s is a
    leave-one-out estimate of P(d(Z, O) = s | Z novel) for a fresh draw Z. Empty when no row is seen once.
    """
    U, cnt = np.unique(X, axis=0, return_counts=True)
    S = U[cnt == 1]
    if len(U) < 2 or not len(S):
        return {}
    dist = []
    for i in range(0, len(S), 256):
        D = (S[i:i + 256, None, :] != U[None]).sum(2)
        D[D == 0] = X.shape[1] + 1  # the row itself
        dist.append(D.min(1))
    s, c = np.unique(np.concatenate(dist), return_counts=True)
    return dict(zip(s.tolist(), (c / c.sum()).tolist()))


def largest_remainder(n, share):
    """Integer split of n over the keys of share, proportional to share by largest remainder."""
    q = {s: math.floor(n * p) for s, p in share.items()}
    for s in sorted(share, key=lambda s: q[s] - n * share[s])[:n - sum(q.values())]:
        q[s] += 1
    return q


def pips(q, n):
    """Inclusion probabilities min(1, c q) summing to n (all ones when n >= len(q))."""
    lam, free = np.ones(len(q)), np.ones(len(q), dtype=bool)
    if n >= len(q):
        return lam
    while True:
        c = (n - (~free).sum()) / q[free].sum()
        over = free & (c * q >= 1)
        if not over.any():
            lam[free] = c * q[free]
            return lam
        free &= ~over


def certify_threshold(n0, n1, r, level, grid_size=20000):
    """Smallest count x with sup_pi (1 - r pi)^n1 P(Bin(n0, pi) >= x) <= level.

    A pattern seen x or more times in the majority sample and never in the minority sample is
    certified as a minority structural zero. If a feasible pattern satisfies p_1 >= r p_0, it is
    certified with probability at most `level`. The supremum is bounded on a grid, so the threshold
    is conservative, and two closed-form thresholds with the same guarantee are also used.
    """
    if r <= 0:
        return n0 + 1
    grid = np.r_[0.0, np.geomspace(1e-9, 1.0, grid_size)]
    with np.errstate(divide="ignore"):
        head = n1 * np.log1p(-r * grid[:-1])  # factor at the left end of each grid cell
    log_g = lambda x: np.max(head + binom.logsf(x - 1, n0, grid[1:]))
    lo, hi, target = 1, n0 + 1, math.log(level)
    while lo < hi:  # the bound decreases in x
        mid = (lo + hi) // 2
        lo, hi = (lo, mid) if log_g(mid) <= target else (mid + 1, hi)
    tau = -math.log(level)
    h = math.ceil(tau / math.log1p(n1 * r / n0)) if n0 > 0 else n0 + 1
    xbar = int(binom.isf(level, n0, tau / (n1 * r))) + 1 if tau <= n1 * r else n0 + 1
    return min(lo, h, xbar)


class Support:
    """Pattern-level support estimate S_hat of the minority class.

    For every set B of at most k features, a pattern on B is admitted when it occurs in the minority
    sample, or in the majority sample without being certified as a structural zero. The family-wise
    error of the certificates is controlled at alpha by a Bonferroni split over all patterns.
    """

    def __init__(self, X0, X1, K, k=2, r=0.5, alpha=0.05):
        self.K, self.k = list(K), k
        d, n0, n1 = len(K), len(X0), len(X1)
        self.subsets = [B for s in range(1, k + 1) for B in itertools.combinations(range(d), s)]
        m = sum(math.prod(self.K[j] for j in B) for B in self.subsets)
        self.x_star = certify_threshold(n0, n1, r, alpha / max(m, 1))
        self.admit, self.n_certified = {}, 0
        for B in self.subsets:
            shape = tuple(self.K[j] for j in B)
            size = math.prod(shape)
            N0 = np.bincount(np.ravel_multi_index(X0[:, B].T, shape), minlength=size)
            N1 = np.bincount(np.ravel_multi_index(X1[:, B].T, shape), minlength=size)
            cert = (N1 == 0) & (N0 >= self.x_star)
            self.n_certified += int(cert.sum())
            # A level seen only in the majority sample is admitted only when some level of that feature
            # is seen once in the minority sample, since otherwise the Good-Turing mass of unseen levels is zero.
            pooled = (N0 > 0) & ((N1 == 1).any() or len(B) > 1)
            self.admit[B] = (((N1 > 0) | pooled) & ~cert).reshape(shape)
        self.by_last = {j: [B for B in self.admit if B[-1] == j] for j in range(d)}  # tables ending at feature j

    def contains(self, X):
        """Boolean mask of the rows of X that lie in S_hat."""
        ok = np.ones(len(X), dtype=bool)
        for B in self.admit:
            ok &= self.admit[B][tuple(X[:, j] for j in B)]
        return ok

    def admissible(self, prefix, j):
        """Admissible levels of feature j given the features 0..j-1 in prefix, an (n, j) array.
        Returns an (n, K_j) boolean mask."""
        mask = np.ones((len(prefix), self.K[j]), dtype=bool)
        for B in self.by_last[j]:
            table = self.admit[B]
            mask &= table[tuple(prefix[:, l] for l in B[:-1])] if len(B) > 1 else table[None, :]
        return mask


class ARTransformer(nn.Module):
    """Class-conditional autoregressive Transformer over categorical features."""

    def __init__(self, K, d_model=64, n_layers=2, n_heads=4, dropout=0.1):
        super().__init__()
        d, T = len(K), int(sum(K))
        offsets = np.concatenate([[0], np.cumsum(K)])
        self.register_buffer("offsets", torch.tensor(offsets[:-1], dtype=torch.long))
        valid = torch.zeros(d, T, dtype=torch.bool)  # valid[j, t]: output unit t is a level of feature j
        for j in range(d):
            valid[j, offsets[j]:offsets[j + 1]] = True
        self.register_buffer("valid", valid)
        self.cls = nn.Embedding(2, d_model)
        self.tok = nn.Embedding(T, d_model)
        self.pos = nn.Parameter(torch.zeros(d, d_model))
        layer = nn.TransformerEncoderLayer(d_model, n_heads, 4 * d_model, dropout,
                                           batch_first=True, norm_first=True)
        self.body = nn.TransformerEncoder(layer, n_layers)
        self.head = nn.Linear(d_model, T)
        self.register_buffer("causal", nn.Transformer.generate_square_subsequent_mask(d))

    def forward(self, x, y):
        """Logits (n, L, T) for features 0..L-1 given a prefix x of length L-1 and class y."""
        L = x.shape[1] + 1
        h = torch.cat([self.cls(y)[:, None], self.tok(x + self.offsets[: L - 1])], 1)
        h = self.body(h + self.pos[:L], mask=self.causal[:L, :L], is_causal=True)
        logits = self.head(h)
        return logits.masked_fill(~self.valid[:L], float("-inf"))


class BOMiS:
    """Support-certified, Good-Turing stratified minority oversampler.

    Usage: BOMiS(seed=0).fit(X0, X1, K).sample(n), with X0 the majority rows, X1 the minority rows
    and K the number of levels of each feature.

    k, r, alpha   pattern order, representation floor and error level of the certificates (Support).
    steps         grid of training lengths. The length is chosen by 5-fold cross-validated held-out
                  log loss of the masked decoder on the minority rows, and the decoder is refitted on all rows.
    d_model, n_layers, n_heads, batch, lr   Transformer and optimizer settings.
    """

    def __init__(self, k=2, r=0.5, alpha=0.05, d_model=64, n_layers=2, n_heads=4,
                 steps=(250, 500, 1000, 2000), val=0.2, batch=256, lr=1e-3, device="cpu", seed=0):
        self.k, self.r, self.alpha = k, r, alpha
        self.arch = dict(d_model=d_model, n_layers=n_layers, n_heads=n_heads)
        self.grid = (steps,) if np.isscalar(steps) else tuple(steps)
        self.val, self.batch, self.lr, self.device, self.seed = val, batch, lr, device, seed

    def fit(self, X0, X1, K):
        self.rng = np.random.default_rng([self.seed, 0])  # sampling stream
        self.K, self.d = list(K), len(K)
        self.support = Support(X0, X1, self.K, self.k, self.r, self.alpha)
        self.M_hat, self.f0 = good_turing(X1), chao1(X1)  # novel weight and number of unseen cells
        self.shells = shell_law(X1)
        self._fit_fresh(X0, X1)
        self.O = np.unique(X1, axis=0)
        self.observed = set(row_keys(self.O))
        # Majority configurations are excluded from the novel rows when the two samples share no configuration.
        self.O0, self.M0_hat = np.unique(X0, axis=0), good_turing(X0)
        O0 = set(row_keys(self.O0))
        self.majority = O0 if not O0 & self.observed else set()
        self.steps = self._select_steps(X0, X1) if len(self.grid) > 1 else self.grid[0]
        self.net = self._train(X0, X1, self.steps, fold=-1)
        if self.M0_hat > 0:  # offset of the posterior log odds (_odds)
            with np.errstate(divide="ignore"):
                u1 = np.log1p(-min(1.0, np.exp(self.log_prob(self.O, y=1, mask=False)).sum()))
                u0 = np.log1p(-min(1.0, np.exp(self.log_prob(self.O0, y=0)).sum()))
                self.offset = np.log(self.M_hat) - u1 - np.log(self.M0_hat) + u0
        return self

    def _fit_fresh(self, X0, X1):
        """Pooled counts of the patterns of order k + 1, kept as one flat count array (table t starts at off[t]).
        ell is the share of minority rows seen once that carry such a pattern seen nowhere else in either
        sample, the leave-one-out estimate of that share under the novel law."""
        B = np.array(list(itertools.combinations(range(self.d), self.k + 1)), dtype=np.int64)
        K = np.asarray(self.K, dtype=np.int64)[B]
        mult = np.cumprod(np.c_[K[:, 1:], np.ones(len(B), np.int64)][:, ::-1], 1)[:, ::-1]
        off = np.r_[0, np.cumsum(K.prod(1))[:-1]]
        self.fresh_N = (B, mult, off, np.zeros(int(K.prod(1).sum()), np.int32))
        P, step = np.vstack([X0, X1]), max(1, (1 << 22) // (len(X0) + len(X1)))
        for lo in range(0, len(B), step):
            codes = self._fresh_codes(P, lo, lo + step).ravel()
            self.fresh_N[3][:] += np.bincount(codes, minlength=len(self.fresh_N[3])).astype(np.int32)
        U, c = np.unique(X1, axis=0, return_counts=True)
        self.ell = float(self._fresh(U[c == 1], seen=1).mean()) if (c == 1).any() else np.nan

    def _select_steps(self, X0, X1):
        """Training length with the smallest cross-validated held-out log loss -log Q(v) of the masked
        decoder, over the held-out minority rows inside the fold's S_hat. The first length when no row is left."""
        perm = np.random.default_rng([self.seed, 1]).permutation(len(X1))
        n_fold = int(round(1 / self.val))
        fold = np.empty(len(X1), dtype=int)
        fold[perm] = np.arange(len(X1)) % n_fold
        full, loss, n = self.support, {}, 0
        for f in range(n_fold):
            Xv, Xt = X1[fold == f], X1[fold != f]
            self.support = Support(X0, Xt, self.K, self.k, self.r, self.alpha)
            Xs = Xv[self.support.contains(Xv)]
            n += len(Xs)
            for s in self.grid:
                nll = -self.log_prob(Xs, net=self._train(X0, Xt, s, fold=f))
                loss[s] = loss.get(s, 0.0) + float(nll.sum())
        self.support = full
        if n == 0:
            return self.grid[0]
        loss = {s: v / n for s, v in loss.items()}
        return min(loss, key=loss.get)

    def _masks(self, X0, X1):
        """Unit masks: minority rows are restricted to the admissible levels of S_hat."""
        offsets = np.concatenate([[0], np.cumsum(self.K)])
        unit_mask = torch.ones(len(X0) + len(X1), offsets[-1], dtype=torch.bool)
        for j in range(self.d):
            unit_mask[len(X0):, offsets[j]:offsets[j + 1]] = torch.from_numpy(self.support.admissible(X1[:, :j], j))
        return unit_mask

    def _train(self, X0, X1, steps, fold):
        """Class-conditional training on both classes with class-balanced row weights. Minority rows are
        trained under the decoding mask, so the trained conditionals are the decoder's conditionals."""
        torch.manual_seed(1000 * self.seed + fold + 1)
        rng = np.random.default_rng([self.seed, 2, fold + 1])
        X = torch.tensor(np.vstack([X0, X1]), dtype=torch.long)
        unit_mask = self._masks(X0, X1)
        y = torch.tensor(np.r_[np.zeros(len(X0)), np.ones(len(X1))], dtype=torch.long)
        w = np.r_[np.full(len(X0), 1.0 / max(len(X0), 1)), np.full(len(X1), 1.0 / len(X1))]
        w /= w.sum()
        net = ARTransformer(self.K, **self.arch).to(self.device)
        opt = torch.optim.AdamW(net.parameters(), lr=self.lr, weight_decay=1e-2)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, self.lr, total_steps=steps)
        shift = net.offsets.cpu()
        net.train()
        for _ in range(steps):
            idx = torch.from_numpy(rng.choice(len(X), self.batch, p=w))
            xb, yb = X[idx].to(self.device), y[idx].to(self.device)
            logits = net(xb[:, :-1], yb)
            logits = logits.masked_fill(~unit_mask[idx].to(self.device)[:, None, :], float("-inf"))
            target = xb + shift.to(self.device)
            loss = nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), target.reshape(-1))
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
        net.eval()
        return net

    def _odds(self, Z):
        """Posterior log odds that an unseen row z is a minority rather than a majority configuration.

        The probability that the next class-y draw is z and unseen is estimated by M_y p_y(z) / p_y(U minus O_y)
        (Katz back-off), with p_y the unmasked class law of the network, O_y the distinct rows of class y and
        M_y the Good-Turing missing mass of class y. +inf when the majority sample has no row seen once.
        """
        if self.M0_hat == 0:
            return np.full(len(Z), np.inf)
        return self.log_prob(Z, y=1, mask=False) + self.offset - self.log_prob(Z, y=0)

    @torch.no_grad()
    def _level_logits(self, prefix, j):
        """Log-probabilities of the levels of feature j given the minority class and prefix."""
        o, out = int(self.net.offsets[j]), []
        for i in range(0, len(prefix), 2048):
            x = torch.tensor(prefix[i:i + 2048], dtype=torch.long, device=self.device)
            y = torch.ones(len(x), dtype=torch.long, device=self.device)
            logits = self.net(x, y)[:, j]
            out.append(torch.log_softmax(logits[:, o:o + self.K[j]], -1).cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, self.K[j]))

    def decode(self, n, rng, max_rounds=1000):
        """Up to n complete draws from the support-constrained decoder. A draw that reaches a feature with
        no admissible level is discarded, and restarts are capped at max_rounds batches."""
        out = []
        for _ in range(max_rounds):
            if sum(len(o) for o in out) >= n:
                break
            X = np.zeros((max(n, 256), self.d), dtype=np.int64)
            alive = np.ones(len(X), dtype=bool)
            for j in range(self.d):
                lp = self._level_logits(X[alive, :j], j)
                mask = self.support.admissible(X[alive, :j], j)
                lp = np.where(mask, lp, -np.inf)
                dead = ~mask.any(1)
                p = np.exp(lp - lp.max(1, keepdims=True, initial=-1e30, where=mask))
                p[dead] = 1.0
                cum = p.cumsum(1)
                cum /= cum[:, -1:]
                u = 1.0 - rng.random((len(p), 1))  # in (0, 1], never picks a zero-mass level
                X[alive, j] = (cum < u).sum(1)
                alive[np.flatnonzero(alive)[dead]] = False
            out.append(X[alive])
        return np.vstack(out)[:n]

    @torch.no_grad()
    def log_prob(self, X, y=1, net=None, mask=True):
        """Log of the decoder law of class y, -inf outside S_hat for the masked minority law.
        One causal pass gives the conditionals of every feature given its prefix."""
        net = net or self.net
        lp = np.zeros(len(X))
        offsets = net.offsets.cpu().numpy()
        for i in range(0, len(X), 2048):
            Xi = X[i:i + 2048]
            x = torch.tensor(Xi, dtype=torch.long, device=self.device)
            cls = torch.full((len(x),), y, dtype=torch.long, device=self.device)
            logits = net(x[:, :-1], cls).cpu().numpy()
            for j in range(self.d):
                lj = logits[:, j, offsets[j]:offsets[j] + self.K[j]]
                if y == 1 and mask:
                    lj = np.where(self.support.admissible(Xi[:, :j], j), lj, -np.inf)
                with np.errstate(invalid="ignore"):
                    lj = lj - np.logaddexp.reduce(lj, axis=1, keepdims=True)
                lp[i:i + 2048] += lj[np.arange(len(Xi)), Xi[:, j]]
        return np.nan_to_num(lp, nan=-np.inf, neginf=-np.inf)

    def _dist(self, Z):
        """Hamming distance of each row of Z to the observed minority rows O."""
        out = np.empty(len(Z), dtype=int)
        step = max(1, 2 ** 24 // (len(self.O) * self.d))
        for i in range(0, len(Z), step):
            out[i:i + step] = (Z[i:i + step, None, :] != self.O[None]).sum(2).min(1)
        return out

    def _shell_cells(self, s, limit):
        """All candidate novel cells at Hamming distance exactly s from O, or None when there are more than limit.

        For an observed row o and edited features J, a level a of j in J must differ from o_j, be admitted
        alone, and be admitted with o_l for every unedited feature l.
        """
        O, d = self.O, self.d
        if math.comb(d, s) > limit:
            return None
        if not hasattr(self, "_viol"):
            def bad(j, l):  # (|O|, K_j): level a of j is not admitted with o_l
                t = self.support.admit.get((min(j, l), max(j, l)))
                if t is None:
                    return np.zeros((len(O), self.K[j]), dtype=bool)
                return ~(t if j < l else t.T)[:, O[:, l]].T
            V = {(j, l): bad(j, l) for j in range(d) for l in range(d) if l != j}
            C = {j: sum(V[j, l].astype(int) for l in range(d) if l != j) for j in range(d)}
            self._viol = (V, C)
        V, C = self._viol
        parts, total = [], 0
        for J in itertools.combinations(range(d), s):
            mk = []
            for j in J:
                m = self.support.admit[(j,)][None] & (C[j] == sum((V[j, l].astype(int) for l in J if l != j), 0))
                m[np.arange(len(O)), O[:, j]] = False
                mk.append(m)
            size = np.prod([m.sum(1) for m in mk], 0)
            total += int(size.sum())
            if total > limit:
                return None
            for i in np.flatnonzero(size):
                P = np.array(list(itertools.product(*[np.flatnonzero(m[i]) for m in mk])))
                Z = np.repeat(O[i:i + 1], len(P), 0)
                Z[:, list(J)] = P
                parts.append(Z)
        if not parts:
            return np.zeros((0, d), dtype=np.int64)
        Z = np.unique(np.vstack(parts), axis=0)
        Z = Z[self.support.contains(Z)]
        Z = Z[self._dist(Z) == s]
        return Z[np.array([key not in self.observed and key not in self.majority for key in row_keys(Z)], dtype=bool)]

    def _fresh_codes(self, Z, lo=0, hi=None):
        """Flat indices (n, tables lo..hi) of the order-(k + 1) patterns of the rows of Z."""
        B, mult, off, _ = self.fresh_N
        return (Z[:, B[lo:hi]] * mult[lo:hi]).sum(-1) + off[lo:hi]

    def _fresh(self, Z, seen=0):
        """True where some pattern of order k + 1 of Z has pooled count `seen` (1 for a row left out)."""
        out = np.zeros(len(Z), dtype=bool)
        T = len(self.fresh_N[0]) if len(Z) else 0
        step = max(1, (1 << 22) // max(len(Z), 1))
        for lo in range(0, T, step):
            out |= (self.fresh_N[3][self._fresh_codes(Z, lo, lo + step)] == seen).any(1)
        return out

    def _novel(self, n_new, budget):
        """n_new novel rows, outside O and outside the majority configurations.

        1. Distance s to O receives the largest-remainder share of n_new under shell_law, and keeps at most
           min(quota, ceil(Chao1 share)) distinct cells.
        2. Cells of a distance that can be listed within the draw budget (_shell_cells) are chosen by a
           Pareto pips sample with inclusion probability min(1, c Q) under the decoder law Q. Other distances
           keep the first distinct decoder draws.
        3. Only cells with nonnegative posterior log odds (_odds) are kept when majority configurations are excluded.
        4. The decoder law is tilted so that the share of cells with an order-(k + 1) pattern seen in neither
           sample equals ell (_fit_fresh).
        5. A short count is completed with the remaining candidates in decreasing posterior odds, with no
           distance above its Chao1 count, and the kept cells are repeated evenly to n_new rows.
        """
        rng = self.rng
        skip = self.observed | self.majority
        share = self.shells or {-1: 1.0}
        quota = {s: q for s, q in largest_remainder(n_new, share).items() if q}
        C = math.ceil(self.f0)  # Chao1 number of unseen cells
        L = min(n_new, max(1, C))
        cap = {s: min(q, max(1, math.ceil(L * share[s]))) for s, q in quota.items()}
        got = {s: {} for s in cap}
        cand, pool = {}, {}  # listed cells per distance; unkept candidates and their log odds
        T, exact, excl_fallback = budget * max(4 * n_new, 1024), set(), False
        filt = bool(self.majority)
        self.__dict__.pop("_viol", None)
        for s in cap if self.shells and n_new > 0 else ():
            Z = self._shell_cells(s, T)
            if Z is not None:
                odds = self._odds(Z) if filt and len(Z) else np.full(len(Z), np.inf)
                cand[s] = (Z, odds, self.log_prob(Z), self._fresh(Z))
                exact.add(s)
        part = {}  # decoder share of fresh cells per distance
        for s, (Z, odds, lp, u) in cand.items():
            ok = odds >= 0
            if ok.any():
                p = np.exp(lp[ok] - lp[ok].max())
                part[s] = float((p * u[ok]).sum() / p.sum())
        logw = np.zeros(2)

        def tilt():
            g = sum(share[s] * v for s, v in part.items()) / sum(share[s] for s in part) if part else np.nan
            if np.isfinite(self.ell) and 0 < g < 1:
                with np.errstate(divide="ignore"):
                    logw[:] = np.log([(1 - self.ell) / (1 - g), self.ell / g])
        full = lambda: all(s in exact or len(got[s]) >= cap[s] for s in cap)
        if full() or n_new <= 0:
            tilt()
        for b in range(budget if n_new > 0 and not full() else 0):
            Z = self.decode(max(4 * n_new, 1024), rng)
            Z = Z[[key not in skip for key in row_keys(Z)]]
            if b == 0 and not len(Z) and skip is not self.observed:
                skip, excl_fallback = self.observed, True  # no draw outside both sets, use S_hat minus O
                continue
            odds = np.full(len(Z), np.inf)
            if filt and skip is not self.observed and len(Z):
                odds = self._odds(Z)
            dist = self._dist(Z) if self.shells else np.full(len(Z), -1)
            u = self._fresh(Z)
            if b == 0 or (excl_fallback and b == 1):  # the first batch gives the fresh share on drawn distances
                for t in set(dist.tolist()) - exact:
                    m = (dist == t) & (odds >= 0)
                    if t in cap and m.any():
                        part[t] = float(u[m].mean())
                tilt()
            keep = rng.random(len(Z)) < np.exp(logw[u.astype(int)] - logw.max())
            for key, z, t, o, a in zip(row_keys(Z), Z, dist.tolist(), odds.tolist(), keep.tolist()):
                if t not in got or t in exact or key in got[t]:
                    continue
                if o >= 0 and a and len(got[t]) < cap[t]:
                    got[t].setdefault(key, z)
                    pool.pop(key, None)
                else:
                    pool[key] = (o, z)
            if len(pool) > 2 * L:  # each distance keeps its best candidates up to what it can still take
                left = sum(cap.values()) - sum(len(g) for g in got.values())
                by_dist = bool(self.shells)
                at = self._dist(np.array([z for o, z in pool.values()])).tolist() if by_dist else [None] * len(pool)
                lim = {t: min(left, math.ceil(share[t] * C) - len(got[t])) for t in got} if by_dist else {None: left}
                n, kept = dict.fromkeys(lim, 0), set()
                for (key, (o, z)), t in sorted(zip(pool.items(), at), key=lambda x: -x[0][1][0]):
                    if n[t] < lim[t]:
                        n[t] += 1
                        kept.add(key)
                pool = {k: v for k, v in pool.items() if k in kept}
            if full():
                break
        for s, (Z, odds, lp, u) in cand.items():
            ok = (odds >= 0) & np.isfinite(logw[u.astype(int)])
            score = lp[ok] + logw[u[ok].astype(int)]
            key = np.zeros(0)
            if ok.any():  # Pareto order sampling, cells with inclusion probability 1 first
                lam, v = pips(np.exp(score - score.max()), cap[s]), rng.random(int(ok.sum()))
                with np.errstate(divide="ignore"):
                    key = np.log(v) - np.log1p(-v) - np.log(lam) + np.log1p(-lam)
            top = np.argsort(key, kind="stable")[:cap[s]]
            got[s] = dict(zip(row_keys(Z[ok][top]), Z[ok][top]))
            pool.update((k, (o, z)) for k, o, z in zip(row_keys(Z), odds.tolist(), Z) if k not in got[s])
        short = sum(cap.values()) - sum(len(g) for g in got.values())
        fill = sorted(pool.items(), key=lambda kv: -kv[1][0])
        if self.shells and short > 0 and fill:  # no distance is filled beyond its Chao1 count
            room = {s: math.ceil(share[s] * C) - len(got[s]) for s in got}
            dist, take = self._dist(np.array([z for _, (o, z) in fill])).tolist(), []
            for passing in (True, False):
                for i, ((key, (o, z)), t) in enumerate(zip(fill, dist)):
                    if (o >= 0) == passing and room.get(t, 0) > 0:
                        take.append(i)
                        room[t] -= 1
            fill = [fill[i] for i in take]
        for key, (o, z) in fill[:max(short, 0)]:
            got.setdefault("fill", {})[key] = z
        R = [np.array(list(g.values()), dtype=np.int64) for g in got.values() if len(g)]
        R = np.vstack(R) if R else np.zeros((0, self.d), dtype=np.int64)
        return R[np.arange(n_new) % len(R)] if len(R) else R

    def sample(self, n, budget=50):
        """n synthetic minority rows. A Binomial(n, G) number are novel rows (_novel), and the rest are
        observed minority rows drawn from the decoder law restricted to O. If the draw budget leaves the
        novel quota unfilled, the gap is filled with observed rows."""
        n_new = int(self.rng.binomial(n, self.M_hat))
        lp = self.log_prob(self.O)
        w_O = np.exp(lp - lp.max())
        w_O /= w_O.sum()
        new = self._novel(n_new, budget)[:n_new] if budget else np.zeros((0, self.d), dtype=np.int64)
        seen = self.O[self.rng.choice(len(self.O), n - len(new), p=w_O)]
        return self.rng.permutation(np.vstack([seen, new]))
