"""Evaluation of generated minority rows G against a population with known supports.

  recovery   p_1 mass of the distinct generated cells in S_1 minus O, divided by the missing mass
             M_0 = p_1(S_1 minus O), with O the observed minority rows (higher is better).
  leak       share of generated rows outside S_1, the structural-zero violations (lower is better).
  control    1 - leak / novel, the share of novel generated rows that are feasible (higher is better),
             NaN when no row is novel.
  uni_tv     mean one-way total variation against the reference rows of p_1 (lower is better).
  pair_tv    mean two-way total variation against the reference rows of p_1 (lower is better).
  auroc      test AUROC of a linear SVM on one-hot features, trained on D_0, D_1 and G (higher is better).
"""
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import LinearSVC

from .populations import keys


def uni_tv(G, R, K):
    """Average one-way total variation between generated rows G and reference rows R."""
    tv = [0.5 * np.abs(np.bincount(G[:, j], minlength=K[j]) / len(G) - np.bincount(R[:, j], minlength=K[j]) / len(R)).sum()
          for j in range(len(K))]
    return float(np.mean(tv))


def pair_tv(G, R, K):
    """Average two-way total variation between generated rows G and reference rows R."""
    d, tv = len(K), []
    for j in range(d):
        for l in range(j + 1, d):
            size = K[j] * K[l]
            g = np.bincount(G[:, j] * K[l] + G[:, l], minlength=size) / len(G)
            r = np.bincount(R[:, j] * K[l] + R[:, l], minlength=size) / len(R)
            tv.append(0.5 * np.abs(g - r).sum())
    return float(np.mean(tv))


def auroc(D0, D1, G, test_X, test_y, K):
    """Test AUROC of a linear SVM on one-hot features, trained on the real rows plus the generated minority rows."""
    enc = OneHotEncoder(categories=[list(range(k)) for k in K], handle_unknown="ignore")
    X = np.vstack([D0, D1] + ([G] if G is not None else []))
    y = np.r_[np.zeros(len(D0)), np.ones(len(X) - len(D0))]
    clf = LinearSVC().fit(enc.fit_transform(X), y)
    return float(roc_auc_score(test_y, clf.decision_function(enc.transform(test_X))))


def evaluate(G, split, pop):
    """All metrics of the generated rows G for one training split of the population pop."""
    obs = set(keys(split["D1"]))
    O = np.unique(split["D1"], axis=0)
    in1 = pop.contains(G, 1)
    novel = np.array([k not in obs for k in keys(G)])
    M0 = 1.0 - float(np.exp(pop.log_prob(O, 1)).sum())
    hit = np.unique(G[in1 & novel], axis=0)
    leak, nov = 1 - float(in1.mean()), float(novel.mean())
    return dict(recovery=float(np.exp(pop.log_prob(hit, 1)).sum()) / M0 if len(hit) else 0.0,
                leak=leak, control=1 - leak / nov if nov > 0 else np.nan, novel=nov,
                uni_tv=uni_tv(G, pop.ref[1], pop.K), pair_tv=pair_tv(G, pop.ref[1], pop.K),
                auroc=auroc(split["D0"], split["D1"], G, split["test_X"], split["test_y"], pop.K))
