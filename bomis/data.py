"""Loading of the twelve categorical datasets.

Raw files are read from data/raw (see scripts/download_data.py). Every column is treated as categorical
and coded 0..K_j-1, and the less frequent label is the minority class (y = 1).
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import arff

RAW = Path(__file__).resolve().parent.parent / "data" / "raw"
DATASETS = ["nursery2", "krk", "monks1", "car", "tictactoe",
            "phish", "naticus", "moto", "krkp", "mushroom", "splice", "vote"]


def _raw(name):
    if name == "phish":
        df = pd.DataFrame(arff.loadarff(RAW / "phish" / "Training Dataset.arff")[0])
        df = df.apply(lambda s: s.str.decode("utf-8"))
        return df.drop(columns="Result"), df["Result"]
    if name == "naticus":
        df = pd.read_csv(RAW / "naticus" / "data.csv")
        return df.drop(columns="Result"), df["Result"]
    if name == "moto":
        df = pd.read_csv(RAW / "moto.csv")
        return df.drop(columns=["Name", "Package", "API", "API_MIN", "class"]), df["class"]
    if name in ("krkp", "mushroom"):
        file = {"krkp": "krkp/kr-vs-kp.data", "mushroom": "mushroom/agaricus-lepiota.data"}[name]
        df = pd.read_csv(RAW / file, header=None, dtype=str)
        lab_col = 0 if name == "mushroom" else df.columns[-1]
        return df.drop(columns=lab_col), df[lab_col]
    if name == "splice":  # exon-intron (EI) against intron-exon (IE) boundaries, one feature per position
        df = pd.read_csv(RAW / "splice" / "splice.data", header=None, dtype=str).apply(lambda s: s.str.strip())
        df = df[df[0].isin(["EI", "IE"])]
        seq = pd.DataFrame([list(x) for x in df[2]], index=df.index)
        return seq, df[0]
    if name == "vote":
        df = pd.read_csv(RAW / "vote" / "house-votes-84.data", header=None, dtype=str)
        return df.drop(columns=0), df[0]
    if name == "nursery2":  # priority against special priority, rows of the full factorial domain
        df = pd.read_csv(RAW / "nursery" / "nursery.data", header=None, dtype=str)
        df = df[df[8].isin(["priority", "spec_prior"])]
        return df.drop(columns=8), df[8]
    if name == "krk":  # optimal depth to mate thirteen against fourteen
        df = pd.read_csv(RAW / "krk" / "krkopt.data", header=None, dtype=str)
        df = df[df[6].isin(["thirteen", "fourteen"])]
        return df.drop(columns=6), df[6]
    if name == "car":  # unacceptable against the other three grades, rows of the full factorial domain
        df = pd.read_csv(RAW / "car" / "car.data", header=None, dtype=str)
        return df.drop(columns=6), df[6].where(df[6] == "unacc", "acc")
    if name == "tictactoe":  # every legal end board, win for x against no win
        df = pd.read_csv(RAW / "tictactoe" / "tic-tac-toe.data", header=None, dtype=str)
        return df.drop(columns=9), df[9]
    if name == "monks1":  # all 432 configurations of the MONK-1 domain
        df = pd.read_csv(RAW / "monks" / "monks-1.test", header=None, sep=r"\s+", dtype=str)
        return df[[1, 2, 3, 4, 5, 6]], df[0]
    raise ValueError(f"unknown dataset {name}")


def load(name):
    """Integer-coded features X, labels y (1 = minority) and the number of levels K of each feature."""
    F, lab = _raw(name)
    F = F.loc[:, F.nunique() > 1].astype(str)
    X = np.stack([pd.factorize(F[c], sort=True)[0] for c in F.columns], 1)
    # MONK-1 has 216 rows per class, and its minority is fixed to the target concept, raw label 1.
    minority = "1" if name == "monks1" else lab.value_counts().idxmin()
    y = (lab.to_numpy() == minority).astype(int)
    K = [int(X[:, j].max()) + 1 for j in range(X.shape[1])]
    return X.astype(np.int64), y, K
