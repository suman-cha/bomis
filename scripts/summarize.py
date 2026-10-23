"""Mean and standard deviation over seeds of every metric, per dataset.

  python scripts/summarize.py [results]

Prints a Markdown table and writes <results>/summary.csv.
"""
import sys
from pathlib import Path

import pandas as pd

METRICS = ["recovery", "leak", "control", "uni_tv", "pair_tv", "auroc"]


def main():
    res = Path(sys.argv[1] if len(sys.argv) > 1 else "results")
    files = [f for f in res.glob("*_*.csv") if f.name != "summary.csv"]
    if not files:
        raise SystemExit(f"no result files in {res}")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    g = df.groupby("dataset")[METRICS]
    summary = g.mean().join(g.std(), rsuffix="_sd").join(df.groupby("dataset").seed.count().rename("n_seeds"))
    summary.to_csv(res / "summary.csv")
    print("| dataset | seeds | " + " | ".join(METRICS) + " |")
    print("|---|---|" + "---|" * len(METRICS))
    for ds, r in summary.iterrows():
        cells = [f"{r[m]:.4f} ± {r[m + '_sd']:.4f}" if pd.notna(r[m + "_sd"]) else f"{r[m]:.4f}" for m in METRICS]
        print(f"| {ds} | {int(r['n_seeds'])} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
