"""Run BOMiS on one dataset for one or more seeds and save the metrics.

  python scripts/run.py --dataset car --seeds 0 1 2

For each seed, D_0 and D_1 are sampled from the population of the dataset, BOMiS generates |D_0| - |D_1|
minority rows, and the metrics of bomis/metrics.py go to results/<dataset>_<seed>.csv.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from bomis import BOMiS  # noqa: E402
from bomis.data import DATASETS  # noqa: E402
from bomis.metrics import evaluate  # noqa: E402
from bomis.populations import SIZES, Population, known_split  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=DATASETS)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--out", default="results")
    ap.add_argument("--save-rows", action="store_true", help="also save the generated rows as .npy")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    pop = Population(args.dataset)
    for seed in args.seeds:
        s = known_split(pop, seed, **SIZES)
        n = len(s["D0"]) - len(s["D1"])
        t0 = time.perf_counter()
        G = BOMiS(seed=seed).fit(s["D0"], s["D1"], pop.K).sample(n)
        row = dict(dataset=args.dataset, seed=seed, n0=len(s["D0"]), n1=len(s["D1"]), n_gen=len(G),
                   seconds=round(time.perf_counter() - t0, 1), **evaluate(G, s, pop))
        pd.DataFrame([row]).to_csv(out / f"{args.dataset}_{seed}.csv", index=False)
        if args.save_rows:
            np.save(out / f"{args.dataset}_{seed}.npy", G.astype(np.int16))
        print(", ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()), flush=True)


if __name__ == "__main__":
    main()
