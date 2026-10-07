"""
Experiment 8 -- certified verification with exact arithmetic (hayes_exact).

For each (q, d): exact integers Psi(x) for every class (CRT of two prime-field computations),
restricted to Legendre intervals (squares for even q). An interval is certified to contain an
irreducible polynomial if Psi(x) > E_bound(q, d) (Lemma 2.3 of v7: E <= d q^a + 2 q^{floor(2d/3)}).
Intervals not certified this way are listed; for those the exact count N_irr is known from the
brute-force data (all of them are tiny cases).

Output: data/exact_scan.csv
Usage:  python code/exact_scan.py [--only q,d] [--max-classes N]
"""
import argparse
import csv
import math
import os
import shutil
import sys
import tempfile
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hayes_exact as X  # noqa: E402
from explicit_scan import GRID, e_bound, legendre_mask  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None)
    ap.add_argument("--max-classes", type=float, default=2e8)
    args = ap.parse_args()
    grid = list(GRID) + [(8, 10)] if not args.only else [tuple(map(int, args.only.split(",")))]
    grid = [g for g in grid if g[1] >= 2]
    work_root = os.environ.get("LEGENDRE_WORK", tempfile.gettempdir())
    rows = []
    for q, d in grid:
        N = q ** (d - 1)
        if N > args.max_classes:
            print(f"skip q={q} d={d}", flush=True)
            continue
        t0 = time.time()
        wd = tempfile.mkdtemp(dir=work_root) if N * (d - 1) * 8 > 1.5e9 else None
        try:
            Psi, ctx, primes = X.psi_exact(q, d, wd)
        finally:
            if wd:
                shutil.rmtree(wd, ignore_errors=True)
        assert int(Psi.sum()) == q ** (2 * d)
        mask = legendre_mask(ctx)
        vals = Psi if mask is None else Psi[mask]
        eb = e_bound(q, d, ctx["p"])
        main_ = q ** (d + 1)
        rows.append({
            "q": q, "d": d, "classes": N, "legendre_intervals": int(vals.size),
            "min_Psi": int(vals.min()), "max_Psi": int(vals.max()), "q^(d+1)": main_,
            "min_Psi_ratio": float(vals.min() / main_), "E_bound": eb,
            "certified": int((vals > eb).sum()), "uncertified": int((vals <= eb).sum()),
            "open_range": int(q <= d - 2), "primes": f"{primes[0]};{primes[1]}",
            "seconds": round(time.time() - t0, 1),
        })
        r = rows[-1]
        print(f"q={q} d={d}: intervals={r['legendre_intervals']} min_ratio={r['min_Psi_ratio']:.4f} "
              f"uncertified={r['uncertified']} [{r['seconds']}s]", flush=True)
    suffix = "" if not args.only else "_" + args.only.replace(",", "_")
    with open(os.path.join(DATA, f"exact_scan{suffix}.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
