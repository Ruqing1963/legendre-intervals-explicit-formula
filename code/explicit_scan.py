"""
Experiment 6 -- every Legendre interval at once, via the explicit formula (hayes_fft_fq).

For each (q, d) the prime-power-weighted count Psi(x) is computed for every class x of monic
polynomials of degree 2d with d-1 prescribed coefficients. Legendre intervals are the classes
[f^2] = [f]^2: all classes for odd q, the squares (all exponents even) for even q.

Certification of N_irr >= 1: by Lemma 2.3 of v7, E(f) <= d q^a + 2 q^{floor(2d/3)} with a = 1 (p odd)
or ceil((d+1)/2) (p = 2), so Psi(x) > that bound implies an irreducible polynomial in I_f.
Rounding: Psi is computed in floating point and rounded; the maximal distance to the nearest
integer is recorded (it must be far below 1/2).

Output: data/explicit_scan.csv  (+ per-case Psi histograms in data/explicit_psi_hist.csv)
Usage:  python code/explicit_scan.py [--only q,d] [--max-classes N]
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
import hayes_fft_fq as H  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")

GRID = ([(2, d) for d in range(2, 26)] + [(3, d) for d in range(2, 17)] + [(4, d) for d in range(2, 14)]
        + [(5, d) for d in range(2, 12)] + [(7, d) for d in range(2, 11)] + [(8, d) for d in range(2, 11)]
        + [(9, d) for d in range(2, 9)])


def legendre_mask(ctx):
    """Boolean mask over flat class indices: True for squares [f]^2 (all classes if p odd)."""
    shape = ctx["shape"]
    N = int(np.prod([float(s) for s in shape]))
    if ctx["p"] % 2 == 1:
        return None
    mask = np.ones(shape, bool)
    for ax, size in enumerate(shape):
        idx = [slice(None)] * len(shape)
        sl = np.arange(size) % 2 == 1
        idx[ax] = sl
        mask[tuple(idx)] = False
    return mask.reshape(N)


def e_bound(q, d, p):
    a = 1 if p % 2 == 1 else math.ceil((d + 1) / 2)
    return d * q ** a + 2 * q ** (2 * d // 3)


def run_case(q, d, work_root):
    t0 = time.time()
    N = q ** (d - 1)
    wd = None
    if N > 4e6:
        wd = tempfile.mkdtemp(dir=work_root)
    try:
        Psi, ctx = H.all_interval_psi(q, d, work_dir=wd)
    finally:
        if wd is not None:
            shutil.rmtree(wd, ignore_errors=True)
    dev = float(np.max(np.abs(Psi - np.rint(Psi))))
    PsiI = np.rint(Psi).astype(np.int64)
    del Psi
    assert int(PsiI.sum()) == q ** (2 * d), "sum over all classes must be q^(2d)"
    mask = legendre_mask(ctx)
    vals = PsiI if mask is None else PsiI[mask]
    main = q ** (d + 1)
    z = (vals - main) / math.sqrt(max(d - 2, 1) * main)
    eb = e_bound(q, d, ctx["p"])
    row = {
        "q": q, "d": d, "p": ctx["p"], "classes": N, "legendre_intervals": int(vals.size),
        "min_Psi_ratio": float(vals.min() / main), "max_Psi_ratio": float(vals.max() / main),
        "mean_Psi_ratio": float(vals.mean() / main),
        "var_Psi_over_q^(d+1)": float(((vals - main).astype(np.float64) ** 2).mean() / main),
        "KR_limit": d - 2 if d >= 4 else "",
        "min_Z": float(z.min()), "max_Z": float(z.max()),
        "E_bound": eb, "certified_all": int(bool((vals > eb).all())),
        "uncertified": int((vals <= eb).sum()),
        "open_range": int(q <= d - 2),
        "max_round_dev": dev, "seconds": round(time.time() - t0, 2),
    }
    hist, edges = np.histogram(z, bins=60, range=(-6, 6))
    return row, hist, edges


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None)
    ap.add_argument("--max-classes", type=float, default=4.1e7)
    args = ap.parse_args()
    grid = GRID
    if args.only:
        q0, d0 = map(int, args.only.split(","))
        grid = [(q0, d0)]
    work_root = os.environ.get("LEGENDRE_WORK", tempfile.gettempdir())
    rows, hrows = [], []
    for q, d in grid:
        if q ** (d - 1) > args.max_classes:
            print(f"skip q={q} d={d}: {q ** (d - 1):.2e} classes", flush=True)
            continue
        row, hist, edges = run_case(q, d, work_root)
        rows.append(row)
        for h, lo in zip(hist, edges[:-1]):
            hrows.append({"q": q, "d": d, "z_lo": round(float(lo), 3), "count": int(h)})
        print(f"q={q} d={d}: intervals={row['legendre_intervals']} min={row['min_Psi_ratio']:.4f} "
              f"var={row['var_Psi_over_q^(d+1)']:.3f} (KR {row['KR_limit']}) certified={row['certified_all']} "
              f"round={row['max_round_dev']:.1e} [{row['seconds']}s]", flush=True)
    suffix = "" if not args.only else "_" + args.only.replace(",", "_")
    with open(os.path.join(DATA, f"explicit_scan{suffix}.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(DATA, f"explicit_psi_hist{suffix}.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(hrows[0].keys()))
        w.writeheader()
        w.writerows(hrows)


if __name__ == "__main__":
    main()
