"""
Check hayes_fft against the exact brute-force data in data/open_regime_orbits*.csv:
for every orbit representative f (q = 2, 3, 4, 5, 7), Psi([f^2]) from the explicit formula must equal
Lambda_sum = 2d N_irr + E computed by testing every polynomial.  Run: python code/test_hayes_fft.py
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hayes_fft_fq as hayes_fft  # noqa: E402
from open_data import load_orbits  # noqa: E402


def main(max_class_count=6e6):
    orb = load_orbits()
    orb = orb[orb.q.isin([2, 3, 4, 5, 7])]
    worst_round = 0.0
    checked = 0
    for (q, d), sub in orb.groupby(["q", "d"]):
        if q ** (d - 1) > max_class_count:
            continue
        t = time.time()
        wd = None
        if q ** (d - 1) > 1e6:
            import tempfile
            wd = tempfile.mkdtemp()
        Psi, ctx = hayes_fft.all_interval_psi(q, d, work_dir=wd)
        dev = float(np.max(np.abs(Psi - np.rint(Psi))))
        worst_round = max(worst_round, dev)
        PsiI = np.rint(Psi).astype(np.int64)
        assert PsiI.sum() == q ** (2 * d), (q, d, PsiI.sum())  # every monic of degree 2d is counted once
        bad = 0
        for _, r in sub.iterrows():
            f = list(map(int, r.f_coeffs_low_first.split()))
            x = hayes_fft.class_index(hayes_fft.square_prefix_series(f, ctx), ctx)
            if PsiI[x] != r.Lambda_sum:
                bad += 1
        checked += len(sub)
        print(f"q={q} d={d}: {len(sub)} orbits, mismatches={bad}, max |Psi - round| = {dev:.2e}, "
              f"min Psi/q^(d+1) over all classes = {PsiI.min() / q ** (d + 1):.4f}  ({time.time() - t:.1f}s)",
              flush=True)
        assert bad == 0
    print(f"all {checked} orbit representatives agree; worst rounding deviation {worst_round:.2e}")


if __name__ == "__main__":
    main()
