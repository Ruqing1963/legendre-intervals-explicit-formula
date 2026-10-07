"""
Exact modular computation (hayes_exact) vs floating-point computation (hayes_fft_fq):
the CRT-reconstructed integers must equal the rounded floats for every class.
Run: python code/test_hayes_exact.py
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hayes_exact as X  # noqa: E402
import hayes_fft_fq as H  # noqa: E402

CASES = [(2, 3), (2, 6), (2, 12), (3, 4), (3, 9), (4, 6), (5, 6), (7, 5), (8, 5), (9, 5), (4, 9), (3, 11)]

if __name__ == "__main__":
    for q, d in CASES:
        t = time.time()
        ex, ctx, primes = X.psi_exact(q, d)
        fl, _ = H.all_interval_psi(q, d)
        fl = np.rint(fl).astype(np.int64)
        assert ex.sum() == q ** (2 * d), (q, d)
        diff = int(np.count_nonzero(ex != fl))
        print(f"q={q} d={d}: classes={ex.size} primes={primes} mismatches={diff} ({time.time() - t:.1f}s)", flush=True)
        assert diff == 0
    print("exact and floating-point computations agree on every class")
