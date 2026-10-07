"""
Experiment 9 -- the "form factor" of the family of Hayes L-functions at fixed q.

F(n) = (1/(q^l - 1)) sum_{chi != 1} |psi_n(chi)|^2 / q^n,  psi_n(chi) = sum_{deg g = n} Lambda(g) chi(g),
for 1 <= n <= 3l, where l = d-1. Since psi_n(chi) = -sum_i omega_i^n, F(n) is the mean of
|Tr U^n|^2 for the normalized zeros; the CUE prediction is min(n, l-1).

Exact diagonal (Proposition 7.1 of the draft): for n <= l every class contains at most one monic
polynomial of degree n, so by Parseval
    F(n) = (q^l S_n - q^{2n}) / ((q^l - 1) q^n),   S_n = sum_{deg g = n} Lambda(g)^2 = sum_{e | n} e^2 I_e(q).

Output: data/form_factor.csv
"""
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hayes_fft_fq as H  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
CASES = [(2, 20), (3, 13), (4, 10), (5, 9), (7, 8), (8, 7), (9, 7)]


def mobius(n):
    r, m, p = 1, n, 2
    while p * p <= m:
        if m % p == 0:
            m //= p
            if m % p == 0:
                return 0
            r = -r
        p += 1
    return -r if m > 1 else r


def n_irreducible(e, q):
    return sum(mobius(k) * q ** (e // k) for k in range(1, e + 1) if e % k == 0) // e


def diagonal(n, q, l):
    S = sum(e * e * n_irreducible(e, q) for e in range(1, n + 1) if n % e == 0)
    return (q ** l * S - q ** (2 * n)) / ((q ** l - 1) * q ** n)


def main():
    rows = []
    for q, d in CASES:
        l = d - 1
        ctx = H.setup(q, l)
        N = q ** l
        shape = ctx["shape"]
        C = []
        for kdeg in range(0, l):
            ind = H._indices_degree(kdeg, l, q, ctx["p"], ctx["k"], ctx["A"], ctx["M"], ctx["NEG"],
                                    ctx["FROB"], ctx["FINV"], ctx["axis_of"], ctx["orders"], ctx["strides"])
            arr = np.zeros(N, np.complex128)
            np.add.at(arr, ind, 1.0)
            C.append(np.fft.fftn(arr.reshape(shape)).reshape(N))
        n_max = 3 * l
        psi = [None] * (n_max + 1)
        for n in range(1, n_max + 1):
            acc = n * C[n] if n < l else np.zeros(N, np.complex128)
            for kk in range(max(1, n - l + 1), n):
                acc = acc - psi[kk] * C[n - kk]
            psi[n] = acc
            F = float(np.sum(np.abs(acc[1:]) ** 2) / (N - 1) / q ** n)
            rows.append({"q": q, "d": d, "l": l, "n": n, "F": F, "CUE": min(n, l - 1),
                         "diagonal_exact": diagonal(n, q, l) if n <= l else ""})
        print(q, d, "F(n) for n=1..", n_max, "done", flush=True)
    with open(os.path.join(DATA, "form_factor.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    # check the exact diagonal formula
    worst = max(abs(r["F"] - r["diagonal_exact"]) for r in rows if r["diagonal_exact"] != "")
    print("max |F - diagonal formula| for n <= l:", worst)


if __name__ == "__main__":
    main()
