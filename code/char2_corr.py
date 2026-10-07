"""
Diagnose the excess variance of Legendre intervals in characteristic 2.

Legendre classes form the subgroup H = G^2 of squares. Over H,
    Var_H(Psi) = q^{-2l} sum_{eta in H^} |sum_{eps in H^perp} psi(chi eps)|^2   (chi any lift of eta),
where H^perp = characters trivial on squares = characters of order <= 2 (coordinates b_j in
{0, order_j/2}). If the psi(chi eps) were uncorrelated this equals the full-group value.
For each quadratic character eps we compute the correlation
    rho(eps) = Re sum_chi psi(chi) conj(psi(chi eps)) / sum_chi |psi(chi)|^2   (chi != 1, chi eps != 1),
and the excess ratio R = Var_H / Var_G.

Output: data/char2_correlations.csv (one row per (q, d, eps) with |rho| > 0.05) and
        data/char2_excess.csv (one row per (q, d)).
"""
import csv
import itertools
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hayes_fft_fq as H  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
CASES = [(2, d) for d in range(4, 17)] + [(4, d) for d in range(3, 10)] + [(8, d) for d in range(3, 8)]


def main():
    corr_rows, ex_rows = [], []
    for q, d in CASES:
        psi, ctx = H.all_character_psi(q, d)
        shape = ctx["shape"]
        A = psi.reshape(shape).copy()
        A.flat[0] = 0.0  # drop the trivial character
        tot = float(np.sum(np.abs(A) ** 2))
        axes_info = [(ax, n) for ax, n in enumerate(shape)]
        # quadratic characters: subsets of axes, shift by n/2 on each chosen axis
        n_eps = 0
        rho_sum = 0.0
        for choice in itertools.product([0, 1], repeat=len(shape)):
            n_eps += 1
            if not any(choice):
                continue
            shift = tuple((n // 2) * c for (ax, n), c in zip(axes_info, choice))
            B = np.roll(A, shift=shift, axis=tuple(range(len(shape))))
            rho = float(np.real(np.sum(A * np.conj(B))) / tot)
            rho_sum += rho
            if abs(rho) > 0.05:
                corr_rows.append({"q": q, "d": d, "eps_axes": "".join(map(str, choice)),
                                  "axis_orders": ";".join(map(str, shape)), "rho": round(rho, 4)})
        # Under independence the variance over H equals the full-group variance; the excess factor
        # is 1 + sum_{eps != 1} rho(eps).
        ex_rows.append({"q": q, "d": d, "quadratic_chars": n_eps, "excess_factor": round(1 + rho_sum, 4)})
        strong = [(r["eps_axes"], r["rho"]) for r in corr_rows if r["q"] == q and r["d"] == d]
        print(q, d, "shape", shape, "excess", round(1 + rho_sum, 3), "strong:", strong[:6], flush=True)
    with open(os.path.join(DATA, "char2_excess.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(ex_rows[0].keys()))
        w.writeheader()
        w.writerows(ex_rows)
    with open(os.path.join(DATA, "char2_correlations.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["q", "d", "eps_axes", "axis_orders", "rho"])
        w.writeheader()
        w.writerows(corr_rows)


if __name__ == "__main__":
    main()
