"""
Experiment 7 -- zeros of all Hayes L-functions L(z, chi), chi a nontrivial character of G_{d-1},
for fixed q, compared with random unitary matrices (CUE).

For each chi, L(z, chi) = sum_{k < d-1} c_k(chi) z^k (c_0 = 1) is computed by hayes_fft_fq.
Its inverse roots omega satisfy |omega| = sqrt(q) or |omega| = 1 (Weil). We record how many of
each kind occur, and for the "nontrivial" ones the angles theta = arg(omega / sqrt(q)).
Statistics, for the characters whose L-function has the generic number N of nontrivial zeros:
  * form factor  F(n) = mean_chi |sum_i exp(i n theta_i)|^2,  CUE prediction min(n, N);
  * nearest-neighbour spacings of the eigenangles, normalized to mean 1, against CUE(N).

Outputs data/zero_stats_summary.csv, data/zero_stats_formfactor.csv, data/zero_stats_spacing.csv
"""
import argparse
import csv
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hayes_fft_fq as H  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
CASES = [(3, 12), (2, 20), (5, 9), (7, 8), (4, 10), (3, 14)]
RNG = np.random.default_rng(20261006)


def inverse_roots(C, idx):
    """Inverse roots of L(z) = sum_k C[k][idx] z^k for a batch of characters (same degree)."""
    coeffs = np.stack([np.asarray(C[k][idx]) for k in range(len(C))], axis=1)  # (batch, l)
    m = coeffs.shape[1] - 1
    # L(z) = prod (1 - omega_i z)  =>  z^m L(1/z) = sum_k c_k z^{m-k} = prod (z - omega_i),
    # monic because c_0 = 1; descending-power coefficients are c_0, c_1, ..., c_m
    mon = coeffs / coeffs[:, :1]
    comp = np.zeros((coeffs.shape[0], m, m), np.complex128)
    comp[:, 0, :] = -mon[:, 1:]
    if m > 1:
        comp[:, np.arange(1, m), np.arange(0, m - 1)] = 1.0
    return np.linalg.eigvals(comp)


def cue_angles(N, count):
    out = []
    for _ in range(count):
        Z = (RNG.standard_normal((N, N)) + 1j * RNG.standard_normal((N, N))) / np.sqrt(2)
        Q, R = np.linalg.qr(Z)
        Q = Q * (np.diagonal(R) / np.abs(np.diagonal(R)))
        out.append(np.angle(np.linalg.eigvals(Q)))
    return np.array(out)


def spacings(angles):
    a = np.sort(np.mod(angles, 2 * np.pi), axis=1)
    N = a.shape[1]
    gaps = np.diff(np.concatenate([a, a[:, :1] + 2 * np.pi], axis=1), axis=1)
    return (gaps * N / (2 * np.pi)).ravel()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-chars", type=int, default=400000)
    args = ap.parse_args()
    summ, ff_rows, sp_rows = [], [], []
    for q, d in CASES:
        t0 = time.time()
        _, ctx, C = H.all_interval_psi(q, d, keep_L=True)
        Nchar = q ** (d - 1)
        idx = np.arange(1, Nchar)
        if idx.size > args.max_chars:
            idx = np.sort(RNG.choice(idx, size=args.max_chars, replace=False))
        l = d - 1
        # effective degree of each L-function = largest k with c_k != 0
        absc = np.stack([np.abs(np.asarray(C[k][idx])) for k in range(l)], axis=1)
        nz = absc > 1e-6
        degs = (l - 1) - np.argmax(nz[:, ::-1], axis=1)
        kinds = {}
        angle_sets = []
        off_circle = 0
        for deg in np.unique(degs):
            sel = idx[degs == deg]
            if deg <= 0:
                kinds[(0, 0)] = kinds.get((0, 0), 0) + sel.size
                continue
            Csub = [C[k] for k in range(deg + 1)]
            roots = inverse_roots(Csub, sel)
            mod = np.abs(roots)
            nontriv = np.abs(mod - np.sqrt(q)) < 1e-4 * np.sqrt(q)
            triv = np.abs(mod - 1) < 1e-4
            off_circle += int((~(nontriv | triv)).sum())
            nn = nontriv.sum(axis=1)
            for a, b in zip(nn, deg - nn):
                kinds[(int(a), int(b))] = kinds.get((int(a), int(b)), 0) + 1
            angle_sets.append((nn, np.angle(roots / np.sqrt(q)), nontriv))
        Ngen = max(kinds, key=lambda kk: kinds[kk])[0]
        # angles for the generic family (Ngen nontrivial zeros)
        ang = []
        for nn, angs, nt in angle_sets:
            rows = np.where(nn == Ngen)[0]
            for r in rows:
                ang.append(angs[r][nt[r]])
        ang = np.array(ang)
        n_gen = ang.shape[0]
        summ.append({"q": q, "d": d, "characters": int(idx.size), "generic_nontrivial_zeros": Ngen,
                     "generic_fraction": n_gen / idx.size,
                     "zero_types": "; ".join(f"{a}+{b}:{c}" for (a, b), c in sorted(kinds.items())),
                     "roots_off_both_circles": off_circle,
                     "seconds": round(time.time() - t0, 1)})
        for n in range(1, 3 * Ngen + 3):
            F = float(np.mean(np.abs(np.exp(1j * n * ang).sum(axis=1)) ** 2))
            ff_rows.append({"q": q, "d": d, "N": Ngen, "n": n, "form_factor": F, "CUE": min(n, Ngen)})
        if Ngen >= 2:
            s = spacings(ang)
            sc = spacings(cue_angles(Ngen, 20000))
            bins = np.linspace(0, 3.5, 36)
            h1, _ = np.histogram(s, bins=bins, density=True)
            h2, _ = np.histogram(sc, bins=bins, density=True)
            for lo, a, b in zip(bins[:-1], h1, h2):
                sp_rows.append({"q": q, "d": d, "N": Ngen, "s_lo": round(float(lo), 3),
                                "density_data": float(a), "density_CUE": float(b)})
        print(summ[-1], flush=True)
    for name, rows in [("zero_stats_summary", summ), ("zero_stats_formfactor", ff_rows), ("zero_stats_spacing", sp_rows)]:
        with open(os.path.join(DATA, f"{name}.csv"), "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)


if __name__ == "__main__":
    main()
