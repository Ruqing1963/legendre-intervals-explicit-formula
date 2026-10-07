"""
hayes_exact.py -- exact (rounding-free) version of the explicit-formula computation.

All quantities in the explicit formula
    c_k(chi) = sum_{deg g = k} chi([g]),   psi_n from Newton (no division),
    Psi(x)  = N^{-1} sum_chi conj(chi(x)) psi_{2d}(chi),   N = q^{d-1},
lie in Z[zeta][1/N] with zeta a primitive p^S-th root of unity (p^S = largest cyclic order of G).
For a prime M = 1 mod p^S and a primitive p^S-th root of unity z in F_M, the ring map
Z[zeta] -> F_M, zeta -> z, turns every identity above into an identity in F_M. Hence the same
computation done in F_M returns Psi(x) mod M exactly. Two primes M1, M2 < 2^31 with
M1 M2 > max Psi (Psi(x) <= 2d q^{d+1}) give Psi(x) itself by the Chinese remainder theorem.

Transforms are separable: one length-p^{s} DFT along each axis of the group, done naively
(the axis lengths are small).
"""
from __future__ import annotations

import os
import sys

import numpy as np
from numba import njit, prange

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hayes_fft_fq as H  # noqa: E402


def _is_prime(n):
    if n < 2:
        return False
    i = 2
    while i * i <= n:
        if n % i == 0:
            return False
        i += 1
    return True


def choose_primes(pS, count=2, upper=2 ** 31):
    """Largest primes M < upper with M = 1 mod pS."""
    out = []
    m = (upper - 1) // pS * pS + 1
    while len(out) < count:
        if m < upper and _is_prime(m):
            out.append(m)
        m -= pS
    return out


def root_of_unity(M, order):
    """A primitive `order`-th root of unity mod prime M (order | M-1)."""
    fac, n, f = [], M - 1, 2
    while f * f <= n:
        if n % f == 0:
            fac.append(f)
            while n % f == 0:
                n //= f
        f += 1
    if n > 1:
        fac.append(n)
    for g in range(2, M):
        if all(pow(g, (M - 1) // r, M) != 1 for r in fac):
            z = pow(g, (M - 1) // order, M)
            return z
    raise RuntimeError


@njit(parallel=True, cache=True)
def _axis_dft(arr, outer, n, inner, wpow, M):
    """In-place DFT of length n along the middle axis of arr viewed as (outer, n, inner).
    wpow[e] = w^e mod M for e in 0..n-1 (w of order n)."""
    for o in prange(outer):
        buf = np.empty(n, np.int64)
        out = np.empty(n, np.int64)
        base = o * n * inner
        for i in range(inner):
            for a in range(n):
                buf[a] = arr[base + a * inner + i]
            for b in range(n):
                s = 0
                for a in range(n):
                    s = (s + buf[a] * wpow[(a * b) % n]) % M
                out[b] = s
            for b in range(n):
                arr[base + b * inner + i] = out[b]


def transform(arr, shape, z_full, pS, M, inverse=False):
    """Multidimensional DFT mod M over the group with the given axis lengths (in place)."""
    N = arr.size
    outer = 1
    for ax, n in enumerate(shape):
        inner = N // (outer * n)
        w = pow(z_full, pS // n, M)
        if inverse:
            w = pow(w, M - 2, M)
        wpow = np.array([pow(w, e, M) for e in range(n)], np.int64)
        _axis_dft(arr, outer, n, inner, wpow, M)
        outer *= n
    return arr


@njit(parallel=True, cache=True)
def _newton_chunk(C, l, n_target, M, top, s0, s1):
    """psi_n = n c_n - sum_{k} psi_k c_{n-k}  (mod M), for characters s0..s1-1; writes psi_{n_target}."""
    for t in prange(s0, s1):
        psi = np.zeros(n_target + 1, np.int64)
        for n in range(1, n_target + 1):
            acc = (n % M) * C[n, t] % M if n < l else 0
            lo = n - l + 1
            if lo < 1:
                lo = 1
            for k in range(lo, n):
                acc = (acc - psi[k] * C[n - k, t]) % M
            psi[n] = acc
        top[t] = psi[n_target]


def psi_mod(q, d, M, ctx=None, work_dir=None):
    """Psi(x) mod M for every class x (flat index order of hayes_fft_fq.setup)."""
    l = d - 1
    if ctx is None:
        ctx = H.setup(q, l)
    shape = ctx["shape"]
    N = q ** l
    pS = max(shape) if shape else 1
    z = root_of_unity(M, pS) if pS > 1 else 1
    if work_dir is not None:
        C = np.memmap(os.path.join(work_dir, f"C_{M}.dat"), dtype=np.int64, mode="w+", shape=(max(l, 1), N))
    else:
        C = np.zeros((max(l, 1), N), np.int64)
    for kdeg in range(0, l):
        ind = H._indices_degree(kdeg, l, q, ctx["p"], ctx["k"], ctx["A"], ctx["M"], ctx["NEG"],
                                ctx["FROB"], ctx["FINV"], ctx["axis_of"], ctx["orders"], ctx["strides"])
        row = np.zeros(N, np.int64)
        np.add.at(row, ind, 1)
        transform(row, shape, z, pS, M)
        C[kdeg] = row
    top = np.zeros(N, np.int64)
    n_target = 2 * d
    if l >= 1:
        chunk = 1 << 20
        for s0 in range(0, N, chunk):
            _newton_chunk(C, l, n_target, M, top, s0, min(N, s0 + chunk))
    top[0] = pow(q, n_target, M)
    del C
    transform(top, shape, z, pS, M, inverse=True)
    Ninv = pow(N % M, M - 2, M)
    top = top * Ninv % M
    return top, ctx


def psi_exact(q, d, work_dir=None):
    """Exact integer Psi(x) for all classes, by two primes and CRT."""
    l = d - 1
    ctx = H.setup(q, l)
    pS = max(ctx["shape"]) if ctx["shape"] else 1
    M1, M2 = choose_primes(pS, 2)
    r1, _ = psi_mod(q, d, M1, ctx, work_dir)
    r2, _ = psi_mod(q, d, M2, ctx, work_dir)
    # CRT: x = r1 + M1 * ((r2 - r1) * M1^{-1} mod M2)
    inv = pow(M1, M2 - 2, M2)
    t = ((r2 - r1) % M2) * inv % M2
    x = r1 + M1 * t  # < M1 M2 < 2^62
    bound = 2 * d * q ** (d + 1)
    assert M1 * M2 > bound
    assert int(x.max()) <= bound, "CRT value outside the a priori range"
    return x, ctx, (M1, M2)
