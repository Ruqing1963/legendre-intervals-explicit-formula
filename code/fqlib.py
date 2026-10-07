"""
fqlib.py -- table-based arithmetic over any small finite field F_q (q = p^k <= 16),
accelerated with numba, used for the "open range" q <= d computations.

Field elements are integers 0..q-1 (base-p digit vectors w.r.t. a polynomial basis);
0 and 1 are the additive and multiplicative identities. Polynomials are int64 arrays,
LOW degree first, with coefficients in 0..q-1.

Main entry points
-----------------
field_tables(q)                -> (A, M, NEG, INV) numpy tables
legendre_orbits(q, d)          -> list of (representative f, orbit size) over all Legendre intervals
interval_count(f, q, tabs)     -> number of irreducible P in I_f = {f^2 + s : deg s <= d}
prime_power_E(f, q, tabs)      -> E(f) = sum over prime powers pi^r in I_f (r >= 2) of deg(pi)
"""
from __future__ import annotations

import itertools
import os
import sys

import numpy as np
from numba import njit, prange

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from small_q_gf import make_field  # noqa: E402

# monic irreducible moduli (low first) for the prime-power fields used here
MODULI = {2: (2, 1, [0, 1]), 3: (3, 1, [0, 1]), 5: (5, 1, [0, 1]), 7: (7, 1, [0, 1]),
          11: (11, 1, [0, 1]), 13: (13, 1, [0, 1]),
          4: (2, 2, [1, 1, 1]), 8: (2, 3, [1, 1, 0, 1]), 9: (3, 2, [1, 0, 1]), 16: (2, 4, [1, 1, 0, 0, 1])}


def field_tables(q):
    p, k, mod = MODULI[q]
    _, add, mul, neg, inv = make_field(p, k, mod)
    A = np.array(add, np.int64)
    M = np.array(mul, np.int64)
    NEG = np.array(neg, np.int64)
    INV = np.array(inv, np.int64)
    return A, M, NEG, INV


def char_of(q):
    return MODULI[q][0]


# ----------------------------------------------------------------------------
# python-level polynomial helpers (used only for orbit bookkeeping)
# ----------------------------------------------------------------------------


def py_mul(a, b, A, M):
    r = [0] * (len(a) + len(b) - 1)
    for i, x in enumerate(a):
        if x:
            for j, y in enumerate(b):
                r[i + j] = int(A[r[i + j], M[x, y]])
    return r


def py_square_prefix(f, A, M):
    """Coefficients of t^{2d-1}, ..., t^{d+1} of f^2 (the data defining I_f)."""
    d = len(f) - 1
    sq = py_mul(f, f, A, M)
    return tuple(sq[2 * d - j] for j in range(1, d))


def py_translate(f, b, A, M):
    """f(t + b) by Horner's rule."""
    res = [0]
    for c in reversed(f):
        # res = res * (t + b) + c
        new = [0] * (len(res) + 1)
        for i, x in enumerate(res):
            new[i + 1] = int(A[new[i + 1], x])
            new[i] = int(A[new[i], M[x, b]])
        new[0] = int(A[new[0], c])
        res = new
    while len(res) > 1 and res[-1] == 0:
        res.pop()
    return res


def py_scale(f, lam, A, M, INV):
    """lambda^{-d} f(lambda t): coefficient c_i -> lambda^{i-d} c_i (monic again)."""
    d = len(f) - 1
    li = int(INV[lam])
    out = []
    for i, c in enumerate(f):
        e = d - i
        w = 1
        for _ in range(e):
            w = int(M[w, li])
        out.append(int(M[c, w]))
    return out


def primitive_element(q, M):
    for g in range(2, q) if q > 2 else [1]:
        x, order = g, 1
        while x != 1:
            x = int(M[x, g])
            order += 1
        if order == q - 1:
            return g
    return 1


def legendre_orbits(q, d):
    """
    All distinct Legendre intervals I_f (f monic of degree d over F_q), grouped into orbits under
    the affine substitutions t -> lambda t + b, which preserve irreducibility and map Legendre
    intervals to Legendre intervals. Returns [(representative f, number of intervals in orbit)].
    """
    A, M, NEG, INV = field_tables(q)
    p = char_of(q)
    reps = {}
    for tail in itertools.product(range(q), repeat=d - 1):
        f = [0] + list(tail) + [1]
        key = py_square_prefix(f, A, M)
        if key not in reps:
            reps[key] = f
    keys = list(reps)
    parent = {k: k for k in keys}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    gens = []
    g = primitive_element(q, M)
    if q > 2:
        gens.append(("scale", g))
    basis = [p ** i for i in range(MODULI[q][1])]  # additive generators 1, x, x^2, ... of F_q
    for b in basis:
        gens.append(("shift", b))
    for k in keys:
        f = reps[k]
        for kind, val in gens:
            if kind == "scale":
                h = py_scale(f, val, A, M, INV)
            else:
                h = py_translate(f, val, A, M)
            hk = py_square_prefix(h, A, M)
            ra, rb = find(k), find(hk)
            if ra != rb:
                parent[ra] = rb
    groups = {}
    for k in keys:
        groups.setdefault(find(k), []).append(k)
    out = []
    for root, members in groups.items():
        rep = min(members)
        out.append((reps[rep], len(members)))
    out.sort(key=lambda t: t[0])
    return out


# ----------------------------------------------------------------------------
# numba kernels
# ----------------------------------------------------------------------------


@njit(cache=True)
def _deg(a, upto):
    d = upto
    while d >= 0 and a[d] == 0:
        d -= 1
    return d


@njit(cache=True)
def _reduce(a, da, m, dm, A, M, NEG):
    """a <- a mod m in place (m monic of degree dm). Returns new degree."""
    for k in range(da, dm - 1, -1):
        c = a[k]
        if c != 0:
            s = k - dm
            for j in range(dm + 1):
                a[s + j] = A[a[s + j], NEG[M[c, m[j]]]]
    return _deg(a, min(da, dm - 1))


@njit(cache=True)
def _mulmod(a, da, b, db, m, dm, A, M, NEG, tmp, out):
    if da < 0 or db < 0:
        for i in range(dm):
            out[i] = 0
        return -1
    for i in range(da + db + 1):
        tmp[i] = 0
    for i in range(da + 1):
        x = a[i]
        if x != 0:
            for j in range(db + 1):
                tmp[i + j] = A[tmp[i + j], M[x, b[j]]]
    dt = _reduce(tmp, da + db, m, dm, A, M, NEG)
    for i in range(dm):
        out[i] = tmp[i] if i <= dt else 0
    return dt


@njit(cache=True)
def _powq(h, dh, q, m, dm, A, M, NEG, tmp, acc, sq, res):
    """res <- h^q mod m."""
    for i in range(dm):
        acc[i] = 0
        sq[i] = h[i] if i <= dh else 0
    acc[0] = 1
    dacc = 0
    dsq = dh
    e = q
    while e > 0:
        if e & 1:
            dacc = _mulmod(acc, dacc, sq, dsq, m, dm, A, M, NEG, tmp, res)
            for i in range(dm):
                acc[i] = res[i]
        e >>= 1
        if e > 0:
            dsq = _mulmod(sq, dsq, sq, dsq, m, dm, A, M, NEG, tmp, res)
            for i in range(dm):
                sq[i] = res[i]
    for i in range(dm):
        res[i] = acc[i]
    return dacc


@njit(cache=True)
def _gcd_deg(a, da, b, db, A, M, NEG, INV, wa, wb):
    """Degree of gcd(a, b) (-1 if both zero)."""
    for i in range(da + 1):
        wa[i] = a[i]
    for i in range(db + 1):
        wb[i] = b[i]
    while db >= 0:
        li = INV[wb[db]]
        while da >= db and da >= 0:
            c = M[wa[da], li]
            if c != 0:
                s = da - db
                for j in range(db + 1):
                    wa[s + j] = A[wa[s + j], NEG[M[c, wb[j]]]]
            da -= 1
            while da >= 0 and wa[da] == 0:
                da -= 1
        top = da if da > db else db
        for i in range(top + 1):
            t = wa[i]
            wa[i] = wb[i]
            wb[i] = t
        t2 = da
        da = db
        db = t2
    return da


@njit(cache=True)
def _is_irreducible(P, n, q, A, M, NEG, INV, W):
    """Ben-Or test: P monic of degree n is irreducible iff gcd(x^{q^i} - x, P) = 1 for i <= n/2."""
    h = W[0]
    tmp = W[1]
    acc = W[2]
    sq = W[3]
    res = W[4]
    hx = W[5]
    wa = W[6]
    wb = W[7]
    for i in range(W.shape[1]):
        h[i] = 0
    h[1] = 1
    dh = 1
    if n == 1:
        return True
    for i in range(1, n // 2 + 1):
        dh = _powq(h, dh, q, P, n, A, M, NEG, tmp, acc, sq, res)
        for j in range(n):
            h[j] = res[j]
        for j in range(n):
            hx[j] = h[j]
        hx[1] = A[hx[1], NEG[1]]
        dhx = _deg(hx, n - 1)
        if dhx < 0:
            return False  # x^{q^i} = x mod P with i < n: P has a factor of degree | i
        g = _gcd_deg(P, n, hx, dhx, A, M, NEG, INV, wa, wb)
        if g > 0:
            return False
    return True


@njit(parallel=True, cache=True)
def _interval_count(base, d, q, A, M, NEG, INV, nchunks):
    n = 2 * d
    total = q ** (d + 1)
    counts = np.zeros(nchunks, np.int64)
    per = (total + nchunks - 1) // nchunks
    for c in prange(nchunks):
        W = np.zeros((8, 2 * n + 2), np.int64)
        P = np.zeros(n + 1, np.int64)
        lo = c * per
        hi = min(total, lo + per)
        cnt = 0
        for idx in range(lo, hi):
            for i in range(n + 1):
                P[i] = base[i]
            r = idx
            for i in range(d + 1):
                P[i] = A[P[i], r % q]
                r //= q
            if _is_irreducible(P, n, q, A, M, NEG, INV, W):
                cnt += 1
        counts[c] = cnt
    return counts.sum()


@njit(cache=True)
def _prime_power_E(prefix, d, q, A, M, NEG, INV):
    """sum of deg(pi) over monic irreducible pi with pi^r (r >= 2) in the interval with this prefix."""
    n = 2 * d
    E = 0
    W = np.zeros((8, 2 * n + 2), np.int64)
    for e in range(1, n):
        if n % e != 0:
            continue
        r = n // e
        total = q ** e
        g = np.zeros(e + 1, np.int64)
        pw = np.zeros(n + 1, np.int64)
        tmp = np.zeros(n + 1, np.int64)
        for idx in range(total):
            x = idx
            for i in range(e):
                g[i] = x % q
                x //= q
            g[e] = 1
            # pw = g^r
            for i in range(n + 1):
                pw[i] = 0
            for i in range(e + 1):
                pw[i] = g[i]
            dp = e
            for _ in range(r - 1):
                for i in range(dp + e + 1):
                    tmp[i] = 0
                for i in range(dp + 1):
                    xi = pw[i]
                    if xi != 0:
                        for j in range(e + 1):
                            tmp[i + j] = A[tmp[i + j], M[xi, g[j]]]
                dp += e
                for i in range(dp + 1):
                    pw[i] = tmp[i]
            ok = True
            for j in range(1, d):
                if pw[n - j] != prefix[j - 1]:
                    ok = False
                    break
            if ok and _is_irreducible(g, e, q, A, M, NEG, INV, W):
                E += e
    return E


def interval_count(f, q, tabs, nchunks=64):
    A, M, NEG, INV = tabs
    d = len(f) - 1
    sq = py_mul(list(f), list(f), A, M)
    base = np.array(sq, np.int64)
    return int(_interval_count(base, d, q, A, M, NEG, INV, nchunks))


def prime_power_E(f, q, tabs):
    A, M, NEG, INV = tabs
    prefix = np.array(py_square_prefix(list(f), A, M) or (0,), np.int64)
    d = len(f) - 1
    return int(_prime_power_E(prefix, d, q, A, M, NEG, INV))
