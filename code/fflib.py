"""
fflib.py -- polynomial arithmetic over prime fields F_p, accelerated with numba.

Conventions
-----------
* A polynomial is a 1-D int64 numpy array of coefficients, LOW degree first:
  [c0, c1, ..., cn]  <->  c0 + c1 t + ... + cn t^n.
* All moduli used for reduction are MONIC.
* Only prime fields q = p are supported (this is all the paper's numerics need).

Main entry points
-----------------
interval_codes(f, p)        factorisation-type code of every P in I_f = {f^2 + s : deg s <= d}
decode_code(code, n)        code -> partition (tuple of factor degrees, descending)
twisted_count(f, p)         enumerate beta in F_{p^{2d}} whose characteristic polynomial lies in I_f
"""
from __future__ import annotations

import numpy as np
from numba import njit, prange

# ----------------------------------------------------------------------------
# basic helpers (pure python)
# ----------------------------------------------------------------------------


def poly_mul_py(a, b, p):
    r = [0] * (len(a) + len(b) - 1)
    for i, x in enumerate(a):
        for j, y in enumerate(b):
            r[i + j] = (r[i + j] + x * y) % p
    return r


def f_squared(f, p):
    """Coefficients (low first) of f(t)^2 mod p; f is monic, low first."""
    return [c % p for c in poly_mul_py(list(f), list(f), p)]


def encode_pattern(counts, n):
    """counts[k] = number of irreducible factors of degree k (k = 1..n)."""
    code = 0
    base = 1
    for k in range(1, n + 1):
        code += counts[k] * base
        base *= n + 1
    return code


def decode_code(code, n):
    """Inverse of encode_pattern; returns the partition as a descending tuple."""
    if code < 0:
        return None
    parts = []
    for k in range(1, n + 1):
        c = code % (n + 1)
        code //= n + 1
        parts.extend([k] * c)
    return tuple(sorted(parts, reverse=True))


def irreducible_code(n):
    return (n + 1) ** (n - 1)


# ----------------------------------------------------------------------------
# numba kernels
# ----------------------------------------------------------------------------


@njit(cache=True)
def _inv_table(p):
    inv = np.zeros(p, np.int64)
    for a in range(1, p):
        r = 1
        e = p - 2
        b = a
        while e > 0:
            if e & 1:
                r = (r * b) % p
            b = (b * b) % p
            e >>= 1
        inv[a] = r
    return inv


@njit(cache=True)
def _deg(a, upto):
    d = upto
    while d >= 0 and a[d] == 0:
        d -= 1
    return d


@njit(cache=True)
def _mulmod(a, da, b, db, m, n, p, tmp, out):
    """out <- a*b mod m  (m monic of degree n; deg a, deg b < n). Returns deg(out)."""
    if da < 0 or db < 0:
        for i in range(n):
            out[i] = 0
        return -1
    dt = da + db
    for i in range(dt + 1):
        tmp[i] = 0
    for i in range(da + 1):
        ai = a[i]
        if ai != 0:
            for j in range(db + 1):
                tmp[i + j] = (tmp[i + j] + ai * b[j]) % p
    for k in range(dt, n - 1, -1):
        c = tmp[k]
        if c != 0:
            s = k - n
            for j in range(n):
                tmp[s + j] = (tmp[s + j] - c * m[j]) % p
            tmp[k] = 0
    top = dt if dt < n - 1 else n - 1
    for i in range(n):
        out[i] = tmp[i] if i <= top else 0
    return _deg(out, n - 1)


@njit(cache=True)
def _powmod(base, dbase, e, m, n, p, tmp, acc, sq, res):
    """res <- base^e mod m. Returns degree of res."""
    for i in range(n):
        acc[i] = 0
        sq[i] = base[i] if i <= dbase else 0
    acc[0] = 1
    dacc = 0
    dsq = dbase
    while e > 0:
        if e & 1:
            dacc = _mulmod(acc, dacc, sq, dsq, m, n, p, tmp, res)
            for i in range(n):
                acc[i] = res[i]
        e >>= 1
        if e > 0:
            dsq = _mulmod(sq, dsq, sq, dsq, m, n, p, tmp, res)
            for i in range(n):
                sq[i] = res[i]
    for i in range(n):
        res[i] = acc[i]
    return dacc


@njit(cache=True)
def _gcd(a, da, b, db, p, inv, wa, wb):
    """Monic gcd of a and b, written into wa; returns its degree (-1 if both zero)."""
    for i in range(da + 1):
        wa[i] = a[i]
    for i in range(db + 1):
        wb[i] = b[i]
    # make sure arrays are clean above degree
    while db >= 0:
        # wa <- wa mod wb
        lead_inv = inv[wb[db]]
        while da >= db and da >= 0:
            c = (wa[da] * lead_inv) % p
            if c != 0:
                s = da - db
                for j in range(db + 1):
                    wa[s + j] = (wa[s + j] - c * wb[j]) % p
            da -= 1
            while da >= 0 and wa[da] == 0:
                da -= 1
        # swap
        for i in range(max(da, db) + 1):
            t = wa[i]
            wa[i] = wb[i]
            wb[i] = t
        t2 = da
        da = db
        db = t2
    if da >= 0:
        li = inv[wa[da]]
        for i in range(da + 1):
            wa[i] = (wa[i] * li) % p
    return da


@njit(cache=True)
def _divexact(a, da, g, dg, p, quot):
    """quot <- a / g (g monic, exact division assumed). Returns deg quot."""
    w = a.copy()
    dq = da - dg
    for i in range(dq + 1):
        quot[i] = 0
    for k in range(da, dg - 1, -1):
        c = w[k]
        if c != 0:
            s = k - dg
            quot[s] = c
            for j in range(dg + 1):
                w[s + j] = (w[s + j] - c * g[j]) % p
    return dq


@njit(cache=True)
def _pattern(P, n, p, inv, counts, W):
    """
    Factorisation pattern of the monic polynomial P (degree n) over F_p.
    counts[k] <- number of irreducible factors of degree k (only if P squarefree).
    Returns 1 if P is squarefree, 0 otherwise.
    W is a (8, 2n+2) int64 work array.
    """
    for k in range(n + 1):
        counts[k] = 0
    # derivative
    der = W[0]
    for i in range(2 * n + 2):
        der[i] = 0
    for i in range(1, n + 1):
        der[i - 1] = (i * P[i]) % p
    dd = _deg(der, n - 1)
    if dd < 0:
        return 0
    g = W[1]
    wb = W[2]
    dg = _gcd(P, n, der, dd, p, inv, g, wb)
    if dg > 0:
        return 0
    # distinct degree factorisation
    cur = W[3]
    for i in range(2 * n + 2):
        cur[i] = 0
    for i in range(n + 1):
        cur[i] = P[i]
    dc = n
    h = W[4]
    tmp = W[5]
    acc = W[6]
    sq = W[7]
    res = np.zeros(2 * n + 2, np.int64)
    hx = np.zeros(2 * n + 2, np.int64)
    quot = np.zeros(2 * n + 2, np.int64)
    for i in range(2 * n + 2):
        h[i] = 0
    h[1] = 1  # h = x
    dh = 1
    k = 0
    while 2 * (k + 1) <= dc:
        k += 1
        if dh >= dc:  # reduce h mod cur
            pass
        dh = _powmod(h, dh, p, cur, dc, p, tmp, acc, sq, res)
        for i in range(dc):
            h[i] = res[i]
        for i in range(dc, 2 * n + 2):
            h[i] = 0
        # hx = h - x
        for i in range(2 * n + 2):
            hx[i] = h[i]
        hx[1] = (hx[1] - 1) % p
        dhx = _deg(hx, dc - 1 if dc - 1 > 1 else 1)
        if dhx < 0:
            # x^{p^k} = x mod cur: cur splits completely into degree-<=k factors;
            # since all factors of degree < k were removed, all have degree k
            counts[k] += dc // k
            dc = 0
            break
        dg = _gcd(cur, dc, hx, dhx, p, inv, g, wb)
        if dg > 0:
            counts[k] += dg // k
            dq = _divexact(cur, dc, g, dg, p, quot)
            for i in range(2 * n + 2):
                cur[i] = 0
            for i in range(dq + 1):
                cur[i] = quot[i]
            dc = dq
            # reduce h modulo the new cur
            if dc > 0:
                for i in range(2 * n + 2):
                    tmp[i] = 0
                for i in range(dh + 1):
                    tmp[i] = h[i]
                for kk in range(dh, dc - 1, -1):
                    c = tmp[kk]
                    if c != 0:
                        s = kk - dc
                        for j in range(dc):
                            tmp[s + j] = (tmp[s + j] - c * cur[j]) % p
                        tmp[kk] = 0
                for i in range(2 * n + 2):
                    h[i] = tmp[i] if i < dc else 0
                dh = _deg(h, dc - 1)
    if dc > 0:
        counts[dc] += 1
    return 1


@njit(parallel=True, cache=True)
def _interval_codes(base, d, p):
    """Pattern code for every P = base + sum_{i<=d} a_i t^i, index = sum a_i p^i."""
    n = 2 * d
    total = p ** (d + 1)
    inv = _inv_table(p)
    codes = np.empty(total, np.int64)
    block = p ** d
    for top in prange(p):
        W = np.zeros((8, 2 * n + 2), np.int64)
        P = np.zeros(2 * n + 2, np.int64)
        counts = np.zeros(n + 1, np.int64)
        for rest in range(block):
            for i in range(n + 1):
                P[i] = base[i]
            r = rest
            for i in range(d):
                P[i] = (P[i] + r % p) % p
                r //= p
            P[d] = (P[d] + top) % p
            ok = _pattern(P, n, p, inv, counts, W)
            idx = top * block + rest
            if ok == 0:
                codes[idx] = -1
            else:
                c = 0
                b = 1
                for k in range(1, n + 1):
                    c += counts[k] * b
                    b *= n + 1
                codes[idx] = c
    return codes


def interval_codes(f, p):
    """Return an int64 array with the factorisation code of each element of I_f."""
    f = [c % p for c in f]
    d = len(f) - 1
    assert f[-1] == 1, "f must be monic"
    base = np.zeros(2 * (2 * d) + 2, np.int64)
    sq = f_squared(f, p)
    base[: len(sq)] = sq
    return _interval_codes(base, d, p)


def count_irreducible(f, p):
    d = len(f) - 1
    codes = interval_codes(f, p)
    return int(np.count_nonzero(codes == irreducible_code(2 * d)))


@njit(cache=True)
def _pattern_single(P, n, p):
    inv = _inv_table(p)
    W = np.zeros((8, 2 * n + 2), np.int64)
    counts = np.zeros(n + 1, np.int64)
    ok = _pattern(P, n, p, inv, counts, W)
    return ok, counts


def factor_pattern(P, p):
    """Pattern of a single monic polynomial P (low first). None if not squarefree."""
    P = [c % p for c in P]
    n = len(P) - 1
    arr = np.zeros(2 * n + 2, np.int64)
    arr[: n + 1] = P
    ok, counts = _pattern_single(arr, n, p)
    if not ok:
        return None
    return decode_code(encode_pattern(list(counts), n), n)


def is_irreducible(P, p):
    pat = factor_pattern(P, p)
    return pat is not None and len(pat) == 1


# ----------------------------------------------------------------------------
# finite field F_{p^N} and the twisted variety W(F_p)
# ----------------------------------------------------------------------------


def find_irreducible(N, p):
    """Smallest (lexicographic) monic irreducible polynomial of degree N over F_p."""
    import itertools

    for tail in itertools.product(range(p), repeat=N):
        P = list(tail[::-1]) + [1]
        if P[0] == 0:
            continue
        if is_irreducible(P, p):
            return P
    raise RuntimeError("no irreducible polynomial found")


def newton_power_sums(P, k, p):
    """Power sums s_1..s_k of the roots of the monic polynomial P (low first), mod p.
    Uses s_j + c_{n-1} s_{j-1} + ... + c_{n-j+1} s_1 + j c_{n-j} = 0  (no division)."""
    n = len(P) - 1
    c = {i: P[i] % p for i in range(n + 1)}
    s = [0] * (k + 1)
    for j in range(1, k + 1):
        val = 0
        for i in range(1, j):
            val += c.get(n - i, 0) * s[j - i] if n - i >= 0 else 0
        if j <= n:
            val += j * c[n - j]
        s[j] = (-val) % p
    return s[1:]


@njit(parallel=True, cache=True)
def _twisted(m, N, d, p, trx, targets):
    """
    For each beta in F_p[x]/(m) (encoded by base-p digits), flag:
      0 : Tr(beta^j) != targets[j] for some 1 <= j <= d-1
      1 : all equal and beta generates F_{p^N}
      2 : all equal and beta lies in a proper subfield
    """
    total = p ** N
    flags = np.zeros(total, np.int8)
    block = p ** (N - 1)
    # proper divisors of N
    ndiv = 0
    divs = np.zeros(N, np.int64)
    for e in range(1, N):
        if N % e == 0:
            divs[ndiv] = e
            ndiv += 1
    for top in prange(p):
        beta = np.zeros(N, np.int64)
        cur = np.zeros(N, np.int64)
        tmp = np.zeros(2 * N + 2, np.int64)
        out = np.zeros(N, np.int64)
        acc = np.zeros(N, np.int64)
        sq = np.zeros(N, np.int64)
        res = np.zeros(N, np.int64)
        fr = np.zeros(N, np.int64)
        for rest in range(block):
            r = rest
            for i in range(N - 1):
                beta[i] = r % p
                r //= p
            beta[N - 1] = top
            db = _deg(beta, N - 1)
            good = True
            for i in range(N):
                cur[i] = beta[i]
            dcur = db
            for j in range(1, d):
                if j > 1:
                    dcur = _mulmod(cur, dcur, beta, db, m, N, p, tmp, out)
                    for i in range(N):
                        cur[i] = out[i]
                t = 0
                for i in range(N):
                    t += cur[i] * trx[i]
                if t % p != targets[j]:
                    good = False
                    break
            idx = top * block + rest
            if not good:
                flags[idx] = 0
                continue
            # subfield test: beta^{p^e} == beta for some proper divisor e
            insub = False
            for i in range(N):
                fr[i] = beta[i]
            dfr = db
            done_e = 0
            for t_i in range(ndiv):
                e = divs[t_i]
                while done_e < e:
                    dfr = _powmod(fr, dfr, p, m, N, p, tmp, acc, sq, res)
                    for i in range(N):
                        fr[i] = res[i]
                    done_e += 1
                same = True
                for i in range(N):
                    if fr[i] != beta[i]:
                        same = False
                        break
                if same:
                    insub = True
                    break
            flags[idx] = 2 if insub else 1
    return flags


def twisted_count(f, p, m=None):
    """
    Count beta in F_{p^{2d}} whose characteristic polynomial over F_p lies in I_f.
    Valid when p > d-1 (Newton identities). Returns dict with
      W  = #W(F_p),  gen = # generators of F_{p^{2d}} among them,  E = # in proper subfields.
    """
    f = [c % p for c in f]
    d = len(f) - 1
    N = 2 * d
    if m is None:
        m = find_irreducible(N, p)
    marr = np.array(m, np.int64)
    trx_list = [N % p] + newton_power_sums(m, N - 1, p)
    trx = np.array(trx_list, np.int64)
    sq = f_squared(f, p)
    tgt = [0] + newton_power_sums(sq, max(d - 1, 1), p)
    targets = np.array(tgt + [0], np.int64)
    flags = _twisted(marr, N, d, p, trx, targets)
    gen = int(np.count_nonzero(flags == 1))
    E = int(np.count_nonzero(flags == 2))
    return {"W": gen + E, "gen": gen, "E": E, "modulus": m}


# ----------------------------------------------------------------------------
# closed forms (Theorem 6.1 of the revised paper)
# ----------------------------------------------------------------------------


def legendre_symbol(a, p):
    a %= p
    if a == 0:
        return 0
    return 1 if pow(a, (p - 1) // 2, p) == 1 else -1


def closed_form(f, p):
    """Exact N_irr for d <= 3 (q = p prime). Returns None when no closed form applies."""
    f = [c % p for c in f]
    d = len(f) - 1
    q = p
    if d == 1:
        return (q * q - q) // 2
    if d == 2 and p % 2 == 1:
        return (q ** 3 - q) // 4
    if d == 3 and p >= 5:
        c2, c1 = f[2], f[1]
        Delta = (c2 * c2 - 3 * c1) % p
        eta_m3 = legendre_symbol(-3, p)
        if Delta == 0:
            six_n = q ** 4 - q - (q - 1) * eta_m3
        else:
            e = legendre_symbol(2 * Delta, p)
            six_n = q ** 4 - e * q * q - q + eta_m3 - 1 + e
        assert six_n % 6 == 0
        return six_n // 6
    return None
