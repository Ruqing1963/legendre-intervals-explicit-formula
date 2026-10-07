"""
Exhaustive check of Legendre intervals over small non-prime fields F_q (q = p^k, tiny q).

Field elements are integers 0..q-1 (base-p digit vectors of a polynomial basis
modulo a fixed irreducible); add/mul tables are precomputed. Irreducibility of a
monic P of degree n is tested with Rabin's criterion:
    x^(q^n) = x mod P  and  gcd(x^(q^(n/r)) - x, P) = 1 for every prime r | n.

Usage:  python code/small_q_gf.py 4 4      (q = 4, d = 4)
"""
import itertools
import sys


def make_field(p, k, modulus):
    """modulus: coefficients (low first) of a monic irreducible of degree k over F_p."""
    q = p ** k

    def to_vec(a):
        return [(a // p ** i) % p for i in range(k)]

    def to_int(v):
        return sum(c * p ** i for i, c in enumerate(v))

    add = [[to_int([(x + y) % p for x, y in zip(to_vec(a), to_vec(b))]) for b in range(q)] for a in range(q)]
    mul = [[0] * q for _ in range(q)]
    for a in range(q):
        va = to_vec(a)
        for b in range(q):
            vb = to_vec(b)
            prod = [0] * (2 * k - 1)
            for i, x in enumerate(va):
                for j, y in enumerate(vb):
                    prod[i + j] = (prod[i + j] + x * y) % p
            for m in range(2 * k - 2, k - 1, -1):
                c = prod[m]
                if c:
                    for i in range(k + 1):
                        prod[m - k + i] = (prod[m - k + i] - c * modulus[i]) % p
            mul[a][b] = to_int(prod[:k])
    neg = [next(b for b in range(q) if add[a][b] == 0) for a in range(q)]
    inv = [0] + [next(b for b in range(q) if mul[a][b] == 1) for a in range(1, q)]
    return q, add, mul, neg, inv


class Poly:
    def __init__(self, field):
        self.q, self.add, self.mul, self.neg, self.inv = field

    def trim(self, a):
        while a and a[-1] == 0:
            a.pop()
        return a

    def sub(self, a, b):
        n = max(len(a), len(b))
        a = a + [0] * (n - len(a))
        b = b + [0] * (n - len(b))
        return self.trim([self.add[x][self.neg[y]] for x, y in zip(a, b)])

    def mulmod(self, a, b, m):
        if not a or not b:
            return []
        r = [0] * (len(a) + len(b) - 1)
        for i, x in enumerate(a):
            if x:
                for j, y in enumerate(b):
                    r[i + j] = self.add[r[i + j]][self.mul[x][y]]
        return self.mod(r, m)

    def mod(self, a, m):
        a = a[:]
        dm = len(m) - 1
        li = self.inv[m[-1]]
        for top in range(len(a) - 1, dm - 1, -1):
            c = self.mul[a[top]][li]
            if c:
                for i in range(dm + 1):
                    a[top - dm + i] = self.add[a[top - dm + i]][self.neg[self.mul[c][m[i]]]]
        return self.trim(a[:dm] if len(a) > dm else a)

    def powq(self, a, m):
        """a^q mod m by repeated squaring/multiplication."""
        e = self.q
        res = [1]
        base = a
        while e:
            if e & 1:
                res = self.mulmod(res, base, m)
            base = self.mulmod(base, base, m)
            e >>= 1
        return res

    def gcd(self, a, b):
        a, b = self.trim(a[:]), self.trim(b[:])
        while b:
            a, b = b, self.mod(a, b)
        return a

    def is_irreducible(self, P):
        n = len(P) - 1
        x = [0, 1]
        primes = [r for r in range(2, n + 1) if n % r == 0 and all(r % s for s in range(2, r))]
        frob = [x]
        h = x
        for _ in range(n):
            h = self.powq(h, P)
            frob.append(h)
        if self.sub(frob[n], x):
            return False
        for r in primes:
            g = self.gcd(P, self.sub(frob[n // r], x))
            if len(g) > 1:
                return False
        return True


FIELDS = {4: (2, 2, [1, 1, 1]), 8: (2, 3, [1, 1, 0, 1]), 9: (3, 2, [1, 0, 1]), 16: (2, 4, [1, 1, 0, 0, 1])}


def legendre_counts(q, d):
    p, k, modulus = FIELDS[q]
    F = make_field(p, k, modulus)
    P = Poly(F)
    results = []
    for tail in itertools.product(range(q), repeat=d - 1):
        f = [0] + list(tail) + [1]  # c_0 irrelevant; c_1..c_{d-1} free
        f2 = [0] * (2 * d + 1)
        for i, a in enumerate(f):
            for j, b in enumerate(f):
                f2[i + j] = F[1][f2[i + j]][F[2][a][b]]
        n_irr = 0
        for s in itertools.product(range(q), repeat=d + 1):
            Q = f2[:]
            for i, c in enumerate(s):
                Q[i] = F[1][Q[i]][c]
            if P.is_irreducible(Q):
                n_irr += 1
        results.append((tuple(f), n_irr))
    return results


if __name__ == "__main__":
    q, d = int(sys.argv[1]), int(sys.argv[2])
    res = legendre_counts(q, d)
    vals = [n for _, n in res]
    print(f"q={q} d={d}: classes={len(vals)} min={min(vals)} max={max(vals)}")
