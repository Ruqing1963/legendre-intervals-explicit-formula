"""
Probe: in characteristic 2, Legendre-interval elements are P = h^2 + t b^2 with h = f + a.
For fixed f and b, is mu(P) (on squarefree P) of the form +-(-1)^{L(a)} with L affine over F_2?
Test the affine identity phi(a1)+phi(a2)+phi(a3) = phi(a1+a2+a3) on random triples.
"""
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fflib  # noqa: E402


def mul(a, b):
    r = [0] * (len(a) + len(b) - 1)
    for i, x in enumerate(a):
        if x:
            for j, y in enumerate(b):
                r[i + j] ^= y
    return r


def add(a, b):
    n = max(len(a), len(b))
    a = a + [0] * (n - len(a))
    b = b + [0] * (n - len(b))
    return [x ^ y for x, y in zip(a, b)]


def mu(P):
    pat = fflib.factor_pattern(P, 2)
    if pat is None:
        return 0
    return (-1) ** len(pat)


def main():
    rng = random.Random(1)
    for d in (8, 10, 12):
        f = [rng.randrange(2) for _ in range(d)] + [1]
        la, lb = d // 2 + 1, (d - 1) // 2 + 1  # numbers of coefficients of a and b
        for trial in range(4):
            b = [rng.randrange(2) for _ in range(lb)]
            if not any(b):
                continue
            tb2 = [0] + mul(b, b)

            def phi(abits):
                h = add(f, abits)
                P = add(mul(h, h), tb2)
                P = P[: 2 * d + 1] + [0] * max(0, 2 * d + 1 - len(P))
                m = mu(P)
                return None if m == 0 else (0 if m == 1 else 1)

            vals = {}
            for x in range(2 ** la):
                abits = [(x >> i) & 1 for i in range(la)]
                vals[x] = phi(abits)
            sq = [x for x, v in vals.items() if v is not None]
            ok = bad = 0
            for _ in range(3000):
                x1, x2, x3 = rng.choice(sq), rng.choice(sq), rng.choice(sq)
                x4 = x1 ^ x2 ^ x3
                if vals.get(x4) is None:
                    continue
                if (vals[x1] ^ vals[x2] ^ vals[x3]) == vals[x4]:
                    ok += 1
                else:
                    bad += 1
            ones = sum(v for v in vals.values() if v is not None)
            # does mu depend only on h mod b ?
            bt = b[:]
            while bt and bt[-1] == 0:
                bt.pop()

            def polymod(a, m):
                a = a[:]
                while len(a) >= len(m):
                    if a[-1]:
                        s = len(a) - len(m)
                        for i, y in enumerate(m):
                            a[s + i] ^= y
                    a.pop()
                while a and a[-1] == 0:
                    a.pop()
                return tuple(a)

            groups = {}
            for x, v in vals.items():
                if v is None:
                    continue
                abits = [(x >> i) & 1 for i in range(la)]
                key = polymod(add(f, abits), bt) if len(bt) > 1 else ()
                groups.setdefault(key, set()).add(v)
            mixed = sum(1 for s in groups.values() if len(s) > 1)
            print(f"d={d} b={b}: squarefree {len(sq)}/{2 ** la}, mu=-1 share {ones / len(sq):.3f}, "
                  f"affine ok/bad={ok}/{bad}; residue classes h mod b: {len(groups)}, mixed {mixed}")


if __name__ == "__main__":
    main()
