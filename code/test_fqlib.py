"""
Cross-checks for fqlib.py (run: python code/test_fqlib.py).

1. interval counts agree with fflib (prime q) and with the pure-python small_q_gf (q = 4, 8, 9);
2. E(f) agrees with the subfield count of the twisted-variety enumeration (fflib.twisted_count);
3. the orbit decomposition covers every Legendre interval, and for odd q the weighted
   mean of sum(Lambda) over all intervals is exactly q^(d+1);
4. counts are invariant under the affine substitutions used to form orbits.
"""
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fflib  # noqa: E402
import fqlib  # noqa: E402
import small_q_gf  # noqa: E402


def test_against_fflib():
    rng = random.Random(1)
    for q, d in [(3, 3), (5, 3), (7, 2), (3, 4), (2, 4), (5, 4)]:
        tabs = fqlib.field_tables(q)
        for _ in range(3):
            f = [rng.randrange(q) for _ in range(d)] + [1]
            a = fqlib.interval_count(f, q, tabs)
            b = fflib.count_irreducible(f, q)
            assert a == b, (q, d, f, a, b)


def test_against_small_q_gf():
    for q, d in [(4, 2), (4, 3), (8, 2), (9, 2)]:
        tabs = fqlib.field_tables(q)
        slow = dict(small_q_gf.legendre_counts(q, d))
        for f, n in list(slow.items())[:6]:
            assert fqlib.interval_count(list(f), q, tabs) == n, (q, d, f)


def test_E_against_twisted():
    for q, f in [(5, [1, 0, 1]), (7, [1, 2, 0, 1]), (5, [0, 1, 3, 1]), (11, [2, 0, 1])]:
        tabs = fqlib.field_tables(q)
        tw = fflib.twisted_count(f, q)
        assert fqlib.prime_power_E(f, q, tabs) == tw["E"], (q, f)


def test_orbits_and_mean():
    for q, d in [(3, 4), (5, 3), (4, 3), (2, 5)]:
        tabs = fqlib.field_tables(q)
        orbits = fqlib.legendre_orbits(q, d)
        total = sum(sz for _, sz in orbits)
        A, M, _, _ = tabs
        distinct = set()
        import itertools
        for tail in itertools.product(range(q), repeat=d - 1):
            distinct.add(fqlib.py_square_prefix([0] + list(tail) + [1], A, M))
        assert total == len(distinct), (q, d, total, len(distinct))
        if q % 2 == 1:
            assert total == q ** (d - 1)  # every prefix is a Legendre prefix in odd characteristic
            s = 0
            for f, sz in orbits:
                n = fqlib.interval_count(f, q, tabs)
                E = fqlib.prime_power_E(f, q, tabs)
                s += sz * (2 * d * n + E)
            assert s == q ** (d - 1) * q ** (d + 1), (q, d, s)


def test_affine_invariance():
    rng = random.Random(2)
    for q, d in [(5, 3), (4, 3), (3, 4)]:
        tabs = fqlib.field_tables(q)
        A, M, NEG, INV = tabs
        for _ in range(3):
            f = [rng.randrange(q) for _ in range(d)] + [1]
            n0 = fqlib.interval_count(f, q, tabs)
            g = fqlib.py_translate(f, rng.randrange(q), A, M)
            h = fqlib.py_scale(f, rng.randrange(1, q), A, M, INV)
            assert fqlib.interval_count(g, q, tabs) == n0
            assert fqlib.interval_count(h, q, tabs) == n0


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
