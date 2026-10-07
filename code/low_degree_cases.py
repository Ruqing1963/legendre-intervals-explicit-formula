"""
The Legendre intervals not certified by the bound Psi > d q^a + 2 q^{floor(2d/3)} in the exact scan:
(q, d) = (2, 2), (2, 3), (2, 4), (2, 6). For every Legendre interval of these cases we test every
element (fqlib.interval_count) and enumerate the prime powers (fqlib.prime_power_E), and record
Psi, E and N_irr. Output: data/low_degree_char2.csv and paper/tables/lowdeg.tex
"""
import csv
import itertools
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fqlib  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    q = 2
    tabs = fqlib.field_tables(q)
    A, M, _, _ = tabs
    rows = []
    for d in (2, 3, 4, 6):
        seen = {}
        for tail in itertools.product(range(q), repeat=d - 1):
            f = [0] + list(tail) + [1]
            key = fqlib.py_square_prefix(f, A, M)
            seen.setdefault(key, f)
        bound = d * q ** ((d + 2) // 2) + 2 * q ** (2 * d // 3)
        for key, f in sorted(seen.items()):
            N = fqlib.interval_count(f, q, tabs)
            E = fqlib.prime_power_E(f, q, tabs)
            rows.append({"q": q, "d": d, "f_coeffs_low_first": " ".join(map(str, f)),
                         "Psi": 2 * d * N + E, "E": E, "N_irr": N, "certification_bound": bound})
    with open(os.path.join(ROOT, "data", "low_degree_char2.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    def poly(fs):
        c = list(map(int, fs.split()))
        terms = []
        for i in range(len(c) - 1, -1, -1):
            if c[i]:
                terms.append("1" if i == 0 else ("t" if i == 1 else f"t^{{{i}}}"))
        return "$" + "+".join(terms) + "$"

    lines = [r"\begin{tabular}{rrlrrrr}", r"\toprule",
             r"$q$ & $d$ & $f$ & $\Psi(f)$ & $E(f)$ & $\Nirr(f)$ & bound \\", r"\midrule"]
    last = None
    for r in rows:
        if last is not None and r["d"] != last:
            lines.append(r"\addlinespace")
        last = r["d"]
        lines.append(f"{r['q']} & {r['d']} & {poly(r['f_coeffs_low_first'])} & {r['Psi']} & {r['E']} & "
                     f"{r['N_irr']} & {r['certification_bound']} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    with open(os.path.join(ROOT, "paper", "tables", "lowdeg.tex"), "w") as fh:
        fh.write("\n".join(lines))
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
