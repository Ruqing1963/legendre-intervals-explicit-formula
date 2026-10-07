"""
Tables, macros and figures for the draft paper/legendre_explicit.tex.

Reads: data/exact_scan*.csv, data/explicit_scan*.csv, data/explicit_psi_hist*.csv,
       data/form_factor.csv, data/zero_stats_*.csv, data/char2_excess.csv
Writes: paper/tables/*.tex, paper/figures/*.pdf|png, data/char2_variance.csv
"""
import glob
import math
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hayes_fft_fq as H  # noqa: E402
from explicit_scan import legendre_mask  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
P2 = os.path.join(ROOT, "paper")
TAB = os.path.join(P2, "tables")
FIG = os.path.join(P2, "figures")

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Arial"], "font.size": 9,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "axes.titlecolor": INK, "axes.titlesize": 10,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "legend.frameon": False, "legend.labelcolor": INK2, "savefig.dpi": 200, "savefig.bbox": "tight",
})
QCOL = {2: SERIES[0], 3: SERIES[1], 4: SERIES[2], 5: SERIES[3], 7: SERIES[4], 8: SERIES[5], 9: SERIES[6]}


def load(pattern):
    files = sorted(glob.glob(os.path.join(DATA, pattern)))
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True).drop_duplicates(["q", "d"], keep="last")


def save(fig, name):
    os.makedirs(FIG, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIG, f"{name}.{ext}"))
    plt.close(fig)


def prime_powers(upto):
    out = []
    for n in range(2, upto + 1):
        p = next(k for k in range(2, n + 1) if n % k == 0)
        m = n
        while m % p == 0:
            m //= p
        if m == 1:
            out.append(n)
    return out


def main():
    os.makedirs(TAB, exist_ok=True)
    ex = load("exact_scan*.csv").sort_values(["q", "d"])
    fl = load("explicit_scan_*.csv") if glob.glob(os.path.join(DATA, "explicit_scan_*.csv")) else None
    fl = pd.concat([pd.read_csv(os.path.join(DATA, "explicit_scan.csv"))] + ([fl] if fl is not None else []),
                   ignore_index=True).drop_duplicates(["q", "d"], keep="last").sort_values(["q", "d"])
    # ---- verification macros
    ok = {(int(r.q), int(r.d)) for _, r in ex.iterrows()}
    maxd = {q: max([d for (qq, d) in ok if qq == q] or [0]) for q in (2, 3, 4, 5, 7, 8, 9)}
    D = 2
    while True:
        dn = D + 1
        if all((q, dn) in ok for q in prime_powers(dn - 2) if q <= dn - 2):
            D = dn
        else:
            break
    unc = ex[ex.uncertified > 0]
    unc_txt = ", ".join(f"$({int(r.q)},{int(r.d)})$" for _, r in unc.iterrows()) or "none"
    unc_txt = ("the cases " + unc_txt) if len(unc) else "none"
    odd = fl[(fl.q % 2 == 1) & (fl.d >= 4)]
    last = odd.loc[odd.groupby("q").d.idxmax()]
    vr = last["var_Psi_over_q^(d+1)"] / (last.d - 2)
    ff = pd.read_csv(os.path.join(DATA, "form_factor.csv"))
    dev = max(abs(r.F / (r.l - 1) - 1) for _, r in ff.iterrows() if r.n > r.l)
    hist = load("explicit_psi_hist*.csv") if False else pd.concat(
        [pd.read_csv(f) for f in glob.glob(os.path.join(DATA, "explicit_psi_hist*.csv"))], ignore_index=True)
    hq, hd = 3, int(fl[fl.q == 3].d.max())
    hint = int(fl[(fl.q == hq) & (fl.d == hd)].legendre_intervals.iloc[0])
    c2 = pd.read_csv(os.path.join(DATA, "char2_excess.csv"))
    macros = {
        "AllQd": D, "AllQdNext": D + 1,
        "MaxDtwo": maxd[2], "MaxDthree": maxd[3], "MaxDfour": maxd[4], "MaxDfive": maxd[5],
        "MaxDseven": maxd[7], "MaxDeight": maxd[8], "MaxDnine": maxd[9],
        "Uncertified": unc_txt,
        "VarRatioOdd": f"between {vr.min():.2f} and {vr.max():.2f}",
        "FormFactorDev": f"{100 * dev:.0f}\\%",
        "HistCase": f"({hq},{hd})", "HistIntervals": f"{hint:,}".replace(",", "{,}"),
        "MaxExcess": f"{c2.excess_factor.max():.0f}",
    }
    with open(os.path.join(TAB, "macros.tex"), "w") as fh:
        for k, v in macros.items():
            fh.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
    # ---- verification table
    lines = [r"\begin{tabular}{rrrrrrr}", r"\toprule",
             r"$q$ & $d$ & intervals & $\min\Psi/q^{d+1}$ & certified & open range & time (s) \\", r"\midrule"]
    show = []
    for q in sorted(ex.q.unique()):
        sub = ex[ex.q == q]
        dmax = sub.d.max()
        show += [r for _, r in sub.iterrows() if r.d >= dmax - 2]
    last_q = None
    for r in show:
        if last_q is not None and r.q != last_q:
            lines.append(r"\addlinespace")
        last_q = r.q
        lines.append(f"{int(r.q)} & {int(r.d)} & {int(r.legendre_intervals):,} & {r.min_Psi_ratio:.4f} & "
                     f"{int(r.certified):,} & {'yes' if r.open_range else 'no'} & {r.seconds:.1f} \\\\".replace(",", "{,}"))
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(TAB, "verification.tex"), "w").write("\n".join(lines))
    # ---- zeros table
    zs = pd.read_csv(os.path.join(DATA, "zero_stats_summary.csv"))
    lines = [r"\begin{tabular}{rrrrrrr}", r"\toprule",
             r"$q$ & $d$ & characters & generic & exact & conductor counts & max dev.\ of $|\omega|/\sqrt q$ \\",
             r"\midrule"]
    max_root_dev = 0.0
    for _, r in zs.sort_values(["q", "d"]).iterrows():
        l = int(r.d) - 1
        exact = (int(r.q) ** l - int(r.q) ** (l - 1)) / (int(r.q) ** l - 1)
        cc_ok = "as predicted" if r.conductor_counts == r.conductor_counts_expected else "MISMATCH"
        max_root_dev = max(max_root_dev, float(r.max_rel_dev_from_sqrt_q))
        lines.append(f"{int(r.q)} & {int(r.d)} & {int(r.characters):,} & {r.generic_fraction:.4f} & {exact:.4f} & "
                     f"{cc_ok} & {float(r.max_rel_dev_from_sqrt_q):.1e} \\\\".replace(",", "{,}"))
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(TAB, "zeros.tex"), "w").write("\n".join(lines))
    with open(os.path.join(TAB, "macros.tex"), "a") as fh:
        fh.write(f"\\newcommand{{\\MaxRootDev}}{{${max_root_dev:.1e}$}}\n".replace("e-0", "\\times10^{-").replace("$}}", "}$}}") if False else
                 f"\\newcommand{{\\MaxRootDev}}{{$%s\\times10^{{%d}}$}}\n" % (f"{max_root_dev / 10 ** math.floor(math.log10(max_root_dev)):.1f}", math.floor(math.log10(max_root_dev))))
    # ---- char 2: full-group vs square-subgroup variance
    rows = []
    for _, r in c2.iterrows():
        q, d = int(r.q), int(r.d)
        if d < 4:
            continue
        Psi, ctx = H.all_interval_psi(q, d)
        Psi = np.rint(Psi)
        mask = legendre_mask(ctx)
        main_ = float(q) ** (d + 1)
        vG = float(np.mean((Psi - main_) ** 2) / main_)
        vH = float(np.mean((Psi[mask] - main_) ** 2) / main_)
        rows.append({"q": q, "d": d, "mean_H": float(np.mean(Psi[mask]) / main_), "var_G": vG, "var_H": vH,
                     "ratio": vH / vG, "excess": float(r.excess_factor), "rho2": float(r.rho_order2_axes),
                     "rhoo": float(r.rho_other), "KR": d - 2})
    cv = pd.DataFrame(rows)
    cv.to_csv(os.path.join(DATA, "char2_variance.csv"), index=False)
    lines = [r"\begin{tabular}{rrrrrrrrr}", r"\toprule",
             r"$q$ & $d$ & mean & Var (all) & Var (Legendre) & ratio & excess & $\Sigma\rho$, order 2 & $\Sigma\rho$, other \\",
             r"\midrule"]
    last_q = None
    for _, r in cv.iterrows():
        if last_q is not None and r.q != last_q:
            lines.append(r"\addlinespace")
        last_q = r.q
        lines.append(f"{int(r.q)} & {int(r.d)} & {r.mean_H:.4f} & {r.var_G:.2f} & {r.var_H:.2f} & {r.ratio:.2f} & "
                     f"{r.excess:.2f} & {r.rho2:+.2f} & {r.rhoo:+.2f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(TAB, "char2.tex"), "w").write("\n".join(lines))
    # ---- figure: variance
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(8.8, 3.5))
    for q in sorted(fl.q.unique()):
        sub = fl[(fl.q == q) & (fl.d >= 4)]
        ax = a1 if q % 2 == 1 else a2
        ax.plot(sub.d, sub["var_Psi_over_q^(d+1)"], "-o", color=QCOL[q], lw=2, ms=4, label=f"q = {q}")
    for ax in (a1, a2):
        dd = np.arange(4, int(fl.d.max()) + 1)
        ax.plot(dd, dd - 2, "--", color=INK2, lw=1.3, label="d - 2 (Keating-Rudnick)")
        ax.set_xlabel("d")
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
        ax.legend(fontsize=7.5, loc="upper left")
    a1.set_ylabel(r"Var$\,\Psi\,/\,q^{d+1}$")
    a1.set_title("odd q", loc="left")
    a2.set_title("even q (Legendre intervals = squares)", loc="left")
    a2.set_yscale("log")
    save(fig, "p2_variance")
    # ---- figure: minima and histogram
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(8.8, 3.5))
    for q in sorted(ex.q.unique()):
        sub = ex[(ex.q == q) & (ex.d >= 3)]
        a1.plot(sub.d, sub.min_Psi_ratio, "-o", color=QCOL[q], lw=1.8, ms=3.5, label=f"q = {q}")
    a1.axhline(1, color=AXIS, lw=1)
    a1.set_ylim(0.3, 1.03)
    a1.set_xlabel("d")
    a1.set_ylabel(r"$\min_f\Psi(f)/q^{d+1}$")
    a1.legend(fontsize=7.5, ncol=2, loc="lower right")
    h = hist[(hist.q == hq) & (hist.d == hd)].sort_values("z_lo")
    width = 12 / 60
    dens = h["count"] / h["count"].sum() / width
    a2.bar(h.z_lo + width / 2, dens, width=width * 0.9, color=SERIES[0])
    xs = np.linspace(-5, 5, 300)
    a2.plot(xs, np.exp(-xs ** 2 / 2) / np.sqrt(2 * np.pi), color=INK2, lw=1.5, label="N(0, 1)")
    a2.set_xlim(-5, 5)
    a2.set_xlabel("Z")
    a2.set_title(f"q = {hq}, d = {hd}", loc="left")
    a2.legend(fontsize=8)
    save(fig, "p2_minratio_hist")
    # ---- figure: zeros
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(8.8, 3.5))
    for (q, d), g in ff.groupby(["q", "d"]):
        l = d - 1
        a1.plot(g.n / l, g.F / (l - 1), "-", color=QCOL[q], lw=1.4, label=f"q={q}, d={d}")
        diag = g[g.diagonal_exact.notna() & (g.diagonal_exact.astype(str) != "")]
        a1.plot(diag.n / l, diag.diagonal_exact.astype(float) / (l - 1), "x", color=QCOL[q], ms=4)
    xs = np.linspace(0, 3, 300)
    a1.plot(xs, np.minimum(xs, 1), "--", color=INK2, lw=1.3, label="CUE")
    a1.set_xlabel(r"$n/\ell$")
    a1.set_ylabel(r"$F(n)/(\ell-1)$")
    a1.legend(fontsize=7, ncol=2, loc="lower right")
    sp = pd.read_csv(os.path.join(DATA, "zero_stats_spacing.csv"))
    for (q, d), g in sp.groupby(["q", "d"]):
        a2.plot(g.s_lo, g.density_data, "-", color=QCOL[q], lw=1.4, label=f"q={q}, d={d}")
    g0 = sp[(sp.q == sp.q.iloc[0]) & (sp.d == sp.d.iloc[0])]
    a2.plot(g0.s_lo, g0.density_CUE, "--", color=INK2, lw=1.5, label="CUE (simulated)")
    a2.set_xlabel("normalized spacing")
    a2.set_ylabel("density")
    a2.legend(fontsize=7, loc="upper right")
    save(fig, "p2_zeros")
    print(macros)
    print(cv.to_string(index=False))


if __name__ == "__main__":
    main()
