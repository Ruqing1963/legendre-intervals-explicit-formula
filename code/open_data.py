"""
Load the open-range results: data/open_regime_summary.csv plus any extra runs
data/open_regime_summary_<q>_<d>.csv produced with `open_regime.py --only q,d --full`.
A full run replaces a sampled row for the same (q, d).
"""
import glob
import os

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")


def _merge(pattern_main, pattern_extra, key_cols):
    main = os.path.join(DATA, pattern_main)
    if not os.path.exists(main):
        return None
    frames = [pd.read_csv(main)]
    for path in sorted(glob.glob(os.path.join(DATA, pattern_extra))):
        frames.append(pd.read_csv(path))
    return frames


def load_summary():
    frames = _merge("open_regime_summary.csv", "open_regime_summary_*.csv", ["q", "d"])
    if frames is None:
        return None
    s = pd.concat(frames, ignore_index=True)
    s["_full"] = (s["mode"] == "all").astype(int)
    s = s.sort_values(["q", "d", "_full"]).drop_duplicates(["q", "d"], keep="last").drop(columns="_full")
    return s.reset_index(drop=True)


def load_orbits():
    frames = _merge("open_regime_orbits.csv", "open_regime_orbits_*.csv", ["q", "d"])
    if frames is None:
        return None
    s = load_summary()
    o = pd.concat(frames, ignore_index=True).drop_duplicates(["q", "d", "f_coeffs_low_first"], keep="last")
    # keep only orbits belonging to the retained (full or sampled) run for each (q, d)
    return o.reset_index(drop=True)
