"""
Reproduce everything for the draft paper/legendre_explicit.tex.

    python code/run_all.py --quick   # tests + tables/figures from the committed CSVs (minutes)
    python code/run_all.py           # all computations from scratch (a few hours, ~10 GB RAM for q=8, d=10)

Large cases write temporary arrays to the directory in the environment variable LEGENDRE_WORK
(default: the system temp directory).
Then:  cd paper && pdflatex legendre_explicit.tex && pdflatex legendre_explicit.tex
"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
TESTS = [["test_fqlib.py"], ["test_hayes_fft.py"], ["test_hayes_exact.py"]]
COMPUTE = [["explicit_scan.py", "--max-classes", "4.1e7"],
           ["explicit_scan.py", "--only", "8,10", "--max-classes", "2e8"],
           ["exact_scan.py"], ["form_factor.py"], ["zero_stats.py"], ["char2_corr.py"]]
MAKE = [["make_paper.py"]]


def main():
    steps = TESTS + (MAKE if "--quick" in sys.argv else COMPUTE + MAKE)
    for s in steps:
        print(f"==> {' '.join(s)}", flush=True)
        subprocess.run([sys.executable, os.path.join(HERE, s[0])] + s[1:], check=True)
    print("all steps finished")


if __name__ == "__main__":
    main()
