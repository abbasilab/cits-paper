"""
hsic_validate_gpu_vs_kpcalg.py

Parity gate: does the GPU full-kernel HSIC (gpu_hsic.hsic_perm_gpu) reproduce
kpcalg::hsic.perm on identical data?

Generates several (x,y) cases (independent / linear / nonlinear), writes them
to CSV, calls Rscript to run kpcalg::hsic.perm, runs the GPU version on the
same data, and prints a side-by-side comparison of the HSIC statistic and the
p-value / decision at alpha=0.05.
"""
from __future__ import annotations
import os, sys, subprocess, json
import numpy as np

_THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _THIS)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(_THIS)), 'shared')); from paths import outdir as _outdir
OUT_DIR = _outdir('exploratory/hsic_gpu')   # was arousal_paper_overleaf/figures/2026-07-06_hsic_gpu_validation
os.makedirs(OUT_DIR, exist_ok=True)
CSV = os.path.join(OUT_DIR, 'hsic_validation_cases.csv')
RREF = os.path.join(OUT_DIR, 'hsic_validation_r_reference.csv')

N = 1000
SIG = 1.0
P = 100
NUMCOL = N // 10  # kpcalg default floor(n/10)
ALPHA = 0.05


def make_cases(seed: int = 0):
    rng = np.random.default_rng(seed)
    cases = {}
    x = rng.standard_normal(N)
    cases['indep']      = (x.copy(), rng.standard_normal(N))                    # X ⟂ Y
    x2 = rng.standard_normal(N)
    cases['linear']     = (x2.copy(), 2 * x2 + 0.5 * rng.standard_normal(N))    # strong linear
    x3 = rng.standard_normal(N)
    cases['nonlinear']  = (x3.copy(), np.sin(3 * x3) + 0.3 * rng.standard_normal(N))
    x4 = rng.uniform(0, 1, N)
    cases['weak_nonlin']= (x4.copy(), np.log(np.abs(x4) + 1e-3) + 0.8 * rng.standard_normal(N))
    return cases


def write_csv(cases):
    cols, names = [], []
    for name, (x, y) in cases.items():
        cols.append(x); names.append(f'{name}_x')
        cols.append(y); names.append(f'{name}_y')
    arr = np.column_stack(cols)
    header = ','.join(names)
    np.savetxt(CSV, arr, delimiter=',', header=header, comments='')
    return names


R_SCRIPT = r'''
suppressMessages(library(kpcalg))
args <- commandArgs(trailingOnly=TRUE)
csv <- args[1]; out <- args[2]; sig <- as.numeric(args[3])
p <- as.integer(args[4]); numCol <- as.integer(args[5])
d <- read.csv(csv, check.names=FALSE)
nm <- colnames(d)
bases <- unique(sub("_(x|y)$", "", nm))
set.seed(12345)
res <- data.frame(case=character(), hsic=numeric(), pval=numeric(),
                  stringsAsFactors=FALSE)
for (b in bases) {
  x <- d[[paste0(b, "_x")]]; y <- d[[paste0(b, "_y")]]
  h <- hsic.perm(x=x, y=y, sig=sig, p=p, numCol=numCol)
  res <- rbind(res, data.frame(case=b, hsic=as.numeric(h$statistic),
                               pval=as.numeric(h$p.value)))
}
write.csv(res, out, row.names=FALSE)
cat("R reference written\n")
'''


def run_r():
    rpath = os.path.join(OUT_DIR, '_hsic_ref.R')
    with open(rpath, 'w') as f:
        f.write(R_SCRIPT)
    cmd = ['Rscript', rpath, CSV, RREF, str(SIG), str(P), str(NUMCOL)]
    print('running R reference ...', flush=True)
    subprocess.run(cmd, check=True)


def main():
    import torch
    from gpu_hsic import hsic_perm_gpu
    assert torch.cuda.is_available(), "CUDA required"

    cases = make_cases()
    write_csv(cases)
    run_r()

    # Load R reference
    rref = {}
    with open(RREF) as f:
        next(f)
        for line in f:
            c, h, pv = line.strip().split(',')
            rref[c.strip('"')] = (float(h), float(pv))

    gen = torch.Generator(device='cuda'); gen.manual_seed(12345)
    print(f"\n{'case':<12} {'R_hsic':>12} {'GPU_hsic':>12} {'relerr':>9} "
          f"| {'R_pval':>7} {'GPU_pval':>8} | {'R_dec':>6} {'GPU_dec':>7} {'match':>6}")
    print('-' * 92)
    rows = []
    all_match = True
    for name, (x, y) in cases.items():
        g = hsic_perm_gpu(x, y, sig=SIG, p=P, generator=gen)
        rh, rp = rref[name]
        relerr = abs(g['hsic'] - rh) / (abs(rh) + 1e-12)
        rdec = 'dep' if rp <= ALPHA else 'indep'
        gdec = 'dep' if g['p_value'] <= ALPHA else 'indep'
        match = 'OK' if rdec == gdec else 'MISMATCH'
        all_match = all_match and (rdec == gdec)
        print(f"{name:<12} {rh:12.6e} {g['hsic']:12.6e} {relerr:9.2%} "
              f"| {rp:7.3f} {g['p_value']:8.3f} | {rdec:>6} {gdec:>7} {match:>6}")
        rows.append({'case': name, 'r_hsic': rh, 'gpu_hsic': g['hsic'],
                     'rel_err': relerr, 'r_pval': rp, 'gpu_pval': g['p_value'],
                     'r_decision': rdec, 'gpu_decision': gdec})

    import csv as _csv
    outcsv = os.path.join(OUT_DIR, 'hsic_validation_comparison.csv')
    with open(outcsv, 'w', newline='') as f:
        w = _csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print('-' * 92)
    print(f"decision agreement: {'ALL MATCH' if all_match else 'SOME MISMATCH'}")
    print(f"saved -> {outcsv}")


if __name__ == '__main__':
    main()
