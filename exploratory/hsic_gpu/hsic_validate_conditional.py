"""
hsic_validate_conditional.py

Parity gate for the FULL GPU conditional-independence test (gpu_cits_ci.hsic_ci_gpu)
vs kpcalg::kernelCItest(ic.method='hsic.perm'): does residualize+HSIC on GPU
reproduce R's regrXonS+hsic.perm decisions across |S| = 0,1,2,3?
"""
from __future__ import annotations
import os, sys, subprocess
import numpy as np

_THIS = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, _THIS)
OUT_DIR = '/home/rbiswas1/microns/arousal_paper_overleaf/figures/2026-07-06_hsic_gpu_validation'
os.makedirs(OUT_DIR, exist_ok=True)
CSV = os.path.join(OUT_DIR, 'cond_cases_data.csv')
RREF = os.path.join(OUT_DIR, 'cond_r_reference.csv')
N, SIG, P, ALPHA = 1000, 1.0, 100, 0.05


def make_data(seed=0):
    rng = np.random.default_rng(seed)
    e = lambda: rng.uniform(0, 1, N)
    z1 = e()
    z2 = 4 * z1 + e()
    z3 = 3 * np.sin(z1) + e()
    z4 = 8 * np.log(np.abs(z2) + 1e-3) + 9 * np.log(np.abs(z3) + 1e-3) + e()
    z5 = e()                                   # independent of the rest
    z6 = 2 * np.sqrt(np.abs(z4)) + e()
    return np.column_stack([z1, z2, z3, z4, z5, z6])  # 0-indexed cols


# (i, j, S) triples, 0-indexed, spanning |S| = 0,1,2,3
TRIPLES = [
    (0, 1, []),            # dep
    (0, 4, []),            # indep
    (1, 2, []),            # dep (common cause z1)
    (1, 2, [0]),           # given z1 -> indep
    (0, 3, []),            # dep
    (0, 3, [1, 2]),        # given z2,z3 -> indep
    (2, 3, [0, 1]),        # |S|=2
    (1, 3, [0, 2]),        # |S|=2
    (0, 3, [1, 2, 4]),     # |S|=3 additive
    (1, 5, [3]),           # z6 via z4
    (0, 5, [3]),           # |S|=1
    (2, 5, [0, 3]),        # |S|=2
]

R_SCRIPT = r'''
suppressMessages(library(kpcalg))
a <- commandArgs(trailingOnly=TRUE)
csv<-a[1]; out<-a[2]; trip<-a[3]; sig<-as.numeric(a[4]); p<-as.integer(a[5])
d <- as.matrix(read.csv(csv)); n <- nrow(d); numCol <- n %/% 10
ss <- list(data=d, ic.method="hsic.perm", sig=sig, p=p, numCol=numCol)
set.seed(999)
lines <- strsplit(trip, ";")[[1]]
res <- data.frame(triple=character(), pval=numeric(), stringsAsFactors=FALSE)
for (ln in lines) {
  parts <- strsplit(ln, "\\|")[[1]]
  i <- as.integer(parts[1]); j <- as.integer(parts[2])
  S <- if (length(parts) >= 3 && nchar(parts[3])>0)
         as.integer(strsplit(parts[3], ",")[[1]]) else integer(0)
  pv <- kernelCItest(x=i, y=j, S=S, suffStat=ss)   # 1-indexed cols
  res <- rbind(res, data.frame(triple=ln, pval=pv))
}
write.csv(res, out, row.names=FALSE)
cat("R conditional reference written\n")
'''


def run_r(triples_1idx):
    rpath = os.path.join(OUT_DIR, '_cond_ref.R')
    open(rpath, 'w').write(R_SCRIPT)
    enc = ';'.join(
        f"{i}|{j}|{','.join(map(str,S)) if S else ''}" for (i, j, S) in triples_1idx)
    subprocess.run(['Rscript', rpath, CSV, RREF, enc, str(SIG), str(P)], check=True)


def main():
    import torch
    from gpu_cits_ci import hsic_ci_gpu
    assert torch.cuda.is_available()

    data = make_data()
    np.savetxt(CSV, data, delimiter=',',
               header=','.join(f'z{i+1}' for i in range(data.shape[1])), comments='')

    # R uses 1-indexed columns
    triples_1idx = [(i + 1, j + 1, [s + 1 for s in S]) for (i, j, S) in TRIPLES]
    print('running R conditional reference ...', flush=True)
    run_r(triples_1idx)

    rref = {}
    with open(RREF) as f:
        next(f)
        for line in f:
            key, pv = line.rsplit(',', 1)
            rref[key.strip().strip('"')] = float(pv)

    gen = torch.Generator(device='cuda'); gen.manual_seed(999)
    print(f"\n{'i':>2} {'j':>2} {'S':<10} {'R_pval':>7} {'GPU_pval':>8} "
          f"{'R_dec':>6} {'GPU_dec':>7} {'match':>8}")
    print('-' * 60)
    n_match = 0
    for (i, j, S) in TRIPLES:
        key = f"{i+1}|{j+1}|{','.join(str(s+1) for s in S) if S else ''}"
        rp = rref[key]
        gp = hsic_ci_gpu(data, i, j, S, sig=SIG, p=P, generator=gen)
        rdec = 'dep' if rp <= ALPHA else 'indep'
        gdec = 'dep' if gp <= ALPHA else 'indep'
        ok = rdec == gdec; n_match += ok
        print(f"{i:>2} {j:>2} {str(S):<10} {rp:7.3f} {gp:8.3f} "
              f"{rdec:>6} {gdec:>7} {'OK' if ok else 'MISMATCH':>8}")
    print('-' * 60)
    print(f"decision agreement: {n_match}/{len(TRIPLES)}")


if __name__ == '__main__':
    main()
