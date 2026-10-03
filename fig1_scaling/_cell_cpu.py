import os, sys, time
for v in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[v] = '32'
sys.path.insert(0,'.'); sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'shared')))
import numpy as np
from scaling_benchmark_lg import lg_var, directed_cs
method, p, N, seed = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
X, GT = lg_var(p, N*4, seed)
from simulation_benchmark_fc_methods_v3 import run_tpc
from pcmci_plus_baseline import run_pcmci_plus
from lpcmci_baseline import run_lpcmci
from kernel_granger_baseline import run_kernel_granger
t0 = time.perf_counter()
if   method=='PCMCI+':   A = run_pcmci_plus(X, tau_max=1, pc_alpha=0.05)
elif method=='LPCMCI':   A = run_lpcmci(X, tau_max=1, pc_alpha=0.05)
elif method=='TPC':      A = run_tpc(X, alpha=0.05)
elif method=='KernelGC': A = run_kernel_granger(X, max_lag=1, alpha=0.05)
print(f"RESULT {time.perf_counter()-t0:.1f} {directed_cs(A, GT):.3f}", flush=True)
