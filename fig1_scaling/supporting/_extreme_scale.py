"""Extreme-scale stress test: climb p = 2000, 5000, 10000 at the minimal-sample
operating point (N=2000, T=8000). Report CS + runtime, and catch the cuPC memory
wall cleanly if/when it hits. Extends the scaling curve as far as one GPU allows."""
import os, sys, time, warnings, gc
warnings.filterwarnings('ignore')
_THIS = os.path.dirname(os.path.abspath(__file__))
# _THIS, the parent fig1_scaling/ (scaling_benchmark_lg) and the repo's shared/
for _p in (_THIS, os.path.dirname(_THIS), os.path.join(os.path.dirname(os.path.dirname(_THIS)), 'shared')):
    if _p not in sys.path: sys.path.insert(0, _p)
import numpy as np, torch
from scaling_benchmark_lg import lg_var, directed_cs
from gpu_cits_lag_cupc_faithful import gpu_cits_lag_cupc_faithful

ALPHA, TAU = 0.05, 1
N_TARGET = 2000
_ = gpu_cits_lag_cupc_faithful(lg_var(10, 1000, 0)[0], ALPHA, TAU)  # warm up

print("=== extreme-scale stress test (N=2000, T=8000) ===", flush=True)
for P in [2000, 5000, 10000]:
    nodes = 2*P*(TAU+1)
    T = N_TARGET * (2*(TAU+1))
    print(f"[p={P}] unrolled nodes={nodes}, generating data (T={T})...", flush=True)
    try:
        X, GT = lg_var(P, T, 0)
    except Exception as e:
        print(f"[p={P}] data-gen failed: {str(e)[:80]}", flush=True); break
    t0 = time.perf_counter()
    try:
        B = gpu_cits_lag_cupc_faithful(X, ALPHA, TAU)
        rt = time.perf_counter() - t0
        cs = directed_cs(B, GT)
        print(f"[p={P}] OK  nodes={nodes}  CS={cs:.3f}  runtime={rt:.1f}s ({rt/60:.1f} min)", flush=True)
    except Exception as e:
        rt = time.perf_counter() - t0
        print(f"[p={P}] WALL after {rt:.1f}s -> {str(e)[:120]}", flush=True)
        print(f"[p={P}] this is the single-GPU cuPC memory ceiling; smaller p is the clean demonstrated max.", flush=True)
        break
    finally:
        del X, GT; gc.collect(); torch.cuda.empty_cache()
print("DONE", flush=True)
