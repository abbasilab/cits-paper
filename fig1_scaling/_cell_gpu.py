import os, sys, time
p, N, gpu, seed = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3], int(sys.argv[4])
os.environ['CUDA_VISIBLE_DEVICES'] = gpu
sys.path.insert(0,'.'); sys.path.insert(0,'/home/rbiswas1/repos/cits')
import numpy as np
from scaling_benchmark_lg import lg_var, directed_cs
from gpu_cits_lag_cupc_faithful import gpu_cits_lag_cupc_faithful
_ = gpu_cits_lag_cupc_faithful(np.random.default_rng(0).standard_normal((10,1000)), 0.05, 1)
X, GT = lg_var(p, N*4, seed)
t0 = time.perf_counter()
B = gpu_cits_lag_cupc_faithful(X, 0.05, 1)
print(f"RESULT {time.perf_counter()-t0:.1f} {directed_cs(B, GT):.3f}", flush=True)
