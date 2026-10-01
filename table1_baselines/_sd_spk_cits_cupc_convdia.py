import numpy as np, sys, os, warnings, csv; warnings.filterwarnings('ignore')
os.environ.setdefault('CUPC_DIR','/home/rbiswas1/repos/cupc')
sys.path.insert(0,'.'); sys.path.insert(0,'/home/rbiswas1/repos/cits')
from glm_spiking_sim import simulate_glm_spiking, directed_cs
from gpu_cits_lag_cupc import gpu_cits_lag_cupc
import argparse; ap=argparse.ArgumentParser(); ap.add_argument('--seeds',type=int,default=50); a=ap.parse_args()
w=csv.writer(open('_sd_spk_cits_cupc_convdia.csv','w',newline='')); w.writerow(['cell','seed','cs'])
for m in ['convergence','diamond']:
    vals=[]
    for s in range(a.seeds):
        X,GT,_=simulate_glm_spiking(s,motif=m)
        B=gpu_cits_lag_cupc(X,alpha=0.05,tau=1)
        p=(np.asarray(B)!=0).astype(int); np.fill_diagonal(p,0); c=directed_cs(p,GT)[0]; vals.append(c); w.writerow([f'CITScupc_{m}',s,c])
    vals=np.array(vals); print(f"{m}: CITS cuPC (nbr-restricted, pcorr) mean={vals.mean():.3f} sd={vals.std():.3f} n={len(vals)}",flush=True)
