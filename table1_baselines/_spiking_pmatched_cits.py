"""Matched +/- recurrence spiking benchmark, CITS (RCIT wrapper, emits self-lags),
full-graph CS (cross + self). Runs on GPU (device=cuda) or CPU. SH env toggles
recurrence as in _spiking_pmatched_cpu.py."""
import numpy as np, sys, os, warnings; warnings.filterwarnings('ignore')
sys.path.insert(0,'.'); sys.path.insert(0,'/home/rbiswas1/repos/cits')
from gpu_cits_lag_rcit import gpu_cits_lag_rcit
MOTIFS={'convergence':[(0,2),(1,2),(2,3)],'diamond':[(0,1),(0,2),(1,3),(2,3)],
        'chain':[(0,1),(1,2),(2,3)],'depression':[(0,2),(1,2),(2,3)]}
SH=float(os.environ.get('SH','-1.5')); NS=int(os.environ.get('NS','50')); T=2000; p=4
DEV=os.environ.get('CITS_DEV','cuda')
MODE='rec' if SH!=0 else 'ff'
ORDER=['convergence','diamond','chain']+(['depression'] if MODE=='rec' else [])
def sim(seed,motif):
    rng=np.random.default_rng(seed); W=np.zeros((p,p))
    for a,b in MOTIFS[motif]: W[a,b]=1.4
    b_i=np.log(0.25); k1,k2=1.0,0.4; s=np.zeros((p,T)); R=np.ones(p); U,tr,dep=0.7,15.0,2.2
    for t in range(1,T):
        p1=s[:,t-1]; p2=s[:,t-2] if t>=2 else np.zeros(p)
        if motif=='depression':
            drive=dep*(W.T@(U*R*p1)); R=R+(1-R)/tr; R=np.clip(R-U*R*p1,0,1)
        else: drive=W.T@(k1*p1+k2*p2)
        eta=b_i+drive+SH*p1; s[:,t]=rng.poisson(np.clip(np.exp(eta),0,4.0))
    GTs=np.eye(p,dtype=int) if SH!=0 else np.zeros((p,p),dtype=int)
    return s.astype(float),(W!=0).astype(int),GTs
def cs(pred,GT):
    TP=FP=0; nt=int(GT.sum()); nn=p*p-nt
    for i in range(p):
        for j in range(p):
            if GT[i,j]: TP+=int(pred[i,j]>0)
            else: FP+=int(pred[i,j]>0)
    return (TP/nt if nt else 0)-(FP/nn if nn else 0)
print(f"MODE={MODE} SH={SH} NS={NS} DEV={DEV}",flush=True)
for motif in ORDER:
    full=[]; self_=[]
    for s in range(NS):
        X,GTc,GTs=sim(s,motif); GTf=((GTc+GTs)>0).astype(int)
        B=(np.asarray(gpu_cits_lag_rcit(X,alpha=0.05,tau=1,K=25,n_perm=100,max_cond_size=5,seed=s,device=DEV,null='gamma'))!=0).astype(int)
        full.append(cs(B,GTf)); self_.append(np.mean([B[i,i]>0 for i in range(p)]))
    print(motif,f"CSfull={np.mean(full):.3f}±{np.std(full):.3f} selfFrac={np.mean(self_):.2f}",flush=True)
print(f"CITS PMATCHED {MODE} DONE",flush=True)
