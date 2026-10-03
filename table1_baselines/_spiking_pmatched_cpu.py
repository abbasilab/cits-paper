"""Matched +/- recurrence spiking benchmark, CPU methods (TPC self-lag, GC1, GC2,
Kernel-GC). Full-graph CS (cross + self). SH env toggles recurrence:
  SH=0.0  -> feedforward control (no self; GTself=0)
  SH=-1.5 -> recurrent realistic (refractory self-history; GTself=eye)
Motifs convergence/diamond/chain are matched across both modes; depression is
recurrent-only (Tsodyks-Markram self-history) and is skipped when SH=0."""
import numpy as np, sys, os, warnings; warnings.filterwarnings('ignore')
sys.path.insert(0,'.'); sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'shared')))
from simulation_benchmark_fc_methods_v3 import pc_skeleton_cpu, orient_v_structures
from kernel_granger_baseline import run_kernel_granger
from scipy import stats
MOTIFS={'convergence':[(0,2),(1,2),(2,3)],'diamond':[(0,1),(0,2),(1,3),(2,3)],
        'chain':[(0,1),(1,2),(2,3)],'depression':[(0,2),(1,2),(2,3)]}
SH=float(os.environ.get('SH','-1.5')); NS=int(os.environ.get('NS','50')); T=2000; p=4
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
    GTc=(W!=0).astype(int)
    GTs=np.eye(p,dtype=int) if SH!=0 else np.zeros((p,p),dtype=int)
    return s.astype(float),GTc,GTs
def tpc_self(X,alpha=0.05):
    chi=np.vstack([X[:,:-1],X[:,1:]]); A,ss=pc_skeleton_cpu(chi,alpha=alpha); G=orient_v_structures(A,ss)
    A_lag=G[:p,p:]; Gc=G[p:,p:]; out=np.zeros((p,p),int)
    for i in range(p):
        for j in range(p):
            if i==j: out[i,i]=int(A_lag[i,i]!=0)
            elif A_lag[i,j] or Gc[i,j]: out[i,j]=1
    return out
def granger(X,cond):
    def ols(Y,Z): Z=np.column_stack([np.ones(len(Z)),Z]); b=np.linalg.lstsq(Z,Y,rcond=None)[0]; r=Y-Z@b; return float(r@r),Z.shape[1]
    A=np.zeros((p,p),int); Xp=X[:,:-1].T; Xt=X[:,1:].T; n=Xt.shape[0]
    for j in range(p):
        y=Xt[:,j]
        for i in range(p):
            if i==j: continue
            if cond: rf,kf=ols(y,Xp);rr,kr=ols(y,Xp[:,[k for k in range(p) if k!=i]])
            else: rf,kf=ols(y,Xp[:,[j,i]]);rr,kr=ols(y,Xp[:,[j]])
            d1=kf-kr;d2=n-kf
            if d1>0 and d2>0 and rf>0 and 1-stats.f.cdf(((rr-rf)/d1)/(rf/d2),d1,d2)<0.05: A[i,j]=1
    return A
def cs(pred,GT):
    TP=FP=0; nt=int(GT.sum()); nn=p*p-nt
    for i in range(p):
        for j in range(p):
            if GT[i,j]: TP+=int(pred[i,j]>0)
            else: FP+=int(pred[i,j]>0)
    return (TP/nt if nt else 0)-(FP/nn if nn else 0)
print(f"MODE={MODE} SH={SH} NS={NS}",flush=True)
for motif in ORDER:
    acc={m:{'full':[],'self':[]} for m in ['TPC','GC1','GC2','KernelGC']}
    for s in range(NS):
        X,GTc,GTs=sim(s,motif); GTf=((GTc+GTs)>0).astype(int)
        preds={'TPC':tpc_self(X),'GC1':granger(X,False),'GC2':granger(X,True),
               'KernelGC':(run_kernel_granger(X,max_lag=2,alpha=0.05)!=0).astype(int)}
        for m,P in preds.items():
            acc[m]['full'].append(cs(P,GTf)); acc[m]['self'].append(np.mean([P[i,i]>0 for i in range(p)]))
    print(motif,{m:f"CSfull={np.mean(d['full']):.3f}±{np.std(d['full']):.3f} selfFrac={np.mean(d['self']):.2f}" for m,d in acc.items()},flush=True)
print(f"CPU PMATCHED {MODE} DONE",flush=True)
