"""Matched +/- recurrence spiking benchmark, PCMCI+/LPCMCI (keep_self=True),
full-graph CS (cross + self). tigra env. SH env toggles recurrence as in
_spiking_pmatched_cpu.py."""
import numpy as np, sys, os, warnings; warnings.filterwarnings('ignore')
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'shared')))  # repo shared/ (_glm_suite_baselines)
from _glm_suite_baselines import _graph_to_adj, TAU_MAX, ALPHA
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
    GTs=np.eye(p,dtype=int) if SH!=0 else np.zeros((p,p),dtype=int)
    return s.astype(float),(W!=0).astype(int),GTs
def pcmciplus(X):
    from tigramite import data_processing as pp
    from tigramite.pcmci import PCMCI
    from tigramite.independence_tests.parcorr import ParCorr
    g=PCMCI(dataframe=pp.DataFrame(X.T.astype(float)),cond_ind_test=ParCorr(),verbosity=0).run_pcmciplus(tau_max=TAU_MAX,pc_alpha=ALPHA)['graph']
    return _graph_to_adj(g,p,keep_self=True)
def lpcmci(X):
    from tigramite import data_processing as pp
    from tigramite.lpcmci import LPCMCI
    from tigramite.independence_tests.parcorr import ParCorr
    g=LPCMCI(dataframe=pp.DataFrame(X.T.astype(float)),cond_ind_test=ParCorr(),verbosity=0).run_lpcmci(tau_max=TAU_MAX,pc_alpha=ALPHA)['graph']
    return _graph_to_adj(g,p,keep_self=True)
def cs(pred,GT):
    TP=FP=0; nt=int(GT.sum()); nn=p*p-nt
    for i in range(p):
        for j in range(p):
            if GT[i,j]: TP+=int(pred[i,j]>0)
            else: FP+=int(pred[i,j]>0)
    return (TP/nt if nt else 0)-(FP/nn if nn else 0)
print(f"MODE={MODE} SH={SH} NS={NS}",flush=True)
for motif in ORDER:
    acc={m:{'full':[],'self':[]} for m in ['PCMCIplus','LPCMCI']}
    for s in range(NS):
        X,GTc,GTs=sim(s,motif); GTf=((GTc+GTs)>0).astype(int)
        for m,fn in [('PCMCIplus',pcmciplus),('LPCMCI',lpcmci)]:
            try: P=(np.asarray(fn(X))!=0).astype(int)
            except Exception: P=np.zeros((p,p),int)
            acc[m]['full'].append(cs(P,GTf)); acc[m]['self'].append(np.mean([P[i,i]>0 for i in range(p)]))
    print(motif,{m:f"CSfull={np.mean(d['full']):.3f}±{np.std(d['full']):.3f} selfFrac={np.mean(d['self']):.2f}" for m,d in acc.items()},flush=True)
print(f"PCMCI PMATCHED {MODE} DONE",flush=True)
