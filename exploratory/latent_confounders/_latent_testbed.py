"""RIGOROUS latent-confounder testbed (NOT for paper yet). All methods.

Observed p=6. Real directed edges among observed (chain 2->3->4->5) PLUS latent
AR(1) confounders. phi (latent autocorrelation) sweeps within->beyond the
Corollary-1 finite-Markov assumption. Two functional regimes (linear-Gaussian;
nonlinear tanh + non-Gaussian noise) probe CITS's nonparametric CI vs parametric
baselines.

Topologies (confounder configs), true observed edges = {2->3,3->4,4->5} in all:
  T1 collinear  : H -> X0 (lag1,1.0), X1 (lag1,1.0)             confounded {0,1}
  T2 asymmetric : H -> X0 (lag1,1.0), X1 (lag2,0.7)            confounded {0,1}
  T3 two-latent : Ha->{X0(l1),X1(l2)}, Hb->{X2(l1),X5(l1)}     confounded {0,1},{2,5}

Methods: CITS (GPU-scalable RCIT), PCMCI+, LPCMCI (parcorr, tau_max=2),
conditional Granger GC2, pairwise Granger GC1.

Scoring on confounded pairs (no true DIRECT edge):
  false_causal = method commits a DIRECTED edge (-->/<--) between the pair  (the error)
  latent_flag  = PAG bidirected <-> on the pair (correct latent call; PAG methods only)
  ambig_link   = only circle marks (o->,o-o,<-o) between the pair (hedge; PAG only)
Structural: TPR on true edges, FPR on true non-edges (excl. confounded pairs), CS=TPR-FPR.
"""
import numpy as np, sys, os, warnings, time; warnings.filterwarnings('ignore')
sys.path.insert(0,'/home/rbiswas1/microns/analysis/functional_circuitry')
sys.path.insert(0,'/home/rbiswas1/repos/cits')
from scipy.stats import f as fdist
from numpy.linalg import lstsq

P=6; T=1500; NOISE=0.5; TAU=2; ALPHA=0.05
TRUE_EDGES=[(2,3),(3,4),(4,5)]
TOPOS={
 'T1_collinear':{'lat':[{'children':[(0,1,1.0),(1,1,1.0)]}], 'conf':[(0,1)]},
 'T2_asymmetric':{'lat':[{'children':[(0,1,1.0),(1,2,0.7)]}], 'conf':[(0,1)]},
 'T3_two_latent':{'lat':[{'children':[(0,1,1.0),(1,2,0.7)]},{'children':[(2,1,1.0),(5,1,1.0)]}], 'conf':[(0,1),(2,5)]},
}
PHIS=[0.0,0.5,0.9]
REGIMES=['linear','nonlinear']

def simulate(seed, topo, phi, regime):
    rng=np.random.default_rng(seed)
    spec=TOPOS[topo]; nlat=len(spec['lat'])
    H=np.zeros((nlat,T))
    for l in range(nlat):
        for t in range(1,T): H[l,t]=phi*H[l,t-1]+rng.standard_normal()
    def noise(n):
        if regime=='linear': return NOISE*rng.standard_normal(n)
        return NOISE*(rng.exponential(1.0,n)-1.0)          # non-Gaussian, zero-mean
    X=np.zeros((P,T)); X[:, :2]=noise(P*2).reshape(P,2)
    for t in range(2,T):
        drive=np.zeros(P)
        for a,b in TRUE_EDGES: drive[b]+=0.9*X[a,t-1]        # true lagged edges
        for l,lat in enumerate(spec['lat']):
            for (child,lag,load) in lat['children']:
                drive[child]+=load*H[l,t-lag]                # latent confounding
        if regime=='nonlinear':
            X[:,t]=np.tanh(drive)+noise(P)
        else:
            X[:,t]=drive+noise(P)
    return X

# ---------------- methods ----------------
_DEV=[None]
def _pick_device():
    if _DEV[0] is not None: return _DEV[0]
    import torch
    for d in ['cuda:1','cuda:0']:
        try:
            if torch.cuda.is_available():
                torch.zeros(8,device=d); _DEV[0]=d; return d
        except Exception: pass
    _DEV[0]='cpu'; return 'cpu'

def m_cits(X):
    from gpu_cits_lag_rcit import gpu_cits_lag_rcit
    dev=_pick_device()
    try:
        B=gpu_cits_lag_rcit(X,alpha=ALPHA,tau=TAU,K=25,n_perm=100,max_cond_size=5,seed=0,device=dev,null='gamma')
    except Exception:
        B=gpu_cits_lag_rcit(X,alpha=ALPHA,tau=TAU,K=25,n_perm=100,max_cond_size=5,seed=0,device='cpu',null='gamma')
    A=(np.asarray(B)!=0).astype(int)
    return {'adj':A,'pag':None}

def _var_lag(X):
    n=T-TAU
    Y=X[:,TAU:]                                   # p x n
    L=np.column_stack([np.ones(n)]+[X[v,TAU-1-d:T-1-d] for d in range(TAU) for v in range(P)])
    return Y,L,n                                  # L cols: 1 + (d0:v0..v5)+(d1:v0..v5)

def m_cond_granger(X):
    Y,L,n=_var_lag(X); k=L.shape[1]; B=np.zeros((P,P),int)
    for j in range(P):
        yj=Y[j]; rf=yj-L@lstsq(L,yj,rcond=None)[0]; rssf=rf@rf; dff=n-k
        for i in range(P):
            if i==j: continue
            drop=[1+d*P+i for d in range(TAU)]     # drop all lags of var i
            keep=[c for c in range(k) if c not in drop]
            Lr=L[:,keep]; rr=yj-Lr@lstsq(Lr,yj,rcond=None)[0]; rssr=rr@rr
            F=((rssr-rssf)/TAU)/(rssf/dff)
            if 1-fdist.cdf(F,TAU,dff)<ALPHA: B[i,j]=1
    return {'adj':B,'pag':None}

def m_pair_granger(X):
    B=np.zeros((P,P),int)
    for j in range(P):
        for i in range(P):
            if i==j: continue
            n=T-TAU; yj=X[j,TAU:]
            own=np.column_stack([np.ones(n)]+[X[j,TAU-1-d:T-1-d] for d in range(TAU)])
            r0=yj-own@lstsq(own,yj,rcond=None)[0]; rss0=r0@r0
            full=np.column_stack([own]+[X[i,TAU-1-d:T-1-d] for d in range(TAU)])
            r1=yj-full@lstsq(full,yj,rcond=None)[0]; rss1=r1@r1
            dff=n-full.shape[1]; F=((rss0-rss1)/TAU)/(rss1/dff)
            if 1-fdist.cdf(F,TAU,dff)<ALPHA: B[i,j]=1
    return {'adj':B,'pag':None}

def _tigra(X, algo):
    from tigramite import data_processing as pp
    from tigramite.independence_tests.parcorr import ParCorr
    df=pp.DataFrame(X.T.astype(float))
    if algo=='pcmci':
        from tigramite.pcmci import PCMCI
        g=PCMCI(dataframe=df,cond_ind_test=ParCorr(),verbosity=0).run_pcmciplus(tau_max=TAU,pc_alpha=ALPHA)['graph']
    else:
        from tigramite.lpcmci import LPCMCI
        g=LPCMCI(dataframe=df,cond_ind_test=ParCorr(),verbosity=0).run_lpcmci(tau_max=TAU,pc_alpha=ALPHA)['graph']
    # presence adjacency for TPR/FPR: arrowhead into j at any tau (incl <-> both ways)
    A=np.zeros((P,P),int)
    for i in range(P):
        for j in range(P):
            if i==j: continue
            for tau in range(0,TAU+1):
                m=g[i,j,tau]
                if m in ('-->','o->'): A[i,j]=1
                if m=='<->': A[i,j]=1
    return {'adj':A,'pag':g}
def m_pcmci(X): return _tigra(X,'pcmci')
def m_lpcmci(X): return _tigra(X,'lpcmci')

METHODS=[('CITS',m_cits),('PCMCI+',m_pcmci),('LPCMCI',m_lpcmci),
         ('CondGranger',m_cond_granger),('PairGranger',m_pair_granger)]

# ---------------- scoring ----------------
def pair_class(res, a, b):
    """Return one of: 'directed','bidirected','ambiguous','none' for the a-b relationship."""
    g=res['pag']
    if g is None:  # directed-only method
        A=res['adj']
        return 'directed' if (A[a,b] or A[b,a]) else 'none'
    directed=bidir=circ=False
    for i,j in [(a,b),(b,a)]:
        for tau in range(0,TAU+1):
            m=g[i,j,tau]
            if m=='': continue
            if m=='<->': bidir=True
            elif m in ('-->','<--'): directed=True
            elif 'o' in m or 'x' in m: circ=True
            else: circ=True
    if directed: return 'directed'
    if bidir: return 'bidirected'
    if circ: return 'ambiguous'
    return 'none'

def structural(res, conf_pairs):
    """Return TPR, FPR_excl (confounded pairs EXCLUDED from non-edges),
    FPR_full (confounded pairs COUNTED as true non-edges -> paper-comparable CS)."""
    A=res['adj']; conf=set(frozenset(p) for p in conf_pairs)
    tp=sum(A[i,j]>0 for i,j in TRUE_EDGES); ntp=len(TRUE_EDGES)
    fp=nfp=0; fpF=nfpF=0
    for i in range(P):
        for j in range(P):
            if i==j or (i,j) in TRUE_EDGES: continue
            nfpF+=1; fpF+=int(A[i,j]>0)                       # full: confounded pairs INCLUDED
            if frozenset({i,j}) in conf: continue
            nfp+=1; fp+=int(A[i,j]>0)                         # excl: confounded pairs removed
    return tp/ntp, fp/max(nfp,1), fpF/max(nfpF,1)

def run_cell(topo, phi, regime, nseed):
    conf=TOPOS[topo]['conf']
    acc={m:{'fc':[],'bi':[],'am':[],'tpr':[],'fpr':[],'fprF':[]} for m,_ in METHODS}
    for s in range(nseed):
        X=simulate(s,topo,phi,regime)
        for name,fn in METHODS:
            try: res=fn(X)
            except Exception as e:
                res={'adj':np.zeros((P,P),int),'pag':None}
            # confounded-pair classification (aggregate across pairs: error if ANY pair is directed)
            cls=[pair_class(res,a,b) for a,b in conf]
            acc[name]['fc'].append(int(any(c=='directed' for c in cls)))
            acc[name]['bi'].append(int(any(c=='bidirected' for c in cls)))
            acc[name]['am'].append(int(any(c=='ambiguous' for c in cls)))
            tpr,fpr,fprF=structural(res,conf)
            acc[name]['tpr'].append(tpr); acc[name]['fpr'].append(fpr); acc[name]['fprF'].append(fprF)
    return acc

def ci95(v):
    v=np.asarray(v,float); m=v.mean(); se=v.std(ddof=1)/np.sqrt(len(v)) if len(v)>1 else 0
    return m,1.96*se

if __name__=='__main__':
    NSEED=int(sys.argv[1]) if len(sys.argv)>1 else 40
    only=sys.argv[2] if len(sys.argv)>2 else None
    print(f"RIGOROUS LATENT TESTBED  p={P} T={T} tau={TAU} alpha={ALPHA} N={NSEED} seeds  device={_pick_device()}",flush=True)
    for regime in REGIMES:
        for topo in TOPOS:
            if only and topo!=only: continue
            for phi in PHIS:
                t0=time.time(); acc=run_cell(topo,phi,regime,NSEED); dt=time.time()-t0
                print(f"\n[{regime}] {topo} phi={phi}  ({dt:.0f}s)",flush=True)
                print(f"  {'method':12s} falseCausal   latent<->   ambig    TPR    FPRx   CSx    FPRfull  CSfull",flush=True)
                for name,_ in METHODS:
                    a=acc[name]; fcm,fcc=ci95(a['fc']); tprm,_=ci95(a['tpr'])
                    fprm,_=ci95(a['fpr']); fprF,_=ci95(a['fprF'])
                    csx=tprm-fprm; csF=tprm-fprF
                    print(f"  {name:12s} {fcm:.2f}±{fcc:.2f}   {np.mean(a['bi']):.2f}      {np.mean(a['am']):.2f}    "
                          f"{tprm:.2f}   {fprm:.2f}   {csx:.2f}   {fprF:.2f}     {csF:.2f}",flush=True)
    print("\nTESTBED DONE",flush=True)
