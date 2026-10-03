"""Curiosity experiment (NOT for paper): latent-confounder robustness of CITS vs the
assumption boundary of Corollary 1.

Setup: latent AR(1) confounder H (autocorrelation phi) drives observed X0 and X1
(NO true X0<->X1 edge). True observed edge: X2 -> X3 (to measure TPR).
Both X0 and X1 are noisy proxies of the same H -> the classic confounding trap.

Corollary 1 needs a finite-Markov observed process with no concurrent effects.
Marginalizing an autocorrelated latent breaks finite-Markovianity, and phi controls
how badly. So we sweep phi:
  phi=0    -> latent is white; observed process stays ~first-order Markov (assumption holds)
  phi->1   -> strong long memory; assumption increasingly violated
Prediction: CITS clean at low phi, degrading with phi; naive lagged methods
(pairwise Granger, lag-1 correlation) fooled across the board.
"""
import numpy as np, sys, os, warnings; warnings.filterwarnings('ignore')
sys.path.insert(0,os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'shared')))   # repo shared/ (was the local cits checkout)
from cits.methods import cits_full
from scipy.stats import f as fdist, pearsonr
from numpy.linalg import lstsq

P=4; T=1000; C=1.0; NOISE=0.5; A_EDGE=0.9
PHIS=[0.0,0.3,0.6,0.9]
NSEED=int(sys.argv[1]) if len(sys.argv)>1 else 25

def simulate(seed, phi):
    rng=np.random.default_rng(seed)
    H=np.zeros(T)
    for t in range(1,T): H[t]=phi*H[t-1]+rng.standard_normal()
    X=rng.standard_normal((P,T))*NOISE
    for t in range(1,T):
        X[0,t]=C*H[t-1]+NOISE*rng.standard_normal()          # H -> X0 (lagged)
        X[1,t]=C*H[t-1]+NOISE*rng.standard_normal()          # H -> X1 (lagged), same driver
        X[3,t]=A_EDGE*X[2,t-1]+NOISE*rng.standard_normal()   # true lagged edge 2->3
    return X

CONF=frozenset({0,1})
def score(Bhat):
    spur=int(any(Bhat[a,b] for a,b in [(0,1),(1,0)]))
    tp=int(Bhat[2,3]>0)
    fp=0;nfp=0
    for i in range(P):
        for j in range(P):
            if i==j or (i,j)==(2,3) or {i,j}==CONF: continue
            nfp+=1; fp+=int(Bhat[i,j]>0)
    return spur,tp,fp/max(nfp,1)

def pairwise_granger(X,alpha=0.05):
    B=np.zeros((P,P),int); n=T-1; Y=X[:,1:]; L=X[:,:-1]
    for j in range(P):
        yj=Y[j]; own=np.column_stack([np.ones(n),L[j]])
        r0=yj-own@lstsq(own,yj,rcond=None)[0]; rss0=r0@r0
        for i in range(P):
            if i==j: continue
            full=np.column_stack([own,L[i]]); r1=yj-full@lstsq(full,yj,rcond=None)[0]; rss1=r1@r1
            F=((rss0-rss1)/1)/(rss1/(n-3))
            if 1-fdist.cdf(F,1,n-3)<alpha: B[i,j]=1
    return B

def lag_corr(X,alpha=0.05):
    B=np.zeros((P,P),int)
    for i in range(P):
        for j in range(P):
            if i==j: continue
            if pearsonr(X[i,:-1],X[j,1:])[1]<alpha: B[i,j]=1
    return B

print(f"phi-sweep, N={NSEED} seeds, T={T}, C={C}, noise={NOISE}",flush=True)
print(f"{'phi':>4} | {'method':8s} spur_X0X1  TPR(2->3)  other_FPR",flush=True)
for phi in PHIS:
    agg={m:{'spur':[],'tp':[],'fp':[]} for m in ['CITS','Granger','LagCorr']}
    for s in range(NSEED):
        X=simulate(s,phi)
        for name,fn in [('CITS',lambda x:cits_full(x,1,0.05,'cond_dep_pcorr')),
                        ('Granger',pairwise_granger),('LagCorr',lag_corr)]:
            sp,tp,fp=score((np.asarray(fn(X))!=0).astype(int))
            agg[name]['spur'].append(sp); agg[name]['tp'].append(tp); agg[name]['fp'].append(fp)
    for m in ['CITS','Granger','LagCorr']:
        a=agg[m]
        print(f"{phi:>4.1f} | {m:8s}   {np.mean(a['spur']):.2f}      {np.mean(a['tp']):.2f}       {np.mean(a['fp']):.2f}",flush=True)
    print("",flush=True)
print("DONE",flush=True)
