"""Unified LSCM edge weight + sign for Fig 2 panel C.
For each true edge, regress the child (current) on its parents (past) + self-history
in the time-windowed data; the coefficient of the parent is the CITS (LSCM) edge
weight. Sign = sign(median weight), reported only where the weight is significantly
!= 0 (OLS t-test) in most datasets; else 'sign n/a' (single-dataset-valid rule).
This is the same signed-LSCM convention used by the neural (Version B) pipeline."""
import numpy as np, sys, json, warnings; warnings.filterwarnings('ignore')
sys.path.insert(0,'.'); sys.path.insert(0,'/home/rbiswas1/repos/cits')
from sim_scm import simulate_extended
from cits.methods import data_transformed
from scipy import stats
tau,NS=1,50
TE={'lingauss1':[(0,2),(1,2),(2,3)],'lingauss2':[(0,1),(0,2),(1,3),(2,3)],
    'nonlinnongauss1':[(0,2),(1,2),(2,3)],'nonlinnongauss2':[(0,1),(0,2),(1,3),(2,3)]}
PARENTS={'lingauss1':{2:[0,1],3:[2]},'nonlinnongauss1':{2:[0,1],3:[2]},
         'lingauss2':{1:[0],2:[0],3:[1,2]},'nonlinnongauss2':{1:[0],2:[0],3:[1,2]}}
def ols_coef_p(y,X,idx):
    n,k=X.shape; X0=np.hstack([np.ones((n,1)),X])
    beta=np.linalg.lstsq(X0,y,rcond=None)[0]; resid=y-X0@beta; sig2=(resid@resid)/(n-k-1)
    se=np.sqrt(np.diag(sig2*np.linalg.inv(X0.T@X0))); t=beta[idx+1]/se[idx+1]
    return float(beta[idx+1]), float(2*stats.t.sf(abs(t),n-k-1))
out={}
for reg,edges in TE.items():
    rec={f"{i+1}->{j+1}":{'raw':[],'p':[]} for (i,j) in edges}
    for s in range(NS):
        d=data_transformed(simulate_extended(reg,1.0,1000,s)[0].astype(float),tau)
        for (i,j) in edges:
            cols=[2*p for p in PARENTS[reg][j]]+[2*j]; idx=cols.index(2*i)
            b,p=ols_coef_p(d[:,2*j+1],d[:,cols],idx)
            k=f"{i+1}->{j+1}"; rec[k]['raw'].append(b); rec[k]['p'].append(p)
    o={}
    for (i,j) in edges:
        k=f"{i+1}->{j+1}"; a=np.array(rec[k]['raw']); pv=np.array(rec[k]['p'])
        med=float(np.median(a)); frac_sig=float(np.mean(pv<0.05))
        sign=('+' if med>0 else '-') if frac_sig>=0.5 else '0'
        o[k]={'med':med,'lo':float(np.percentile(a,2.5)),'hi':float(np.percentile(a,97.5)),
              'sign':sign,'frac_sig':frac_sig}
    out[reg]=o
    print(reg,{k:f"{v['med']:+.2f}[{v['lo']:.2f},{v['hi']:.2f}] sig{v['frac_sig']:.2f} {v['sign']}" for k,v in o.items()},flush=True)
json.dump(out,open('_fig2_edgesign.json','w'),indent=1)
print("EDGESIGN DONE",flush=True)
