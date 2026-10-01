"""Data-driven edge DIRECTION for Fig 2 panel B (CITS-estimate row).
Nonparametric partial Spearman: residualize child and parent on the OTHER
parents (+ self-history) with a cross-fitted kNN, then rank-correlate.
Sign = increasing(+)/decreasing(-); |rho|<=THR -> non-monotone (none)."""
import numpy as np, sys, json, warnings; warnings.filterwarnings('ignore')
sys.path.insert(0,'.'); sys.path.insert(0,'/home/rbiswas1/repos/cits')
from sim_scm import simulate_extended
from cits.methods import data_transformed
from scipy import stats
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import KFold
tau,NSEED,THR=1,20,0.10
TE={'lingauss1':[(0,2),(1,2),(2,3)],'lingauss2':[(0,1),(0,2),(1,3),(2,3)],
    'nonlinnongauss1':[(0,2),(1,2),(2,3)],'nonlinnongauss2':[(0,1),(0,2),(1,3),(2,3)]}
PARENTS={'lingauss1':{2:[0,1],3:[2]},'nonlinnongauss1':{2:[0,1],3:[2]},
         'lingauss2':{1:[0],2:[0],3:[1,2]},'nonlinnongauss2':{1:[0],2:[0],3:[1,2]}}
def cf_resid(y,C):
    if C.shape[1]==0: return y-y.mean()
    r=np.zeros_like(y)
    for tr,te in KFold(5,shuffle=True,random_state=0).split(C):
        m=KNeighborsRegressor(n_neighbors=15).fit(C[tr],y[tr]); r[te]=y[te]-m.predict(C[te])
    return r
out={}
for reg,edges in TE.items():
    rec={}
    for (i,j) in edges:
        others=[p for p in PARENTS[reg][j] if p!=i]
        ctrl_cols=[2*p for p in others]+[2*j]     # other parents (lagged) + self-history
        vals=[]
        for s in range(NSEED):
            X=simulate_extended(reg,1.0,1000,s)[0].astype(float); d=data_transformed(X,tau)
            y=d[:,2*j+1]; xi=d[:,2*i]; C=d[:,ctrl_cols]
            vals.append(stats.spearmanr(cf_resid(xi,C),cf_resid(y,C)).correlation)
        m=float(np.mean(vals))
        direction='+' if m>THR else ('-' if m<-THR else '0')
        rec[f"{i+1}->{j+1}"]={'spearman':m,'dir':direction}
    out[reg]=rec
    print(reg,{k:f"{v['spearman']:+.2f}{v['dir']}" for k,v in rec.items()},flush=True)
json.dump(out,open('_fig2_edgedir.json','w'),indent=1)
print("EDGEDIR DONE",flush=True)
