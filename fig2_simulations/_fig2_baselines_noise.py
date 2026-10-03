import numpy as np, sys, os, warnings, csv; warnings.filterwarnings('ignore')
sys.path.insert(0,'.'); sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'shared')))
from paths import out as _out
from sim_scm import simulate_extended
from simulation_benchmark_fc_methods_v3 import run_tpc, run_pc
from directed_metrics import compute_directed_metrics
from scipy import stats
def ols(Y,X): X=np.column_stack([np.ones(len(X)),X]); b,_,_,_=np.linalg.lstsq(X,Y,rcond=None); r=Y-X@b; return float(r@r),X.shape[1]
def granger(X,cond):
    p,T=X.shape;A=np.zeros((p,p),int);Xp=X[:,:-1].T;Xt=X[:,1:].T;n=Xt.shape[0]
    for j in range(p):
        y=Xt[:,j]
        for i in range(p):
            if i==j: continue
            if cond: rf,kf=ols(y,Xp[:,list(range(p))]);rr,kr=ols(y,Xp[:,[k for k in range(p) if k!=i]])
            else: rf,kf=ols(y,Xp[:,[j,i]]);rr,kr=ols(y,Xp[:,[j]])
            d1=kf-kr;d2=n-kf
            if d1<=0 or d2<=0 or rf<=0: continue
            if 1-stats.f.cdf(((rr-rf)/d1)/(rf/d2),d1,d2)<0.05: A[i,j]=1
    return A
def m3(pred,out):
    _,gl_uw,gl_w,gc_uw,gc_w,gb_uw,gb_lw,gb_cw=out
    m=compute_directed_metrics(pred,gl_w,gc_w,gb_lw,gb_cw,gl_uw,gc_uw,gb_uw)
    return m['directed_TPR_strict'],m['directed_FPR_strict'],m['directed_CS_strict']
REG=['lingauss1','lingauss2','nonlinnongauss1','nonlinnongauss2']
NOISE=[0.1,0.5,1,1.5,2,2.5,3,3.5]; NS=50
w=csv.writer(open(_out('fig2_simulations', '_fig2_baselines_noise50.csv'),'w',newline='')); w.writerow(['method','regime','noise','seed','TPR','FPR','CS'])
for reg in REG:
  for nz in NOISE:
    for s in range(NS):
        out=simulate_extended(reg,nz,1000,s); X=out[0]
        for name,pred in [('GC1',granger(X,False)),('GC2',granger(X,True)),('PC',run_pc(X,0.05)),('TPC',run_tpc(X,0.05))]:
            p=(np.asarray(pred)!=0).astype(int); np.fill_diagonal(p,0); t,f,c=m3(p,out); w.writerow([name,reg,nz,s,t,f,c])
  print(f"done {reg}",flush=True)
print("BASELINES NOISE DONE",flush=True)
