import numpy as np, sys, os, warnings, csv; warnings.filterwarnings('ignore')
sys.path.insert(0,'.'); sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'shared')))
from paths import out as _out
from glm_spiking_sim import simulate_glm_spiking, directed_cs as gdcs
from sim_scm import simulate_extended
from simulation_benchmark_fc_methods_v3 import run_tpc
from kernel_granger_baseline import run_kernel_granger
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
def dcs_ar(pred,out):
    _,gl_uw,gl_w,gc_uw,gc_w,gb_uw,gb_lw,gb_cw=out
    return compute_directed_metrics(pred,gl_w,gc_w,gb_lw,gb_cw,gl_uw,gc_uw,gb_uw)['directed_CS_strict']
w=csv.writer(open(_out('table1_baselines', '_sd_local.csv'),'w',newline='')); w.writerow(['cell','seed','cs'])
SPK={'convergence':'conv','diamond':'cce','depression':'depr'}
for m,tag in SPK.items():
    for s in range(50):
        X,GT,_=simulate_glm_spiking(s,motif=m)
        for name,pred in [('GC1',granger(X,False)),('GC2',granger(X,True)),
                          ('TPC',run_tpc(X,0.05)),('KernelGC',run_kernel_granger(X.astype(np.float64),max_lag=2,alpha=0.05))]:
            p=(np.asarray(pred)!=0).astype(int); np.fill_diagonal(p,0); w.writerow([f'{name}_spk_{tag}',s,gdcs(p,GT)[0]])
for reg in ['lingauss1','lingauss2','nonlinnongauss1','nonlinnongauss2']:
    for s in range(50):
        out=simulate_extended(reg,1.0,1000,s); X=out[0]
        p=(granger(X,True)!=0).astype(int); np.fill_diagonal(p,0); w.writerow([f'GC2_{reg}',s,dcs_ar(p,out)])
print("local sd done",flush=True)
