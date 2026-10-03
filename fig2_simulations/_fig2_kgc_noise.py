import numpy as np, sys, os, warnings, csv; warnings.filterwarnings('ignore')
sys.path.insert(0,'.'); sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'shared')))
from paths import out as _out
from sim_scm import simulate_extended
from kernel_granger_baseline import run_kernel_granger
from directed_metrics import compute_directed_metrics
def m3(pred,out):
    _,gl_uw,gl_w,gc_uw,gc_w,gb_uw,gb_lw,gb_cw=out
    m=compute_directed_metrics(pred,gl_w,gc_w,gb_lw,gb_cw,gl_uw,gc_uw,gb_uw)
    return m['directed_TPR_strict'],m['directed_FPR_strict'],m['directed_CS_strict']
REG=['lingauss1','lingauss2','nonlinnongauss1','nonlinnongauss2']; NOISE=[0.1,0.5,1,1.5,2,2.5,3,3.5]
w=csv.writer(open(_out('fig2_simulations', '_fig2_kgc_noise50.csv'),'w',newline='')); w.writerow(['method','regime','noise','seed','TPR','FPR','CS'])
for reg in REG:
  for nz in NOISE:
    for s in range(50):
        out=simulate_extended(reg,nz,1000,s); X=out[0].astype(np.float64)
        p=(run_kernel_granger(X,max_lag=2,alpha=0.05)!=0).astype(int); np.fill_diagonal(p,0); t,f,c=m3(p,out); w.writerow(['KernelGC',reg,nz,s,t,f,c])
  print("done",reg,flush=True)
print("KGC NOISE DONE",flush=True)
