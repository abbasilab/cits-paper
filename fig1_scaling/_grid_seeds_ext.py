"""Extra accuracy seeds for uniform-n CS bars. CS-only (runtimes NOT used; panel a
keeps certified seeds 0-2). Phased so there is a clean uniform checkpoint:
  Phase 1 = seeds 3,4  -> uniform n=5    Phase 2 = seeds 5..9 -> uniform n=10
All cells here are FEASIBLE (they completed at n=3), so NO per-cell kill budget.
Memory-gated so the box cannot OOM. Resumable: skips (method,p,N,seed) already done.
Output: grid_seeds_ext.csv (method,p,N,seed,status,cs)."""
import os, sys, time, subprocess, tempfile, psutil
import pandas as pd
_THIS = os.path.dirname(os.path.abspath(__file__)); os.chdir(_THIS)
sys.path.insert(0, os.path.abspath(os.path.join(_THIS, '..', 'shared')))
from paths import out as _out, outdir, result
PHASES     = [[3,4],[5,6,7,8,9]]
MAX_CONC   = 8
MEM_RESERVE= 220            # GB free floor
CPU_THREADS= 16
GPUS       = ['0','3','4','5']; MAX_PER_GPU = 1
OUT        = _out('fig1_scaling', 'grid_seeds_ext.csv')
TMP        = tempfile.mkdtemp(prefix='seedext_', dir=outdir('fig1_scaling/tmp'))
feas = pd.read_csv(result('fig1_scaling', '_feasible_cells.csv'))
done=set(); results=[]
if os.path.exists(OUT):
    dd=pd.read_csv(OUT); results=dd.to_dict('records')
    done={(r.method,int(r.p),int(r.N),int(r.seed)) for r in dd.itertuples()}

def env_cpu():
    e=dict(os.environ)
    for v in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'): e[v]=str(CPU_THREADS)
    return e
def launch(key,gpu=None):
    m,p,N,s=key; fo=open(os.path.join(TMP,f'{m}_{p}_{N}_{s}.out'),'w')
    if gpu is None: cmd=['python3','_cell_cpu.py',m,str(p),str(N),str(s)]; env=env_cpu()
    else: cmd=['python3','_cell_gpu.py',str(p),str(N),gpu,str(s)]; env=dict(os.environ)
    return {'key':key,'pr':subprocess.Popen(cmd,stdout=fo,stderr=subprocess.DEVNULL,text=True,env=env),'fo':fo,'gpu':gpu,'t0':time.time()}
def reap(job):
    job['fo'].flush(); job['fo'].close(); m,p,N,s=job['key']; cs=None; st='err'
    try:
        for ln in open(job['fo'].name):
            if ln.startswith('RESULT'): _,rt,c=ln.split(); cs=float(c); st='ok'
    except Exception: pass
    return {'method':m,'p':p,'N':N,'seed':s,'status':st,'cs':round(cs,3) if cs is not None else None}

def run_jobs(cpu_jobs,gpu_jobs,tag):
    running=[]; gpu_busy={g:0 for g in GPUS}; ok=err=0; t0=time.time()
    cpu_jobs.sort(key=lambda k:-(k[1]*k[2]))     # big first
    while cpu_jobs or gpu_jobs or running:
        for job in running[:]:
            if job['pr'].poll() is not None:
                row=reap(job); results.append(row); running.remove(job)
                if job['gpu'] is not None: gpu_busy[job['gpu']]-=1
                ok+=row['status']=='ok'; err+=row['status']!='ok'
                pd.DataFrame(results).to_csv(OUT,index=False)
                if (ok+err)%20==0 or (cpu_jobs==[] and gpu_jobs==[]):
                    print(f"  [{tag}] {ok} ok {err} err, {len(cpu_jobs)+len(gpu_jobs)} left, {time.time()-t0:.0f}s, RAM {psutil.virtual_memory().available/1e9:.0f}GB",flush=True)
        for g in GPUS:
            while gpu_jobs and gpu_busy[g]<MAX_PER_GPU: running.append(launch(gpu_jobs.pop(),gpu=g)); gpu_busy[g]+=1
        while cpu_jobs and len([j for j in running if j['gpu'] is None])<MAX_CONC and psutil.virtual_memory().available/1e9>MEM_RESERVE:
            running.append(launch(cpu_jobs.pop(0))); time.sleep(0.2)
        time.sleep(2)
    print(f"  [{tag}] phase done: {ok} ok {err} err in {time.time()-t0:.0f}s",flush=True)

for ph,seeds in enumerate(PHASES,1):
    cpu=[]; gpu=[]
    for r in feas.itertuples():
        for s in seeds:
            k=(r.method,int(r.p),int(r.N),s)
            if k in done: continue
            (gpu if r.method=='CITS-GPU' else cpu).append(k)
    print(f"=== PHASE {ph} seeds={seeds}: CPU {len(cpu)} GPU {len(gpu)} ===",flush=True)
    run_jobs(cpu,gpu,f'ph{ph}')
    n_now = 3 + sum(len(PHASES[i]) for i in range(ph))
    print(f"=== CHECKPOINT: uniform n={n_now} available ===",flush=True)
print("ALL DONE",flush=True)
