"""Extend TPC & KernelGC to p=1000 (the two baselines feasible at p=500) to
demonstrate (not extrapolate) whether they wall before CITS-GPU's p=1000.
Fork-free fresh subprocess, 3-seed w/ auto-continue, 30-min cap, certified."""
import os, sys, time, subprocess
import pandas as pd
BUDGET=1800; NCPU=os.cpu_count(); SEEDS=[0,1,2]
METHODS=['TPC','KernelGC']; P=[1000]; N_GRID=[125,250,500]
OUT='grid_ext.csv'; rows=[]
def run_cell(cmd):
    loads=[]; pr=subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=subprocess.DEVNULL,text=True)
    t0=time.perf_counter()
    while pr.poll() is None and (time.perf_counter()-t0)<BUDGET:
        loads.append(os.getloadavg()[0]); time.sleep(2)
    if pr.poll() is None:
        pr.terminate()
        try: pr.wait(10)
        except Exception: pr.kill()
        return 'timeout',BUDGET,float('nan'),max(loads) if loads else 0
    out=pr.stdout.read(); rt=cs=float('nan'); st='err'
    for l in out.splitlines():
        if l.startswith('RESULT'): _,r,c=l.split(); rt,cs,st=float(r),float(c),'ok'
    return st,rt,cs,max(loads) if loads else os.getloadavg()[0]
print(f"=== extension: TPC/KernelGC at p=1000, {NCPU} cores, 30-min cap ===",flush=True)
for m in METHODS:
    for p in P:
        for N in N_GRID:
            for s in SEEDS:
                st,rt,cs,lmax=run_cell(['python3','_cell_cpu.py',m,str(p),str(N),str(s)])
                clean=(NCPU-lmax)>=40
                rows.append({'method':m,'p':p,'N':N,'seed':s,'status':st,
                    'runtime_sec':round(rt,1),'cs':round(cs,3) if cs==cs else None,
                    'load_max':round(lmax,1),'clean':clean})
                pd.DataFrame(rows).to_csv(OUT,index=False)
                body=f"{rt:8.1f}s CS={cs:.3f}" if st=='ok' else st.upper()
                print(f"  {m:9s} p={p} N={N:4d} s{s}: {body} [{'CLEAN' if clean else 'DIRTY'}]",flush=True)
                if st=='timeout': break   # wall -> skip extra seeds
print("DONE",flush=True)
