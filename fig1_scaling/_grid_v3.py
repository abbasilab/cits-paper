"""Grid v3 - fork-free, multi-seed. Each (method,p,N,seed) cell runs in a FRESH
subprocess (no fork). Per cell: run seed 0; if it completes, auto-run seeds 1,2
(CS error bars on feasible cells); if seed 0 times out, skip extra seeds (wall
confirmed - no need to replicate a timeout). Records CS+runtime+cert per seed.
CPU: 32 threads, certified >=40 cores free. GPU: exclusive. Sequential (solo)."""
import os, sys, time, subprocess
_THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_THIS, '..', 'shared')))
from paths import out as _out
import pandas as pd, numpy as np

BUDGET = 1800; NCPU = os.cpu_count(); CLEAN_MIN_FREE = 40
GPU = '3'; SEEDS = [0, 1, 2]
P_CPU = [5, 10, 25, 50, 100, 250, 500]
P_GPU = [5, 10, 25, 50, 100, 250, 500, 1000]
N_GRID = [125, 250, 500, 1000]
CPU_METHODS = ['PCMCI+', 'TPC', 'KernelGC', 'LPCMCI']
OUT = _out('fig1_scaling', 'grid_v3.csv')
rows = []


def run_cell(cmd):
    loads = []
    pr = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
    t0 = time.perf_counter()
    while pr.poll() is None and (time.perf_counter()-t0) < BUDGET:
        loads.append(os.getloadavg()[0]); time.sleep(2)
    if pr.poll() is None:
        pr.terminate()
        try: pr.wait(10)
        except Exception: pr.kill()
        return 'timeout', BUDGET, float('nan'), max(loads) if loads else 0
    out = pr.stdout.read(); rt = cs = float('nan'); st = 'err'
    for line in out.splitlines():
        if line.startswith('RESULT'):
            _, r, c = line.split(); rt, cs, st = float(r), float(c), 'ok'
    return st, rt, cs, max(loads) if loads else os.getloadavg()[0]


def do_cell(method, p, N, worker_cmd):
    """seed 0 first; auto-continue to seeds 1,2 only if seed 0 completed."""
    got_timeout = False
    for s in SEEDS:
        st, rt, cs, lmax = run_cell(worker_cmd(s))
        clean = (NCPU - lmax) >= CLEAN_MIN_FREE if method != 'CITS-GPU' else True
        rows.append({'method':method,'p':p,'N':N,'seed':s,'status':st,
                     'runtime_sec':round(rt,1),'cs':round(cs,3) if cs==cs else None,
                     'load_max':round(lmax,1),'clean':clean})
        pd.DataFrame(rows).to_csv(OUT, index=False)
        body = f"{rt:8.1f}s CS={cs:.3f}" if st=='ok' else st.upper()
        print(f"  {method:9s} p={p:4d} N={N:5d} s{s}: {body} [{'CLEAN' if clean else 'DIRTY'}]", flush=True)
        if st == 'timeout':
            got_timeout = True; break        # skip remaining seeds for a wall
    return got_timeout


print("=== GRID v3 (fork-free, 3-seed, seed0->auto seeds1,2), CS+runtime+cert ===", flush=True)
for method in CPU_METHODS:
    prev_all_to = False
    for p in P_CPU:
        if prev_all_to:
            print(f"  {method:9s} p={p:4d}: SKIP (all N walled at smaller p)", flush=True); continue
        n_to = 0
        for N in N_GRID:
            to = do_cell(method, p, N, lambda s: ['python3','_cell_cpu.py',method,str(p),str(N),str(s)])
            if to: n_to += 1
        prev_all_to = (n_to == len(N_GRID))
    print("", flush=True)

print(f"--- CITS-GPU (exclusive GPU {GPU}) ---", flush=True)
for p in P_GPU:
    for N in N_GRID:
        do_cell('CITS-GPU', p, N, lambda s: ['python3','_cell_gpu.py',str(p),str(N),GPU,str(s)])
print("DONE", flush=True)
