"""Scaling driver: run stim FC (both variants) over all (scan,field,stimulus),
sharded across GPUs. Saves per-cell FC npz + appends metrics rows to a CSV.
Usage: stim_run.py --gpu 0 --nshards 2 --shard 0 --variants AB
"""
import os, sys, time, argparse, glob, json
ap = argparse.ArgumentParser()
ap.add_argument('--gpu', default='0'); ap.add_argument('--nshards', type=int, default=2)
ap.add_argument('--shard', type=int, default=0); ap.add_argument('--variants', default='AB')
args = ap.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu           # before any cuda import
os.environ.setdefault('OMP_NUM_THREADS', '8')

sys.path.insert(0, '/home/rbiswas1/microns/analysis/stimulus_fc')
import numpy as np, pandas as pd
import stim_fc_pipeline as P

OUT = '/home/rbiswas1/microns/analysis/stimulus_fc/out'
FCDIR = f'{OUT}/fc'; os.makedirs(FCDIR, exist_ok=True); os.makedirs(OUT, exist_ok=True)
STIMS = ['clip', 'Monet', 'Trippy']
tasks = [(s, sc, f, st) for (s, sc) in P.scans() for f in range(1, 9) for st in STIMS]
mine = tasks[args.shard::args.nshards]
csv_path = f'{OUT}/results_gpu{args.gpu}_shard{args.shard}.csv'
rows = []
print(f"[shard {args.shard}/{args.nshards} gpu{args.gpu}] {len(mine)} tasks", flush=True)
for k, (s, sc, f, st) in enumerate(mine):
    tag = f's{s}sc{sc}f{f}_{st}'
    fcnpz = f'{FCDIR}/{tag}.npz'
    if os.path.exists(fcnpz) and os.path.exists(csv_path):
        continue
    t0 = time.time()
    try:
        out, status = P.process(s, sc, f, st, variants=tuple(args.variants))
    except Exception as e:
        import traceback; print(f"  {tag} ERR {str(e)[:120]}", flush=True); traceback.print_exc(); continue
    if out is None:
        print(f"  {tag} skip: {status}", flush=True); continue
    save = {}
    if 'fcA' in out: save['fcA'] = out.pop('fcA').astype(np.float32)
    if 'fcB' in out: save['fcB'] = out.pop('fcB').astype(np.float32)
    np.savez_compressed(fcnpz, **save)
    rows.append(out)
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"  [{k+1}/{len(mine)}] {tag} p={out['p']} nwin={out['n_windows']} "
          f"A={out.get('A_secs','-')}s B={out.get('B_secs','-')}s ({time.time()-t0:.1f}s)", flush=True)
print("SHARD_DONE", flush=True)
