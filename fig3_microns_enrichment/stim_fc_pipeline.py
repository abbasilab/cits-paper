"""
stim_fc_pipeline.py  (runs on gpu-2)

Stimulus-type FC (Clip/Monet/Trippy), TWO variants, per (session,scan,field,stimulus):
  Variant A: cuPC lagged skeleton (gpu_cits_lag_cupc_faithful) + LSCM weights
             (cits.methods.causaleff_lscm + cits_weighted_rolled, thresh |w|<max/10).
  Variant B: contemporaneous CITS with cuPC lagged  = cuPC rolled B as the lagged parent
             identity + PC-contemporaneous (pc_skeleton_raw, GPU) + orient v-structures
             only (NO_MEEK) + build_union + lscm_refit_cpdag once per child.

Pooling: per stimulus trial, center per-neuron, truncate to multiple of w=2(tau+1),
concatenate -> X_pooled (p, T). Because trial boundaries land on w-aligned block
edges, chi / data_transform / data_transformed windows are all WITHIN-trial.
(Caveat: lscm_refit's consecutive-sample lag regression has ~1 cross-junction row
per trial boundary, ~1-2% of rows -- Variant B only.)
"""
import os, sys, glob, time, json, argparse
import numpy as np, pandas as pd
from itertools import combinations, product

AD = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'shared'))   # repo shared/ (was analysis/functional_circuitry)
for p in (AD,):
    if p not in sys.path: sys.path.insert(0, p)
from paths import MICRONS_SAVES, microns_meta

SAVES = MICRONS_SAVES   # was microns_data/saves on the GPU server (same layout as /data1 saves)
NPY = f'{SAVES}/calcium_npy'
AREA_FILE = microns_meta('all_unit_areas.csv')
AREAS = ['AL', 'LM', 'RL', 'V1']
TAU = 1
W = 2 * (TAU + 1)   # 4
ALPHA = 0.05
MIN_UNITS = 10
THRESH = 10.0       # |w| < max|w|/THRESH -> 0
N_MAX = 800         # cap pooled windows for BOTH variants (N*~500 plateau).
N_B_MAX = N_MAX     # Variant B: dense contemp graph at large N blows up IDA/LSCM.
N_A_MAX = N_MAX     # Variant A: cuPC skeleton stays dense at high N (huge CI power)
                    # -> PC conditioning-set explosion (>3min at N=5376). Same cap as B
                    # also makes A vs B a same-sample-size comparison.

_area_df = pd.read_csv(AREA_FILE).set_index(['session', 'scan_idx', 'unit_id'])['brain_area'].to_dict()

def scans():
    out = []
    for f in sorted(glob.glob(f'{NPY}/calcium_session*_scan*.npy')):
        import re; m = re.search(r'session(\d+)_scan(\d+)', f); out.append((int(m.group(1)), int(m.group(2))))
    return out

def load_scan(s, sc):
    cal = np.load(f'{NPY}/calcium_session{s}_scan{sc}.npy', mmap_mode='r')
    ids = np.load(f'{NPY}/ids_session{s}_scan{sc}.npy')
    fields = np.load(f'{NPY}/fields_session{s}_scan{sc}.npy')
    union = set(np.load(f'{NPY}/unionids_session{s}_scan{sc}.npy').tolist())
    return cal, ids, fields, union

def field_rows(ids, fields, union, field):
    """rows (into calcium) for this field's union neurons, sorted by ID; + their IDs."""
    idx = [i for i in range(len(ids)) if fields[i] == field and int(ids[i]) in union]
    idx.sort(key=lambda i: int(ids[i]))
    return np.array(idx, dtype=int), np.array([int(ids[i]) for i in idx])

def load_trials(s, sc, stim):
    d = pd.read_pickle(f'{SAVES}/{stim}_timepoints_session{s}_scan{sc}.pkl')
    a = np.asarray(d)
    return [(int(a[k, 0]), int(a[k, 1])) for k in range(len(a))]

def build_pooled(cal, rows, trials, Tmax):
    """Per-trial: extract cal[rows, start:end], center per-neuron, truncate to
    multiple of W, concat. Returns X (p, T) float64."""
    blocks = []
    for (st, en) in trials:
        en = min(en, Tmax)
        if en - st < W: continue
        b = np.asarray(cal[rows, st:en], dtype=np.float64)     # (p, L)
        b = b - b.mean(axis=1, keepdims=True)                  # center per-neuron
        L = (b.shape[1] // W) * W
        if L < W: continue
        blocks.append(b[:, :L])
    if not blocks: return None
    return np.concatenate(blocks, axis=1)                      # (p, T)

def area_index(neuron_ids, s, sc):
    ai = {a: [] for a in AREAS}
    for i, nid in enumerate(neuron_ids):
        a = _area_df.get((s, sc, int(nid)))
        if a in ai: ai[a].append(i)
    return ai

# ---------- metrics ----------
def _ds(flat):
    nz = flat[flat != 0]
    return (len(nz) / len(flat) if len(flat) else np.nan,
            np.mean(np.abs(nz)) if len(nz) else np.nan)

def metrics(FC, ai):
    m = FC.astype(float).copy(); np.fill_diagonal(m, 0.0)
    row = {}
    row['overall_d'], row['overall_s'] = _ds(m.flatten())
    # within (lateral)
    lw = [];
    for a in AREAS:
        si = ai[a]
        if len(si) >= 2:
            sub = FC[np.ix_(si, si)].astype(float).copy(); np.fill_diagonal(sub, 0.0)
            lw.append(sub.flatten())
    row['within_d'], row['within_s'] = _ds(np.concatenate(lw)) if lw else (np.nan, np.nan)
    # between (hierarchical)
    hw = []
    for a, b in combinations(AREAS, 2):
        if ai[a] and ai[b]:
            hw.append(FC[np.ix_(ai[a], ai[b])].astype(float).flatten())
            hw.append(FC[np.ix_(ai[b], ai[a])].astype(float).flatten())
    row['between_d'], row['between_s'] = _ds(np.concatenate(hw)) if hw else (np.nan, np.nan)
    # 16 directed area-pairs
    for a in AREAS:
        for b in AREAS:
            key = f'{a}->{b}'
            if ai[a] and ai[b]:
                blk = FC[np.ix_(ai[a], ai[b])].astype(float).copy()
                if a == b: np.fill_diagonal(blk, 0.0)
                d, s_ = _ds(blk.flatten())
            else:
                d, s_ = np.nan, np.nan
            row[f'{key}_d'] = d; row[f'{key}_s'] = s_
    return row

# ---------- FC variants ----------
def fc_variant_A(X):
    """cuPC lagged skeleton + LSCM weights -> signed weighted (p,p)."""
    import cits.methods as cm
    from _cupc_wrapper import pc_skeleton_cupc
    from gpu_cits_lag_cupc_faithful import _build_chi_nonoverlap
    import networkx as nx
    p, T = X.shape
    _ta = time.time()
    U = _build_chi_nonoverlap(X, TAU)                          # (N, p*w)
    G, _sep, _inact, _lvl = pc_skeleton_cupc(U, alpha=ALPHA, max_level=14, verbose=False)
    print(f"  [A] cuPC skeleton: {time.time()-_ta:.1f}s (N={U.shape[0]}, edges={int((G!=0).sum()//2)})", flush=True); _ta = time.time()
    t = 2 * TAU + 1
    # rolled binary skeleton B (for reference)
    # build (tau+1) unrolled DiGraph Uu for LSCM (matches cits_full_weighted)
    Uu = np.zeros((p * (TAU + 1), p * (TAU + 1)))
    for v1 in range(p):
        for v2 in range(p):
            for t1 in range(t - TAU, t):
                if G[t1 * p + v1, t * p + v2] != 0:
                    Uu[(TAU + 1) * v1 + (t1 - t + TAU), (TAU + 1) * v2 + TAU] = 1
    g = nx.from_numpy_array(Uu, create_using=nx.DiGraph())
    data_trans = cm.data_transformed(X, TAU)                   # (N, p*(tau+1))
    ce_A = cm.causaleff_lscm(g, data_trans)
    print(f"  [A] LSCM weights: {time.time()-_ta:.1f}s (edges={g.number_of_edges()})", flush=True)
    ce_B = cm.cits_weighted_rolled(ce_A, p, TAU)
    mx = np.abs(ce_B).max()
    if mx > 0: ce_B[np.abs(ce_B) < mx / THRESH] = 0.0
    return ce_B

def fc_variant_B(X, cits_B_binary, tag=""):
    """Contemporaneous CITS with cuPC lagged: cuPC B = lagged parent identity + PC-contemp
    (NO_MEEK) + union + lscm refit -> signed weighted (p,p). Per-step timed."""
    from _pc_raw import pc_skeleton_raw
    from _pc_orientation import orient_v_structures
    from _lscm_refit import lscm_refit_cpdag
    from _union_cpdag import build_union
    Xtp = X.T   # (T, p)
    def _t(msg, t0): print(f"  [B{tag}] {msg}: {time.time()-t0:.1f}s", flush=True); return time.time()
    t0 = time.time()
    pc_skel, pc_r0, sep_sets, inactive = pc_skeleton_raw(Xtp, alpha=ALPHA, use_gpu=True, verbose=False)
    n_contemp = int((pc_skel != 0).sum() // 2)
    print(f"  [B{tag}] contemp pc_skeleton_raw: {time.time()-t0:.1f}s ({n_contemp} edges)", flush=True); t0 = time.time()
    pc_G = orient_v_structures(pc_skel, sep_sets)              # NO_MEEK (safe)
    t0 = _t("orient_v_structures", t0)
    union_parents, union_skel, edge_type = build_union(cits_B_binary.astype(float), pc_skel, pc_G, sign_amb_mat=None, tau=TAU)
    t0 = _t("build_union", t0)
    union_B, union_sa = lscm_refit_cpdag(Xtp, pc_G, extra_parents_per_child=union_parents, verbose=False)
    t0 = _t("lscm_refit_cpdag", t0)
    return union_B, union_skel, edge_type

def cupc_rolled_binary(X):
    from gpu_cits_lag_cupc_faithful import gpu_cits_lag_cupc_faithful
    return gpu_cits_lag_cupc_faithful(X, alpha=ALPHA, tau=TAU)

def process(s, sc, field, stim, variants=('A', 'B'), Tmax=40000):
    cal, ids, fields, union = load_scan(s, sc)
    rows, nids = field_rows(ids, fields, union, field)
    if len(rows) < MIN_UNITS:
        return None, f"few_units({len(rows)})"
    ai = area_index(nids, s, sc)
    if sum(len(v) >= 2 for v in ai.values()) < 1:
        return None, "no_area>=2"
    trials = load_trials(s, sc, stim)
    X = build_pooled(cal, rows, trials, Tmax)
    if X is None or X.shape[1] < 2 * W:
        return None, "no_pooled_data"
    p, T = X.shape
    nwin = T // W
    out = {'session': s, 'scan': sc, 'field': field, 'stim': stim, 'p': p, 'n_windows': nwin}
    if 'A' in variants:
        tA = time.time()
        XA = X[:, :N_A_MAX * W] if nwin > N_A_MAX else X    # Variant A: capped N
        out['A_nwin'] = XA.shape[1] // W
        fcA = fc_variant_A(XA)
        out['fcA'] = fcA; out['A_secs'] = round(time.time() - tA, 1)
        for k, v in metrics(fcA, ai).items(): out[f'A_{k}'] = v
    if 'B' in variants:
        tB = time.time()
        XB = X[:, :N_B_MAX * W] if nwin > N_B_MAX else X   # Variant B: capped N
        out['B_nwin'] = XB.shape[1] // W
        t0 = time.time(); cB = cupc_rolled_binary(XB)
        print(f"  [B] cuPC-lagged: {time.time()-t0:.1f}s (N={XB.shape[1]//W})", flush=True)
        fcB, skelB, etB = fc_variant_B(XB, cB)
        out['fcB'] = fcB; out['B_secs'] = round(time.time() - tB, 1)
        for k, v in metrics(fcB, ai).items(): out[f'B_{k}'] = v
    return out, "ok"

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--validate', action='store_true')
    ap.add_argument('--s', type=int, default=4); ap.add_argument('--sc', type=int, default=7)
    ap.add_argument('--field', type=int, default=2); ap.add_argument('--stim', default='clip')
    ap.add_argument('--gpu', default='0'); ap.add_argument('--variant', default='AB')
    args = ap.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    if args.validate:
        t0 = time.time()
        out, status = process(args.s, args.sc, args.field, args.stim, variants=tuple(args.variant))
        if out is None:
            print("VALIDATE status:", status); sys.exit(1)
        print(f"VALIDATE ok: s{args.s}sc{args.sc}f{args.field} {args.stim} p={out['p']} nwin={out['n_windows']} t={time.time()-t0:.1f}s")
        for var in ('A', 'B'):
            if f'{var}_overall_d' in out:
                fc = out[f'fc{var}']
                print(f"  Variant {var}: FC {fc.shape} nnz={int((fc!=0).sum())} "
                      f"overall_d={out[f'{var}_overall_d']:.4f} within_d={out[f'{var}_within_d']:.4f} "
                      f"between_d={out[f'{var}_between_d']:.4f} within_s={out[f'{var}_within_s']:.4f}")
