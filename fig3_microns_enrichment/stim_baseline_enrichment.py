"""
Stimulus-provenance FC-EM enrichment for the two BASELINE estimators
(lagged-correlation and conditional-Granger GC2), computed on the SAME stimulus
calcium input that fed the CITS Version-B stimulus FC (fcB), with the SAME EM
fields, neurons, preprocessing (build_pooled + N_B_MAX cap), FC-present union
across the 3 stimulus graphs, and EM pairing as panelA_fc_em_enrichment_versionB.py.

Only the FC ESTIMATOR differs:
  - 'lagged1' : directed lagged Pearson, lag=1, Fisher-z + BH-FDR<0.05
                (compute_corr_baseline_fold.lagged_corr_adj, faithful)
  - 'laggedmax': max over lags {1..5} of |lagged corr|, then same BH-FDR<0.05
                (reconstruction of the '_max' arousal baseline; exact lag set
                 was an inline session heredoc, not preserved -> flagged)
  - 'granger' : conditional Granger VAR(1), edge present iff |t|>=1.96
                (regenerate_gc_raw_2026-05-01_safe.mvgc_one, faithful)

Usage: python stim_baseline_enrichment.py <lagged1|laggedmax|granger>
"""
import os, sys, glob, pickle, time
os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('MKL_NUM_THREADS', '4')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '4')
import numpy as np
import pandas as pd
from scipy.stats import fisher_exact, norm

# ---------- paths ----------
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'shared')))
from paths import outdir as _outdir, OUT_ROOT, MICRONS_SAVES, arousal_figs
SAVES = os.path.join(MICRONS_SAVES, '')
NPY = f'{SAVES}calcium_npy'
EM_KEYS_NPZ = arousal_figs('2026-04-29', 'fig4', 'bootstrap_sf_correlation_13sess.npz')
# Version-B FC written by stim_run.py (was staged from gpu-2 into a session scratchpad)
STAGE_FC = os.path.join(OUT_ROOT, 'fig3_microns_enrichment', 'out', 'fc')
OUTDIR = _outdir('fig3_microns_enrichment')
STIMS = ['clip', 'Monet', 'Trippy']

# ---------- CITS stimulus preprocessing constants (from stim_fc_pipeline.py) ----------
TAU = 1
W = 2 * (TAU + 1)     # 4
N_B_MAX = 800         # windows cap that produced fcB
TMAX = 40000
MIN_UNITS = 10

N_BOOT = 5000
LAGS_MAX = [1, 2, 3, 4, 5]   # reconstruction lag set for the '_max' variant


# ============ input builders (mirror stim_fc_pipeline.py exactly) ============
def load_scan(s, sc):
    cal = np.load(f'{NPY}/calcium_session{s}_scan{sc}.npy', mmap_mode='r')
    ids = np.load(f'{NPY}/ids_session{s}_scan{sc}.npy')
    fields = np.load(f'{NPY}/fields_session{s}_scan{sc}.npy')
    union = set(np.load(f'{NPY}/unionids_session{s}_scan{sc}.npy').tolist())
    return cal, ids, fields, union


def field_rows(ids, fields, union, field):
    idx = [i for i in range(len(ids)) if fields[i] == field and int(ids[i]) in union]
    idx.sort(key=lambda i: int(ids[i]))
    return np.array(idx, dtype=int), [int(ids[i]) for i in idx]


def load_trials(s, sc, stim):
    d = pd.read_pickle(f'{SAVES}{stim}_timepoints_session{s}_scan{sc}.pkl')
    a = np.asarray(d)
    return [(int(a[k, 0]), int(a[k, 1])) for k in range(len(a))]


def build_pooled(cal, rows, trials, Tmax=TMAX):
    blocks = []
    for (st, en) in trials:
        en = min(en, Tmax)
        if en - st < W:
            continue
        b = np.asarray(cal[rows, st:en], dtype=np.float64)
        b = b - b.mean(axis=1, keepdims=True)     # center per-neuron per-trial
        L = (b.shape[1] // W) * W
        if L < W:
            continue
        blocks.append(b[:, :L])
    if not blocks:
        return None
    return np.concatenate(blocks, axis=1)         # (p, T)


def build_XB(s, sc, field, stim):
    """Reproduce the exact XB (p, T) that fed fcB for this (field, stim)."""
    cal, ids, fields, union = load_scan(s, sc)
    rows, nids = field_rows(ids, fields, union, field)
    if len(rows) < MIN_UNITS:
        return None, nids
    trials = load_trials(s, sc, stim)
    X = build_pooled(cal, rows, trials, TMAX)
    if X is None or X.shape[1] < 2 * W:
        return None, nids
    nwin = X.shape[1] // W
    XB = X[:, :N_B_MAX * W] if nwin > N_B_MAX else X
    return XB, nids


# ============ estimators ============
def zscore_rows(X):
    return (X - X.mean(1, keepdims=True)) / (X.std(1, keepdims=True) + 1e-9)


def _fdr_directed_present(C, n_eff, p_thresh=0.05):
    """Binary DIRECTED present mask over off-diagonal pairs, BH-FDR (faithful to
    compute_corr_baseline_fold.fdr_adj_directed)."""
    p = C.shape[0]
    mask = ~np.eye(p, dtype=bool)
    idx = np.where(mask)
    r = np.clip(C[mask], -0.999999, 0.999999)
    z = np.arctanh(r) * np.sqrt(max(n_eff, 1))
    pv = 2 * norm.sf(np.abs(z))
    m = len(pv)
    pv_sorted = np.sort(pv)
    bh = pv_sorted <= (np.arange(1, m + 1) / m * p_thresh)
    adj = np.zeros((p, p), dtype=bool)
    if bh.any():
        cut = pv_sorted[np.where(bh)[0].max()]
        sig = pv <= cut
        adj[idx[0][sig], idx[1][sig]] = True
    return adj


def present_lagged(XB, lag=1):
    past = zscore_rows(XB[:, :-lag])
    fut = zscore_rows(XB[:, lag:])
    n = past.shape[1]
    C = (past @ fut.T) / n
    np.fill_diagonal(C, 0.0)
    return _fdr_directed_present(C, n - 3)


def present_lagged_max(XB, lags=LAGS_MAX):
    """Per-directed-pair max |lagged corr| across lags, then BH-FDR<0.05.
    n_eff uses the shortest overlap (n = T - max(lags)). Reconstruction."""
    p = XB.shape[0]
    Cmax = np.zeros((p, p))
    for lag in lags:
        past = zscore_rows(XB[:, :-lag])
        fut = zscore_rows(XB[:, lag:])
        n = past.shape[1]
        C = (past @ fut.T) / n
        np.fill_diagonal(C, 0.0)
        m = np.abs(C) > np.abs(Cmax)
        Cmax[m] = C[m]
    n_eff = XB.shape[1] - max(lags) - 3
    return _fdr_directed_present(Cmax, n_eff)


def present_granger(XB, t_thresh=1.96):
    """Conditional Granger VAR(1); present iff |t|>=1.96. Faithful to
    regenerate_gc_raw_2026-05-01_safe.mvgc_one (union-neuron conditioning set)."""
    data = XB - XB.mean(axis=1, keepdims=True)   # center per neuron (matches process_trial)
    X = np.asarray(data.T, dtype=np.float64)     # (T, N)
    T, N = X.shape
    if N > T - 3:
        var = X.var(axis=0)
        keep = np.argsort(var)[::-1][: max(2, T - 3)]
        X_used = X[:, keep]; idx_remap = keep
    else:
        X_used = X; idx_remap = np.arange(N)
    Tn, Nn = X_used.shape
    Y = X_used[1:]
    Xl = X_used[:-1]
    Z = np.column_stack([np.ones(Tn - 1), Xl])
    ZtZ = Z.T @ Z
    B, *_ = np.linalg.lstsq(Z, Y, rcond=None)
    A1 = B[1:].T
    R = Y - Z @ B
    dof = (Tn - 1) - (Nn + 1)
    sigma2 = (R ** 2).sum(axis=0) / dof
    ZtZ_inv = np.linalg.inv(ZtZ)
    diag_inv = np.diag(ZtZ_inv)[1:]
    se = np.sqrt(np.outer(sigma2, diag_inv))
    with np.errstate(divide='ignore', invalid='ignore'):
        tvals = A1 / se
    present_used = np.abs(tvals) >= t_thresh
    present = np.zeros((N, N), dtype=bool)
    ri = np.repeat(idx_remap, len(idx_remap))
    ci = np.tile(idx_remap, len(idx_remap))
    present[ri, ci] = present_used.ravel()
    np.fill_diagonal(present, False)
    return present


ESTIMATORS = {'lagged1': present_lagged,
              'laggedmax': present_lagged_max,
              'granger': present_granger}


# ============ EM pairing (identical to panelA) ============
def counts_for_field(s, sc, field, present_fn, matched_df, sc_pairs):
    # Only use the stimuli whose fcB was staged (identical support as panelA CITS)
    avail = [st for st in STIMS if os.path.exists(f'{STAGE_FC}/s{s}sc{sc}f{field}_{st}.npz')]
    if not avail:
        return None
    XB0, nids = build_XB(s, sc, field, avail[0])
    if XB0 is None:
        return None
    p = len(nids)
    present = np.zeros((p, p), dtype=bool)
    got = 0
    for st in avail:
        XB, nids_st = build_XB(s, sc, field, st)
        if XB is None or XB.shape[0] != p:
            continue
        # sanity: XB neuron count must equal staged fcB dimension
        fcB = np.load(f'{STAGE_FC}/s{s}sc{sc}f{field}_{st}.npz')['fcB']
        if fcB.shape != (p, p):
            continue
        present |= present_fn(XB)
        got += 1
    if got == 0:
        return None
    # EM pairing (panelA)
    id_to_idx = {u: i for i, u in enumerate(nids)}
    fu = matched_df[(matched_df['session'] == s) &
                    (matched_df['scan_idx'] == sc) &
                    (matched_df['field'] == field)]
    idx_to_pt = {}
    for u, pt in zip(fu['unit_id'].astype(np.int64).values,
                     fu['pt_root_id_v1718'].astype(np.int64).values):
        if int(u) in id_to_idx:
            idx_to_pt[id_to_idx[int(u)]] = int(pt)
    m_idx = sorted(idx_to_pt.keys())
    fcp = fps = fca = fas = 0
    for i in m_idx:
        for j in m_idx:
            if i == j:
                continue
            fc = bool(present[i, j])
            syn = (idx_to_pt[i], idx_to_pt[j]) in sc_pairs
            if fc:
                fcp += 1; fps += int(syn)
            else:
                fca += 1; fas += int(syn)
    return {'field_key': (s, sc, field), 'n_matched': len(m_idx), 'n_stim': got,
            'fc_plus': fcp, 'fcplus_scplus': fps,
            'fc_minus': fca, 'fcminus_scplus': fas}


def katz_ci(a, n1, c, n2):
    fold = (a / n1) / (c / n2)
    se = np.sqrt(1 / a - 1 / n1 + 1 / c - 1 / n2)
    return fold, np.exp(np.log(fold) - 1.96 * se), np.exp(np.log(fold) + 1.96 * se)


def main():
    method = sys.argv[1] if len(sys.argv) > 1 else 'lagged1'
    present_fn = ESTIMATORS[method]
    print(f"=== stimulus baseline enrichment: {method} ===", flush=True)

    d = np.load(EM_KEYS_NPZ, allow_pickle=True)
    em_keys = [tuple(int(x) for x in k) for k in d['field_keys']]
    matched_df = pickle.load(open(SAVES + 'matched_df_v1718.pkl', 'rb'))
    matched_df = matched_df.dropna(subset=['pt_root_id_v1718']).copy()
    matched_df['pt_root_id_v1718'] = matched_df['pt_root_id_v1718'].astype(np.int64)
    syn = pickle.load(open(SAVES + 'synapses_matcheddf_frompre_v1718.pkl', 'rb'))
    sc_pairs = set()
    for pre_id, df_s in syn.items():
        if hasattr(df_s, 'columns') and 'post_pt_root_id' in df_s.columns:
            for post in df_s['post_pt_root_id'].astype(np.int64).values:
                sc_pairs.add((int(pre_id), int(post)))
    print(f"  em_keys={len(em_keys)}  sc_pairs={len(sc_pairs)}", flush=True)

    rows = []
    for (s, sc, field) in em_keys:
        t0 = time.time()
        r = counts_for_field(s, sc, field, present_fn, matched_df, sc_pairs)
        if r is not None:
            rows.append(r)
            print(f"  s{s}sc{sc}f{field}: n_matched={r['n_matched']} n_stim={r['n_stim']} "
                  f"fc+={r['fc_plus']} (syn {r['fcplus_scplus']}) fc-={r['fc_minus']} "
                  f"(syn {r['fcminus_scplus']})  [{time.time()-t0:.1f}s]", flush=True)
    pf = pd.DataFrame(rows)
    pf.to_csv(f'{OUTDIR}/stim_baseline_perfield_{method}.csv', index=False)
    print(f"  usable fields = {len(pf)}", flush=True)

    agg = pf[['fc_plus', 'fcplus_scplus', 'fc_minus', 'fcminus_scplus']].sum().to_numpy()
    fcp, fps, fca, fas = (int(x) for x in agg)
    fold, katz_lo, katz_hi = katz_ci(fps, fcp, fas, fca)
    _, fisher_p = fisher_exact([[fps, fcp - fps], [fas, fca - fas]])

    # field-cluster bootstrap (seed 42, N=5000)
    A = pf['fcplus_scplus'].values.astype(float); N1 = pf['fc_plus'].values.astype(float)
    Cc = pf['fcminus_scplus'].values.astype(float); N2 = pf['fc_minus'].values.astype(float)
    n = len(pf); rng = np.random.default_rng(42); folds = []
    for _ in range(N_BOOT):
        ii = rng.integers(0, n, n)
        a = A[ii].sum(); n1 = N1[ii].sum(); c = Cc[ii].sum(); n2 = N2[ii].sum()
        if n1 > 0 and c > 0 and n2 > 0:
            folds.append((a / n1) / (c / n2))
    blo, bhi = np.percentile(np.array(folds), [2.5, 97.5])

    print(f"\n  RESULT [{method}]  n_fields={n}", flush=True)
    print(f"  FC-present: {fps}/{fcp} = {fps/fcp*100:.3f}%", flush=True)
    print(f"  FC-absent : {fas}/{fca} = {fas/fca*100:.3f}%", flush=True)
    print(f"  FOLD={fold:.4f}  Fisher p={fisher_p:.3e}  "
          f"Katz95=({katz_lo:.4f},{katz_hi:.4f})  boot95=({blo:.4f},{bhi:.4f})", flush=True)


if __name__ == '__main__':
    main()
