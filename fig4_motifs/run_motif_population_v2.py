#!/usr/bin/env python3
"""
Reproducible lagged CITS graph + population conditional-independence demonstration.

STEP 1: infer a lagged directed adjacency over the active neurons using the CURRENT
        cits code's own CI machinery (data_transform windowing + Fisher-z partial
        correlation), but via a SCALABLE lagged PC-skeleton (bounded conditioning)
        instead of the infeasible powerset search in cits_unrolled/cits_full_weighted.

STEP 2: population CI demonstration on the reproducible graph.
STEP 3: pick a corrected common-source motif.

Deterministic: no randomness anywhere (PC-stable skeleton). Fixed data -> fixed graph.
"""
import sys, os, json
import numpy as np
import pickle as pkl
from itertools import combinations
from scipy import stats

sys.path.insert(0, '/home/rbiswas1/repos/cits')
from cits.methods import data_transform, partial_corr

# ----------------------------------------------------------------------------
# config
# ----------------------------------------------------------------------------
SESS = 791319847
STIM = 'natural_scenes'
BIN = 0.01
IDX = 0
TAU = 1
ALPHA = 0.05
MAX_K = 4           # bounded conditioning-set size (scalable; pure PC would be unbounded)
DATA_DIR = '/home/rbiswas1/citsproject/data'
OUT_DIR = '/home/rbiswas1/microns/CITS_manuscript/figures'
np.random.seed(0)   # not used by anything deterministic; set for good hygiene

X_FILE = f'{DATA_DIR}/ID{SESS}_{STIM}_bin_{BIN}_X_idx-{IDX}.p'
MASK_FILE = f'{DATA_DIR}/ID{SESS}_{STIM}_units2use_stim_{STIM}.p'

# ----------------------------------------------------------------------------
# load + standardize exactly as the pipeline does (per-unit z-score)
# ----------------------------------------------------------------------------
raw = np.asarray(pkl.load(open(X_FILE, 'rb')), dtype=float)       # (T, N_all)
mask = np.asarray(pkl.load(open(MASK_FILE, 'rb')))
units_idx = np.where(mask)[0] if mask.dtype == bool else mask     # global unit ids
data = raw[:, units_idx]                                          # (T, p)
mu = data.mean(axis=0); sd = data.std(axis=0); sd[sd == 0] = 1.0
data = (data - mu) / sd
T, p = data.shape
X = data.T                                                        # (p, T) for cits

# windowed samples used by the CITS CI test (lag-1 unrolled representation)
chi = data_transform(X, TAU)                                      # (p*2*(tau+1), N)
N = chi.shape[1]
n_slots = 2 * (TAU + 1)                                           # =4 for tau=1
# slot indexing: variable v at slot s -> row  s*p + v  in chi
SRC_SLOT = 2 * TAU        # =2 : lagged (t-1) source slot
TGT_SLOT = 2 * TAU + 1    # =3 : present (t) target slot

print(f'p={p} active units, T={T}, N_windows={N}, tau={TAU}, alpha={ALPHA}, max_k={MAX_K}')

def row(v, slot):
    return slot * p + v

def fisher_z_pval(r, n, k):
    """Two-sided Fisher-z p-value for partial correlation r with n samples, k conditioners."""
    if r >= 1.0:
        return 0.0
    if r <= -1.0:
        return 0.0
    z = 0.5 * np.log((1 + r) / (1 - r))
    stat = np.sqrt(max(n - k - 3, 1)) * np.abs(z)
    return 2 * (1 - stats.norm.cdf(stat))

def lagged_pcorr(u, v, cond_neurons):
    """Partial corr between source u at slot t-1 and target v at slot t,
    conditioning on `cond_neurons` at slot t-1 (lagged separating set).
    Uses the cits partial_corr on the windowed chi. Returns (r, pval)."""
    A = row(u, SRC_SLOT)
    B = row(v, TGT_SLOT)
    S = set(row(c, SRC_SLOT) for c in cond_neurons)
    r = partial_corr(A, B, S, chi)
    return r, fisher_z_pval(r, N, len(S))

# ----------------------------------------------------------------------------
# STEP 1: scalable lagged PC-skeleton  (fast_cits_pcorr)
# For each target v, prune candidate lagged parents by PC-stable adjacency search.
# Candidate parents = all neurons u != v at slot t-1 (autapses excluded from graph).
# Direction is fixed by time (t-1 -> t), so no orientation step is needed.
# ----------------------------------------------------------------------------
def fast_cits_pcorr(alpha=ALPHA, max_k=MAX_K):
    """Returns adjacency and sepset dict.
    sepset[(u,v)] = the EXACT conditioning set (tuple of neuron indices at t-1) at
    which the lagged CI test for source u(t-1) -> target v(t) PASSED (p>alpha),
    causing the edge to be removed. For pairs already marginally independent the
    recorded sepset is the empty set (removed at order 0)."""
    adj = np.zeros((p, p), dtype=int)   # adj[u,v]=1 : u(t-1) -> v(t)
    sepset = {}                          # (u,v) local -> tuple(S)  for removed edges
    for v in range(p):
        # order-0: marginal lagged dependence
        parents = []
        for u in range(p):
            if u == v:
                continue
            r, pv = lagged_pcorr(u, v, [])
            if pv <= alpha:
                parents.append(u)
            else:
                sepset[(u, v)] = tuple()          # removed at order 0 (marginally indep)
        # increasing conditioning-set size (PC-stable: decide removals per level)
        ell = 1
        while ell <= max_k:
            if len(parents) - 1 < ell:
                break
            to_remove = []
            cur = list(parents)
            for u in cur:
                others = [w for w in cur if w != u]
                for S in combinations(others, ell):
                    r, pv = lagged_pcorr(u, v, S)
                    if pv > alpha:
                        to_remove.append(u)
                        sepset[(u, v)] = tuple(S)  # EXACT separating set used
                        break
            parents = [u for u in parents if u not in to_remove]
            ell += 1
        for u in parents:
            adj[u, v] = 1
    return adj, sepset

print('\n[STEP 1] running scalable lagged PC-skeleton (fast_cits_pcorr)...')
adj, sepset = fast_cits_pcorr()
n_edges = int(adj.sum())
n_possible = p * (p - 1)
density = n_edges / n_possible
print(f'  edges = {n_edges} / {n_possible} ordered pairs, density = {density:.4f}')

# parents per target
parents_of = {v: [u for u in range(p) if adj[u, v]] for v in range(p)}

# ----------------------------------------------------------------------------
# STEP 2: population CI demonstration on the algorithm's OWN decision.
# For every removed (non-adjacent) ordered pair (u -> v), condition on the EXACT
# separating set S the PC-skeleton recorded at edge removal. This transparently
# demonstrates the algorithm's own conditional-independence criterion at
# population scale (self-consistent by construction: that is the intent).
# For surviving edges we condition on the target's other inferred parents (the
# largest set the edge was tested against and survived) to show they stay dependent.
# ----------------------------------------------------------------------------
print('\n[STEP 2] population CI demonstration (exact recorded separating sets)...')

def cond_on_parents(u, v):
    """conditional lagged r of u->v given v's other inferred parents."""
    S = [w for w in parents_of[v] if w != u]
    return lagged_pcorr(u, v, S)

records = []  # per ordered pair
for v in range(p):
    for u in range(p):
        if u == v:
            continue
        rm, pm = lagged_pcorr(u, v, [])
        is_edge = int(adj[u, v])
        if is_edge:
            # surviving edge: no sepset made it independent; condition on other parents
            rc, pc = cond_on_parents(u, v)
            S_used = tuple(w for w in parents_of[v] if w != u)
            cond_kind = 'other_parents'
        else:
            # removed edge: condition on the EXACT recorded separating set
            S_used = sepset.get((u, v), tuple())
            rc, pc = lagged_pcorr(u, v, S_used)
            cond_kind = 'recorded_sepset'
        records.append({
            'u': int(units_idx[u]), 'v': int(units_idx[v]),
            'u_local': u, 'v_local': v,
            'is_edge': is_edge,
            'marg_r': rm, 'marg_p': pm,
            'cond_r': rc, 'cond_p': pc,
            'n_cond': len(S_used),
            'sepset_global': [int(units_idx[w]) for w in S_used],
            'cond_kind': cond_kind,
        })

def summarize(recs, label):
    if len(recs) == 0:
        return {'label': label, 'n': 0}
    marg = np.array([r['marg_r'] for r in recs])
    cond = np.array([r['cond_r'] for r in recs])
    condp = np.array([r['cond_p'] for r in recs])
    n_indep_p = int((condp > ALPHA).sum())
    n_below01 = int((np.abs(cond) < 0.1).sum())
    return {
        'label': label, 'n': len(recs),
        'n_cond_indep_by_pval': n_indep_p,
        'frac_cond_indep_by_pval': round(n_indep_p / len(recs), 4),
        'n_cond_below_r0.1': n_below01,
        'frac_cond_below_r0.1': round(n_below01 / len(recs), 4),
        'mean_abs_marg': round(float(np.abs(marg).mean()), 4),
        'mean_abs_cond': round(float(np.abs(cond).mean()), 4),
        'mean_drop': round(float((np.abs(marg) - np.abs(cond)).mean()), 4),
    }

nonedges = [r for r in records if r['is_edge'] == 0]
edges = [r for r in records if r['is_edge'] == 1]

step2 = {}
# (a) non-adjacent pairs, marginally dependent at various thresholds
step2['a_nonedges_marg_pdep'] = summarize(
    [r for r in nonedges if r['marg_p'] <= ALPHA], 'nonedges marg p<=0.05')
for thr in (0.1, 0.2, 0.3):
    step2[f'a_nonedges_marg>{thr}'] = summarize(
        [r for r in nonedges if abs(r['marg_r']) > thr], f'nonedges |marg r|>{thr}')
# (b) edges-stay contrast
step2['b_edges'] = summarize(edges, 'edges (should STAY dependent)')
step2['b_edges_marg>0.1'] = summarize(
    [r for r in edges if abs(r['marg_r']) > 0.1], 'edges |marg r|>0.1')

print('  non-edges |marg r|>0.1 :', step2['a_nonedges_marg>0.1'])
print('  non-edges |marg r|>0.2 :', step2['a_nonedges_marg>0.2'])
print('  edges                  :', step2['b_edges'])

# ----------------------------------------------------------------------------
# STEP 2c + STEP 3: common-source triples  A <- C -> B , A,B NON-adjacent
# condition on shared parent C's lagged activity.
# We look for triples where C is an inferred parent of both A and B, A and B are
# non-adjacent, marginal A-B lagged dep is high, and conditioning on C (and full
# parent set) drops it.
# ----------------------------------------------------------------------------
print('\n[STEP 2c / STEP 3] common-source triples...')

triples = []
for C in range(p):
    children = [w for w in range(p) if adj[C, w]]
    for A, B in combinations(children, 2):
        # require A,B non-adjacent both directions
        if adj[A, B] or adj[B, A]:
            continue
        # marginal lagged dependence A(t-1)->B(t) and B(t-1)->A(t)
        r_ab_m, p_ab_m = lagged_pcorr(A, B, [])
        r_ba_m, p_ba_m = lagged_pcorr(B, A, [])
        # conditional on shared parent C only
        r_ab_c, p_ab_c = lagged_pcorr(A, B, [C])
        r_ba_c, p_ba_c = lagged_pcorr(B, A, [C])
        # conditional on full separating set (B's other parents / A's other parents)
        r_ab_full, p_ab_full = cond_on_parents(A, B)
        r_ba_full, p_ba_full = cond_on_parents(B, A)
        # conditional on the EXACT recorded separating set the algorithm used
        S_ab = sepset.get((A, B), tuple())
        S_ba = sepset.get((B, A), tuple())
        r_ab_sep, p_ab_sep = lagged_pcorr(A, B, S_ab)
        r_ba_sep, p_ba_sep = lagged_pcorr(B, A, S_ba)
        triples.append({
            'C': int(units_idx[C]), 'A': int(units_idx[A]), 'B': int(units_idx[B]),
            'C_local': C, 'A_local': A, 'B_local': B,
            'r_AB_marg': r_ab_m, 'p_AB_marg': p_ab_m,
            'r_BA_marg': r_ba_m, 'p_BA_marg': p_ba_m,
            'r_AB_condC': r_ab_c, 'p_AB_condC': p_ab_c,
            'r_BA_condC': r_ba_c, 'p_BA_condC': p_ba_c,
            'r_AB_condfull': r_ab_full, 'p_AB_condfull': p_ab_full,
            'r_BA_condfull': r_ba_full, 'p_BA_condfull': p_ba_full,
            'r_AB_condsep': r_ab_sep, 'p_AB_condsep': p_ab_sep,
            'r_BA_condsep': r_ba_sep, 'p_BA_condsep': p_ba_sep,
            'sepset_AB_global': [int(units_idx[w]) for w in S_ab],
            'sepset_BA_global': [int(units_idx[w]) for w in S_ba],
            'max_marg': max(abs(r_ab_m), abs(r_ba_m)),
            'max_condC': max(abs(r_ab_c), abs(r_ba_c)),
            'max_condsep': max(abs(r_ab_sep), abs(r_ba_sep)),
        })

# 2c fraction explained away by conditioning on shared parent C
marg_dep_triples = [t for t in triples if t['max_marg'] > 0.1]
expl_away = [t for t in marg_dep_triples if t['max_condC'] < 0.1]
step2c = {
    'n_common_source_triples': len(triples),
    'n_marg_dep(>0.1)': len(marg_dep_triples),
    'n_explained_away_by_C(<0.1)': len(expl_away),
    'frac_explained_away_by_C': round(len(expl_away) / len(marg_dep_triples), 4) if marg_dep_triples else None,
    'mean_max_marg': round(float(np.mean([t['max_marg'] for t in marg_dep_triples])), 4) if marg_dep_triples else None,
    'mean_max_condC': round(float(np.mean([t['max_condC'] for t in marg_dep_triples])), 4) if marg_dep_triples else None,
}
print('  ', step2c)

# STEP 3: best example motif = high marginal, near-zero conditional given C
ranked = sorted(marg_dep_triples,
                key=lambda t: (t['max_marg'] - t['max_condC']), reverse=True)
best = ranked[:5]
print('\n  Top corrected motifs (A<-C->B, A,B non-adjacent):')
for t in best:
    print(f"   C={t['C']} A={t['A']} B={t['B']} | "
          f"marg AB={t['r_AB_marg']:.3f}/BA={t['r_BA_marg']:.3f} "
          f"condC AB={t['r_AB_condC']:.3f}/BA={t['r_BA_condC']:.3f} "
          f"condfull AB={t['r_AB_condfull']:.3f}/BA={t['r_BA_condfull']:.3f} "
          f"condsep AB={t['r_AB_condsep']:.3f}(S={t['sepset_AB_global']})/"
          f"BA={t['r_BA_condsep']:.3f}(S={t['sepset_BA_global']})")

# ----------------------------------------------------------------------------
# STEP 4: motif-type enumeration + internal CI-pattern verification.
# Generalizes the two hand-picked examples: for every connected 3-node (and
# 4-node multi-parent) motif type present in the inferred graph, verify the
# expected internal conditional-independence pattern using the algorithm's OWN
# recorded separating sets. Self-consistent by construction (interpretability).
# ----------------------------------------------------------------------------
print('\n[STEP 4] motif-type CI-pattern verification...')

def adjacent(a, b):
    return adj[a, b] == 1 or adj[b, a] == 1

def pair_eval(x, y):
    """Return dict for ordered lagged relation x(t-1)->y(t).
    Non-edge: conditioned on the EXACT recorded separating set (p by construction >alpha).
    Edge:     conditioned on target y's OTHER inferred parents (should stay dependent)."""
    marg, pm = lagged_pcorr(x, y, [])
    S = sepset.get((x, y), None)
    if S is None:              # (x,y) is a retained edge
        cond, pc = cond_on_parents(x, y)
        S_used = tuple(w for w in parents_of[y] if w != x)
        is_edge = True
    else:                      # (x,y) was removed; S is the recorded sepset
        cond, pc = lagged_pcorr(x, y, S)
        S_used = S
        is_edge = False
    return {'marg': marg, 'cond': cond, 'p': pc, 'S': S_used, 'is_edge': is_edge}

def indep_pair(a, b):
    """Undirected CI summary for a predicted-independent, non-adjacent pair {a,b}.
    Both ordered directions were removed; each has its own recorded sepset."""
    ab = pair_eval(a, b); ba = pair_eval(b, a)
    return {
        'marg_ab': ab['marg'], 'cond_ab': ab['cond'], 'p_ab': ab['p'], 'S_ab': ab['S'],
        'marg_ba': ba['marg'], 'cond_ba': ba['cond'], 'p_ba': ba['p'], 'S_ba': ba['S'],
        'max_marg': max(abs(ab['marg']), abs(ba['marg'])),
        'max_cond': max(abs(ab['cond']), abs(ba['cond'])),
        'indep_bypval': (ab['p'] > ALPHA) and (ba['p'] > ALPHA),
        'S_union': tuple(sorted(set(ab['S']) | set(ba['S']))),
    }

def edge_stays(x, y):
    """True if retained edge x->y remains dependent (p<=alpha) given y's other parents."""
    ev = pair_eval(x, y)
    return ev['p'] <= ALPHA, ev

def gid(w):
    return int(units_idx[w])

def repr_pack(inst):
    """serialize one motif instance to global ids for the figure."""
    return inst

MARG_DEP = 0.1   # only count instances where the predicted-indep pair is actually
                 # marginally dependent (otherwise there is nothing to explain away)

# ---- 1. FORK  A <- C -> B  (A,B non-adjacent) -----------------------------
fork = []
for C in range(p):
    ch = [w for w in range(p) if adj[C, w]]
    for A, B in combinations(ch, 2):
        if adjacent(A, B):
            continue
        ip = indep_pair(A, B)
        if ip['max_marg'] <= MARG_DEP:
            continue
        sep_has_sep = (C in ip['S_ab']) or (C in ip['S_ba'])
        eCA, _ = edge_stays(C, A); eCB, _ = edge_stays(C, B)
        fork.append({'type': 'fork', 'C': gid(C), 'A': gid(A), 'B': gid(B),
                     'indep_pair': (gid(A), gid(B)), 'separator': gid(C),
                     'sep_contains_separator': bool(sep_has_sep),
                     'indep_bypval': bool(ip['indep_bypval']),
                     'edges_stay': bool(eCA and eCB),
                     'marg_indep': round(ip['max_marg'], 3),
                     'cond_indep': round(ip['max_cond'], 3),
                     'S_used_global': [gid(w) for w in ip['S_union']],
                     'pairs': {  # for figure: 3 pairs marginal/conditional |r|
                         'A_B': [round(ip['max_marg'], 3), round(ip['max_cond'], 3)],
                         'C_A': [round(abs(pair_eval(C, A)['marg']), 3), round(abs(pair_eval(C, A)['cond']), 3)],
                         'C_B': [round(abs(pair_eval(C, B)['marg']), 3), round(abs(pair_eval(C, B)['cond']), 3)],
                     }})

# ---- 2. CHAIN  A -> B -> C  (A,C non-adjacent) ----------------------------
chain = []
for B in range(p):
    pa = [w for w in range(p) if adj[w, B]]
    ch = [w for w in range(p) if adj[B, w]]
    for A in pa:
        for C in ch:
            if A == C or adjacent(A, C):
                continue
            ip = indep_pair(A, C)
            if ip['max_marg'] <= MARG_DEP:
                continue
            sep_has_sep = (B in ip['S_ab']) or (B in ip['S_ba'])
            eAB, _ = edge_stays(A, B); eBC, _ = edge_stays(B, C)
            chain.append({'type': 'chain', 'A': gid(A), 'B': gid(B), 'C': gid(C),
                          'indep_pair': (gid(A), gid(C)), 'separator': gid(B),
                          'sep_contains_separator': bool(sep_has_sep),
                          'indep_bypval': bool(ip['indep_bypval']),
                          'edges_stay': bool(eAB and eBC),
                          'marg_indep': round(ip['max_marg'], 3),
                          'cond_indep': round(ip['max_cond'], 3),
                          'S_used_global': [gid(w) for w in ip['S_union']],
                          'pairs': {
                              'A_C': [round(ip['max_marg'], 3), round(ip['max_cond'], 3)],
                              'A_B': [round(abs(pair_eval(A, B)['marg']), 3), round(abs(pair_eval(A, B)['cond']), 3)],
                              'B_C': [round(abs(pair_eval(B, C)['marg']), 3), round(abs(pair_eval(B, C)['cond']), 3)],
                          }})

# ---- 3. MULTI-PARENT PATH  A <- P1 (-) P2 -> B  (P1,P2 adj; A,B non-adj) ---
path = []
adj_pairs = [(i, j) for i in range(p) for j in range(i + 1, p) if adjacent(i, j)]
for P1, P2 in adj_pairs:
    chA = [w for w in range(p) if adj[P1, w] and w not in (P1, P2)]
    chB = [w for w in range(p) if adj[P2, w] and w not in (P1, P2)]
    for A in chA:
        for B in chB:
            if A == B or adjacent(A, B):
                continue
            ip = indep_pair(A, B)
            if ip['max_marg'] <= MARG_DEP:
                continue
            sep_has_both = ({P1, P2} <= set(ip['S_ab'])) or ({P1, P2} <= set(ip['S_ba']))
            sep_has_either = len(({P1, P2}) & (set(ip['S_ab']) | set(ip['S_ba']))) > 0
            eP1A, _ = edge_stays(P1, A); eP2B, _ = edge_stays(P2, B)
            path.append({'type': 'path', 'P1': gid(P1), 'P2': gid(P2), 'A': gid(A), 'B': gid(B),
                         'indep_pair': (gid(A), gid(B)), 'separators': (gid(P1), gid(P2)),
                         'sep_contains_both': bool(sep_has_both),
                         'sep_contains_either': bool(sep_has_either),
                         'indep_bypval': bool(ip['indep_bypval']),
                         'edges_stay': bool(eP1A and eP2B),
                         'marg_indep': round(ip['max_marg'], 3),
                         'cond_indep': round(ip['max_cond'], 3),
                         'S_used_global': [gid(w) for w in ip['S_union']],
                         'pairs': {
                             'A_B': [round(ip['max_marg'], 3), round(ip['max_cond'], 3)],
                             'P1_A': [round(abs(pair_eval(P1, A)['marg']), 3), round(abs(pair_eval(P1, A)['cond']), 3)],
                             'P2_B': [round(abs(pair_eval(P2, B)['marg']), 3), round(abs(pair_eval(P2, B)['cond']), 3)],
                         }})

# ---- 4. COLLIDER  A -> C <- B  (A,B non-adjacent) -------------------------
# Reverse pattern: A,B expected MARGINALLY independent -> recorded sepset EMPTY
# (order-0 removal); algorithm does NOT condition on the common child C.
# Berkson check: conditioning ON C should INDUCE dependence.
collider = []
for C in range(p):
    pa = [w for w in range(p) if adj[w, C]]
    for A, B in combinations(pa, 2):
        if adjacent(A, B):
            continue
        S_ab = sepset.get((A, B), None)
        S_ba = sepset.get((B, A), None)
        sep_empty = (S_ab is not None and len(S_ab) == 0) and (S_ba is not None and len(S_ba) == 0)
        rab_m, _ = lagged_pcorr(A, B, [])
        rba_m, _ = lagged_pcorr(B, A, [])
        # induce dependence by conditioning on the collider C
        rab_c, pab_c = lagged_pcorr(A, B, [C])
        rba_c, pba_c = lagged_pcorr(B, A, [C])
        max_marg = max(abs(rab_m), abs(rba_m))
        max_condC = max(abs(rab_c), abs(rba_c))
        eAC, _ = edge_stays(A, C); eBC, _ = edge_stays(B, C)
        collider.append({'type': 'collider', 'A': gid(A), 'B': gid(B), 'C': gid(C),
                         'nonadj_pair': (gid(A), gid(B)), 'collider_node': gid(C),
                         'sepset_empty': bool(sep_empty),
                         'edges_stay': bool(eAC and eBC),
                         'max_marg': round(max_marg, 3),
                         'max_condC': round(max_condC, 3),
                         'condC_increases_dep': bool(max_condC > max_marg + 0.02)})

def typesummary(insts, sep_key, need_marg_dep=True):
    if len(insts) == 0:
        return {'n': 0}
    n = len(insts)
    s = {
        'n': n,
        'frac_indep_pair_CI_bypval': round(np.mean([i['indep_bypval'] for i in insts]), 4),
        'frac_sepset_contains_expected_separator': round(np.mean([i[sep_key] for i in insts]), 4),
        'frac_direct_edges_stay_dependent': round(np.mean([i['edges_stay'] for i in insts]), 4),
        'mean_marg_indep_pair': round(float(np.mean([i['marg_indep'] for i in insts])), 4),
        'mean_cond_indep_pair': round(float(np.mean([i['cond_indep'] for i in insts])), 4),
        'mean_drop_indep_pair': round(float(np.mean([i['marg_indep'] - i['cond_indep'] for i in insts])), 4),
    }
    # coordinator's internal CI pattern: predicted pair explained away (CI under
    # recorded sepset) AND all direct edges retained. Does NOT require the sepset to
    # equal the specific textbook separator node.
    s['frac_correct_CI_pattern'] = round(
        np.mean([i['indep_bypval'] and i['edges_stay'] for i in insts]), 4)
    # stricter: additionally the recorded MINIMAL sepset contains the expected
    # separator node(s). Lower because PC records minimal sets and many other nodes
    # can also d-separate a pair in a recurrent graph.
    s['frac_correct_pattern_strict'] = round(
        np.mean([i['indep_bypval'] and i[sep_key] and i['edges_stay'] for i in insts]), 4)
    return s

fork_sum = typesummary(fork, 'sep_contains_separator')
chain_sum = typesummary(chain, 'sep_contains_separator')
path_sum = typesummary(path, 'sep_contains_both')

# collider summary (reverse logic)
if len(collider) > 0:
    coll_sum = {
        'n': len(collider),
        'frac_sepset_empty_order0': round(np.mean([c['sepset_empty'] for c in collider]), 4),
        'frac_direct_edges_stay_dependent': round(np.mean([c['edges_stay'] for c in collider]), 4),
        'mean_marg_nonadj_pair': round(float(np.mean([c['max_marg'] for c in collider])), 4),
        'mean_condC_nonadj_pair': round(float(np.mean([c['max_condC'] for c in collider])), 4),
        'frac_condC_induces_dependence': round(np.mean([c['condC_increases_dep'] for c in collider]), 4),
        'frac_correct_collider_pattern': round(
            np.mean([c['sepset_empty'] and c['edges_stay'] for c in collider]), 4),
        'note': 'Collider parents are removed at order 0 (empty sepset); the algorithm does NOT '
                'condition on the common child C. Conditioning ON C tends to INDUCE dependence (Berkson).',
    }
else:
    coll_sum = {'n': 0}

def pick_repr(insts, sep_key, exclude_pairs=frozenset()):
    """clean representative: strict-correct, largest drop, indep pair not already used."""
    good = [i for i in insts if i.get('indep_bypval') and i.get(sep_key) and i.get('edges_stay')
            and frozenset(i['indep_pair']) not in exclude_pairs]
    if not good:
        good = [i for i in insts if frozenset(i['indep_pair']) not in exclude_pairs] or insts
    return sorted(good, key=lambda i: (i['marg_indep'] - i['cond_indep']), reverse=True)[0] if good else None

used = set()
repr_fork = pick_repr(fork, 'sep_contains_separator', used)
if repr_fork: used.add(frozenset(repr_fork['indep_pair']))
repr_chain = pick_repr(chain, 'sep_contains_separator', used)
if repr_chain: used.add(frozenset(repr_chain['indep_pair']))
repr_path = pick_repr(path, 'sep_contains_both', used)
repr_coll = None
coll_good = [c for c in collider if c['sepset_empty'] and c['edges_stay']]
if coll_good:
    repr_coll = sorted(coll_good, key=lambda c: c['max_condC'] - c['max_marg'], reverse=True)[0]
elif collider:
    repr_coll = sorted(collider, key=lambda c: c['max_condC'] - c['max_marg'], reverse=True)[0]

print(f"  FORK      n={fork_sum['n']:5d}  CI-pattern={fork_sum.get('frac_correct_CI_pattern')}  "
      f"strict={fork_sum.get('frac_correct_pattern_strict')}  "
      f"sepset-has-C={fork_sum.get('frac_sepset_contains_expected_separator')}  "
      f"edges-stay={fork_sum.get('frac_direct_edges_stay_dependent')}  "
      f"drop={fork_sum.get('mean_drop_indep_pair')}")
print(f"  CHAIN     n={chain_sum['n']:5d}  CI-pattern={chain_sum.get('frac_correct_CI_pattern')}  "
      f"strict={chain_sum.get('frac_correct_pattern_strict')}  "
      f"sepset-has-B={chain_sum.get('frac_sepset_contains_expected_separator')}  "
      f"edges-stay={chain_sum.get('frac_direct_edges_stay_dependent')}  "
      f"drop={chain_sum.get('mean_drop_indep_pair')}")
print(f"  PATH      n={path_sum['n']:5d}  CI-pattern={path_sum.get('frac_correct_CI_pattern')}  "
      f"strict={path_sum.get('frac_correct_pattern_strict')}  "
      f"sepset-has-P1P2={path_sum.get('frac_sepset_contains_expected_separator')}  "
      f"edges-stay={path_sum.get('frac_direct_edges_stay_dependent')}  "
      f"drop={path_sum.get('mean_drop_indep_pair')}")
print(f"  COLLIDER  n={coll_sum['n']:5d}  sepset-empty(order0)={coll_sum.get('frac_sepset_empty_order0')}  "
      f"edges-stay={coll_sum.get('frac_direct_edges_stay_dependent')}  "
      f"condC-induces-dep={coll_sum.get('frac_condC_induces_dependence')}")
for nm, r in [('fork', repr_fork), ('chain', repr_chain), ('path', repr_path), ('collider', repr_coll)]:
    print(f"    repr {nm}: {r}")

step4 = {
    'description': "Motif-structured, by-construction demonstration of the algorithm's own CI "
                   "decisions. For each motif type the predicted-independent pair is verified "
                   "conditionally independent under its RECORDED separating set, while the direct-edge "
                   "pairs remain dependent. Shows the inferred graph's CI logic is coherent motif-by-motif.",
    'fork': {'summary': fork_sum, 'representative': repr_fork,
             'definition': 'A <- C -> B; A,B non-adjacent; expect A _||_ B | {C}, edges C->A,C->B stay'},
    'chain': {'summary': chain_sum, 'representative': repr_chain,
              'definition': 'A -> B -> C; A,C non-adjacent; expect A _||_ C | {B}, edges A->B,B->C stay'},
    'path': {'summary': path_sum, 'representative': repr_path,
             'definition': 'A <- P1 (-) P2 -> B; P1,P2 adjacent, A,B non-adjacent; expect A _||_ B | {P1,P2}'},
    'collider': {'summary': coll_sum, 'representative': repr_coll,
                 'definition': 'A -> C <- B; A,B non-adjacent; REVERSE: A,B marg. independent, empty '
                               'sepset, algorithm does NOT condition on C (conditioning on C induces dep.)'},
}

# ----------------------------------------------------------------------------
# save
# ----------------------------------------------------------------------------
out = {
    'meta': {
        'session': SESS, 'stimulus': STIM, 'block_idx': IDX, 'data_idx': IDX,
        'method': 'fast_cits_pcorr: scalable lagged CITS PC-stable skeleton',
        'ci_test': 'Fisher-z partial correlation on cits data_transform windowed samples (lag-1)',
        'edge_type': 'LAGGED source slot2 (t-1) -> target slot3 (t); no contemporaneous',
        'conditioning': "non-edges: EXACT separating set the PC-skeleton recorded at edge removal; "
                        "edges: target's other inferred lagged parents (survived, not removed)",
        'framing': "Demonstrates the algorithm's OWN conditional-independence criterion at population "
                   "scale. Self-consistent by construction: each removed edge is shown independent under "
                   "the very set that removed it. Intent is interpretability, not an out-of-sample test.",
        'tau': TAU, 'alpha': ALPHA, 'max_k': MAX_K,
        'n_active_neurons': p, 'N_windows': N,
        'deterministic': True, 'note': 'PC-stable, no randomness; reproducible for fixed data',
    },
    'step1_graph': {
        'n_edges': n_edges, 'n_ordered_pairs': n_possible,
        'density': round(density, 4),
        'n_nonedges': n_possible - n_edges,
    },
    'step2_population': {
        'n_ordered_pairs': len(records),
        'n_edges': len(edges), 'n_nonedges': len(nonedges),
        **step2,
    },
    'step2c_common_source': step2c,
    'step3_example_motifs': best,
    'step4_motif_types': step4,
    'active_unit_global_ids': [int(x) for x in units_idx],
}

os.makedirs(OUT_DIR, exist_ok=True)
with open(f'{OUT_DIR}/motif_population_v2.json', 'w') as f:
    json.dump(out, f, indent=1)
np.save(f'{OUT_DIR}/motif_population_v2_adjacency.npy', adj)
print(f'\nSaved {OUT_DIR}/motif_population_v2.json and _adjacency.npy')

# stash records for the plotting step
with open(f'{OUT_DIR}/motif_population_v2_records.pkl', 'wb') as f:
    pkl.dump({'records': records, 'triples': triples, 'adj': adj,
              'units_idx': units_idx, 'best': best, 'out': out}, f)
print('Done STEP1-3.')
