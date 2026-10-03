"""
simulation_benchmark_fc_methods_v3.py

Replaces the +contemp variants with biologically-calibrated +mixed variants.

Edge-type mix (mimicking calcium imaging at ~7 Hz):
  50% contemp-only  -- synaptic effect fully within-frame
  30% both          -- mixed-timescale (lag + contemp, weight split equally)
  20% lag-only      -- pure slow / sub-threshold

Per-model edge assignments (full ground truth, NOT additions to base):

  lingauss1+mixed    (3 edges):
    contemp-only:  0->2 (w=2), 1->2 (w=1)
    both:          2->3  lag_w=1, contemp_w=1
    lag-only:      none

  lingauss2+mixed    (4 edges, DIAMOND-PRESERVING):
    lag-only:      0->1 (w=2)   [preserves diamond confounding via X_0(t-1)]
    both:          0->2  lag_w=1, contemp_w=1  (split base w=2)
    contemp-only:  1->3 (w=1)
    lag-only:      2->3 (w=1)

  nonlinnongauss1+mixed  (3 edges):
    contemp-only:  0->2 sin(w=4), 1->2 sin(w=-3)
    both:          2->3 sin  lag_w=1.5, contemp_w=1.5
    lag-only:      none

  nonlinnongauss2+mixed  (4 edges, DIAMOND-PRESERVING):
    lag-only:      0->1 lin(w=4)  [preserves diamond confounding via X_0(t-1)]
    both:          0->2 sin  lag_w=1.5, contemp_w=1.5  (split base w=3)
    contemp-only:  1->3 log(w=8)
    lag-only:      2->3 log(w=9)

  ctrnn+mixed   (3 edges):
    contemp-only:  0->2 (w=100), 1->2 (w=100)
    both:          2->3  lag_coupling=50, contemp_coupling=50
    lag-only:      none

Correctness definition for "both" edges (lenient):
  A method recovers the "both" edge if EITHER the lag or the contemp half is
  detected in its output skeleton.

Methods:
  1. Granger           -- pairwise lag-1 F-test
  2. TPC               -- thresholded partial correlation (rolling window)
  3. PC                -- vanilla PC on flat contemporaneous data
  4. CITS              -- cits.methods.cits_full (lag-1 window only)
  5. VerB              -- CITS + PC-contemp (lag-0 only) + Meek + LSCM refit
  6. CITSplus_v1       -- CITS + PC-contemp (lag-0 only, no Meek)
  7. CITSplus_v2       -- CITS + PC-contemp (tau+1 stacked window, no Meek)
  8. CITSplus_v_latest -- CITS-optimal (2tau+1 window) + PC-contemp (2tau+1 chi),
                          no Meek; 33% more chi samples than v2

Output:
  figures/2026-06-02_simulation_benchmark_v3_mixed/simulation_results_v3.csv
  figures/2026-06-02_simulation_benchmark_v3_mixed/method_comparison_v3.{png,pdf}
  figures/2026-06-02_simulation_benchmark_v3_mixed/per_edge_type_v3.{png,pdf}
  figures/2026-06-02_simulation_benchmark_v3_mixed/f1_vs_regime_v3.{png,pdf}

Usage:
  python simulation_benchmark_fc_methods_v3.py [--workers N] [--seeds N] [--smoke]
"""

from __future__ import annotations

import os
import sys
import time
import argparse
import warnings
import itertools
import multiprocessing as mp
from multiprocessing import Pool

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

warnings.filterwarnings('ignore')

# -- Paths ---------------------------------------------------------------------
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:   # shared/ (paths.py)
    sys.path.insert(0, _THIS_DIR)
from paths import OUT_ROOT as _OUT_ROOT
# outputs of this module's own __main__ benchmark (was arousal_paper_overleaf/figures/2026-06-02_simulation_benchmark_v3_mixed)
OUT_DIR    = os.path.join(_OUT_ROOT, 'shared',
                          '2026-06-02_simulation_benchmark_v3_mixed')
OUT_CSV    = os.path.join(OUT_DIR, 'simulation_results_v3.csv')

for _p in [_THIS_DIR]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# -- Analysis-dir private modules ----------------------------------------------
from _pc_orientation    import orient_v_structures, apply_meek_rules
from _lscm_refit        import lscm_refit_cpdag
from _pc_raw_v2         import pc_skeleton_raw_v2
from cits_plus_v_latest import run_cits_plus_v_latest as _run_cits_plus_v_latest

# -- CITS methods (lag-windowed) -----------------------------------------------
from cits import methods as cits_methods
from cits.simulate_timeseries import simulate

N_NEURONS = 4
ALPHA = 0.05
LAG   = 1

# ==============================================================================
#  Simulation model definitions
#
#  Returns:
#    X                 (p, T) float
#    gt_lag_uw         (p, p) int   -- lag-ONLY edges
#    gt_lag_w          (p, p) float
#    gt_contemp_uw     (p, p) int   -- contemp-ONLY edges
#    gt_contemp_w      (p, p) float
#    gt_both_uw        (p, p) int   -- edges with BOTH lag + contemp components
#    gt_both_lag_w     (p, p) float
#    gt_both_contemp_w (p, p) float
#
#  For base (lag-only) models: gt_contemp_* and gt_both_* are all zeros.
# ==============================================================================

def simulate_extended(model_name: str, noise: float, T: int, seed: int):
    rng = np.random.default_rng(seed)
    p = N_NEURONS
    is_mixed   = model_name.endswith('+mixed')
    base_model = model_name.replace('+mixed', '')

    # ctrnn_mod is structurally identical to ctrnn but with a faster
    # self-decay (alpha ~ 0.55 instead of ~0.85) chosen to match the lag-1
    # autocorrelation observed in real MICrONS calcium traces. We implement
    # it by branching on a flag inside the ctrnn block (see below) so the
    # rest of the simulator (mixed/lag-only logic) is unchanged.
    is_mod = (base_model == 'ctrnn_mod')
    if is_mod:
        base_model = 'ctrnn'

    smspikes = np.zeros((p, T))

    gt_lag_w          = np.zeros((p, p))
    gt_contemp_w      = np.zeros((p, p))
    gt_both_lag_w     = np.zeros((p, p))
    gt_both_contemp_w = np.zeros((p, p))

    # -- lingauss1 -------------------------------------------------------------
    if base_model == 'lingauss1':
        for i in range(p):
            smspikes[i, 0] = rng.normal(scale=noise)

        if not is_mixed:
            # Base: 3 lag-only edges  0->2 (w=2), 1->2 (w=-1), 2->3 (w=2)
            for t in range(1, T):
                smspikes[0, t] = rng.normal(scale=noise) + 1.0
                smspikes[1, t] = rng.normal(scale=noise) - 1.0
                smspikes[2, t] = (2.0 * smspikes[0, t-1]
                                  - 1.0 * smspikes[1, t-1]
                                  + rng.normal(scale=noise))
                smspikes[3, t] = 2.0 * smspikes[2, t-1] + rng.normal(scale=noise)

            gt_lag_w[0, 2] = 2.0
            gt_lag_w[1, 2] = -1.0
            gt_lag_w[2, 3] = 2.0

        else:
            # Mixed:
            #   contemp-only: 0->2 (w=2), 1->2 (w=1)
            #   both:         2->3  lag_w=1, contemp_w=1
            #   lag-only:     none
            for t in range(1, T):
                smspikes[0, t] = rng.normal(scale=noise) + 1.0
                smspikes[1, t] = rng.normal(scale=noise) - 1.0
                smspikes[2, t] = (2.0 * smspikes[0, t]
                                  + 1.0 * smspikes[1, t]
                                  + rng.normal(scale=noise))
                smspikes[3, t] = (1.0 * smspikes[2, t-1]
                                  + 1.0 * smspikes[2, t]
                                  + rng.normal(scale=noise))

            gt_contemp_w[0, 2] = 2.0
            gt_contemp_w[1, 2] = 1.0
            gt_both_lag_w[2, 3]     = 1.0
            gt_both_contemp_w[2, 3] = 1.0

    # -- lingauss2 -------------------------------------------------------------
    elif base_model == 'lingauss2':
        for i in range(p):
            smspikes[i, 0] = rng.normal(scale=noise)

        if not is_mixed:
            # Base: 4 lag-only edges
            for t in range(1, T):
                smspikes[0, t] = rng.normal(scale=noise) + 1.0
                smspikes[1, t] = (-1.0 + 2.0 * smspikes[0, t-1]
                                  + rng.normal(scale=noise))
                smspikes[2, t] = 2.0 * smspikes[0, t-1] + rng.normal(scale=noise)
                smspikes[3, t] = (smspikes[1, t-1] + smspikes[2, t-1]
                                  + rng.normal(scale=noise))

            gt_lag_w[0, 1] = 2.0
            gt_lag_w[0, 2] = 2.0
            gt_lag_w[1, 3] = 1.0
            gt_lag_w[2, 3] = 1.0

        else:
            # Mixed (corrected to PRESERVE diamond-DAG confounding):
            #   lag-only:     0->1 (w=2)  [keeps X_0(t-1) as common past parent of X_1]
            #   both:         0->2  lag_w=1, contemp_w=1  (split base w=2)
            #   contemp-only: 1->3 (w=1)
            #   lag-only:     2->3 (w=1)
            # Diamond test: X_1(t) depends only on X_0(t-1); X_2(t) depends on
            # X_0(t-1) AND X_0(t). The X_1(t)-X_2(t) correlation through
            # X_0(t-1) is separated only with τ+1 conditioning (CITS+ v2).
            for t in range(1, T):
                smspikes[0, t] = rng.normal(scale=noise) + 1.0
                smspikes[1, t] = (-1.0 + 2.0 * smspikes[0, t-1]
                                  + rng.normal(scale=noise))
                smspikes[2, t] = (1.0 * smspikes[0, t-1]
                                  + 1.0 * smspikes[0, t]
                                  + rng.normal(scale=noise))
                smspikes[3, t] = (1.0 * smspikes[1, t]
                                  + 1.0 * smspikes[2, t-1]
                                  + rng.normal(scale=noise))

            gt_lag_w[0, 1] = 2.0
            gt_both_lag_w[0, 2]     = 1.0
            gt_both_contemp_w[0, 2] = 1.0
            gt_contemp_w[1, 3] = 1.0
            gt_lag_w[2, 3] = 1.0

    # -- nonlinnongauss1 -------------------------------------------------------
    elif base_model == 'nonlinnongauss1':
        for i in range(p):
            smspikes[i, 0] = rng.random()

        if not is_mixed:
            # Base: 3 lag-only edges  0->2 sin(4), 1->2 sin(-3), 2->3 sin(3)
            for t in range(1, T):
                smspikes[0, t] = rng.random() * noise
                smspikes[1, t] = rng.random() * noise
                smspikes[2, t] = (4.0 * np.sin(smspikes[0, t-1])
                                  - 3.0 * np.sin(smspikes[1, t-1])
                                  + rng.random() * noise)
                smspikes[3, t] = (3.0 * np.sin(smspikes[2, t-1])
                                  + rng.random() * noise)

            gt_lag_w[0, 2] =  1.0
            gt_lag_w[1, 2] = -1.0
            gt_lag_w[2, 3] =  1.0

        else:
            # Mixed:
            #   contemp-only: 0->2 sin(4), 1->2 sin(-3)
            #   both:         2->3 sin  lag_w=1.5, contemp_w=1.5
            #   lag-only:     none
            for t in range(1, T):
                smspikes[0, t] = rng.random() * noise
                smspikes[1, t] = rng.random() * noise
                smspikes[2, t] = (4.0 * np.sin(smspikes[0, t])
                                  - 3.0 * np.sin(smspikes[1, t])
                                  + rng.random() * noise)
                smspikes[3, t] = (1.5 * np.sin(smspikes[2, t-1])
                                  + 1.5 * np.sin(smspikes[2, t])
                                  + rng.random() * noise)

            gt_contemp_w[0, 2] =  1.0
            gt_contemp_w[1, 2] = -1.0
            gt_both_lag_w[2, 3]     = 1.0
            gt_both_contemp_w[2, 3] = 1.0

    # -- nonlinnongauss2 -------------------------------------------------------
    elif base_model == 'nonlinnongauss2':
        for i in range(p):
            smspikes[i, 0] = rng.normal(scale=noise)

        if not is_mixed:
            # Base: 4 lag-only edges
            for t in range(1, T):
                smspikes[0, t] = rng.normal(scale=noise)
                smspikes[1, t] = 4.0 * smspikes[0, t-1] + rng.normal(scale=noise)
                smspikes[2, t] = (3.0 * np.sin(smspikes[0, t-1])
                                  + rng.normal(scale=noise))
                v1 = np.abs(smspikes[1, t-1]) + 1e-6
                v2 = np.abs(smspikes[2, t-1]) + 1e-6
                smspikes[3, t] = (8.0 * np.log(v1) + 9.0 * np.log(v2)
                                  + rng.normal(scale=noise))

            gt_lag_w[0, 1] = 1.0
            gt_lag_w[0, 2] = 1.0
            gt_lag_w[1, 3] = 1.0
            gt_lag_w[2, 3] = 1.0

        else:
            # Mixed (corrected to PRESERVE diamond-DAG confounding):
            #   lag-only:     0->1 lin(w=4)  [keeps diamond through X_0(t-1)]
            #   both:         0->2 sin  lag_w=1.5, contemp_w=1.5  (split base w=3)
            #   contemp-only: 1->3 log(w=8)
            #   lag-only:     2->3 log(w=9)
            for t in range(1, T):
                smspikes[0, t] = rng.normal(scale=noise)
                smspikes[1, t] = 4.0 * smspikes[0, t-1] + rng.normal(scale=noise)
                smspikes[2, t] = (1.5 * np.sin(smspikes[0, t-1])
                                  + 1.5 * np.sin(smspikes[0, t])
                                  + rng.normal(scale=noise))
                v1_now = np.abs(smspikes[1, t])   + 1e-6
                v2_lag = np.abs(smspikes[2, t-1]) + 1e-6
                smspikes[3, t] = (8.0 * np.log(v1_now)
                                  + 9.0 * np.log(v2_lag)
                                  + rng.normal(scale=noise))

            gt_lag_w[0, 1] = 1.0
            gt_both_lag_w[0, 2]     = 1.0
            gt_both_contemp_w[0, 2] = 1.0
            gt_contemp_w[1, 3] = 1.0
            gt_lag_w[2, 3] = 1.0

    # -- ctrnn / ctrnn_mod -----------------------------------------------------
    # ctrnn (default tau_ct=10.0) has mean lag-1 autocorr rho ~ 0.85.
    # ctrnn_mod (tau_ct=5.0) is calibrated to mean lag-1 autocorr rho ~ 0.55,
    # matching the autocorrelation observed in real MICrONS calcium traces.
    # Everything else (couplings, tanh nonlinearity, 4 neurons, mixed logic)
    # is unchanged.
    elif base_model == 'ctrnn':
        e      = np.exp(1)
        tau_ct = 5.0 if is_mod else 10.0

        if not is_mixed:
            # Base: CTRNN with full lag coupling 0->2, 1->2, 2->3 (w=100 each)
            w_c = np.zeros((p, p))
            w_c[0, 2] = 100.0
            w_c[1, 2] = 100.0
            w_c[2, 3] = 100.0
            u = np.zeros((p, T))
            for n in range(T - 1):
                for i in range(p):
                    In = rng.normal(1.0, noise)
                    u[i, n+1] = (u[i, n]
                                 - (e * u[i, n]) / tau_ct
                                 + e * np.sum(w_c[:, i] * np.tanh(u[:, n])) / tau_ct
                                 + e * In / tau_ct)
            smspikes = u

            # Self-dynamics (CTRNN decay term) counted as lag self-edges
            for i in range(p):
                gt_lag_w[i, i] = 1.0
            gt_lag_w[0, 2] = 1.0
            gt_lag_w[1, 2] = 1.0
            gt_lag_w[2, 3] = 1.0

        else:
            # Mixed:
            #   contemp-only: 0->2 (w=100), 1->2 (w=100)  fast tau
            #   both:         2->3  lag_coupling=50, contemp_coupling=50
            #   lag-only:     none  (beyond self-dynamics)
            #
            # Implementation:
            #   Lag step: CTRNN update with only 2->3 lag coupling (w=50);
            #             self-dynamics are implicit.
            #   Contemp step: add instantaneous 0->2, 1->2, and 2->3 contemp half.
            tau_fast = 2.0
            w_lag = np.zeros((p, p))
            w_lag[2, 3] = 50.0   # lag half of "both" edge

            u = np.zeros((p, T))
            for n in range(T - 1):
                u_next = np.zeros(p)
                for i in range(p):
                    In = rng.normal(1.0, noise)
                    lag_input = np.sum(
                        [w_lag[k, i] * np.tanh(u[k, n]) for k in range(p)])
                    u_next[i] = (u[i, n]
                                 - (e * u[i, n]) / tau_ct
                                 + e * lag_input / tau_ct
                                 + e * In / tau_ct)
                # Contemp coupling added instantaneously
                u_next[2] += (e / tau_fast) * np.tanh(u_next[0])  # 0->2 contemp
                u_next[2] += (e / tau_fast) * np.tanh(u_next[1])  # 1->2 contemp
                u_next[3] += (e / tau_fast) * np.tanh(u_next[2])  # 2->3 contemp half
                u[:, n+1] = u_next
            smspikes = u

            # GT: contemp-only 0->2, 1->2; "both" 2->3; lag-only: none
            gt_contemp_w[0, 2] = 1.0
            gt_contemp_w[1, 2] = 1.0
            gt_both_lag_w[2, 3]     = 1.0
            gt_both_contemp_w[2, 3] = 1.0
            # Self-dynamics (CTRNN decay) are lag
            for i in range(p):
                gt_lag_w[i, i] = 1.0

    else:
        raise ValueError(f"Unknown base model: {base_model}")

    X = smspikes

    gt_lag_uw     = (gt_lag_w         != 0).astype(int)
    gt_contemp_uw = (gt_contemp_w     != 0).astype(int)
    gt_both_uw    = (gt_both_lag_w    != 0).astype(int)

    return (X,
            gt_lag_uw, gt_lag_w,
            gt_contemp_uw, gt_contemp_w,
            gt_both_uw, gt_both_lag_w, gt_both_contemp_w)


# ==============================================================================
#  Self-contained PC skeleton (Fisher-z, CPU-only)
# ==============================================================================

def _fisher_r_crit(N: int, k: int, alpha: float = 0.05) -> float:
    df = N - k - 3
    if df <= 0:
        return 1.0
    t_crit = scipy_stats.t.ppf(1.0 - alpha / 2.0, df)
    r_crit = t_crit / np.sqrt(df + t_crit**2)
    return float(r_crit)


def _partial_corr(X_data: np.ndarray, i: int, j: int, cond_set: tuple) -> float:
    if len(cond_set) == 0:
        r = np.corrcoef(X_data[i], X_data[j])[0, 1]
        return float(r)
    S = np.array(list(cond_set), dtype=int)
    Z = X_data[S].T
    N = Z.shape[0]
    ones = np.ones((N, 1))
    Z_aug = np.hstack([ones, Z])
    try:
        B, _, _, _ = np.linalg.lstsq(Z_aug, X_data[i], rcond=None)
        res_i = X_data[i] - Z_aug @ B
        B2, _, _, _ = np.linalg.lstsq(Z_aug, X_data[j], rcond=None)
        res_j = X_data[j] - Z_aug @ B2
    except np.linalg.LinAlgError:
        return 0.0
    if res_i.std() < 1e-12 or res_j.std() < 1e-12:
        return 0.0
    r = np.corrcoef(res_i, res_j)[0, 1]
    return float(r)


def pc_skeleton_cpu(X_data: np.ndarray, alpha: float = 0.05):
    """Plain PC skeleton on (p, N) data matrix."""
    p, N = X_data.shape
    A = np.ones((p, p), dtype=int)
    np.fill_diagonal(A, 0)
    sep_sets = {}

    l = 0
    while True:
        edges_to_test = []
        for i in range(p):
            for j in range(p):
                if i == j or A[i, j] == 0:
                    continue
                nbrs_i = [k for k in range(p) if k != j and A[i, k] != 0]
                if len(nbrs_i) >= l:
                    edges_to_test.append((i, j))

        if not edges_to_test:
            break

        tested_pairs = set()
        for (i, j) in edges_to_test:
            if A[i, j] == 0:
                continue
            pair_key = (min(i, j), max(i, j))
            if pair_key in tested_pairs:
                continue
            tested_pairs.add(pair_key)

            nbrs_i = [k for k in range(p) if k != j and A[i, k] != 0]
            r_crit = _fisher_r_crit(N, l, alpha)

            found_sep = False
            for S_list in itertools.combinations(nbrs_i, l):
                S = tuple(S_list)
                r = _partial_corr(X_data, i, j, S)
                if abs(r) < r_crit:
                    A[i, j] = 0
                    A[j, i] = 0
                    sep_sets[(i, j)] = S
                    sep_sets[(j, i)] = S
                    found_sep = True
                    break

            if not found_sep:
                nbrs_j = [k for k in range(p) if k != i and A[j, k] != 0]
                for S_list in itertools.combinations(nbrs_j, l):
                    S = tuple(S_list)
                    r = _partial_corr(X_data, i, j, S)
                    if abs(r) < r_crit:
                        A[i, j] = 0
                        A[j, i] = 0
                        sep_sets[(i, j)] = S
                        sep_sets[(j, i)] = S
                        break

        l += 1
        if l >= p - 1:
            break

    return A, sep_sets


# ==============================================================================
#  Method implementations (identical to v2)
# ==============================================================================

def run_granger(X: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    p, T = X.shape
    A = np.zeros((p, p), dtype=int)
    for j in range(p):
        y = X[j, 1:]
        ones = np.ones((T-1, 1))
        X_restr_aug = np.hstack([ones, X[j, :-1].reshape(-1, 1)])
        B_r, _, _, _ = np.linalg.lstsq(X_restr_aug, y, rcond=None)
        res_r = y - X_restr_aug @ B_r
        ss_r  = (res_r**2).sum()
        for i in range(p):
            if i == j:
                continue
            X_full = np.hstack([ones, X[j, :-1].reshape(-1, 1),
                                       X[i, :-1].reshape(-1, 1)])
            B_f, _, _, _ = np.linalg.lstsq(X_full, y, rcond=None)
            res_f = y - X_full @ B_f
            ss_f  = (res_f**2).sum()
            df_f  = T - 1 - 3
            if df_f <= 0 or ss_f <= 0:
                continue
            F_stat = ((ss_r - ss_f) / 1.0) / (ss_f / df_f)
            if F_stat < 0:
                continue
            p_val = 1.0 - scipy_stats.f.cdf(F_stat, 1, df_f)
            if p_val < alpha:
                A[i, j] = 1
    return A


def run_tpc(X: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    p, T = X.shape
    chi = np.vstack([X[:, :-1], X[:, 1:]])
    A_full, sep_sets_full = pc_skeleton_cpu(chi, alpha=alpha)
    G_full = orient_v_structures(A_full, sep_sets_full)
    A_lag = G_full[:p, p:]        # lag block: t-1 -> t (directed by time)
    G_c   = G_full[p:, p:]        # contemp block: directed by v-structures
    A_out = np.zeros((p, p), dtype=int)
    for i in range(p):
        for j in range(p):
            if i == j:
                continue
            if A_lag[i, j] or G_c[i, j]:
                A_out[i, j] = 1
    return A_out


def run_pc(X: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    p, T = X.shape
    A, sep_sets = pc_skeleton_cpu(X, alpha=alpha)
    G = orient_v_structures(A, sep_sets)
    np.fill_diagonal(G, 0)
    return G


def run_cits(X: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    p, T = X.shape
    try:
        adj = cits_methods.cits_full(X, LAG, alpha)
    except Exception as e:
        warnings.warn(f"CITS failed: {e}")
        adj = np.zeros((p, p), dtype=int)
    adj_bin = (np.asarray(adj) != 0).astype(int)
    np.fill_diagonal(adj_bin, 0)
    return adj_bin


def run_verB(X: np.ndarray, alpha: float = 0.05, meek: bool = True) -> np.ndarray:
    """Ver B (with Meek) or CITSplus_v1 (no Meek, lag-0 only contemp)."""
    p, T = X.shape
    try:
        cits_adj = cits_methods.cits_full(X, LAG, alpha)
    except Exception as e:
        warnings.warn(f"CITS in VerB failed: {e}")
        cits_adj = np.zeros((p, p), dtype=int)
    cits_bin = (np.asarray(cits_adj) != 0).astype(int)
    np.fill_diagonal(cits_bin, 0)

    stride = max(1, 4)
    X_contemp = X[:, ::stride]
    if X_contemp.shape[1] < p + 3:
        X_contemp = X

    A_pc, sep_sets_pc = pc_skeleton_cpu(X_contemp, alpha=alpha)
    G_cpdag = orient_v_structures(A_pc, sep_sets_pc)
    if meek:
        G_cpdag = apply_meek_rules(G_cpdag)

    union_skel = np.zeros((p, p), dtype=int)
    for i in range(p):
        for j in range(p):
            if i == j:
                continue
            if cits_bin[i, j]:
                union_skel[i, j] = 1
            if G_cpdag[i, j]:
                union_skel[i, j] = 1
    np.fill_diagonal(union_skel, 0)
    return union_skel


def run_cits_plus_v2(X: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """CITS+ v2: CITS-lagged + PC-contemp with tau+1 stacked window, no Meek."""
    p, T = X.shape

    try:
        cits_adj = cits_methods.cits_full(X, LAG, alpha)
    except Exception as e:
        warnings.warn(f"CITS in CITSplus_v2 failed: {e}")
        cits_adj = np.zeros((p, p), dtype=int)
    cits_bin = (np.asarray(cits_adj) != 0).astype(int)
    np.fill_diagonal(cits_bin, 0)

    X_T = X.T  # (T, p)
    A_c2, r_mat, sep_sets_v2, inactive = pc_skeleton_raw_v2(
        X_T, alpha=alpha, tau=1, backend='python', verbose=False)

    sep_sets_lag0_only = {}
    for (i, j), S in sep_sets_v2.items():
        sep_sets_lag0_only[(i, j)] = tuple(k for k in S if k >= 0)

    G_c2 = orient_v_structures(A_c2, sep_sets_lag0_only)

    union_skel = np.zeros((p, p), dtype=int)
    for i in range(p):
        for j in range(p):
            if i == j:
                continue
            if cits_bin[i, j]:
                union_skel[i, j] = 1
            if G_c2[i, j]:
                union_skel[i, j] = 1
    np.fill_diagonal(union_skel, 0)
    return union_skel


def run_cits_plus_v_latest_bench(X: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """CITS+ v_latest: 2tau+1-optimal CITS-lag + PC-contemp, no Meek.

    Uses cits_full_optimal (N = T//(2tau+1), +33% samples vs v2) for the
    lagged step and pc_skeleton_raw_v_latest for the contemporaneous step.
    Both steps use the same optimal 2tau+1 chi.
    """
    return _run_cits_plus_v_latest(X, alpha=alpha, tau=LAG, backend='python')


# Method registry
METHOD_FNS = {
    'Granger'          : lambda X: run_granger(X, ALPHA),
    'TPC'              : lambda X: run_tpc(X, ALPHA),
    'PC'               : lambda X: run_pc(X, ALPHA),
    'CITS'             : lambda X: run_cits(X, ALPHA),
    'VerB'             : lambda X: run_verB(X, ALPHA, meek=True),
    'CITSplus_v1'      : lambda X: run_verB(X, ALPHA, meek=False),
    'CITSplus_v2'      : lambda X: run_cits_plus_v2(X, ALPHA),
    'CITSplus_v_latest': lambda X: run_cits_plus_v_latest_bench(X, ALPHA),
}


# ==============================================================================
#  Metrics: 3-way edge-type breakdown
# ==============================================================================

def compute_metrics(pred: np.ndarray,
                    gt_lag_uw: np.ndarray,
                    gt_contemp_uw: np.ndarray,
                    gt_both_uw: np.ndarray) -> dict:
    """
    gt_lag_uw     -- lag-ONLY edges (directed; symmetrised internally)
    gt_contemp_uw -- contemp-ONLY edges
    gt_both_uw    -- edges with BOTH lag + contemp components

    For "both" edges the lenient criterion applies: recovered if EITHER
    lag or contemp component is detected (i.e. edge appears in pred skeleton).
    Strict criterion uses the same skeleton check since most methods return
    undirected skeletons.
    """
    p = pred.shape[0]

    gt_full = ((gt_lag_uw + gt_lag_uw.T
                + gt_contemp_uw + gt_contemp_uw.T
                + gt_both_uw + gt_both_uw.T) > 0).astype(int)
    np.fill_diagonal(gt_full, 0)

    pred_skel = ((pred + pred.T) > 0).astype(int)
    np.fill_diagonal(pred_skel, 0)

    # Overall skeleton metrics
    tp = fp = fn = 0
    for i in range(p):
        for j in range(i+1, p):
            has_edge  = pred_skel[i, j] > 0
            true_edge = gt_full[i, j] > 0
            if has_edge and true_edge:
                tp += 1
            elif has_edge and not true_edge:
                fp += 1
            elif not has_edge and true_edge:
                fn += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 1.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)
    shd = fp + fn

    gt_lag_skel     = ((gt_lag_uw     + gt_lag_uw.T)     > 0).astype(int)
    gt_contemp_skel = ((gt_contemp_uw + gt_contemp_uw.T) > 0).astype(int)
    gt_both_skel    = ((gt_both_uw    + gt_both_uw.T)    > 0).astype(int)
    np.fill_diagonal(gt_lag_skel,     0)
    np.fill_diagonal(gt_contemp_skel, 0)
    np.fill_diagonal(gt_both_skel,    0)

    contemp_only_tp = contemp_only_fp = 0
    lag_only_tp     = lag_only_fp     = 0
    both_tp_lenient = both_tp_strict  = both_fp = 0
    spurious_fp = 0

    for i in range(p):
        for j in range(i+1, p):
            has_edge        = pred_skel[i, j] > 0
            is_contemp_only = (gt_contemp_skel[i, j] > 0
                               and gt_both_skel[i, j] == 0
                               and gt_lag_skel[i, j] == 0)
            is_lag_only     = (gt_lag_skel[i, j] > 0
                               and gt_both_skel[i, j] == 0
                               and gt_contemp_skel[i, j] == 0)
            is_both         = gt_both_skel[i, j] > 0

            if is_contemp_only:
                if has_edge:
                    contemp_only_tp += 1
            elif is_lag_only:
                if has_edge:
                    lag_only_tp += 1
            elif is_both:
                if has_edge:
                    both_tp_lenient += 1
                    both_tp_strict  += 1   # same: skeleton check
            else:
                # True negative or spurious FP
                if has_edge:
                    spurious_fp += 1

    # Assign contemp_only_fp as spurious FP count (generic)
    contemp_only_fp = spurious_fp

    return {
        'edges_tp'            : tp,
        'edges_fp'            : fp,
        'edges_fn'            : fn,
        'edges_precision'     : round(precision, 6),
        'edges_recall'        : round(recall, 6),
        'edges_F1'            : round(f1, 6),
        'edges_SHD'           : shd,
        'contemp_only_tp'     : contemp_only_tp,
        'contemp_only_fp'     : contemp_only_fp,
        'lag_only_tp'         : lag_only_tp,
        'lag_only_fp'         : lag_only_fp,
        'both_tp_lenient'     : both_tp_lenient,
        'both_tp_strict'      : both_tp_strict,
        'both_fp'             : both_fp,
    }


# ==============================================================================
#  Single-run worker
# ==============================================================================

def _run_one(args):
    """Worker: runs all 8 methods on one (model, T, noise, seed) config."""
    model_name, T, noise, seed = args
    rows = []
    is_mixed = int(model_name.endswith('+mixed'))

    try:
        (X,
         gt_lag_uw, gt_lag_w,
         gt_contemp_uw, gt_contemp_w,
         gt_both_uw, gt_both_lag_w, gt_both_contemp_w
        ) = simulate_extended(model_name, noise, T, seed)
    except Exception as e:
        warnings.warn(
            f"Sim failed {model_name} T={T} noise={noise} seed={seed}: {e}")
        return []

    for method_name, method_fn in METHOD_FNS.items():
        t0 = time.perf_counter()
        try:
            pred = method_fn(X)
        except Exception as e:
            warnings.warn(
                f"Method {method_name} failed on {model_name} seed={seed}: {e}")
            pred = np.zeros((N_NEURONS, N_NEURONS), dtype=int)
        runtime = time.perf_counter() - t0

        metrics = compute_metrics(pred, gt_lag_uw, gt_contemp_uw, gt_both_uw)
        row = {
            'model'      : model_name,
            'is_mixed'   : is_mixed,
            'T'          : T,
            'noise'      : noise,
            'seed'       : seed,
            'method'     : method_name,
            'runtime_sec': round(runtime, 4),
        }
        row.update(metrics)
        rows.append(row)

    return rows


# ==============================================================================
#  Main
# ==============================================================================

MODEL_NAMES = [
    'lingauss1',       'lingauss1+mixed',
    'lingauss2',       'lingauss2+mixed',
    'nonlinnongauss1', 'nonlinnongauss1+mixed',
    'nonlinnongauss2', 'nonlinnongauss2+mixed',
    'ctrnn',           'ctrnn+mixed',
]

T_VALUES     = [1000, 5000]
NOISE_VALUES = [1.0, 0.5]
N_SEEDS      = 50


def main():
    parser = argparse.ArgumentParser(
        description='FC simulation benchmark v3 (mixed edge types)')
    parser.add_argument('--workers', type=int, default=min(mp.cpu_count(), 24))
    parser.add_argument('--seeds', type=int, default=N_SEEDS)
    parser.add_argument('--smoke', action='store_true',
                        help='Smoke test: lingauss1+mixed and lingauss2+mixed, '
                             'T=1000, noise=1.0, 5 seeds')
    parser.add_argument('--check-autocorr', action='store_true',
                        help='Print mean lag-1 autocorrelation for ctrnn '
                             'and ctrnn_mod paradigms and exit.')
    args = parser.parse_args()

    if args.check_autocorr:
        for mname in ['ctrnn', 'ctrnn_mod', 'ctrnn+mixed', 'ctrnn_mod+mixed']:
            rhos_all = []
            for seed in range(5):
                X, *_ = simulate_extended(mname, 1.0, 5000, seed)
                for i in range(X.shape[0]):
                    rhos_all.append(
                        np.corrcoef(X[i, :-1], X[i, 1:])[0, 1])
            mean_rho = float(np.mean(rhos_all))
            print(f"  {mname:20s}: mean rho_lag1 = {mean_rho:.3f} "
                  f"(N={len(rhos_all)} neuron-seeds)", flush=True)
        return

    os.makedirs(OUT_DIR, exist_ok=True)

    if args.smoke:
        models  = ['lingauss1+mixed', 'lingauss2+mixed']
        t_vals  = [1000]
        n_vals  = [1.0]
        n_seeds = 5
    else:
        models  = MODEL_NAMES
        t_vals  = T_VALUES
        n_vals  = NOISE_VALUES
        n_seeds = args.seeds

    tasks = []
    for model, T, noise in itertools.product(models, t_vals, n_vals):
        for seed in range(n_seeds):
            tasks.append((model, T, noise, seed))

    total     = len(tasks)
    n_methods = len(METHOD_FNS)
    print(f"[benchmark_v3] {total} configs x {n_methods} methods "
          f"= {total * n_methods} evaluations", flush=True)
    print(f"[benchmark_v3] workers={args.workers}", flush=True)

    t_start  = time.time()
    all_rows = []

    if args.workers > 1:
        with Pool(processes=args.workers) as pool:
            for i, result in enumerate(
                    pool.imap_unordered(_run_one, tasks, chunksize=4)):
                all_rows.extend(result)
                if (i + 1) % max(1, total // 20) == 0:
                    pct     = 100.0 * (i+1) / total
                    elapsed = time.time() - t_start
                    eta     = elapsed / (i+1) * (total - i - 1)
                    print(f"  {i+1}/{total} ({pct:.0f}%) "
                          f"elapsed={elapsed:.0f}s ETA={eta:.0f}s", flush=True)
    else:
        for i, task in enumerate(tasks):
            result = _run_one(task)
            all_rows.extend(result)
            if (i + 1) % max(1, total // 20) == 0:
                pct     = 100.0 * (i+1) / total
                elapsed = time.time() - t_start
                print(f"  {i+1}/{total} ({pct:.0f}%) elapsed={elapsed:.0f}s",
                      flush=True)

    elapsed_total = time.time() - t_start
    print(f"[benchmark_v3] Done in {elapsed_total:.1f}s. "
          f"Rows: {len(all_rows)}", flush=True)

    df = pd.DataFrame(all_rows)
    df.to_csv(OUT_CSV, index=False)
    print(f"[benchmark_v3] Saved -> {OUT_CSV}", flush=True)

    try:
        _make_figures(df)
    except Exception as e:
        print(f"[benchmark_v3] Figure generation failed: {e}", flush=True)
        import traceback; traceback.print_exc()


# ==============================================================================
#  Figures
# ==============================================================================

def _make_figures(df: pd.DataFrame):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    METHODS = ['Granger', 'TPC', 'PC', 'CITS', 'VerB', 'CITSplus_v1',
               'CITSplus_v2', 'CITSplus_v_latest']
    BASE_MODELS = ['lingauss1', 'lingauss2', 'nonlinnongauss1',
                   'nonlinnongauss2', 'ctrnn']
    COLORS = {
        'Granger'          : '#1f77b4',
        'TPC'              : '#ff7f0e',
        'PC'               : '#2ca02c',
        'CITS'             : '#d62728',
        'VerB'             : '#9467bd',
        'CITSplus_v1'      : '#8c564b',
        'CITSplus_v2'      : '#e377c2',
        'CITSplus_v_latest': '#17becf',
    }

    # -- Figure 1: F1 per method per model, base vs +mixed --------------------
    fig, axes = plt.subplots(2, 5, figsize=(20, 7), sharey=True)
    fig.suptitle(
        'F1 Score by Method and Model  (T=1000, noise=1.0, N=50 seeds)\n'
        'Top row: lag-only base models | Bottom row: +mixed variants',
        fontsize=11)

    for col, bm in enumerate(BASE_MODELS):
        for row_idx, suffix in enumerate(['', '+mixed']):
            ax = axes[row_idx, col]
            model_key = bm + suffix
            sub = df[(df['model'] == model_key)
                     & (df['T'] == 1000)
                     & (df['noise'] == 1.0)]

            if sub.empty:
                ax.text(0.5, 0.5, 'no data', ha='center', va='center',
                        transform=ax.transAxes)
                continue

            means, sems = [], []
            for m in METHODS:
                vals = sub.loc[sub['method'] == m, 'edges_F1'].values
                means.append(np.mean(vals) if len(vals) > 0 else 0)
                sems.append(np.std(vals) / np.sqrt(max(1, len(vals))))

            xs     = np.arange(len(METHODS))
            colors = [COLORS[m] for m in METHODS]
            bars   = ax.bar(xs, means, yerr=sems, capsize=3,
                            color=colors, alpha=0.85, width=0.7)
            for highlight_m in ('CITSplus_v2', 'CITSplus_v_latest'):
                h_idx = METHODS.index(highlight_m)
                bars[h_idx].set_linewidth(2.0)
                bars[h_idx].set_edgecolor('black')

            ax.set_xticks(xs)
            ax.set_xticklabels(METHODS, rotation=50, ha='right', fontsize=6.0)
            ax.set_ylim(0, 1.05)
            ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
            if col == 0:
                label = 'Lag-only' if suffix == '' else '+Mixed'
                ax.set_ylabel(f'{label}\nF1')
            if row_idx == 0:
                ax.set_title(bm.replace('nonlin', 'nl.'), fontsize=9)

    plt.tight_layout()
    for ext in ('png', 'pdf'):
        path = os.path.join(OUT_DIR, f'method_comparison_v3.{ext}')
        fig.savefig(path, dpi=150, bbox_inches='tight')
        print(f"[figures] Saved {path}", flush=True)
    plt.close(fig)

    # -- Figure 2: Per-edge-type recovery (+mixed variants) -------------------
    fig2, axes2 = plt.subplots(3, 5, figsize=(20, 10))
    fig2.suptitle(
        'Per-Edge-Type Recovery (+mixed variants, T=1000, noise=1.0)\n'
        'Row 1: Contemp-only TP | Row 2: Both-edge TP (lenient) | Row 3: Lag-only TP',
        fontsize=11)

    row_metrics = ['contemp_only_tp', 'both_tp_lenient', 'lag_only_tp']
    row_labels  = ['Contemp-only TP', 'Both-edge TP\n(lenient)', 'Lag-only TP']

    for col, bm in enumerate(BASE_MODELS):
        model_m = bm + '+mixed'
        sub = df[(df['model'] == model_m)
                 & (df['T'] == 1000)
                 & (df['noise'] == 1.0)].copy()

        for row_idx, (metric, ylabel) in enumerate(
                zip(row_metrics, row_labels)):
            ax = axes2[row_idx, col]

            if sub.empty:
                ax.text(0.5, 0.5, 'no data', ha='center', va='center',
                        transform=ax.transAxes)
                continue

            means, sems = [], []
            for m in METHODS:
                vals = sub.loc[sub['method'] == m, metric].values.astype(float)
                means.append(np.mean(vals))
                sems.append(np.std(vals) / np.sqrt(max(1, len(vals))))

            xs     = np.arange(len(METHODS))
            colors = [COLORS[m] for m in METHODS]
            bars   = ax.bar(xs, means, yerr=sems, capsize=3,
                            color=colors, alpha=0.85, width=0.7)
            for highlight_m in ('CITSplus_v2', 'CITSplus_v_latest'):
                h_idx = METHODS.index(highlight_m)
                bars[h_idx].set_linewidth(2.0)
                bars[h_idx].set_edgecolor('black')

            top = max(means) * 1.25 + 0.05 if any(m > 0 for m in means) else 1.25
            ax.set_xticks(xs)
            ax.set_xticklabels(METHODS, rotation=50, ha='right', fontsize=6.0)
            ax.set_ylim(0, max(1.25, top))
            if col == 0:
                ax.set_ylabel(ylabel, fontsize=8)
            if row_idx == 0:
                ax.set_title(bm.replace('nonlin', 'nl.'), fontsize=9)

    patches = [mpatches.Patch(color=COLORS[m], label=m) for m in METHODS]
    fig2.legend(handles=patches, loc='lower center', ncol=8, fontsize=8,
                bbox_to_anchor=(0.5, -0.04))
    plt.tight_layout()
    for ext in ('png', 'pdf'):
        path = os.path.join(OUT_DIR, f'per_edge_type_v3.{ext}')
        fig2.savefig(path, dpi=150, bbox_inches='tight')
        print(f"[figures] Saved {path}", flush=True)
    plt.close(fig2)

    # -- Figure 3: F1 vs T and noise ------------------------------------------
    t_vals_in_data = sorted(df['T'].unique())
    n_vals_in_data = sorted(df['noise'].unique())

    fig3, axes3 = plt.subplots(1, 2, figsize=(14, 4))
    fig3.suptitle('F1 vs Sample Size and SNR\n(averaged over all models, v3)',
                  fontsize=11)

    for ax_idx, (split_col, split_vals, xlabel) in enumerate([
        ('T',     t_vals_in_data, 'Timepoints T'),
        ('noise', n_vals_in_data, 'Noise level'),
    ]):
        ax = axes3[ax_idx]
        for m in METHODS:
            ys, xs_plot = [], []
            for sv in split_vals:
                sub = df[(df['method'] == m) & (df[split_col] == sv)]
                if sub.empty:
                    continue
                ys.append(sub['edges_F1'].mean())
                xs_plot.append(sv)
            if ys:
                is_lead = m in ('CITSplus_v2', 'CITSplus_v_latest')
                lw = 2.5 if is_lead else 1.5
                ax.plot(xs_plot, ys, marker='o', color=COLORS[m], label=m,
                        linewidth=lw, zorder=3 if is_lead else 2)
        ax.set_xlabel(xlabel)
        ax.set_ylabel('Mean F1 (all models)')
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=7, ncol=2)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    for ext in ('png', 'pdf'):
        path = os.path.join(OUT_DIR, f'f1_vs_regime_v3.{ext}')
        fig3.savefig(path, dpi=150, bbox_inches='tight')
        print(f"[figures] Saved {path}", flush=True)
    plt.close(fig3)

    # -- Top-line summary printed to stdout -----------------------------------
    print("\n===== TOP-LINE SUMMARY (v3 benchmark) =====", flush=True)

    ref_T     = 1000
    ref_noise = 1.0
    sub_main  = df[(df['T'] == ref_T) & (df['noise'] == ref_noise)]
    if sub_main.empty:
        ref_T     = df['T'].iloc[0]
        ref_noise = df['noise'].iloc[0]
        sub_main  = df[(df['T'] == ref_T) & (df['noise'] == ref_noise)]

    print(f"\nMean F1 by method (all models, T={ref_T}, noise={ref_noise}):",
          flush=True)
    for m in METHODS:
        vals = sub_main[sub_main['method'] == m]['edges_F1']
        if len(vals) > 0:
            print(f"  {m:14s}: {vals.mean():.3f} +/- {vals.std():.3f}",
                  flush=True)

    print(f"\nMean F1 on +mixed models only (T={ref_T}, noise={ref_noise}):",
          flush=True)
    sub_m = sub_main[sub_main['is_mixed'] == 1]
    for m in METHODS:
        vals = sub_m[sub_m['method'] == m]['edges_F1']
        if len(vals) > 0:
            print(f"  {m:14s}: {vals.mean():.3f} +/- {vals.std():.3f}",
                  flush=True)

    print(f"\nPer-edge-type recovery (+mixed, T={ref_T}, noise={ref_noise}):",
          flush=True)
    for metric in ['contemp_only_tp', 'both_tp_lenient', 'lag_only_tp']:
        print(f"  [{metric}]", flush=True)
        for m in METHODS:
            vals = sub_m[sub_m['method'] == m][metric]
            if len(vals) > 0:
                print(f"    {m:14s}: {vals.mean():.3f}", flush=True)


if __name__ == '__main__':
    main()
