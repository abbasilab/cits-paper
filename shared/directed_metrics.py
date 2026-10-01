"""
directed_metrics.py

Directed CS metric for the FC simulation benchmark.

Extends compute_metrics() from simulation_benchmark_fc_methods_v3.py with
directed-edge scoring.  The undirected metrics are returned unchanged for
backward compatibility.

Directed scoring
================
We evaluate all p*(p-1) ordered pairs (i, j) with i != j.

For each ordered pair:

  GT_dir[i, j] = 1  if any true directed edge i -> j exists in any of the
                     four weighted GT matrices (lag-only, contemp-only,
                     both-lag, both-contemp).
  pred[i, j]        predicted adjacency (as returned by benchmark methods).

Two variants are provided:

  STRICT  -- no partial credit for undirected predictions.
    GT=1, pred_ij=1, pred_ji=0  ->  TP  += 1
    GT=1, pred_ij=1, pred_ji=1  ->  TP  += 0,  FN += 1   (direction missed)
    GT=1, pred_ij=0             ->  FN  += 1
    GT=0, pred_ij=1             ->  FP  += 1

  LENIENT -- half credit for undirected predictions (CPDAG outputs).
    GT=1, pred_ij=1, pred_ji=0  ->  TP  += 1,  FN += 0
    GT=1, pred_ij=1, pred_ji=1  ->  TP  += 0.5, FN += 0.5
    GT=1, pred_ij=0             ->  FN  += 1
    GT=0, pred_ij=1             ->  FP  += 1

FPR denominator: n_neg = p*(p-1) - n_true_directed_edges
  (number of ordered pairs with no true directed edge).

Note on bidirectionality: none of the built-in simulation models generate
bidirectional true edges (GT[i,j]=1 AND GT[j,i]=1 for the same pair).  The
metric handles this case correctly if it arises: each direction is scored
independently because we iterate all ordered pairs.

CS = TPR - FPR  (Youden J statistic, range [-1, 1], ideal = 1.0)

Usage
-----
    from directed_metrics import compute_directed_metrics

    metrics = compute_directed_metrics(
        pred,
        gt_lag_w, gt_contemp_w,
        gt_both_lag_w, gt_both_contemp_w,
        gt_lag_uw, gt_contemp_uw, gt_both_uw,   # for backward-compat undirected
    )

    # directed keys:
    #   directed_tp_strict, directed_fp_strict, directed_fn_strict
    #   directed_F1_strict, directed_TPR_strict, directed_FPR_strict
    #   directed_CS_strict
    #   ... (same with _lenient suffix)

    # undirected keys (identical to compute_metrics output):
    #   edges_tp, edges_fp, edges_fn, edges_F1, edges_precision, edges_recall
    #   edges_SHD, contemp_only_tp, lag_only_tp, ...
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_gt_directed(gt_lag_w: np.ndarray,
                       gt_contemp_w: np.ndarray,
                       gt_both_lag_w: np.ndarray,
                       gt_both_contemp_w: np.ndarray) -> np.ndarray:
    """Return binary (p, p) directed GT: GT[i,j]=1 if any i->j edge exists."""
    combined = (np.abs(gt_lag_w)
                + np.abs(gt_contemp_w)
                + np.abs(gt_both_lag_w)
                + np.abs(gt_both_contemp_w))
    gt_dir = (combined != 0).astype(int)
    np.fill_diagonal(gt_dir, 0)   # exclude self-edges
    return gt_dir


def _directed_counts(pred: np.ndarray,
                     gt_dir: np.ndarray,
                     lenient: bool) -> tuple[float, float, float]:
    """
    Return (tp, fp, fn) for directed scoring.

    Parameters
    ----------
    pred    : (p, p) binary/int adjacency from a benchmark method.
    gt_dir  : (p, p) binary directed GT (from _build_gt_directed).
    lenient : if True, undirected predictions get 0.5 credit when GT is 1.
    """
    p = pred.shape[0]
    tp = fp = fn = 0.0

    for i in range(p):
        for j in range(p):
            if i == j:
                continue
            gt_ij   = int(gt_dir[i, j] != 0)
            pred_ij = int(pred[i, j] != 0)
            pred_ji = int(pred[j, i] != 0)

            if gt_ij == 1:
                if pred_ij == 1 and pred_ji == 0:
                    # Correctly directed
                    tp += 1.0
                elif pred_ij == 1 and pred_ji == 1:
                    # Undirected prediction; partial credit only in lenient mode
                    if lenient:
                        tp += 0.5
                        fn += 0.5
                    else:
                        fn += 1.0
                else:
                    # pred_ij == 0: missed entirely
                    fn += 1.0
            else:
                # GT[i,j] = 0
                if pred_ij == 1:
                    # Spurious directed prediction
                    fp += 1.0
                # pred_ij == 0: true negative; not counted

    return tp, fp, fn


def _directed_rates(tp: float, fp: float, fn: float,
                    n_neg: int) -> dict:
    """Compute TPR, FPR, CS, F1 from directed counts."""
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 1.0
    fpr = fp / n_neg     if n_neg      > 0 else 0.0
    cs  = tpr - fpr

    prec = tp / (tp + fp)   if (tp + fp) > 0 else 1.0
    rec  = tpr
    f1   = (2 * prec * rec / (prec + rec)
            if (prec + rec) > 0 else 0.0)

    return dict(TPR=round(tpr, 6), FPR=round(fpr, 6),
                CS=round(cs, 6), F1=round(f1, 6),
                tp=round(tp, 4), fp=round(fp, 4), fn=round(fn, 4))


# ---------------------------------------------------------------------------
# Undirected metrics (copied from simulation_benchmark_fc_methods_v3.py
# so this module is self-contained; returns identical output)
# ---------------------------------------------------------------------------

def _compute_undirected_metrics(pred: np.ndarray,
                                gt_lag_uw: np.ndarray,
                                gt_contemp_uw: np.ndarray,
                                gt_both_uw: np.ndarray) -> dict:
    p = pred.shape[0]

    gt_full = ((gt_lag_uw + gt_lag_uw.T
                + gt_contemp_uw + gt_contemp_uw.T
                + gt_both_uw + gt_both_uw.T) > 0).astype(int)
    np.fill_diagonal(gt_full, 0)

    pred_skel = ((pred + pred.T) > 0).astype(int)
    np.fill_diagonal(pred_skel, 0)

    tp = fp = fn = 0
    for i in range(p):
        for j in range(i + 1, p):
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

    contemp_only_tp = lag_only_tp = both_tp_lenient = both_tp_strict = 0
    spurious_fp = 0

    for i in range(p):
        for j in range(i + 1, p):
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
                    both_tp_strict  += 1
            else:
                if has_edge:
                    spurious_fp += 1

    return {
        'edges_tp'        : tp,
        'edges_fp'        : fp,
        'edges_fn'        : fn,
        'edges_precision' : round(precision, 6),
        'edges_recall'    : round(recall, 6),
        'edges_F1'        : round(f1, 6),
        'edges_SHD'       : shd,
        'contemp_only_tp' : contemp_only_tp,
        'contemp_only_fp' : spurious_fp,
        'lag_only_tp'     : lag_only_tp,
        'lag_only_fp'     : 0,
        'both_tp_lenient' : both_tp_lenient,
        'both_tp_strict'  : both_tp_strict,
        'both_fp'         : 0,
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_directed_metrics(pred: np.ndarray,
                             gt_lag_w: np.ndarray,
                             gt_contemp_w: np.ndarray,
                             gt_both_lag_w: np.ndarray,
                             gt_both_contemp_w: np.ndarray,
                             gt_lag_uw: np.ndarray | None = None,
                             gt_contemp_uw: np.ndarray | None = None,
                             gt_both_uw: np.ndarray | None = None) -> dict:
    """
    Compute both directed and undirected metrics for a single prediction.

    Parameters
    ----------
    pred              : (p, p) int/binary adjacency matrix from a method.
                        pred[i, j] = 1 means the method predicts a directed
                        edge i -> j.
    gt_lag_w          : (p, p) weighted directed GT for lag-only edges.
    gt_contemp_w      : (p, p) weighted directed GT for contemp-only edges.
    gt_both_lag_w     : (p, p) weighted directed GT, lag component of
                        'both'-type edges.
    gt_both_contemp_w : (p, p) weighted directed GT, contemp component of
                        'both'-type edges.
    gt_lag_uw         : (p, p) binary undirected GT for lag-only edges.
                        If None, derived from gt_lag_w != 0.
    gt_contemp_uw     : similar; derived from gt_contemp_w if None.
    gt_both_uw        : similar; derived from gt_both_lag_w if None.

    Returns
    -------
    dict with keys:
      Undirected (backward-compatible):
        edges_tp, edges_fp, edges_fn, edges_precision, edges_recall,
        edges_F1, edges_SHD, contemp_only_tp, lag_only_tp, ...

      Directed (strict and lenient variants):
        directed_tp_strict, directed_fp_strict, directed_fn_strict,
        directed_TPR_strict, directed_FPR_strict,
        directed_CS_strict, directed_F1_strict,
        directed_tp_lenient, directed_fp_lenient, directed_fn_lenient,
        directed_TPR_lenient, directed_FPR_lenient,
        directed_CS_lenient, directed_F1_lenient,
        n_true_directed, n_neg_directed

    Scoring variants
    ----------------
    STRICT  : A predicted undirected edge (pred[i,j]=1 AND pred[j,i]=1) when
              GT has i->j scores as FN (no credit for direction).
    LENIENT : The same case scores as TP=0.5, FN=0.5 (partial credit; useful
              for CPDAG-outputting methods that cannot orient all edges).

    Use STRICT for comparing methods that always output directed graphs
    (Granger, TPC, CITS).  Use LENIENT when comparing against methods whose
    CPDAG output is not fully oriented (PC, VerB with Meek).
    """
    pred = np.asarray(pred, dtype=int)
    p    = pred.shape[0]

    # Build undirected GT matrices if not supplied
    if gt_lag_uw     is None:
        gt_lag_uw     = (gt_lag_w     != 0).astype(int)
    if gt_contemp_uw is None:
        gt_contemp_uw = (gt_contemp_w != 0).astype(int)
    if gt_both_uw    is None:
        gt_both_uw    = (gt_both_lag_w != 0).astype(int)

    # ---- Undirected metrics (backward-compatible) ----------------------------
    undirected = _compute_undirected_metrics(
        pred, gt_lag_uw, gt_contemp_uw, gt_both_uw)

    # ---- Directed GT ---------------------------------------------------------
    gt_dir = _build_gt_directed(
        gt_lag_w, gt_contemp_w, gt_both_lag_w, gt_both_contemp_w)

    n_true_directed = int(gt_dir.sum())
    n_neg           = p * (p - 1) - n_true_directed   # ordered pairs w/o true edge

    # ---- Directed scoring ----------------------------------------------------
    tp_s, fp_s, fn_s = _directed_counts(pred, gt_dir, lenient=False)
    tp_l, fp_l, fn_l = _directed_counts(pred, gt_dir, lenient=True)

    rates_s = _directed_rates(tp_s, fp_s, fn_s, n_neg)
    rates_l = _directed_rates(tp_l, fp_l, fn_l, n_neg)

    directed = {
        'n_true_directed'      : n_true_directed,
        'n_neg_directed'       : n_neg,
        # strict
        'directed_tp_strict'   : rates_s['tp'],
        'directed_fp_strict'   : rates_s['fp'],
        'directed_fn_strict'   : rates_s['fn'],
        'directed_TPR_strict'  : rates_s['TPR'],
        'directed_FPR_strict'  : rates_s['FPR'],
        'directed_CS_strict'   : rates_s['CS'],
        'directed_F1_strict'   : rates_s['F1'],
        # lenient
        'directed_tp_lenient'  : rates_l['tp'],
        'directed_fp_lenient'  : rates_l['fp'],
        'directed_fn_lenient'  : rates_l['fn'],
        'directed_TPR_lenient' : rates_l['TPR'],
        'directed_FPR_lenient' : rates_l['FPR'],
        'directed_CS_lenient'  : rates_l['CS'],
        'directed_F1_lenient'  : rates_l['F1'],
    }

    return {**undirected, **directed}
