"""
lpcmci_baseline.py

Wraps tigramite LPCMCI (Gerhardus & Runge, NeurIPS 2020) as a drop-in FC method
for the simulation benchmark. LPCMCI extends PCMCI+ to allow for latent
confounders (unobserved common causes) without assuming causal sufficiency.

Reference:
  A. Gerhardus, J. Runge, "High-recall causal discovery for autocorrelated time
  series with latent confounders", NeurIPS 2020.

Interface mirrors pcmci_plus_baseline.run_pcmci_plus:
    run_lpcmci(X, tau_max=2, pc_alpha=0.05, seed=0, ci_test='parcorr')

Direction convention (mirrors PCMCI+ convention where sensible):
- LPCMCI outputs a PAG-like graph with mark types: '-->', '<--', '<->', 'o->',
  '<-o', 'o-o', ''. We treat directed as directed, and 'certainly not tail'-only
  ambiguous marks ('<->', 'o->', 'o-o') as *presence-of-edge* in both directions
  (i.e., a predicted directed edge in both i->j and j->i). This is the same
  posture as our PCMCI+ wrapper takes for 'o-o'.

- Lag-side (tau>0): any non-empty mark → A[i,j]=1. Direction is unambiguous
  because time enforces X_i(t-tau) → X_j(t).
- Contemp (tau=0): unambiguous directed → single-direction A; ambiguous → both.
"""
from __future__ import annotations

import numpy as np


def run_lpcmci(
    X: np.ndarray,
    tau_max: int = 2,
    pc_alpha: float = 0.05,
    seed: int = 0,
    ci_test: str = 'parcorr',
) -> np.ndarray:
    """
    Run LPCMCI on multivariate time series X (p x T).

    Returns (p, p) int adjacency matrix.
    """
    from tigramite import data_processing as pp
    from tigramite.lpcmci import LPCMCI
    from tigramite.independence_tests.parcorr import ParCorr

    p, T = X.shape
    data = X.T.astype(np.float64)
    dataframe = pp.DataFrame(data)

    if ci_test == 'parcorr':
        ci = ParCorr()
    elif ci_test == 'gpdc':
        from tigramite.independence_tests.gpdc import GPDC
        ci = GPDC()
    else:
        raise ValueError(f"Unknown ci_test: {ci_test!r}. Use 'parcorr' or 'gpdc'.")

    lpcmci = LPCMCI(dataframe=dataframe, cond_ind_test=ci, verbosity=0)
    result = lpcmci.run_lpcmci(tau_max=tau_max, pc_alpha=pc_alpha)

    graph = result['graph']  # shape (p, p, tau_max+1), dtype object (strings)

    adj = np.zeros((p, p), dtype=int)

    # --- Lag side (tau >= 1): time-order fixes direction ---
    for tau in range(1, tau_max + 1):
        for i in range(p):
            for j in range(p):
                if i == j:
                    continue
                if graph[i, j, tau] != '':
                    adj[i, j] = 1

    # --- Contemp side (tau = 0) ---
    for i in range(p):
        for j in range(p):
            if i == j:
                continue
            mark = graph[i, j, 0]
            if mark == '-->':
                adj[i, j] = 1
            elif mark == '<--':
                adj[j, i] = 1
            elif mark in ('o-o', '<->', 'o->', '<-o'):
                # Ambiguous / bidirected: presence of edge, direction unclear
                adj[i, j] = 1
                adj[j, i] = 1

    np.fill_diagonal(adj, 0)
    return adj
