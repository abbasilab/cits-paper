"""
pcmci_plus_baseline.py

Wraps tigramite PCMCI+ (Runge 2020, UAI) as a drop-in FC method for
the simulation benchmark.

Reference:
  J. Runge, "Discovering contemporaneous and lagged causal relations in
  autocorrelated nonlinear time series datasets", UAI 2020.
  https://proceedings.mlr.press/v124/runge20a.html

Interface:
    run_pcmci_plus(X, tau_max=2, pc_alpha=0.05, seed=0)

    Parameters
    ----------
    X : ndarray, shape (p, T)
        Multivariate time series.  Variables are rows (benchmark convention).
    tau_max : int
        Maximum lag to consider.  Use tau_max=2 to match our methods.
    pc_alpha : float
        Significance level for the PC skeleton phase.  Use 0.05 to match.
    seed : int
        Unused (ParCorr is deterministic), kept for API compatibility.
    ci_test : str
        'parcorr'  -- linear partial correlation (default, fast)
        'gpdc'     -- Gaussian-process distance correlation (nonlinear)

    Returns
    -------
    adj : ndarray, shape (p, p), dtype int
        Directed/partially-directed adjacency matrix (0/1).
        Direction is preserved from tigramite's PCMCI+ output:
          - Directed contemp: A[i,j]=1, A[j,i]=0
          - Undirected contemp ('o-o'): A[i,j]=A[j,i]=1
          - Lag edges '-->' X_i(t-tau)->X_j(t): A[i,j]=1 only
          - No edge: A[i,j]=A[j,i]=0
        Diagonal is always zero.

Notes on PCMCI+ graph encoding
-------------------------------
tigramite returns graph[i, j, tau] with string labels:
  ''      -- no edge
  '-->'   -- directed edge  X_i(t-tau) -> X_j(t)   (lag>0)
             or contemporaneous  X_i(t) -> X_j(t)   (tau=0)
  '<--'   -- reverse contemporaneous  X_i(t) <- X_j(t)  (tau=0 only)
  'o-o'   -- unoriented contemporaneous edge          (tau=0 only)
  '-'     -- edge mark (internal, should not appear in final graph)

Direction convention applied here:
  - graph[i,j,tau]='-->' (tau>0): X_i(t-tau)->X_j(t), so A[i,j]=1
  - graph[i,j,0]='-->': X_i(t)->X_j(t), so A[i,j]=1
  - graph[i,j,0]='<--': X_i(t)<-X_j(t), so A[j,i]=1
  - graph[i,j,0]='o-o': undirected, so A[i,j]=A[j,i]=1
"""
from __future__ import annotations

import numpy as np


def run_pcmci_plus(
    X: np.ndarray,
    tau_max: int = 2,
    pc_alpha: float = 0.05,
    seed: int = 0,
    ci_test: str = 'parcorr',
) -> np.ndarray:
    """
    Run PCMCI+ on multivariate time series X (p x T).

    Returns (p, p) int adjacency matrix preserving direction from tigramite.
    Directed edges are asymmetric; undirected contemp edges ('o-o') are symmetric.
    """
    from tigramite import data_processing as pp
    from tigramite.pcmci import PCMCI
    from tigramite.independence_tests.parcorr import ParCorr

    # X is (p, T); tigramite expects (T, p)
    p, T = X.shape
    data = X.T.astype(np.float64)

    dataframe = pp.DataFrame(data)

    if ci_test == 'parcorr':
        ci = ParCorr()
    elif ci_test == 'gpdc':
        from tigramite.independence_tests.gpdc import GPDC
        ci = GPDC()
    elif ci_test == 'cmiknn':
        # Fully nonparametric CMI(kNN) with shuffle significance test -- the
        # tigramite-recommended choice for nonlinear, non-Gaussian data
        # (matches CITS's HSIC in generality). Uses off-the-shelf defaults.
        from tigramite.independence_tests.cmiknn import CMIknn
        ci = CMIknn(significance='shuffle_test')
    else:
        raise ValueError(
            f"Unknown ci_test: {ci_test!r}. Use 'parcorr', 'gpdc', or 'cmiknn'.")

    pcmci = PCMCI(dataframe=dataframe, cond_ind_test=ci, verbosity=0)
    result = pcmci.run_pcmciplus(tau_max=tau_max, pc_alpha=pc_alpha)

    graph = result['graph']   # shape (p, p, tau_max+1), dtype object (strings)

    adj = np.zeros((p, p), dtype=int)

    # --- Lag side: tau = 1 .. tau_max ---
    # graph[i, j, tau] = '-->' means X_i(t-tau) -> X_j(t), so A[i,j]=1.
    for tau in range(1, tau_max + 1):
        for i in range(p):
            for j in range(p):
                if i == j:
                    continue
                if graph[i, j, tau] != '':
                    adj[i, j] = 1

    # --- Contemp side: tau = 0 ---
    # graph[i,j,0]='-->' means X_i(t)->X_j(t): A[i,j]=1
    # graph[i,j,0]='<--' means X_i(t)<-X_j(t): A[j,i]=1
    # graph[i,j,0]='o-o' means undirected:      A[i,j]=A[j,i]=1
    for i in range(p):
        for j in range(p):
            if i == j:
                continue
            mark = graph[i, j, 0]
            if mark == '-->':
                adj[i, j] = 1
            elif mark == '<--':
                adj[j, i] = 1
            elif mark == 'o-o':
                adj[i, j] = 1
                adj[j, i] = 1

    np.fill_diagonal(adj, 0)

    return adj
