"""
gpu_rcit.py

PyTorch implementation of Random Fourier Features HSIC for conditional
independence testing (Strobl, Zhang, Visweswaran 2019). GPU-resident
counterpart to ``rcit.py``.

Phase 2 deliverable of the GPU-RCIT-PC pipeline (see
``GPU_RCIT_PC_DESIGN.md``). This module exposes single-call helpers; the
batched-tile kernel used by the PC stagewise loop will be built on top of
``gpu_rcit_test`` in Phase 3.

API
---
    gpu_rff_embed(x_gpu, K, sigma, seed, device)
    gpu_median_bandwidth(X_gpu, max_samples=200)
    gpu_rcit_test(x_gpu, y_gpu, z_gpu=None, K=25, null='gamma',
                  n_perm=100, seed=0, device='cuda:0')
    gpu_rcit_pvalue_for_pc(C_gpu, i, j, S, N, K=25, null='gamma', seed=0)

Defaults match ``rcit.py`` (K=25, ridge=1e-2, n_perm=100). Default null is
``'gamma'`` (Strobl-2019 moment-matched gamma); ``'perm'`` is available as
a slower fallback that mirrors the CPU permutation null.
"""
from __future__ import annotations

import math
from typing import Iterable, Optional, Union

import numpy as np
import torch

# Ridge regularizer for kernel-ridge residualization in RFF space. Matches
# the CPU default in ``rcit.py``.
_RIDGE_LAMBDA = 1e-2


# ----------------------------------------------------------------------
# Utilities
# ----------------------------------------------------------------------


def _as_tensor(x, device, dtype=torch.float32) -> torch.Tensor:
    """Cast to a contiguous tensor on ``device`` with the given ``dtype``.

    Phase 4.6: ``dtype`` is now load-bearing for the float32 acceleration
    path. Pass ``torch.float64`` to retain pre-Phase-4.6 numerics.
    """
    if isinstance(x, torch.Tensor):
        t = x.to(device=device, dtype=dtype)
    else:
        t = torch.as_tensor(np.asarray(x), device=device, dtype=dtype)
    if t.ndim == 1:
        t = t[:, None]
    return t.contiguous()


def _make_generator(device, seed: int) -> torch.Generator:
    """Reproducible torch.Generator on the given device."""
    g = torch.Generator(device=device)
    g.manual_seed(int(seed))
    return g


# ----------------------------------------------------------------------
# Bandwidth heuristic
# ----------------------------------------------------------------------


def gpu_median_bandwidth(X_gpu: torch.Tensor, max_samples: int = 200,
                          seed: int = 0) -> float:
    """Median pairwise distance heuristic for Gaussian-kernel bandwidth.

    Matches the CPU helper ``rcit._median_bandwidth``. The CPU helper draws
    a fixed-seed subsample (``np.random.default_rng(0)``) of up to 200
    rows; we mirror that here so bandwidths agree across backends.

    Parameters
    ----------
    X_gpu : torch.Tensor
        ``(N,)`` or ``(N, d)`` data on a torch device.
    max_samples : int
        Subsample size for the median heuristic.
    seed : int
        Seed for the subsample draw.

    Returns
    -------
    float
        Median pairwise Euclidean distance (clamped to >= 1e-12).
    """
    if X_gpu.ndim == 1:
        X_gpu = X_gpu[:, None]
    n = X_gpu.shape[0]
    if n > max_samples:
        # Match CPU: rng = np.random.default_rng(0); idx = rng.choice(...)
        rng = np.random.default_rng(int(seed))
        idx = rng.choice(n, size=max_samples, replace=False)
        idx_t = torch.as_tensor(idx, device=X_gpu.device, dtype=torch.long)
        X_sub = X_gpu.index_select(0, idx_t)
    else:
        X_sub = X_gpu
    d = torch.cdist(X_sub, X_sub)
    # Take strictly upper triangular entries (mirror scipy.spatial.distance.pdist).
    iu = torch.triu_indices(d.shape[0], d.shape[1], offset=1,
                            device=X_gpu.device)
    dists = d[iu[0], iu[1]]
    med = torch.median(dists).item()
    if not math.isfinite(med) or med < 1e-12:
        med = 1.0
    return float(med)


# ----------------------------------------------------------------------
# RFF embedding
# ----------------------------------------------------------------------


def gpu_rff_embed(x_gpu: torch.Tensor, K: int, sigma: float,
                   seed: int, device: Optional[Union[str, torch.device]] = None,
                   generator: Optional[torch.Generator] = None) -> torch.Tensor:
    """Random Fourier Features embedding on GPU.

    For input ``x`` of shape ``(N,)`` or ``(N, d)``, returns

        phi(x) = (1/sqrt(K)) * [cos(W x + b), sin(W x + b)]

    of shape ``(N, 2K)``. The Fourier projection ``W`` is drawn from
    ``N(0, 1/sigma^2)``, matching the CPU helper ``rcit._rff_embed``.

    Parameters
    ----------
    x_gpu : torch.Tensor
        Input data on a torch device.
    K : int
        Number of random Fourier features (output dimension is ``2K``).
    sigma : float
        Bandwidth of the underlying Gaussian kernel.
    seed : int
        Seed for the Fourier projection (used only if ``generator`` is None).
    device : str or torch.device, optional
        Device for the random projection. If None, uses ``x_gpu.device``.
    generator : torch.Generator, optional
        If provided, used instead of creating a new generator from ``seed``.

    Returns
    -------
    torch.Tensor
        Embedded data of shape ``(N, 2K)`` on ``device``.
    """
    if x_gpu.ndim == 1:
        x_gpu = x_gpu[:, None]
    if device is None:
        device = x_gpu.device
    else:
        device = torch.device(device)
        x_gpu = x_gpu.to(device)
    N, d = x_gpu.shape
    if generator is None:
        generator = _make_generator(device, seed)
    inv_sigma = 1.0 / float(sigma)
    W = torch.randn((d, K), generator=generator, device=device,
                    dtype=x_gpu.dtype) * inv_sigma
    b = torch.rand((K,), generator=generator, device=device,
                   dtype=x_gpu.dtype) * (2.0 * math.pi)
    proj = x_gpu @ W + b  # (N, K)
    phi = torch.empty((N, 2 * K), device=device, dtype=x_gpu.dtype)
    phi[:, :K] = torch.cos(proj)
    phi[:, K:] = torch.sin(proj)
    phi.div_(math.sqrt(K))
    return phi


# ----------------------------------------------------------------------
# Residualization and HSIC statistic
# ----------------------------------------------------------------------


def _residualize_gpu(phi: torch.Tensor, phi_z: torch.Tensor,
                      ridge_lambda: float = _RIDGE_LAMBDA) -> torch.Tensor:
    """Kernel-ridge residualize ``phi`` against ``phi_z`` in RFF space.

    Solves ``(phi_z^T phi_z + lam I) B = phi_z^T phi`` and returns
    ``phi - phi_z @ B``. Mirrors ``rcit._residualize``.
    """
    M = phi_z.shape[1]
    A = phi_z.T @ phi_z
    A = A + ridge_lambda * torch.eye(M, device=A.device, dtype=A.dtype)
    rhs = phi_z.T @ phi
    try:
        B = torch.linalg.solve(A, rhs)
    except RuntimeError:
        # Float64 fallback for poorly conditioned systems.
        B = torch.linalg.solve(A.double(), rhs.double()).to(phi.dtype)
    return phi - phi_z @ B


def _Tstat_gpu(phi_x: torch.Tensor, phi_y: torch.Tensor) -> torch.Tensor:
    """HSIC RFF statistic ``N * ||phi_x^T phi_y / N||_F^2`` (returns scalar tensor)."""
    N = phi_x.shape[0]
    Sxy = (phi_x.T @ phi_y) / N
    return N * (Sxy ** 2).sum()


# ----------------------------------------------------------------------
# Null distributions
# ----------------------------------------------------------------------


def _gamma_pvalue(phi_x: torch.Tensor, phi_y: torch.Tensor) -> float:
    """Strobl-2019 gamma moment match for the RFF-HSIC null.

    Computes the shape and rate of a gamma distribution matched to the
    first two moments of the test statistic
    ``T = N * ||phi_x^T phi_y / N||_F^2`` under H0, then returns the right
    tail probability at the observed ``T``.

    Both inputs are assumed already centered.
    """
    N = phi_x.shape[0]
    Sxy = (phi_x.T @ phi_y) / N  # (2K, 2K)
    T_obs = (N * (Sxy ** 2).sum()).item()
    # Per-side centered Gram (in feature space). The HSIC null moments
    # depend only on the trace of the feature-space covariance per side.
    Cxx = (phi_x.T @ phi_x) / N
    Cyy = (phi_y.T @ phi_y) / N
    tr_Cxx = torch.diagonal(Cxx).sum().item()
    tr_Cyy = torch.diagonal(Cyy).sum().item()
    tr_Cxx2 = (Cxx * Cxx).sum().item()
    tr_Cyy2 = (Cyy * Cyy).sum().item()
    mu = tr_Cxx * tr_Cyy
    var = 2.0 * tr_Cxx2 * tr_Cyy2
    if not math.isfinite(mu) or not math.isfinite(var) or var <= 0.0 or mu <= 0.0:
        # Degenerate; treat as fully independent.
        return 1.0
    shape = (mu * mu) / var
    rate = mu / var
    # Right-tail probability P(Gamma(shape, rate) >= T_obs)
    # = gammaincc(shape, rate * T_obs).
    if T_obs <= 0.0:
        return 1.0
    pval = torch.special.gammaincc(
        torch.tensor(shape, dtype=torch.float64),
        torch.tensor(rate * T_obs, dtype=torch.float64),
    ).item()
    # Clip into (0, 1].
    if not math.isfinite(pval):
        return 1.0
    pval = max(min(pval, 1.0), 1e-300)
    return pval


def _perm_pvalue(phi_x: torch.Tensor, phi_y: torch.Tensor,
                  n_perm: int, generator: torch.Generator) -> float:
    """Permutation null: shuffle rows of phi_y, compare T_b to T_obs.

    Mirrors ``rcit.rcit_test`` permutation block (with add-one smoothing).
    """
    N = phi_x.shape[0]
    T_obs = _Tstat_gpu(phi_x, phi_y).item()
    count_geq = 1  # add-one smoothing
    # Pre-compute phi_x^T once.
    phi_xT = phi_x.T
    for _ in range(n_perm):
        idx = torch.randperm(N, generator=generator, device=phi_y.device)
        phi_y_perm = phi_y.index_select(0, idx)
        Sxy = (phi_xT @ phi_y_perm) / N
        T_b = (N * (Sxy ** 2).sum()).item()
        if T_b >= T_obs:
            count_geq += 1
    return count_geq / (n_perm + 1)


# ----------------------------------------------------------------------
# Public single-test API
# ----------------------------------------------------------------------


def gpu_rcit_test(x_gpu, y_gpu, z_gpu=None, K: int = 25,
                   null: str = 'gamma', n_perm: int = 100,
                   seed: int = 0,
                   device: Union[str, torch.device] = 'cuda:0',
                   dtype: torch.dtype = torch.float32) -> float:
    """RFF-HSIC (conditional) independence test on GPU.

    Parameters
    ----------
    x_gpu, y_gpu : torch.Tensor or array-like
        ``(N,)`` or ``(N, d)``.
    z_gpu : torch.Tensor, array-like, or None
        Conditioning variable. ``None`` triggers a marginal test.
    K : int
        Number of Fourier features per variable.
    null : {'gamma', 'perm'}
        Null-distribution method. ``'gamma'`` (default) uses the
        moment-matched gamma null and is closed-form. ``'perm'`` uses the
        permutation null of length ``n_perm`` and mirrors the CPU
        reference.
    n_perm : int
        Permutation count (used only when ``null='perm'``).
    seed : int
        Master seed. RFF projections for ``x``, ``y``, (and ``z``) use
        derived sub-seeds for reproducibility independent of order.
    device : str or torch.device
        Compute device. Tensors that already live on a different device
        are moved.
    dtype : torch.dtype, default torch.float32
        Internal floating-point dtype for the GPU tensors. ``float32`` is
        the Phase 4.6 default; pass ``torch.float64`` to recover the
        pre-4.6 numerics.

    Returns
    -------
    float
        p-value under H0: ``x`` is independent of ``y`` (given ``z``).
    """
    device = torch.device(device)
    x = _as_tensor(x_gpu, device, dtype=dtype)
    y = _as_tensor(y_gpu, device, dtype=dtype)
    if x.shape[0] != y.shape[0]:
        raise ValueError("x and y must have the same N")
    N = x.shape[0]
    if N < 8:
        # Too few samples to estimate moments / kernel bandwidth.
        return 1.0

    # Bandwidths: median heuristic. Use seed=0 for the subsample to mirror
    # CPU behavior (which always seeds bandwidth subsampling with 0).
    sigma_x = gpu_median_bandwidth(x, seed=0)
    sigma_y = gpu_median_bandwidth(y, seed=0)
    # Derive deterministic sub-seeds for RFF projections from the master seed
    # so changing variable order does not change embeddings catastrophically.
    seed_x = (int(seed) * 1_000_003 + 11) & 0x7FFFFFFF
    seed_y = (int(seed) * 1_000_003 + 23) & 0x7FFFFFFF
    seed_z = (int(seed) * 1_000_003 + 37) & 0x7FFFFFFF
    seed_perm = (int(seed) * 1_000_003 + 41) & 0x7FFFFFFF
    phi_x = gpu_rff_embed(x, K, sigma_x, seed=seed_x, device=device)
    phi_y = gpu_rff_embed(y, K, sigma_y, seed=seed_y, device=device)
    phi_x = phi_x - phi_x.mean(dim=0, keepdim=True)
    phi_y = phi_y - phi_y.mean(dim=0, keepdim=True)

    if z_gpu is not None:
        z = _as_tensor(z_gpu, device, dtype=dtype)
        if z.shape[0] != N:
            raise ValueError("z must have N rows")
        sigma_z = gpu_median_bandwidth(z, seed=0)
        phi_z = gpu_rff_embed(z, K, sigma_z, seed=seed_z, device=device)
        phi_z = phi_z - phi_z.mean(dim=0, keepdim=True)
        phi_x = _residualize_gpu(phi_x, phi_z, ridge_lambda=_RIDGE_LAMBDA)
        phi_y = _residualize_gpu(phi_y, phi_z, ridge_lambda=_RIDGE_LAMBDA)

    if null == 'gamma':
        return _gamma_pvalue(phi_x, phi_y)
    elif null in ('perm', 'permutation'):
        g_perm = _make_generator(device, seed_perm)
        return _perm_pvalue(phi_x, phi_y, n_perm=n_perm, generator=g_perm)
    else:
        raise ValueError(f"Unknown null: {null!r}")


# ----------------------------------------------------------------------
# PC-style signature
# ----------------------------------------------------------------------


def gpu_rcit_pvalue_for_pc(C_gpu, i: int, j: int,
                            S: Union[Iterable[int], tuple, list],
                            N: int, K: int = 25,
                            null: str = 'gamma', seed: int = 0,
                            device: Union[str, torch.device] = 'cuda:0',
                            n_perm: int = 100,
                            dtype: torch.dtype = torch.float32) -> float:
    """PC-style RCIT signature, mirrors ``rcit.rcit_pvalue_for_pc``.

    Parameters
    ----------
    C_gpu : torch.Tensor or array-like
        ``(N, V)`` data matrix.
    i, j : int
        Column indices of the variables being tested.
    S : iterable of int
        Column indices of the conditioning set.
    N : int
        Sample count (kept for API parity; ``C_gpu.shape[0]`` is the
        authoritative value).
    K : int
        Number of Fourier features per variable.
    null : {'gamma', 'perm'}
        Null distribution method.
    seed : int
        Master seed.
    device : str or torch.device
        Compute device.
    n_perm : int
        Permutation count (used only when ``null='perm'``).

    Returns
    -------
    float
        p-value of H0: ``C[:, i] ⊥ C[:, j] | C[:, S]``.
    """
    device = torch.device(device)
    C = _as_tensor(C_gpu, device, dtype=dtype)
    if C.shape[0] != N:
        # Allow mismatch but use the tensor's actual N.
        pass
    x = C[:, i]
    y = C[:, j]
    if len(tuple(S)) == 0:
        z = None
    else:
        S_idx = torch.as_tensor(list(S), device=device, dtype=torch.long)
        z = C.index_select(1, S_idx)
    return gpu_rcit_test(x, y, z_gpu=z, K=K, null=null, n_perm=n_perm,
                          seed=seed, device=device, dtype=dtype)


# ----------------------------------------------------------------------
# Validation
# ----------------------------------------------------------------------


def _run_validation(device: str = 'cuda:0', n_seeds: int = 30,
                     N: int = 1000) -> bool:
    """Validate GPU RCIT against CPU RCIT on canonical test cases."""
    try:
        from rcit import rcit_test as cpu_rcit_test
    except ImportError as exc:  # pragma: no cover
        print(f"[FATAL] Could not import CPU rcit: {exc}")
        return False

    print("=== GPU RCIT validation ===\n")
    print(f"Device:      {device}")
    print(f"Seeds:       {n_seeds}")
    print(f"N per case:  {N}\n")

    cases = ['Independent', 'Linear', 'Nonlinear (sin)', 'Conditional |Z']
    all_pass = True
    summary_rows = []

    for case_i, name in enumerate(cases, start=1):
        cpu_pvals = np.zeros(n_seeds)
        gpu_pvals = np.zeros(n_seeds)
        for s in range(n_seeds):
            rng = np.random.default_rng(1000 + s)
            if name == 'Independent':
                x = rng.normal(size=N); y = rng.normal(size=N); z = None
            elif name == 'Linear':
                x = rng.normal(size=N); y = 0.5 * x + rng.normal(size=N); z = None
            elif name == 'Nonlinear (sin)':
                x = rng.normal(size=N); y = np.sin(2 * x) + 0.3 * rng.normal(size=N); z = None
            else:  # Conditional |Z
                z = rng.normal(size=N)
                x = z + 0.5 * rng.normal(size=N)
                y = z + 0.5 * rng.normal(size=N)
            # CPU
            cpu_pvals[s] = cpu_rcit_test(x, y, z=z, seed=s)
            # GPU (use the SAME seed; gamma null so no permutation)
            gpu_pvals[s] = gpu_rcit_test(x, y, z_gpu=z, seed=s, device=device,
                                          null='gamma')

        # Paired comparison: Pearson correlation, mean |Δp|, agreement at α=0.05.
        # Correlation can be ill-defined when one vector has zero variance
        # (e.g., both vectors at the gamma floor for strong dependence). Guard.
        cpu_var = float(np.var(cpu_pvals))
        gpu_var = float(np.var(gpu_pvals))
        if cpu_var < 1e-30 or gpu_var < 1e-30:
            pearson = float('nan')
        else:
            pearson = float(np.corrcoef(cpu_pvals, gpu_pvals)[0, 1])
        mean_abs_diff = float(np.mean(np.abs(cpu_pvals - gpu_pvals)))
        cpu_reject = cpu_pvals < 0.05
        gpu_reject = gpu_pvals < 0.05
        agree = int((cpu_reject == gpu_reject).sum())

        print(f"Test {case_i} ({name}):")
        print(f"  CPU mean p: {cpu_pvals.mean():.4f}  GPU mean p: {gpu_pvals.mean():.4f}")
        print(f"  CPU min/max: [{cpu_pvals.min():.4f}, {cpu_pvals.max():.4f}]  "
              f"GPU min/max: [{gpu_pvals.min():.4f}, {gpu_pvals.max():.4f}]")
        print(f"  Pearson r (CPU vs GPU p-values): "
              f"{pearson if not math.isnan(pearson) else 'n/a'}")
        print(f"  Mean |Δp|: {mean_abs_diff:.4f}")
        agree_mark = '✓' if agree / n_seeds >= 0.90 else '✗'
        print(f"  Agreement at α=0.05: {agree}/{n_seeds} {agree_mark}")

        # Pass criteria depend on the case. For strong-dependence cases the
        # CPU permutation null bottoms out at 1/(n_perm+1) ≈ 0.0099 for many
        # seeds → near-zero variance; the gamma null can return p ≈ 1e-30.
        # In that regime the Pearson criterion is not meaningful, so we
        # demand that BOTH backends reject H0 in every seed instead.
        if name == 'Independent':
            ok = (agree / n_seeds >= 0.85) and (mean_abs_diff < 0.15)
            ok = ok and (not math.isnan(pearson) and pearson > 0.5)
        elif name == 'Conditional |Z':
            # Marginal would reject, conditional (this is what's tested) should not.
            # Agreement on H0 retention is what matters.
            ok = (agree / n_seeds >= 0.85) and (mean_abs_diff < 0.15)
            ok = ok and (math.isnan(pearson) or pearson > 0.3)
        else:
            # Strong dependence: both should reject in all seeds.
            ok = (cpu_reject.sum() == n_seeds) and (gpu_reject.sum() == n_seeds)
        summary_rows.append((name, ok))
        if not ok:
            all_pass = False
        print()

    print("--- Summary ---")
    for name, ok in summary_rows:
        print(f"  {name:25s}: {'PASS' if ok else 'FAIL'}")
    print()
    print("OVERALL: " + ("PASSED (all tests within tolerance)" if all_pass else
                          "FAILED (see per-test results above)"))
    return all_pass


def _speed_benchmark(device: str = 'cuda:0', N: int = 1000) -> None:
    """Wall-clock comparison of CPU vs GPU on a single RCIT call."""
    import time
    try:
        from rcit import rcit_test as cpu_rcit_test
    except ImportError:
        return

    rng = np.random.default_rng(0)
    x = rng.normal(size=N)
    y = 0.5 * x + rng.normal(size=N)
    z = rng.normal(size=N)

    print("\n=== Speed sanity check ===\n")
    print(f"N = {N}, K = 25, with conditioning")

    # CPU: permutation null (its default).
    t0 = time.perf_counter()
    p_cpu = cpu_rcit_test(x, y, z=z, K=25, seed=0)
    t_cpu = time.perf_counter() - t0
    print(f"  CPU (permutation null, n_perm=100): {t_cpu*1000:7.2f} ms  "
          f"(p = {p_cpu:.4g})")

    # GPU: gamma null (default).
    # Warm up the GPU (kernel compilation, allocator).
    _ = gpu_rcit_test(x, y, z_gpu=z, K=25, seed=0, device=device, null='gamma')
    if str(device).startswith('cuda'):
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    p_gpu = gpu_rcit_test(x, y, z_gpu=z, K=25, seed=0, device=device, null='gamma')
    if str(device).startswith('cuda'):
        torch.cuda.synchronize()
    t_gpu = time.perf_counter() - t0
    print(f"  GPU (gamma null, closed-form):      {t_gpu*1000:7.2f} ms  "
          f"(p = {p_gpu:.4g})")

    # Also benchmark GPU with permutation null for an apples-to-apples view.
    _ = gpu_rcit_test(x, y, z_gpu=z, K=25, seed=0, device=device,
                      null='perm', n_perm=100)
    if str(device).startswith('cuda'):
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    p_gpu_perm = gpu_rcit_test(x, y, z_gpu=z, K=25, seed=0, device=device,
                                null='perm', n_perm=100)
    if str(device).startswith('cuda'):
        torch.cuda.synchronize()
    t_gpu_perm = time.perf_counter() - t0
    print(f"  GPU (permutation null, n_perm=100): {t_gpu_perm*1000:7.2f} ms  "
          f"(p = {p_gpu_perm:.4g})")

    print(f"\n  Speedup (CPU perm vs GPU gamma): {t_cpu / max(t_gpu, 1e-9):.2f}x")
    print(f"  Speedup (CPU perm vs GPU perm):  {t_cpu / max(t_gpu_perm, 1e-9):.2f}x")


if __name__ == '__main__':
    import os
    import sys

    # Make the CPU reference (rcit.py) importable when running this script
    # directly.
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    dev = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {dev}\n")

    ok = _run_validation(device=dev, n_seeds=30, N=1000)
    _speed_benchmark(device=dev, N=1000)

    sys.exit(0 if ok else 1)
