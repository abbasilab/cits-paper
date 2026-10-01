"""
gpu_hsic.py

GPU (torch) reimplementation of kpcalg's marginal HSIC permutation test
(`hsic.perm`), used as the conditional-independence statistic inside CITS
for non-Gaussian regimes.

Parity target (kpcalg::hsic.perm, via kernelCItest):
    - RBF kernel k(u,v) = exp(-(1/sig) * ||u-v||^2), default sig=1.
    - Biased HSIC estimator: trace((H Kx H)(H Ky H)) / n^2,
      with centering H = I - (1/n) 11^T.
    - Permutation null of size p (default 100): permute the sample order of y,
      recompute the statistic; p-value = mean( [perm_stats, hsic] >= hsic ).
    - kpcalg approximates Kx, Ky by incomplete Cholesky (numCol = n/10) purely
      for speed. On GPU we use the FULL kernel, i.e. the exact estimator that
      incomplete Cholesky approximates. This is the SAME statistic, not RCIT
      (no random-feature approximation).

Everything runs in float64 on CUDA for numerical parity with R's doubles.
"""
from __future__ import annotations
import torch


def _rbf_centered(v: torch.Tensor, sig: float) -> torch.Tensor:
    """Centered RBF Gram matrix H K H for a 1-D sample vector v (length n).

    k(a,b) = exp(-(1/sig) * (a-b)^2).  Returns H K H  (n x n), float64.
    """
    v = v.reshape(-1, 1)                      # (n,1)
    d2 = (v - v.T) ** 2                        # (n,n) squared distances
    K = torch.exp(-(1.0 / sig) * d2)           # (n,n) RBF kernel
    n = K.shape[0]
    # Center: H K H  where H = I - (1/n) 11^T.  Row/col mean subtraction form.
    Kr = K.mean(dim=0, keepdim=True)           # column means (1,n)
    Kc = K.mean(dim=1, keepdim=True)           # row means (n,1)
    Kall = K.mean()                            # grand mean
    return K - Kr - Kc + Kall                  # = H K H


def hsic_perm_gpu(x: torch.Tensor, y: torch.Tensor, sig: float = 1.0,
                  p: int = 100, device: str = "cuda",
                  generator: torch.Generator | None = None) -> dict:
    """Full-kernel HSIC permutation test on GPU. Mirrors kpcalg::hsic.perm.

    :param x, y: 1-D tensors / arrays of equal length n.
    :param sig:  kernel width parameter (kpcalg default 1).
    :param p:    number of permutations (kpcalg default 100).
    :returns: dict with 'hsic' (statistic) and 'p_value'.
    """
    x = torch.as_tensor(x, dtype=torch.float64, device=device).reshape(-1)
    y = torch.as_tensor(y, dtype=torch.float64, device=device).reshape(-1)
    n = x.shape[0]

    HKx = _rbf_centered(x, sig)                # (n,n)
    HLy = _rbf_centered(y, sig)                # (n,n)

    # Observed biased HSIC = trace(HKx @ HLy)/n^2 = sum(HKx * HLy)/n^2
    # (both symmetric, so elementwise product summed equals the trace).
    hsic = (HKx * HLy).sum() / (n * n)

    # Permutation null: permute samples of y  ->  P HLy P^T = HLy[perm][:,perm].
    perms = torch.stack([torch.randperm(n, device=device, generator=generator)
                         for _ in range(p)])   # (p, n)
    perm_stats = torch.empty(p, dtype=torch.float64, device=device)
    for i in range(p):
        pi = perms[i]
        HLp = HLy[pi][:, pi]
        perm_stats[i] = (HKx * HLp).sum() / (n * n)

    # p-value including the observed statistic (matches R's mean(c(perm,hsic)>=hsic)).
    ge = (torch.cat([perm_stats, hsic.reshape(1)]) >= hsic).double().mean()
    return {"hsic": float(hsic.item()), "p_value": float(ge.item())}


def hsic_perm_gpu_batched(x, y, sig: float = 1.0, p: int = 100,
                          device: str = "cuda",
                          generator: torch.Generator | None = None) -> dict:
    """Same as hsic_perm_gpu but batches all permutations into one gather.

    Faster for large p; memory ~ O(p * n^2) so guard for big n.
    """
    x = torch.as_tensor(x, dtype=torch.float64, device=device).reshape(-1)
    y = torch.as_tensor(y, dtype=torch.float64, device=device).reshape(-1)
    n = x.shape[0]
    HKx = _rbf_centered(x, sig)
    HLy = _rbf_centered(y, sig)
    hsic = (HKx * HLy).sum() / (n * n)

    perms = torch.stack([torch.randperm(n, device=device, generator=generator)
                         for _ in range(p)])            # (p,n)
    # HLy[perm][:,perm] for each perm, then elementwise with HKx, sum.
    idx_r = perms.unsqueeze(2).expand(p, n, n)          # (p,n,n) row index
    idx_c = perms.unsqueeze(1).expand(p, n, n)          # (p,n,n) col index
    HLp = HLy[idx_r, idx_c]                             # (p,n,n) permuted centered L
    perm_stats = (HKx.unsqueeze(0) * HLp).sum(dim=(1, 2)) / (n * n)  # (p,)
    ge = (torch.cat([perm_stats, hsic.reshape(1)]) >= hsic).double().mean()
    return {"hsic": float(hsic.item()), "p_value": float(ge.item())}
