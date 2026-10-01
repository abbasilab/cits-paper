"""
glm_spiking_sim.py

Biologically-realistic electrophysiology benchmark: a GLM-coupled Poisson
spiking network with a KNOWN directed connectivity matrix (Pillow-style coupled
GLM), the standard synthetic model for spike-train connectivity inference.
Unlike CTRNN, the causal influence is an explicit lag>=1 coupling filter, so
causal DIRECTION is identifiable from time order.

Per neuron i, bin t:
    eta_i(t) = b_i + self_hist*s_i(t-1) + drive_i(t)
    lambda_i(t) = clip(exp(eta_i(t)), 0, lam_cap)
    s_i(t) ~ Poisson(lambda_i(t))

drive depends on the motif:
  'convergence' : 0->2, 1->2, 2->3           additive excitation
  'diamond'     : 0->1, 0->2, 1->3, 2->3     additive (common cause + effect)
  'coincidence' : 0->2, 1->2, 2->3 ; neuron 2 is driven by the PRODUCT
                  s0(t-1)*s1(t-1) (dendritic coincidence, nonlinear) rather
                  than the sum -- linear/pairwise methods miss the 0->2,1->2
                  edges; a nonparametric CI test (HSIC) recovers them.
"""
from __future__ import annotations
import numpy as np

MOTIFS = {
    'convergence': [(0, 2), (1, 2), (2, 3)],
    'diamond':     [(0, 1), (0, 2), (1, 3), (2, 3)],
    'coincidence': [(0, 2), (1, 2), (2, 3)],
    'depression':  [(0, 2), (1, 2), (2, 3)],   # convergence topology, depressing synapses
}
P_DEFAULT = 4


def simulate_glm_spiking(seed: int, motif: str = 'convergence', T: int = 3000,
                         base_rate: float = 0.06, w: float = 1.4,
                         self_hist: float = -0.5, lam_cap: float = 3.0):
    """Return (X counts p x T, GT directed p x p, edge list)."""
    p = P_DEFAULT
    rng = np.random.default_rng(seed)
    edges = MOTIFS[motif]
    W = np.zeros((p, p))
    for (a, b) in edges:
        W[a, b] = w
    b_i = np.log(base_rate)
    k1, k2 = 1.0, 0.4
    s = np.zeros((p, T))
    R = np.ones(p)                                     # Tsodyks-Markram resource
    U, tau_rec, dep_gain = 0.7, 15.0, 2.2              # depression params
    for t in range(1, T):
        p1 = s[:, t - 1]
        p2 = s[:, t - 2] if t >= 2 else np.zeros(p)
        filt = k1 * p1 + k2 * p2
        if motif == 'depression':
            # postsynaptic drive uses the DEPRESSED release (U*R*spike), a
            # saturating history-dependent (nonlinear) function of the input.
            released = U * R * p1
            drive = dep_gain * (W.T @ released)
            R = R + (1.0 - R) / tau_rec                 # recover
            R = np.clip(R - U * R * p1, 0.0, 1.0)        # deplete on spikes
        else:
            drive = W.T @ filt
            if motif == 'coincidence':
                drive[2] = w * (filt[0] * filt[1])
        eta = b_i + drive + self_hist * p1
        lam = np.clip(np.exp(eta), 0, lam_cap)
        s[:, t] = rng.poisson(lam)
    GT = (W != 0).astype(int)
    return s.astype(np.float64), GT, edges


def directed_cs(pred, GT, incl_self=False):
    p = GT.shape[0]
    TP = FP = FN = 0
    n_true = int(GT.sum())
    n_neg = (p * p - p if not incl_self else p * p) - n_true
    for i in range(p):
        for j in range(p):
            if i == j and not incl_self:
                continue
            if GT[i, j]:
                TP += 1 if pred[i, j] else 0
                FN += 0 if pred[i, j] else 1
            else:
                FP += 1 if pred[i, j] else 0
    tpr = TP / n_true if n_true else 0.0
    fpr = FP / n_neg if n_neg else 0.0
    return tpr - fpr, tpr, fpr


if __name__ == '__main__':
    for m in MOTIFS:
        s, GT, e = simulate_glm_spiking(0, motif=m)
        print(f"{m:12s} rate/bin={np.round(s.mean(1),3)} edges={e}")
