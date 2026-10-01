import numpy as np
N_NEURONS = 4

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

