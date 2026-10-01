import pandas as pd, numpy as np
from scipy.stats import wilcoxon

df = pd.read_csv("/home/rbiswas1/microns/analysis/stimulus_fc/results/stimulus_fc_combined.csv")
print("stim values:", df['stim'].unique(), " nrows:", len(df))

# map stim labels (clip lowercase) to requested display names
stim_map = {'clip': 'Clip', 'Monet': 'Monet', 'Trippy': 'Trippy'}

metrics = {'density': ('B_within_d', 'B_between_d'), 'strength': ('B_within_s', 'B_between_s')}

def rank_biserial(x, y):
    # matched-pairs rank-biserial from signed ranks (Wilcoxon), zeros dropped
    d = np.asarray(x) - np.asarray(y)
    d = d[d != 0]
    n = len(d)
    ranks = pd.Series(np.abs(d)).rank().values
    w_plus = ranks[d > 0].sum()
    w_minus = ranks[d < 0].sum()
    tot = w_plus + w_minus
    r = 1 - 2*w_minus/tot
    return r, w_plus, w_minus

def boot_mean_ci(vals, seed=42, N=5000):
    vals = np.asarray(vals, float)
    rng = np.random.default_rng(seed)
    n = len(vals)
    means = np.array([vals[rng.integers(0, n, n)].mean() for _ in range(N)])
    return vals.mean(), np.percentile(means, [2.5, 97.5])

print("\n===== TASK 9: paired Wilcoxon within vs between =====")
print(f"{'stim':7} {'metric':9} {'n':>4} {'W':>10} {'p':>12} {'rank-biserial r':>16}")
for raw, disp in stim_map.items():
    sub = df[df['stim'] == raw]
    for mname, (wc, bc) in metrics.items():
        d = sub[[wc, bc]].dropna()
        n = len(d)
        w_arr = d[wc].values; b_arr = d[bc].values
        res = wilcoxon(w_arr, b_arr)  # default: wilcox, zsplit? scipy default zero_method='wilcox'
        r, wp, wm = rank_biserial(w_arr, b_arr)
        print(f"{disp:7} {mname:9} {n:>4} {res.statistic:>10.2f} {res.pvalue:>12.3e} {r:>16.4f}")

print("\n===== Within / between means with field-cluster bootstrap 95% CI =====")
for raw, disp in stim_map.items():
    sub = df[df['stim'] == raw]
    print(f"\n-- {disp} --")
    for mname, (wc, bc) in metrics.items():
        for label, col in [('within', wc), ('between', bc)]:
            vals = sub[col].dropna().values
            m, ci = boot_mean_ci(vals)
            print(f"  {mname:8} {label:8} mean={m:.6f}  95%CI=({ci[0]:.6f}, {ci[1]:.6f})  n={len(vals)}")
