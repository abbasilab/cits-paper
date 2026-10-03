import numpy as np, pandas as pd
from scipy.stats import fisher_exact
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'shared')))
from paths import arousal_figs, result

# arousal-pipeline per-field CSVs (CITS, GC2; not produced in this repo)
BASE = os.path.join(arousal_figs('2026-06-01_versionBsafe', 'exp_analysis', 'fig1_sc'), '')

def katz_ci(a, n1, c, n2):
    rr = (a/n1)/(c/n2)
    se = np.sqrt(1/a - 1/n1 + 1/c - 1/n2)
    lo = np.exp(np.log(rr) - 1.96*se)
    hi = np.exp(np.log(rr) + 1.96*se)
    return rr, lo, hi

def analyze(name, a, n1, c, n2):
    b = n1 - a; d = n2 - c
    rr, lo, hi = katz_ci(a, n1, c, n2)
    _, p = fisher_exact([[a, b], [c, d]])
    print(f"\n=== {name} ===")
    print(f"  a=fcplus_scplus={a}, n1=fc_plus={n1}, c=fcminus_scplus={c}, n2=fc_minus={n2}")
    print(f"  present_rate={a/n1:.6f}, absent_rate={c/n2:.6f}")
    print(f"  FOLD={rr:.4f}   Fisher p={p:.3e}   Katz 95% CI=({lo:.4f}, {hi:.4f})")
    return rr

# --- CITS ---
analyze("CITS (sc_enrichment_per_state_versionBsafe.csv, state=all)", 280, 22495, 2769, 810339)
# --- lagged corr ---
analyze("Lagged correlation (sc_enrichment_per_state_laggedcorr_max.csv, all)", 2419, 590183, 630, 242651)
# --- GC2 ---
analyze("Conditional Granger GC2 (sc_enrichment_per_state_granger.csv, all)", 819, 214260, 2230, 618574)
# --- within/between ---
# within: a=w_syn=2656, n1=w_tot=432618 ; between: c=b_syn=393, n2=b_tot=400216
analyze("Within-vs-between EM (sc_within_vs_between_synapse_versionBsafe.csv)", 2656, 432618, 393, 400216)

# ================= FIELD-CLUSTER BOOTSTRAP =================
print("\n\n########## FIELD-CLUSTER BOOTSTRAP 95% CI (N=5000, seed=42) ##########")

def boot_enrich(df):
    # df has columns fc_plus, fcplus_scplus, fc_minus, fcminus_scplus per field
    keys = df['field_key'].values
    A = df['fcplus_scplus'].values.astype(float)
    N1 = df['fc_plus'].values.astype(float)
    C = df['fcminus_scplus'].values.astype(float)
    N2 = df['fc_minus'].values.astype(float)
    n = len(df)
    rng = np.random.default_rng(42)
    folds = []
    for _ in range(5000):
        idx = rng.integers(0, n, n)
        a = A[idx].sum(); n1 = N1[idx].sum(); c = C[idx].sum(); n2 = N2[idx].sum()
        if n1 > 0 and c > 0 and n2 > 0:
            folds.append((a/n1)/(c/n2))
    folds = np.array(folds)
    return np.percentile(folds, [2.5, 97.5]), n

def boot_within_between(df):
    W_tot = df['w_tot'].values.astype(float); W_syn = df['w_syn'].values.astype(float)
    B_tot = df['b_tot'].values.astype(float); B_syn = df['b_syn'].values.astype(float)
    n = len(df)
    rng = np.random.default_rng(42)
    ratios = []
    for _ in range(5000):
        idx = rng.integers(0, n, n)
        wt = W_tot[idx].sum(); ws = W_syn[idx].sum(); bt = B_tot[idx].sum(); bs = B_syn[idx].sum()
        if wt > 0 and bt > 0 and bs > 0:
            ratios.append((ws/wt)/(bs/bt))
    ratios = np.array(ratios)
    return np.percentile(ratios, [2.5, 97.5]), n

# CITS per-field (versionBsafe), state=all
df_cits = pd.read_csv(BASE+"sc_enrichment_perfield_versionBsafe.csv")
df_cits_all = df_cits[df_cits['state']=='all']
ci, n = boot_enrich(df_cits_all)
print(f"CITS (perfield versionBsafe, all, {n} fields): bootstrap 95% CI = ({ci[0]:.4f}, {ci[1]:.4f})")

# GC2 per-field (granger), state=all
df_g = pd.read_csv(BASE+"sc_enrichment_perfield_granger.csv")
df_g_all = df_g[df_g['state']=='all']
ci, n = boot_enrich(df_g_all)
print(f"GC2 (perfield granger, all, {n} fields): bootstrap 95% CI = ({ci[0]:.4f}, {ci[1]:.4f})")

# within/between per-field
# produced by bootstrap_em_within_vs_between_versionBsafe.py (committed copy in source_data/)
df_wb = pd.read_csv(result('fig3_microns_enrichment', "sc_within_vs_between_perfield_versionBsafe.csv"))
ci, n = boot_within_between(df_wb)
print(f"Within/between (perfield, {n} fields): bootstrap 95% CI = ({ci[0]:.4f}, {ci[1]:.4f})")

print("\nNOTE: no per-field CSV exists for lagged correlation -> bootstrap CI not available for that baseline.")
