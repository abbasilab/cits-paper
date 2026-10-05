"""Aggregate the 2-variant stimulus FC sweep -> within/between + area-pair figures.
Variant A = cits-gpu (cuPC) lagged + LSCM weights. Variant B = contemporaneous CITS with cits-gpu.
Per stimulus (Clip/Monet/Trippy): mean + field-bootstrap 95% CI (descriptive), and paired
within-field stats across the 124 fields: Friedman omnibus + pairwise Wilcoxon (FDR-BH)."""
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from scipy.stats import friedmanchisquare, wilcoxon
from itertools import combinations
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'shared')))
from paths import out as _out, result

d = pd.read_csv(result('fig3_microns_enrichment', 'results/stimulus_fc_combined.csv'))
d['fid'] = d.session.astype(str)+'_'+d.scan.astype(str)+'_'+d.field.astype(str)
STIMS = ['clip','Monet','Trippy']; COL={'clip':'#0072B2','Monet':'#D55E00','Trippy':'#009E73'}
AREAS=['V1','LM','RL','AL']; PAIRS=[f'{a}->{b}' for a in AREAS for b in AREAS]
rng = np.random.default_rng(0)

def boot_ci(vals, B=10000):
    vals=np.asarray(vals); idx=rng.integers(0,len(vals),(B,len(vals)))
    means=vals[idx].mean(1); return vals.mean(), np.percentile(means,2.5), np.percentile(means,97.5)

def fdr(ps):
    ps=np.asarray(ps); o=np.argsort(ps); r=np.empty_like(o); r[o]=np.arange(1,len(ps)+1)
    q=ps*len(ps)/r; # BH
    q_sorted=q[o]; q_sorted=np.minimum.accumulate(q_sorted[::-1])[::-1]; out=np.empty_like(q); out[o]=q_sorted
    return np.clip(out,0,1)

def paired_matrix(metric):
    """(n_fields, 3) aligned by field for the 3 stimuli; drop fields missing any stim."""
    piv = d.pivot_table(index='fid', columns='stim', values=metric)
    piv = piv.dropna(subset=STIMS)
    return piv[STIMS].values, piv.index

def stimulus_stats(metric):
    M,_ = paired_matrix(metric)
    if len(M)<3: return None
    fr = friedmanchisquare(M[:,0],M[:,1],M[:,2])
    pw={}
    for i,j in combinations(range(3),2):
        try: pw[(STIMS[i],STIMS[j])] = wilcoxon(M[:,i],M[:,j]).pvalue
        except Exception: pw[(STIMS[i],STIMS[j])] = np.nan
    keys=list(pw); q=fdr([pw[k] for k in keys])
    return {'friedman_p':fr.pvalue,'n':len(M),'pairwise':{keys[i]:(pw[keys[i]],q[i]) for i in range(len(keys))}}

# ---- print stats table ----
print("=== stimulus differences (paired, n=124 fields) ===")
for var in ['A','B']:
    for base in ['within_d','within_s','between_d','between_s']:
        m=f'{var}_{base}'; s=stimulus_stats(m)
        if s is None: continue
        sig=[f"{a}v{b}:q={q:.3f}{'*' if q<0.05 else ''}" for (a,b),(p,q) in s['pairwise'].items()]
        print(f"{var} {base:10s} Friedman p={s['friedman_p']:.2e}  | "+"  ".join(sig))

# ---- FIGURE 1 (per variant): within vs between, density & strength ----
def fig_within_between(var, fname):
    fig,axes=plt.subplots(1,2,figsize=(9,4.2))
    for ax,(base,lab) in zip(axes,[('_d','edge density'),('_s','edge strength')]):
        x=np.arange(2); w=0.25
        for k,stim in enumerate(STIMS):
            for wi,lvl in enumerate(['within','between']):
                col=f'{var}_{lvl}{base}'; vals=d[d.stim==stim][col].dropna().values
                mu,lo,hi=boot_ci(vals)
                ax.bar(wi+(k-1)*w, mu, w, color=COL[stim], label=stim if wi==0 else None,
                       yerr=[[mu-lo],[hi-mu]], capsize=2, error_kw={'lw':1})
        ax.set_xticks(x); ax.set_xticklabels(['within-area','between-area']); ax.set_ylabel(lab)
        ax.set_title(lab, fontsize=10, fontweight='bold'); ax.spines[['top','right']].set_visible(False)
    axes[0].legend(frameon=False, fontsize=9, title='stimulus')
    fig.suptitle(f'Variant {var}: within vs between-area FC by stimulus', fontsize=11, fontweight='bold')
    fig.tight_layout(); fig.savefig(fname, dpi=190, bbox_inches='tight'); print("saved",fname)

# ---- FIGURE 2 (per variant): 16 area-pair density ----
def fig_area_pairs(var, fname):
    fig,ax=plt.subplots(figsize=(13,4.2)); x=np.arange(len(PAIRS)); w=0.26
    for k,stim in enumerate(STIMS):
        mus=[];los=[];his=[]
        for pr in PAIRS:
            vals=d[d.stim==stim][f'{var}_{pr}_d'].dropna().values
            mu,lo,hi=boot_ci(vals,B=2000); mus.append(mu);los.append(mu-lo);his.append(hi-mu)
        ax.bar(x+(k-1)*w, mus, w, color=COL[stim], label=stim, yerr=[los,his], capsize=1.5, error_kw={'lw':0.7})
    ax.set_xticks(x); ax.set_xticklabels(PAIRS, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('edge density'); ax.legend(frameon=False, fontsize=9, title='stimulus')
    ax.set_title(f'Variant {var}: directed area-pair FC density by stimulus', fontsize=11, fontweight='bold')
    ax.spines[['top','right']].set_visible(False)
    fig.tight_layout(); fig.savefig(fname, dpi=190, bbox_inches='tight'); print("saved",fname)

for var in ['A','B']:
    fig_within_between(var, _out('fig3_microns_enrichment', f'stim_fc_variant{var}_within_between.png'))
    fig_area_pairs(var, _out('fig3_microns_enrichment', f'stim_fc_variant{var}_area_pairs.png'))
print("DONE")
