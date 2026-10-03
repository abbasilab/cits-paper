"""Build tab:cs_supp (combined score by method, N and p; mean of 3 seeds) from the
aggregated scaling grid. A row with no completed seed prints as $\\times$ (all seeds
exceeded the 30-minute budget); a (method, p, N) never run prints as --.

    python make_tab_cs_supp.py > tab_cs_supp.tex
"""
import os
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(HERE, '..', 'fig1_scaling', 'source_data', 'cits_scaling_aggregated.csv')
METHODS = [('CITS-GPU', r'\textbf{CITS}'), ('PCMCI+', 'PCMCI+'), ('LPCMCI', 'LPCMCI'),
           ('TPC', 'TPC'), ('KernelGC', 'Kernel GC')]
PS = [5, 10, 25, 50, 100, 250, 500, 1000]
NS = [125, 250, 500, 1000]

df = pd.read_csv(CSV, float_precision='round_trip')  # exact parse; 0.9049999… must round to 0.90
cell = {(r.method, int(r.p), int(r.N)): r for r in df.itertuples()}


def fmt(m, p, n):
    r = cell.get((m, p, n))
    if r is None:
        return '--'
    if not (r.n_seeds_ok > 0) or pd.isna(r.cs_mean):
        return r'$\times$'
    return f'{r.cs_mean:.2f}'


lines = []
for k, (m, label) in enumerate(METHODS):
    if k:
        lines.append(r'\midrule')
    for i, n in enumerate(NS):
        lines.append(' & '.join([label if i == 0 else '', str(n)] + [fmt(m, p, n) for p in PS]) + r' \\')
print('\n'.join(lines))
