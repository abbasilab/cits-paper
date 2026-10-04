# cits-paper: analysis code for the CITS methods paper

This repository holds the analysis code behind the paper *Causal Inference in Time Series (CITS)*
(manuscript `CITS_manuscript/main_nn.tex`). It contains the scripts and notebooks that produced
the paper's figures and tables: the simulation benchmarks, the GPU scaling benchmark, the
MICrONS functional-connectivity versus electron-microscopy (FC–EM) comparison, and the Allen
Neuropixels analyses.

The CITS algorithm itself is a separate package and is **not** included here:

- Package: <https://github.com/abbasilab/cits> (`pip install "cits>=1.8.3"`)
- GPU runs also need the cuPC library (<https://github.com/LIS-Laboratory/cupc>); see [Environment](#environment).

**License:** code under the [PolyForm Noncommercial License 1.0.0](LICENSE) (free for noncommercial use; for commercial use, contact the authors); data tables in `source_data/` under [CC BY 4.0](LICENSE-DATA). Copyright (c) 2023-2026 Rahul Biswas. If you use this code or data, please cite the CITS paper (see `CITATION.cff`).

**Status.** This is a snapshot of working scripts, copied on 2026-10-01 from unversioned
locations (provenance in the [appendix](#appendix-file-provenance)). The scripts were **not
refactored**, except that all input and output paths now go through `shared/paths.py` (see
[Environment variables](#environment-variables)); computations, seeds and parameters are
unchanged. Notebook outputs were stripped. Credentials were redacted (see
[Credential redactions](#credential-redactions)). `CHANGES_paths.md` lists every file whose paths
were changed.

---

## Repository layout

| Folder | Contents |
|---|---|
| `shared/` | Helper modules imported by scripts in two or more folders: cuPC wrapper, GPU CITS (lagged cuPC and RCIT versions), Version-B pieces (contemporaneous PC, v-structure orientation, union CPDAG, LSCM refit), simulators (`sim_scm.py`, `glm_spiking_sim.py`), metrics, and baselines (PCMCI+, LPCMCI, kernel Granger, TPC/Granger inside `simulation_benchmark_fc_methods_v3.py`). |
| `fig1_scaling/` | Fig 1 D–F: GPU scaling benchmark (grid driver, fork-free per-cell workers, figure). `supporting/` holds earlier runtime sweeps and a stress test that are not plotted. |
| `fig2_simulations/` | Fig 2: noise sweeps for every method, edge-sign estimates, figure assembly. |
| `table1_baselines/` | Table `tab:modern_baselines`: autoregressive block, spiking-network blocks, motif glyphs. Also the source of `tab:selfedge_supp`. |
| `fig3_microns_enrichment/` | Fig 3: MICrONS stimulus FC (CITS Version B), FC–EM synapse enrichment, within/between-area and area-pair panels. `data_prep/` holds the MICrONS data fetch and conversion steps. |
| `fig4_motifs/` | Fig 4: reproducible CITS graph, motif examples and population conditional-independence statistics. `legacy/` holds the original notebook behind the caption's motif numbers. |
| `fig5_neuropixels/` | Fig 5: method-comparison montage (A), stimulus-specific graphs and region distribution (B, C), text statistics. `notebooks/` holds the original Neuropixels data-preparation notebooks; `legacy/` holds an earlier plotting notebook. |
| `supplement/` | `tab:tau_saturation_supp` (Markov-order sweep) and `fig:scaling_supp` (runtime grid). `tab:cs_supp` and `tab:selfedge_supp` come from `fig1_scaling/` and `table1_baselines/`. |
| `exploratory/` | Code that does **not** produce any figure or table of `main_nn.tex`: the GPU HSIC re-implementation and its validation, and latent-confounder simulations. Kept for reference. |
| `*/source_data/` | Small CSV tables (<200 KB each) that directly hold plotted or tabulated numbers. See [Source data](#source-data). |

## Running the code

Every script adds the repository's `shared/` folder to `sys.path` itself (helpers are imported
by bare module name, as in the original flat directory `analysis/functional_circuitry/`), and
imports CITS from the installed `cits` package. Run each script from its own folder:

```bash
git clone https://github.com/abbasilab/cits && pip install "./cits[rcit]"   # CITS v1.9.0 from source
pip install -r requirements.txt          # plus cuPC and R packages, see Environment
# data access (allensdk, caveclient, datajoint, rpy2) needs a separate env: requirements-data.txt
export CITS_PAPER_OUT=$PWD/outputs       # optional; this is the default
cd fig2_simulations && python _fig2_kgc_noise.py   # writes outputs/fig2_simulations/_fig2_kgc_noise50.csv
```

Outputs go to `$CITS_PAPER_OUT/<figure folder>/` with the original file names. Large inputs are
read from the data roots described in [Environment variables](#environment-variables). Scripts
that re-plot or re-tabulate committed tables read them from `$CITS_PAPER_OUT/<folder>/` when
present and otherwise from the committed `source_data/` copies. `_make_scaling_figure.py`,
`_make_supp_figure.py`, `make_tab_cs_supp.py`, `fig3A_enrichment_stimulus.py` and
`plot_sc_within_vs_between_GRAY.py` therefore run on a fresh clone without any data download.
`_fig2_orig_assemble.py` also needs `_fig2_baselines_noise50.csv` and `_fig2_edgesign.json`, and
`plot_em_areapair_synapse_GRAY.py` needs the bootstrap NPZ; these are not committed
(see [Source data](#source-data)).

---

## Figure and table map

Notation: `<OUT>` = `$CITS_PAPER_OUT` (default `outputs/` in the repository). `$MICRONS_SAVES`,
`$MICRONS_META`, `$AROUSAL_FIGS`, `$NEUROPIXELS_DATA` and `$NEUROPIXELS_CACHE` are the input roots
of [Environment variables](#environment-variables). "fallback `source_data/`" means the script reads
`<OUT>/<folder>/<file>` if it exists and otherwise the committed copy in `<folder>/source_data/`.
In the manuscript folder, `figures/fig1.pdf`, `fig4.pdf` and `fig5.pdf` are blank placeholders.
Figs 1, 3, 4 and 5 were assembled from the panel files below in a slide deck.

### Fig 1 (`fig:graphexample`, `fig:scaling`): CITS framework and GPU scaling

| Panel | Script(s) | Inputs | Outputs | Command |
|---|---|---|---|---|
| A–C (schematic) | none (hand-drawn illustration) | – | – | – |
| D–F (runtime, combined score, sample complexity vs p) | `fig1_scaling/_make_scaling_figure.py` | `grid_v3.csv`, `grid_ext.csv`, `grid_seeds_ext.csv` from `<OUT>/fig1_scaling/` (fallback `source_data/`) | `<OUT>/fig1_scaling/scaling_grid_figure.{png,pdf}`, `<OUT>/fig1_scaling/grid_v3_aggregated.csv` | `cd fig1_scaling && python _make_scaling_figure.py` |
| benchmark grid (3 seeds, 30-min budget) | `_grid_v3.py` → `_cell_cpu.py` (PCMCI+, LPCMCI, TPC, Kernel GC; 32 threads) and `_cell_gpu.py` (CITS on one GPU; GPU id hard-coded to `'3'`) | data simulated by `scaling_benchmark_lg.lg_var` | `<OUT>/fig1_scaling/grid_v3.csv` | `python _grid_v3.py` (from `fig1_scaling/`; it launches `_cell_*.py` by relative name) |
| p = 1000 extension for TPC / Kernel GC | `_grid_ext.py` | – | `<OUT>/fig1_scaling/grid_ext.csv` | `python _grid_ext.py` |
| extra CS-only seeds 3–9 | `_grid_seeds_ext.py` | `<OUT>/fig1_scaling/_feasible_cells.csv` (not included; the `(method, p, N)` cells with status `ok` in `grid_v3.csv`/`grid_ext.csv`) | `<OUT>/fig1_scaling/grid_seeds_ext.csv` (per-cell logs in `<OUT>/fig1_scaling/tmp/`) | `python _grid_seeds_ext.py` |

The PNG labels its panels a (combined score), b (runtime at N = 1000), c (N\*). They appear in
the paper as Fig 1E, 1D and 1F. `supporting/` holds `_extreme_scale.py` (p = 2000–10000 stress
test), `_baseline_clean_runtime.py` and `_baseline_wall_budget.py` (earlier baseline sweeps
superseded by `_grid_v3.py`). None of them feeds a figure; their CSVs go to
`<OUT>/fig1_scaling/supporting/`.

### Fig 2 (`fig:compsim`, `fig:edge_wts`): simulation benchmark

| Panel | Script(s) | Inputs | Outputs | Command |
|---|---|---|---|---|
| A–C (assembly) | `fig2_simulations/_fig2_orig_assemble.py` | the five per-seed CSVs below and `_fig2_edgesign.json` from `<OUT>/fig2_simulations/` (fallback `source_data/`) | `<OUT>/fig2_simulations/fig2_orig.{png,pdf}` | `python _fig2_orig_assemble.py` |
| full figure A–F (as in the paper) | `fig2_simulations/make_fig2_full.py` | `fig2_orig.pdf` above + the scaling grid (relabelled D–F) | `<OUT>/fig2_simulations/fig2.pdf` | `python make_fig2_full.py` |
| GC1, GC2, PC, TPC (η ∈ {0.1, …, 3.5}, 50 seeds) | `_fig2_baselines_noise.py` | `sim_scm.simulate_extended` | `<OUT>/fig2_simulations/_fig2_baselines_noise50.csv` | `python _fig2_baselines_noise.py` |
| CITS, linear paradigms (exhaustive conditioning-set search, `cits.methods.cits_full`, partial correlation; the paper's small-graph CITS) | `_fig2_cits_noise_linear_full.py` (supersedes `_fig2_cits_noise_perregime.py`, which used the neighbour-restricted cuPC search) | same | `<OUT>/fig2_simulations/_fig2_cits_noise50.csv` (holds the two linear paradigms) | `python _fig2_cits_noise_perregime.py` |
| CITS, non-linear paradigms (RCIT, packaged `cits.cits_rcit`, no cap on the conditioning set) | `_fig2_cits_noise_nlng.py` | same | `<OUT>/fig2_simulations/_fig2_cits_noise50_nlng.csv` | `CITS_DEV=cuda:0 python _fig2_cits_noise_nlng.py` (or `CITS_DEV=cpu`) |
| Kernel GC | `_fig2_kgc_noise.py` | same | `<OUT>/fig2_simulations/_fig2_kgc_noise50.csv` | `python _fig2_kgc_noise.py` |
| PCMCI+, LPCMCI (ParCorr) | `_fig2_pcmci_noise.py` | same | `<OUT>/fig2_simulations/_fig2_pcmci_noise50.csv` | `python _fig2_pcmci_noise.py` |
| C (edge weights and signs) | `_fig2_edgesign.py` | same | `<OUT>/fig2_simulations/_fig2_edgesign.json` | `python _fig2_edgesign.py` |
| text: signs agree with a partial-rank estimate | `_fig2_edgedir.py` | same | `<OUT>/fig2_simulations/_fig2_edgedir.json` | `python _fig2_edgedir.py` |

**Fig 2 provenance note (2026-10-04).** The linear CITS curves now use the exhaustive search (CS 1.000 at every noise level); the neighbour-restricted cuPC search had given 0.993 on Linear Gaussian 1 through collider false positives.

**Earlier note (2026-10-03).** The committed CITS source data come from a rerun with the
released code. Linear Gaussian 1: the original run gave CS 0.996 at every noise level; the rerun gives
0.993 (3 of 50 seeds differ by one borderline false-positive edge). The rerun matches an independent
CPU PC-stable reference exactly, is identical across GPUs and machines, and is what the paper reports;
the original run predates a change to the simulator module and cannot be regenerated. Non-linear
paradigms: rerun without the conditioning-set cap (`max_cond_size=None`), matching Table 1; this
changes 3 of 800 runs (η = 2.5, 1.5, 3.0; at most 0.008) and nothing at η = 1.

### Table 1 (`tab:modern_baselines`)

**Baseline implementations (2026-10-03).** For the small-scale benchmarks (Fig 2 A/B, Table 1), TPC
and kernel Granger causality use the authors' implementations:
`table1_baselines/tpc_official_pkg.py` runs `timeawarepc` 2.0.2 (`cfc_tpc`, Fisher-z, α = 0.05,
no bootstrap; needs R `pcalg` through rpy2), and `table1_baselines/kgc_official_marinazzo.py` runs
Marinazzo's original `causality.m` (github.com/danielemarinazzo/KernelGrangerCausality, commit
c043147; polynomial kernel of order 2, model order 2) unchanged in GNU Octave 10.3, with
`octave_shims/corr.m` supplying MATLAB's `[r, p] = corr(x, y)`. Both run in parallel
(`N_WORKERS`). Per-seed results: `table1_baselines/source_data/{tpc,kgc}_official_*.csv`. The
scaling benchmark (Fig 2 D–F) keeps the faster in-repo reimplementations (`run_tpc`,
`kernel_granger_baseline.py`), because the original implementations cannot run at p = 100–1000.
: combined score versus state-of-the-art baselines

| Block | Script(s) | Inputs | Outputs | Command |
|---|---|---|---|---|
| Autoregressive rows (assembly) | `table1_baselines/_sd_assemble.py` | `_sd_ar_lingauss_cupc.csv`, `_sd_gpu2_rcit.csv`, `_sd_local.csv`, `_sd_tigra.csv`, `_sd_spk_cits_cupc_convdia.csv` from `<OUT>/table1_baselines/`, and `simulation_results_directed_{granger,tpc_original,kernel_gc,pcmci_plus,lpcmci}.csv` from `<OUT>/table1_baselines/` (fallback: the author's original run in `$AROUSAL_FIGS/2026-06-06_directed_cs/`) | `<OUT>/table1_baselines/_sd_assembled.csv` (only its four autoregressive paradigms are used) | `python _sd_assemble.py` |
| CITS on Linear Gaussian 1 and 2 (cuPC) | `_sd_ar_lingauss_cupc.py` | `sim_scm` | `<OUT>/table1_baselines/_sd_ar_lingauss_cupc.csv` | `python _sd_ar_lingauss_cupc.py` |
| CITS on Non-linear Non-Gaussian 1 and 2 (GPU RCIT) | `_sd_gpu2_rcit.py` (the name indicates the gpu-2 server) | `sim_scm`, `glm_spiking_sim` | `<OUT>/table1_baselines/_sd_gpu2_rcit.csv` | `python _sd_gpu2_rcit.py` |
| GC2 (autoregressive) | `_sd_local.py` | `sim_scm`, `glm_spiking_sim` | `<OUT>/table1_baselines/_sd_local.csv` | `python _sd_local.py` |
| GC1 (`granger`), TPC (`tpc_original`) | `directed_benchmark_cpu.py` | `simulation_benchmark_fc_methods_v3.simulate_extended` | `<OUT>/table1_baselines/simulation_results_directed_{granger,tpc_original}.csv` | `python directed_benchmark_cpu.py --methods 18,19` |
| PCMCI+ | `directed_benchmark_gpu.py` (method 14) | same | `<OUT>/table1_baselines/simulation_results_directed_pcmci_plus.csv` | `python directed_benchmark_gpu.py --methods 14` |
| Kernel GC | `directed_benchmark_kernel_gc.py` | same | `<OUT>/table1_baselines/simulation_results_directed_kernel_gc.csv` | `python directed_benchmark_kernel_gc.py` |
| LPCMCI | `directed_benchmark_lpcmci.py` | same | `<OUT>/table1_baselines/simulation_results_directed_lpcmci.csv` | `python directed_benchmark_lpcmci.py` |
| Spiking networks, recurrence-free control and recurrent | `_spiking_pmatched_cpu.py` (TPC, GC1, GC2, Kernel GC), `_spiking_pmatched_pcmci.py` (PCMCI+, LPCMCI; `tigra` env), `_spiking_pmatched_cits.py` (CITS RCIT on GPU; ran on gpu-2) | simulator inside each script | stdout only (`CSfull=mean±sd selfFrac=...` per motif) | `SH=0 python <script>` (control) and `SH=-1.5 python <script>` (recurrent) |
| Motif glyphs | `plot_spiking_motifs.py` | – | `<OUT>/table1_baselines/fig_motif_*.{pdf,png}` | `python plot_spiking_motifs.py` |

`_sd_tigra.py` and `_sd_spk_cits_cupc_convdia.py` produce spiking rows of an earlier table
version. Those rows are superseded by the `_spiking_pmatched_*` runs, but `_sd_assemble.py`
reads both CSVs, so the scripts are kept. `sim_scm.simulate_extended` is a standalone copy of
`simulation_benchmark_fc_methods_v3.simulate_extended` (identical AST).

**Recorded spiking results.** The spiking runs write to stdout only. The CPU and PCMCI logs
survive only in an ephemeral session directory, so their values are transcribed here
(`CSfull` mean ± s.d. over 50 simulations; `self` = fraction of neurons given a self-edge).

| Motif | Mode | TPC | GC1 | GC2 | Kernel GC | PCMCI+ | LPCMCI |
|---|---|---|---|---|---|---|---|
| convergence | control | 0.821±0.077 (self 0.49) | 0.966±0.051 | 0.969±0.051 | 0.848±0.011 | 0.889±0.100 (self 0.10) | 0.911±0.086 (self 0.07) |
| diamond (common cause and effect) | control | 0.643±0.131 (0.46) | 0.848±0.066 | 0.827±0.055 | 0.917±0.000 | 0.730±0.076 (0.24) | 0.752±0.070 (0.22) |
| chain | control | 0.668±0.152 (0.66) | 0.931±0.069 | 0.957±0.051 | 0.846±0.000 | 0.883±0.086 (0.13) | 0.889±0.076 (0.12) |
| convergence | recurrent | 0.863±0.160 (0.88) | 0.237±0.099 | 0.351±0.081 | 0.206±0.000 | 0.869±0.123 (1.00) | 0.869±0.140 (1.00) |
| diamond | recurrent | 0.745±0.071 (0.95) | 0.343±0.082 | 0.225±0.112 | 0.375±0.000 | 0.757±0.107 (1.00) | 0.725±0.127 (1.00) |
| chain | recurrent | 0.957±0.116 (0.96) | 0.197±0.086 | 0.389±0.058 | 0.206±0.000 | 0.891±0.114 (1.00) | 0.889±0.102 (1.00) |
| depression | recurrent | 0.759±0.108 (0.71) | 0.375±0.064 | 0.377±0.071 | 0.391±0.063 | 0.876±0.129 (1.00) | 0.871±0.128 (1.00) |

GC1, GC2 and Kernel GC never emit self-edges (self = 0.00). CITS per-seed results are in
`table1_baselines/source_data/cits_rcit_uncapped_perseed.csv` (`make_cits_rcit_table1.py`, which
now writes `<OUT>/table1_baselines/cits_rcit_uncapped_perseed.csv`). Means:
control 0.992 / 1.000 / 1.000 (spurious self-edges 0.33%); recurrent 0.993 / 0.990 / 0.994 /
0.969 (self-edge TPR 99.1%). These come from the packaged `cits.cits_rcit` with no cap on the
conditioning-set size (the scripts below, as updated 2026-10-03; the original runs used
`max_cond_size=5`, which differs in 2 of 450 runs: common cause & effect 0.998 / 0.993).
The table's ±s.d. values are from this rerun.

### Fig 3 (`fig:microns_stim`): MICrONS stimulus FC versus the EM connectome

| Panel | Script(s) | Inputs | Outputs | Command |
|---|---|---|---|---|
| A (visual areas on the cortical surface) | none (rendered image) | – | – | – |
| B (spatial layout, Clip) | `fig3_microns_enrichment/plot_fig3B_cfc_spatial_clip.py` | Version-B FC `<OUT>/fig3_microns_enrichment/out/fc/s{s}sc{sc}f{f}_clip.npz` (key `fcB`, from `stim_run.py`; was staged from gpu-2 into a session scratchpad), calcium metadata `$MICRONS_SAVES/calcium_npy/{ids,fields,unionids}_session*_scan*.npy`, `$MICRONS_META/all_unit_coords.pkl`, `$MICRONS_META/all_unit_areas.csv` | `<OUT>/fig3_microns_enrichment/fig3B_cfc_spatial_clip.{pdf,png}` | `python plot_fig3B_cfc_spatial_clip.py` |
| C (FC–EM synapse enrichment, 3 methods) | `fig3A_enrichment_stimulus.py` (file name says 3A; it is panel C) | `panelA_perfield_counts.csv` (CITS), `stim_baseline_perfield_granger.csv` (GC2), `stim_baseline_perfield_lagged01.csv` (lagged correlation, lag 0 ∪ lag 1), all from `<OUT>/fig3_microns_enrichment/` (fallback `source_data/`) | `<OUT>/fig3_microns_enrichment/fig3A_enrichment_stimulus.{pdf,png}` | `python fig3A_enrichment_stimulus.py` |
| C, CITS per-field counts | `panelA_perfield.py` | FC as for B, `$MICRONS_SAVES/{matched_df_v1718,synapses_matcheddf_frompre_v1718}.pkl`, EM field keys `$AROUSAL_FIGS/2026-04-29/fig4/bootstrap_sf_correlation_13sess.npz` | `<OUT>/fig3_microns_enrichment/panelA_perfield_counts.csv` (and `panelA_fc_em_enrichment_versionB.{csv,png}`) | `python panelA_perfield.py` |
| C, GC2 and lag-1 correlation per-field counts | `stim_baseline_enrichment.py` | calcium `$MICRONS_SAVES/calcium_npy/`, `$MICRONS_SAVES/{stim}_timepoints_*.pkl` (same windows as CITS), FC availability from `<OUT>/fig3_microns_enrichment/out/fc/`, EM tables and keys as above | `<OUT>/fig3_microns_enrichment/stim_baseline_perfield_{granger,lagged1,laggedmax}.csv` | `python stim_baseline_enrichment.py granger` (and `lagged1`) |
| C, lag 0 ∪ lag 1 correlation counts and paired Δ vs CITS | `corr_l0l1_final.py` (imports `stim_baseline_enrichment` from its own folder) | as above, plus `panelA_perfield_counts.csv` (fallback `source_data/`) | `<OUT>/fig3_microns_enrichment/stim_baseline_perfield_lagged01.csv` | `python corr_l0l1_final.py` |
| C, pooled CITS fold summary | `panelA_fc_em_enrichment_versionB.py` | as for `panelA_perfield.py` | `<OUT>/fig3_microns_enrichment/panelA_fc_em_enrichment_versionB.{csv,png}` | `python panelA_fc_em_enrichment_versionB.py` |
| D (within vs between, per stimulus) and F (directed area pairs) | `_plot_arousal_style.py` | `<OUT>/fig3_microns_enrichment/results/stimulus_fc_combined.csv` (concatenation of the `stim_run.py` shards `<OUT>/fig3_microns_enrichment/out/results_gpu{0,1}_shard{0,1}.csv`) | `<OUT>/fig3_microns_enrichment/stim_fc_variant{A,B}_within_between.png`, `stim_fc_variant{A,B}_area_pairs.png`, `report.md` (Variant B is the paper's) | `python _plot_arousal_style.py` |
| D, statistics quoted in the text (Wilcoxon, rank-biserial r) | `task9.py` | `<OUT>/fig3_microns_enrichment/results/stimulus_fc_combined.csv` | stdout | `python task9.py` |
| E (EM within vs between) | `bootstrap_em_within_vs_between_versionBsafe.py`, then `plot_sc_within_vs_between_GRAY.py` | EM tables in `$MICRONS_SAVES`, `$MICRONS_META/all_unit_areas.csv`, EM field keys, arousal-pipeline FC universe `$MICRONS_SAVES/{cits_plus_pc_versionBsafe_2026-05-28/,statement_dfs_session*_scan*.pkl}` (see gaps) | `<OUT>/fig3_microns_enrichment/sc_within_vs_between_{synapse,perfield}_versionBsafe.csv`; `<OUT>/fig3_microns_enrichment/sc_within_vs_between_synapse_GRAY.png` (the plot reads the CSV with fallback `source_data/`) | `python bootstrap_em_within_vs_between_versionBsafe.py && python plot_sc_within_vs_between_GRAY.py` |
| E, "6.3-fold, 95% CI 5.6–7.0" in the text | `task6.py` (Katz CI on pooled counts; it also prints superseded arousal-provenance folds) | `sc_within_vs_between_perfield_versionBsafe.csv` (fallback `source_data/`); the arousal-pipeline `sc_enrichment_perfield_{versionBsafe,granger}.csv` in `$AROUSAL_FIGS/2026-06-01_versionBsafe/exp_analysis/fig1_sc/` | stdout | `python task6.py` |
| G (EM area pairs) | `bootstrap_em_areapair_synapse_versionBsafe.py`, then `plot_em_areapair_synapse_GRAY.py` | as for E | `<OUT>/fig3_microns_enrichment/em_areapair_synapse_versionBsafe.{csv,npz}`; `<OUT>/fig3_microns_enrichment/em_areapair_synapse_GRAY.png` (the plot needs the NPZ, which is not committed) | `python bootstrap_em_areapair_synapse_versionBsafe.py && python plot_em_areapair_synapse_GRAY.py` |
| FC computation, 124 fields × 3 stimuli (Variant A = cuPC lagged; Variant B = cuPC lagged + PC-contemporaneous + union + LSCM refit) | `stim_run.py` (imports `stim_fc_pipeline.py`) | `$MICRONS_SAVES/calcium_npy/` arrays, `$MICRONS_SAVES/{stim}_timepoints_*.pkl`, `$MICRONS_META/all_unit_areas.csv` | `<OUT>/fig3_microns_enrichment/out/fc/s{s}sc{sc}f{f}_{stim}.npz` (`fcA`, `fcB`), `<OUT>/fig3_microns_enrichment/out/results_gpu{G}_shard{K}.csv` | `python stim_run.py --gpu 0 --nshards 2 --shard 0 --variants AB` (and `--gpu 1 --shard 1`) |
| earlier summaries of the same FC | `stim_aggregate.py`, `aggregate_and_plot.py` | shards / combined CSV | `<OUT>/fig3_microns_enrichment/out/{summary,stats}.csv`, `out/figures/`; `<OUT>/fig3_microns_enrichment/stim_fc_variant*.png` | – |

`data_prep/` (MICrONS inputs, run in order). Every step writes to
`<OUT>/fig3_microns_enrichment/data_prep/` (logs in its `logs/` subfolder). A step that needs the
previous step's file reads it from there if present, and otherwise from `$MICRONS_SAVES`.
Upstream pickles (`merged_df_*`, `filtered_*_neurons_*`) are read from `$MICRONS_SAVES`.

1. `fetch_all_areas.py`, `fetch_all_coords.py`: DataJoint (`microns_phase3.nda.AreaMembership`, `nda.ScanUnit`) → `all_unit_areas.csv`, `all_unit_coords.pkl`. Needs `DJ_USER` and `DJ_PASS`.
2. `cits_finalize.ipynb`, cells 3–5: CAVE `coregistration_manual_v4` at materialization 1181 → `matched_df.pkl`. Cells 6–8 query areas and v1181 synapses (superseded by step 4). The rest of the notebook is unrelated exploration. Run it from its folder.
3. `build_matched_df_v1718.py`: re-roots `matched_df.pkl` supervoxels to materialization 1718 → `matched_df_v1718.pkl`.
4. `build_synapses_pkl_v1718.py`: `synapses_pni_2` at materialization 1718 for every matched presynaptic root → `synapses_matcheddf_frompre_v1718.pkl`.
5. `_convert_merged.py <merged_df_session{s}_scan{sc}.pkl>`, `_extract_idfield.py`, `_extract_unionids.py`: calcium traces and unit metadata → `calcium_npy/{calcium,ids,fields,unionids,units}_session{s}_scan{sc}.npy`.

The Fig 3 scripts read these files from `$MICRONS_SAVES` and `$MICRONS_META`. To use freshly
built ones, set both variables to `<OUT>/fig3_microns_enrichment/data_prep` and put the other
`$MICRONS_SAVES` inputs (`{stim}_timepoints_*.pkl`, `statement_dfs_*.pkl`,
`cits_plus_pc_versionBsafe_2026-05-28/`) next to them.

The upstream pickles `merged_df_session*_scan*.pkl`, `{clip,Monet,Trippy}_timepoints_session*_scan*.pkl`
and `filtered_{stim}_neurons_session*_scan*.pkl` are not produced by any script here (see [Known gaps](#known-gaps-and-discrepancies)).

### Fig 4 (`fig:corrplots`): conditional-independence signatures in neural motifs

| Panel | Script(s) | Inputs | Outputs | Command |
|---|---|---|---|---|
| graph + population records | `fig4_motifs/run_motif_population_v2.py` (reproducible lagged CITS graph `fast_cits_pcorr`, τ = 1, α = 0.05) | `$NEUROPIXELS_DATA/ID791319847_natural_scenes_bin_0.01_X_idx-0.p`, `..._units2use_stim_natural_scenes.p` | `<OUT>/fig4_motifs/motif_population_v2_records.pkl`, `motif_population_v2.json`, `motif_population_v2_adjacency.npy` | `python run_motif_population_v2.py` |
| A (worked examples) | `plot_motif_examples_scatter_v2.py` | records + data above | `<OUT>/fig4_motifs/motif_examples_scatter_v2.{png,pdf}` | `python plot_motif_examples_scatter_v2.py` |
| B–D (adjacent / non-adjacent / collider populations) | `plot_motif_population_v2_split.py` | same | `<OUT>/fig4_motifs/motif_population_v2_{a,b,c}.{png,pdf}` | `python plot_motif_population_v2_split.py` |
| combined population figure (not used) | `plot_motif_population_v2.py` | same | `<OUT>/fig4_motifs/motif_population_v2.{png,pdf}`, `motif_population_detail_v2.png` | – |
| legacy motif numbers quoted in the caption | `legacy/script copy.ipynb` (cells 6–14: triple and quadruple searches), `legacy/run_lagged_search.py` | `$NEUROPIXELS_DATA`; original CITS results (`citsproject/save/`), expected in `<OUT>/fig4_motifs/legacy/save/` | `<OUT>/fig4_motifs/legacy/` (`lagged_triples_results_*.pkl`, `figs/panels/*.png`) | – |

### Fig 5 (`fig:resneuropixels`, `fig:stimtypegraphs`): Allen Neuropixels, session 791319847

| Panel | Script(s) | Inputs | Outputs | Command |
|---|---|---|---|---|
| data: 10 ms binning, active-unit masks, 68-unit union frame | `notebooks/script.ipynb` cell 44 (AllenSDK download, presentation-wise 10 ms counts → `ID791319847_{stim}_bin_0.01_P.p`, masks `..._units2use_stim_{stim}.p`, units nonzero in >20% of bins); cell 6 (union of active units across the 4 stimuli, ordered by `ecephys_structure_acronym`) | Allen Brain Observatory cache (`$NEUROPIXELS_CACHE`) | `<OUT>/fig5_neuropixels/notebooks/data/*` (was `citsproject/data/`; the scripts read `$NEUROPIXELS_DATA`) | run the notebook cells from `notebooks/` |
| A, CITS column (Version B on the first 90 s of concatenated presentations) | `_montage_cc_cits.py` | `$NEUROPIXELS_DATA/P_raw_{stim}.npy`, masks, `cits_v2_union_{units,labels}.npy` (from `<OUT>/fig5_neuropixels/` if present, else `$NEUROPIXELS_DATA`) | `<OUT>/fig5_neuropixels/cmp_cc_{stim}_CITS_68.npy` | `python _montage_cc_cits.py` |
| A, GC2 column (R bruceR conditional Granger, VAR(1)) | `_montage_cc_gc2.py` | same | `<OUT>/fig5_neuropixels/cmp_cc_{stim}_GC2_68.npy`, `cmp_cc_{stim}_GC2adj_68.npy` | `conda activate gc2r && python _montage_cc_gc2.py` |
| A, Pearson and GC1 columns; significance masking of all four | `_montage_sig_magnitude.py` | same + the `cmp_cc_*` arrays | `<OUT>/fig5_neuropixels/cmp_ccsig_{stim}_{CORR,GC1,GC2,CITS}_68.npy` | `python _montage_sig_magnitude.py` |
| A (render) | `render_fine_ccsig.py` | `cmp_ccsig_*` | `<OUT>/fig5_neuropixels/cmp_montage_4col_ccsig.png`, `region_legend_fine.png` | `python render_fine_ccsig.py` |
| text: density, Pearson support, \|r\| percentile | `_verify_fig5_stats.py` | `cmp_ccsig_*`, `P_raw_*` | stdout | `python _verify_fig5_stats.py` |
| B, C (per-90 s-window Version B, edges in ≥90% of windows) | `_stimtypes_90swin_compute.py` (imports `_neuropixels_versionB_pooled.py`) | `P_raw_{stim}.npy`, masks, union frame | `<OUT>/fig5_neuropixels/cits_v2_directedB_W90WIN_{stim}_{fwd,w,pboot}68.npy` | `python _stimtypes_90swin_compute.py` |
| B, C (render) | `regen_cfc_stimtypes.py` | the W90WIN arrays | `<OUT>/fig5_neuropixels/cfc_stimtypes_directed_th90_w90win90_thin.{pdf,png}` (per-panel PNGs and a preview in `<OUT>/fig5_neuropixels/panels/`) | `CFC_PREFIX=cits_v2_directedB_W90WIN CFC_THRESH=0.9 CFC_OUTTAG=_w90win90_thin python regen_cfc_stimtypes.py` |
| text: ADF stationarity (116 series, BH-FDR) | `task7.py` | `$NEUROPIXELS_DATA` | stdout | `python task7.py` |

`notebooks/script_matchbarplot.ipynb` holds the original plotting code (graph layout cell 24,
region bars cell 28) that `regen_cfc_stimtypes.py` reproduces verbatim. Its cell 3 imports
`ace_tools`, a helper that is not installable. `notebooks/neuropixels_testresults.py` is
imported by the notebooks. The three Neuropixels notebooks read and write their data folder
`<OUT>/fig5_neuropixels/notebooks/data/` (was `data/`), results `<OUT>/fig5_neuropixels/notebooks/save/`
(was `save/` and `D:\...\mice-aibs\save\`), and figures under `<OUT>/fig5_neuropixels/`. To run
them on existing prepared data, copy or link that data into `<OUT>/fig5_neuropixels/notebooks/data/`;
to feed the scripts with freshly prepared data, set `NEUROPIXELS_DATA` to that folder. `legacy/scc_clustering.ipynb` is an earlier graph-plotting notebook
and does not feed the final figure.

### Supplementary material

| Item | Script(s) | Inputs | Outputs | Command |
|---|---|---|---|---|
| `tab:tau_saturation_supp` (CS of CITS at τ = 1, 2, 3; 20 sims; GPU RCIT, \|S\| ≤ 5) | `supplement/tau_sensitivity_gpu.py` | `simulation_benchmark_fc_methods_v3.simulate_extended` | `<OUT>/supplement/simulation_results_directed_tau_sensitivity_gpu.csv` (the CTRNN rows are not tabulated) | `python tau_sensitivity_gpu.py` |
| `tab:selfedge_supp` | the `_spiking_pmatched_*` runs in `table1_baselines/` | – | the `self` fractions in the table above (TPC 88% / 54% = mean over motifs; PCMCI+ 16%; LPCMCI 14%) | see Table 1 |
| `fig:scaling_supp` (runtime vs p at each N) | `supplement/_make_supp_figure.py` | `grid_v3.csv`, `grid_ext.csv` from `<OUT>/fig1_scaling/` (fallback `fig1_scaling/source_data/`) | `<OUT>/supplement/scaling_grid_supp.png`, `<OUT>/supplement/scaling_supp.{png,pdf}` | `python _make_supp_figure.py` |
| `tab:cs_supp` (CS by method, N, p; mean of 3 seeds) | `supplement/make_tab_cs_supp.py` | `fig1_scaling/source_data/cits_scaling_aggregated.csv` | table body (LaTeX), identical to the paper | `python make_tab_cs_supp.py > tab_cs_supp.tex` |

`make_tab_cs_supp.py` prints every cell of `tab:cs_supp` (verified identical to the paper):
`f"{cs_mean:.2f}"`, × when no seed finished within the budget, -- when not run.
`cits_scaling_aggregated.csv` itself was built by an unsaved one-liner: concatenate `grid_v3.csv`
and `grid_ext.csv`, then group by `(method, p, N)` to get the mean and s.d. of `cs` over
`status == "ok"` seeds and the mean `runtime_sec`.

---

## Data sources

- **Allen Brain Observatory, Visual Coding Neuropixels.** Session **791319847** (116-day-old
  male, six probes, 555 units). Stimuli: natural scenes, static gratings and Gabor patches;
  flashes are used only to build the 68-unit union frame. Access is through AllenSDK
  `EcephysProjectCache` (`notebooks/script.ipynb`) and the session NWB file. Spikes are binned
  at 10 ms over stimulus presentations; Fig 5 uses the unsmoothed counts.
- **MICrONS Minnie65.** Two-photon calcium traces and unit metadata from the MICrONS phase-3
  DataJoint database (`microns_phase3`, <https://github.com/cajal/microns_phase3_nda>).
  The EM reconstruction comes from CAVE datastack `minnie65_public`. The coregistration
  table `coregistration_manual_v4` was queried at materialization **1181** and re-rooted to
  materialization **1718**. Synapses come from `synapses_pni_2` at materialization **1718**.
  The analysis covers 124 imaging fields, 39 of them EM-coregistered, under three stimuli
  (Clip, Monet, Trippy).
- **Simulated data.**
  - Four-variable autoregressive paradigms (Linear Gaussian 1/2, Non-linear Non-Gaussian 1/2):
    `shared/sim_scm.py` (`simulate_extended`), identical to the function in
    `shared/simulation_benchmark_fc_methods_v3.py`.
  - GLM-coupled Poisson spiking networks: the `sim()` function inside each
    `table1_baselines/_spiking_pmatched_*.py`; an earlier version is `shared/glm_spiking_sim.py`.
  - Random linear-Gaussian VAR(1) graphs for scaling: `fig1_scaling/scaling_benchmark_lg.py` (`lg_var`).

## Environment

Main environment (`microns` conda env): **Python 3.10.13**, packages pinned in
`requirements.txt`. The analyses imported CITS from a local checkout of
<https://github.com/abbasilab/cits> (v1.8.x; the env's PyPI `cits==1.3` was shadowed by
`sys.path`). Use `cits>=1.8.3`.

Auxiliary environments:

| Env | Used for | Versions |
|---|---|---|
| `tigra` | `_sd_tigra.py`, `_spiking_pmatched_pcmci.py`, `_glm_suite_baselines.py` | Python 3.10.21, tigramite 5.2.10.1 |
| `gc2r` | `fig5_neuropixels/_montage_cc_gc2.py` (bruceR conditional Granger through rpy2) | Python 3.12.14, R 4.3.3, r-bruceR 2023.9, r-vars 1.6-1, rpy2 3.5.11. Create with `conda create -n gc2r -c conda-forge --solver classic r-base=4.3 r-bruceR r-vars rpy2 numpy pandas h5py scipy` and run with the env activated so rpy2 uses its R. |
| `micronsnew` | DataJoint fetches in `fig3_microns_enrichment/data_prep/` | Python 3.9.21, datajoint 0.12.9, caveclient 7.11.1, `microns_phase3` from <https://github.com/cajal/microns_phase3_nda> (local MICrONS DataJoint database at `127.0.0.1:3306`) |
| R + kpcalg | `exploratory/hsic_gpu/hsic_validate_gpu_vs_kpcalg.py` only (calls `Rscript`) | kpcalg 1.0.1 |

GPU:

- cuPC: <https://github.com/LIS-Laboratory/cupc> (commit 8ed927c), built with
  `nvcc -O3 --shared -Xcompiler -fPIC -o Skeleton.so cuPC-S.cu` (CUDA 12.2). Point the `CUPC_DIR`
  environment variable at the folder holding `Skeleton.so` (default `~/repos/cupc`); both the
  `cits` package and `shared/_cupc_wrapper.py` read it.
- PyTorch 2.9.1 with CUDA 12.8 wheels for the RCIT and HSIC kernels.
- The scaling benchmark used one exclusive GPU for CITS and 32 CPU threads for the baselines.
  Every grid cell ran in a fresh subprocess; never fork after NumPy/BLAS initialization.

---

## Environment variables

All input and output locations are defined once, in `shared/paths.py`. Each root is an
environment variable with a default inside the repository, so a fresh clone runs without editing
any script. Every script puts `shared/` on `sys.path` itself and imports CITS from the installed
`cits` package (no script prepends a local CITS checkout any more).

| Variable | Default | Holds |
|---|---|---|
| `CITS_PAPER_OUT` | `<repo>/outputs` | **Every file a script writes.** Layout: `<OUT>/<figure folder>/<original file name>`, e.g. `outputs/fig1_scaling/grid_v3.csv`, `outputs/fig3_microns_enrichment/out/fc/*.npz`, `outputs/fig3_microns_enrichment/data_prep/calcium_npy/*.npy`. Also the place where a script looks first for an intermediate made by another script of this repository. |
| `CITS_PAPER_DATA` | `<repo>/data` | Parent of the input roots below (only used for their defaults). |
| `MICRONS_SAVES` | `$CITS_PAPER_DATA/microns_saves` | MICrONS calcium arrays and EM tables: `calcium_npy/`, `merged_df_session*_scan*.pkl`, `filtered_{stim}_neurons_*.pkl`, `{clip,Monet,Trippy}_timepoints_*.pkl`, `matched_df*.pkl`, `synapses_matcheddf_frompre_v1718.pkl`, `statement_dfs_session*_scan*.pkl`, `cits_plus_pc_versionBsafe_2026-05-28/`. |
| `MICRONS_META` | `$CITS_PAPER_DATA/microns_meta` | `all_unit_areas.csv`, `all_unit_coords.pkl`. |
| `AROUSAL_FIGS` | `$CITS_PAPER_DATA/arousal_figures` | Outputs of the companion arousal-paper pipeline that Fig 3 reads: the 39 EM field keys `2026-04-29/fig4/bootstrap_sf_correlation_13sess.npz`, and (for `task6.py`) `2026-06-01_versionBsafe/exp_analysis/fig1_sc/sc_enrichment_perfield_{versionBsafe,granger}.csv`. `_sd_assemble.py` also falls back to `2026-06-06_directed_cs/` here for the per-seed baseline CSVs. |
| `NEUROPIXELS_DATA` | `$CITS_PAPER_DATA/neuropixels` | Prepared Allen Neuropixels data for session 791319847: `ID791319847_{stim}_bin_0.01_P.p`, `..._units2use_stim_{stim}.p`, `..._bin_0.01_X_idx-{i}.p`, `P_raw_{stim}.npy`, and the union frame `cits_v2_union_{units,labels}.npy`. |
| `NEUROPIXELS_CACHE` | `$CITS_PAPER_DATA/allen_cache` | AllenSDK `EcephysProjectCache` folder (`manifest.json`) used by the Neuropixels notebooks. |
| `CUPC_DIR` | `~/repos/cupc` | Folder with the compiled cuPC library `Skeleton.so` (same default as the `cits` package). |

Helpers in `shared/paths.py`: `out(subdir, name)` and `outdir(subdir)` return paths under
`CITS_PAPER_OUT` and create the folder; `data(...)`, `microns_saves(...)`, `microns_meta(...)`,
`arousal_figs(...)` and `neuropixels(...)` build input paths; `result(subdir, name, fallback=None)`
returns `<OUT>/<subdir>/<name>` if it exists, else the fallback (default: the committed
`<subdir>/source_data/<name>`).

### Reproducing the paper's runs exactly (author's machine)

Point the input roots at the original data layout. Outputs then go to a fresh folder, and nothing
in the original directories is overwritten:

```bash
export CITS_PAPER_OUT=$HOME/cits-paper-outputs
export MICRONS_SAVES=/data1/rb1/microns/saves             # on gpu-2: /home/rbiswas1/microns_data/saves
export MICRONS_META=/home/rbiswas1/microns                # all_unit_areas.csv, all_unit_coords.pkl
export AROUSAL_FIGS=/home/rbiswas1/microns/arousal_paper_overleaf/figures
export NEUROPIXELS_DATA=/home/rbiswas1/citsproject/data
export CUPC_DIR=/home/rbiswas1/repos/cupc
```

A few intermediates lived outside these roots (in the manuscript figure folder, the stimulus-FC
folder or a session scratchpad). To reuse them instead of recomputing, copy or link them into the
output tree:

| Original location | Link or copy to |
|---|---|
| `/home/rbiswas1/microns/analysis/stimulus_fc/out/` (`fc/*.npz`, shard CSVs; on gpu-2) | `$CITS_PAPER_OUT/fig3_microns_enrichment/out/` |
| `/home/rbiswas1/microns/analysis/stimulus_fc/results/stimulus_fc_combined.csv` | `$CITS_PAPER_OUT/fig3_microns_enrichment/results/` |
| `/home/rbiswas1/microns/CITS_manuscript/figures/cits_v2_union_{units,labels}.npy` | `$NEUROPIXELS_DATA/` or `$CITS_PAPER_OUT/fig5_neuropixels/` |
| `/home/rbiswas1/microns/CITS_manuscript/figures/{cmp_*,cits_v2_*}.npy` (Fig 5 arrays) | `$CITS_PAPER_OUT/fig5_neuropixels/` |
| `/home/rbiswas1/microns/CITS_manuscript/figures/motif_population_v2_records.pkl` | `$CITS_PAPER_OUT/fig4_motifs/` |
| `/home/rbiswas1/citsproject/save/` (legacy CITS results) | `$CITS_PAPER_OUT/fig4_motifs/legacy/save/` and `$CITS_PAPER_OUT/fig5_neuropixels/notebooks/save/` |
| `/home/rbiswas1/citsproject/data/` (for the Neuropixels notebooks) | `$CITS_PAPER_OUT/fig5_neuropixels/notebooks/data/` |
| `/home/rbiswas1/microns/analysis/functional_circuitry/{grid_v3,grid_ext,grid_seeds_ext,_feasible_cells}.csv`, `_fig2_*`, `_sd_*` CSVs/JSON | `$CITS_PAPER_OUT/fig1_scaling/`, `fig2_simulations/`, `table1_baselines/` |

The staged calcium metadata that Fig 3B/3C read from the session scratchpad
(`panelA_stage/npy/{ids,fields,unionids}_*.npy`) is now read from `$MICRONS_SAVES/calcium_npy/`
(the files that `data_prep/` builds; the staged copies were taken from that folder on gpu-2).

### Where the old hard-coded paths went

| Old path | Now |
|---|---|
| `sys.path` insert of `/home/rbiswas1/repos/cits` | removed; the installed `cits` package is used |
| `sys.path` insert of `/home/rbiswas1/microns/analysis/functional_circuitry` | the repository's `shared/` |
| `sys.path` insert of `/home/rbiswas1/repos/cupc` | removed (it holds no Python module); cuPC is found through `CUPC_DIR` |
| `sys.path` insert of `.../CITS_manuscript/figures` or `.../analysis/stimulus_fc` | the script's own folder |
| `/home/rbiswas1/microns/CITS_manuscript/figures` (and `final_figures_2026-08-31/`) | `<OUT>/<figure folder>/` |
| `/home/rbiswas1/microns/arousal_paper_overleaf/figures/2026-06-06_directed_cs` | `<OUT>/table1_baselines/` (directed benchmarks), `<OUT>/supplement/` (τ sweep) |
| `/home/rbiswas1/microns/arousal_paper_overleaf/figures/2026-06-01_versionBsafe/...` (EM outputs) | `<OUT>/fig3_microns_enrichment/` |
| `/home/rbiswas1/microns/arousal_paper_overleaf/figures/2026-07-06_hsic_gpu_validation` | `<OUT>/exploratory/hsic_gpu/` |
| `/home/rbiswas1/microns/arousal_paper_overleaf/figures/2026-06-02_simulation_benchmark*`, `2026-05-23` | `<OUT>/shared/...` (outputs of the modules' own `__main__` runs) |
| other `.../arousal_paper_overleaf/figures/...` inputs | `$AROUSAL_FIGS/...` |
| `/home/rbiswas1/microns/analysis/stimulus_fc` (`out/`, `results/`, panel PNG/CSV) | `<OUT>/fig3_microns_enrichment/` (`out/`, `results/`) |
| `/data1/rb1/microns/saves/`, `/home/rbiswas1/microns_data/saves` | `$MICRONS_SAVES` (reads); data-prep writes go to `<OUT>/fig3_microns_enrichment/data_prep/` |
| `/data1/rb1/microns/saves/cits_pc_contemporaneous_2026-05-23/` (cache) | `<OUT>/shared/cits_pc_contemporaneous_2026-05-23/` |
| `/home/rbiswas1/microns/all_unit_{areas.csv,coords.pkl}` | `$MICRONS_META` (reads); `fetch_all_*.py` write to `<OUT>/fig3_microns_enrichment/data_prep/` |
| session scratchpad `/tmp/claude-1004/...` (staged FC, per-field CSVs, scratch PNGs) | the producing script's output path (`<OUT>/fig3_microns_enrichment/out/fc/`, `<OUT>/fig3_microns_enrichment/*.csv`, `<OUT>/fig5_neuropixels/panels/`) |
| `/home/rbiswas1/citsproject/data` | `$NEUROPIXELS_DATA` |
| `data/`, `save/`, `figs/`, `D:\OneDrive - UW\...` in the Neuropixels notebooks | `<OUT>/fig5_neuropixels/notebooks/{data,save,figs,savedir}/` (`fig4_motifs/legacy/script copy.ipynb` reads `$NEUROPIXELS_DATA`); AllenSDK manifest in `$NEUROPIXELS_CACHE` |
| `/tmp/cave_migration_logs/` | `<OUT>/fig3_microns_enrichment/data_prep/logs/` |
| working directory (`grid_*.csv`, `_fig2_*`, `_sd_*`) | `<OUT>/fig1_scaling/`, `<OUT>/fig2_simulations/`, `<OUT>/table1_baselines/` |

Still unresolved (see `CHANGES_paths.md`): `/tmp/cits_pc_skeleton_prototype.py`, a missing serial
reference loaded only by the HSIC branch of `shared/cits_pc_skeleton_{numba,optimized}.py`.
`shared/_glm_suite_baselines.py` names the `tigra` interpreter only in a usage string.

## Credential redactions

Every copied file was scanned for passwords, tokens, API keys, private keys and e-mail
addresses. Three files contained DataJoint credentials (a database user name and password).
In the copies these are replaced by environment-variable reads:

| File (in this repo) | Lines | Change |
|---|---|---|
| `fig3_microns_enrichment/data_prep/fetch_all_areas.py` | 7–8 | `database.user` / `database.password` → `os.environ['DJ_USER']` / `os.environ['DJ_PASS']` |
| `fig3_microns_enrichment/data_prep/fetch_all_coords.py` | 12–13 (plus `import os` added at line 5) | same |
| `fig3_microns_enrichment/data_prep/cits_finalize.ipynb` | cell 1 (commented-out lines) | same, inside the comments |

The database host (`127.0.0.1:3306` / `localhost:3307`) is a local address and was left as is.
No other credentials were found. The original source files were not modified.

## Source data

Small tables copied verbatim (each < 200 KB):

| File | Holds |
|---|---|
| `fig1_scaling/source_data/grid_v3.csv` | per-seed CS and runtime, 3-seed grid (Fig 1D–F, `fig:scaling_supp`) |
| `fig1_scaling/source_data/grid_ext.csv` | TPC / Kernel GC at p = 1000 (timeouts) |
| `fig1_scaling/source_data/grid_seeds_ext.csv` | extra CS-only seeds 3–9 (Fig 1E error bars) |
| `fig1_scaling/source_data/grid_v3_aggregated.csv` | per-(method, p, N) CS mean / s.d. / seed count and runtime, as plotted |
| `fig1_scaling/source_data/cits_scaling_aggregated.csv` | 3-seed CS mean / s.d., runtime, timeouts (`tab:cs_supp`) |
| `fig2_simulations/source_data/_fig2_cits_noise50.csv` | CITS per-seed TPR/FPR/CS, linear paradigms |
| `fig2_simulations/source_data/_fig2_cits_noise50_nlng.csv` | CITS per-seed, non-linear paradigms |
| `fig2_simulations/source_data/_fig2_kgc_noise50.csv` | Kernel GC per-seed |
| `fig2_simulations/source_data/_fig2_pcmci_noise50.csv` | PCMCI+ and LPCMCI per-seed |
| `table1_baselines/source_data/_sd_assembled.csv` | Table 1 autoregressive block (mean, s.d., n); its spiking rows are superseded |
| `fig3_microns_enrichment/source_data/panelA_perfield_counts.csv` | CITS FC-present/absent pairs and synapses per EM field (Fig 3C) |
| `fig3_microns_enrichment/source_data/stim_baseline_perfield_granger.csv` | GC2 per-field counts (Fig 3C) |
| `fig3_microns_enrichment/source_data/stim_baseline_perfield_lagged01.csv` | lag 0 ∪ lag 1 correlation per-field counts (Fig 3C bar) |
| `fig3_microns_enrichment/source_data/stim_baseline_perfield_lagged1.csv` | lag-1 correlation per-field counts (CI and density quoted in the text) |
| `fig3_microns_enrichment/source_data/panelA_fc_em_enrichment_versionB.csv` | pooled CITS fold, CIs, Fisher p |
| `fig3_microns_enrichment/source_data/sc_within_vs_between_synapse_versionBsafe.csv` | EM within/between rates and ratio (Fig 3E) |
| `fig3_microns_enrichment/source_data/sc_within_vs_between_perfield_versionBsafe.csv` | per-field EM within/between counts |
| `fig3_microns_enrichment/source_data/em_areapair_synapse_versionBsafe.csv` | EM area-pair synapse fractions and CIs (Fig 3G) |
| `supplement/source_data/simulation_results_directed_tau_sensitivity_gpu.csv` | per-seed CS for τ = 1, 2, 3 (`tab:tau_saturation_supp`) |

Excluded on purpose:

- **Too large (> 200 KB):** `_fig2_baselines_noise50.csv` (GC1/GC2/PC/TPC for Fig 2, 248 KB),
  `stimulus_fc_combined.csv` (Fig 3D/F per-field metrics, 359 KB; one of its two shards is also
  over the limit).
- **Raw data and arrays** (npy/npz/pkl/p/nwb): MICrONS calcium (`calcium_npy/`, about 22 GB),
  EM tables (`synapses_matcheddf_frompre_v1718.pkl`, about 0.7 GB), staged FC
  (`panelA_stage/`), Neuropixels counts (`*_bin_0.01_P.p`, `P_raw_*.npy`, 0.4–0.7 GB each),
  and all intermediate matrices (`cmp_*`, `cits_v2_*`, motif records).
- **JSON and logs:** `_fig2_edgesign.json`, `_fig2_edgedir.json`, `motif_population_v2.json`,
  and run logs. The spiking-benchmark log values are transcribed above.
- **Per-seed baseline CSVs** of the `directed_benchmark_*.py` runs (originally in
  `arousal_paper_overleaf/figures/2026-06-06_directed_cs/`); only their assembled table was copied.
- **All images and PDFs.**

---

## Known gaps and discrepancies

Items marked **(check)** may need a manuscript or figure fix before publication.

1. **Fig 1A–C and Fig 3A** are schematic illustrations (model diagram; map of visual areas). They were drawn by hand and have no generating code.
2. **Fig 1D (check).** The caption describes "mean runtime per graph at N\*" with × marks at the
   30-minute wall. `_make_scaling_figure.py`, and the PNG placed in the figure deck, instead
   plot runtime at a fixed N = 1000 with no × markers. The in-text "≈33 s per graph at
   p = 1000" matches N = 250–500 (32.9–33.4 s in `grid_v3.csv`); at N = 1000 the plotted value
   is about 61 s.
3. **Stale manuscript figure files (check).** `CITS_manuscript/figures/fig1.pdf`, `fig4.pdf` and
   `fig5.pdf` are blank 300×200-pt placeholders. `figures/fig3.pdf` (2026-09-02) is an older
   5-panel layout that still shows the superseded arousal-provenance baselines (Granger 1.1× n.s.,
   lagged correlation 1.6×), not the 7-panel figure the caption describes.
4. **Fig 3C lagged-correlation bar (check).** The figure plots the lag 0 ∪ lag 1 variant
   (`stim_baseline_perfield_lagged01.csv`: fold 2.09, field-bootstrap CI 1.66–2.96, density
   95.4%). The caption and text quote CI 1.7–2.7 and density 93%, which match the lag-1-only run
   (`stim_baseline_perfield_lagged1.csv`: fold 2.07, CI 1.74–2.72, density 93.1%). Both round
   to 2.1-fold.
5. **Fig 3C paired Δ.** The paired field-cluster bootstrap Δ values in the text were computed
   inline; only CITS versus lag 0 ∪ lag 1 is saved (`corr_l0l1_final.py`). Recomputing from
   the per-field CSVs (seed 42, 5000 draws) gives CITS−GC2 Δ = 2.17 [1.66, 2.60], matching the
   text. CITS−lag-1 gives Δ = 1.51 [0.68, 2.07] with two-sided p ≈ 0.002; the text says p = 0.001.
6. **Fig 3E CI (check).** "6.3-fold, 95% CI 5.6–7.0" is a Katz interval on pooled pairs
   (`task6.py`). The field-cluster bootstrap (`sc_within_vs_between_synapse_versionBsafe.csv`)
   gives 6.25 [5.47, 7.55]. The Methods state that pooled pair-level tests are anticonservative.
7. **Fig 4 caption (check).** The caption and text describe motifs 105←125→247 and
   106←102↔118→109 (r ≈ 0.435, n = 324; r = 0.523; r ≈ 0.068). These numbers come from the legacy
   notebook `fig4_motifs/legacy/script copy.ipynb`. Project notes record a mislabeled pair
   (106–109 is adjacent) and a biased conditioning statistic in that notebook. The v2 scripts,
   and the panel in the figure deck, show FORK 359←272→400 and PATH 110←400–496→497.
8. **Fig 4 input (check).** `run_motif_population_v2.py` uses the 13-s block
   `ID791319847_natural_scenes_bin_0.01_X_idx-0.p` from the original pipeline. Project notes
   describe these X blocks as Gaussian-smoothed, while the Methods describe unsmoothed 10 ms
   counts.
9. **Table 1 versus Fig 2A.** Both use 50 simulations of the same simulator at η = 1, but
   from different runs. Three autoregressive cells differ: LPCMCI on Linear Gaussian 1
   (Table 0.904, Fig 2 CSV 0.911), LPCMCI on Non-linear Non-Gaussian 1 (0.787 vs 0.796), and
   Kernel GC on Non-linear Non-Gaussian 2 (0.515±0.06 vs 0.500±0.00). All other cells match.
10. **Table 1 spiking blocks.** The CITS ±s.d. values come from a gpu-2 log that is not
    available locally. The CPU and PCMCI logs existed only in an ephemeral session directory
    and are transcribed above.
11. **Unsaved inline steps.**
    - `cits_scaling_aggregated.csv` and the `tab:cs_supp` LaTeX table.
    - `_feasible_cells.csv`.
    - `results/stimulus_fc_combined.csv` (concatenation of the two shards; place it at
      `<OUT>/fig3_microns_enrichment/results/stimulus_fc_combined.csv`).
    - `P_raw_{stim}.npy`: NumPy copies of `ID791319847_{stim}_bin_0.01_P.p`, same sizes up to the pickle overhead.
    - `cits_v2_union_{units,labels}.npy`: same logic as `notebooks/script.ipynb` cell 6.
    - The environment settings of the "thin" Fig 5B render (`CFC_MAXW` / `CFC_MINW`; the script defaults 2.6 / 0.9 are assumed).
12. **MICrONS upstream data not covered.**
    - `merged_df_session*_scan*.pkl`, `{clip,Monet,Trippy}_timepoints_session*_scan*.pkl` and
      `filtered_{stim}_neurons_session*_scan*.pkl` (the >12%-active unit filter) were produced by
      `~/microns/microns/single_TPC_Hasika.ipynb`. That is a collaborator's mixed exploratory
      notebook and was **not copied**. It contains a hard-coded CAVE token, which should be
      revoked or rotated.
    - The 39 EM field keys come from an arousal-paper output (`bootstrap_sf_correlation_13sess.npz`).
    - The EM within/between and area-pair scripts restrict pairs to the arousal Version-B-safe
      FC universe (`cits_plus_pc_versionBsafe_2026-05-28/`, `statement_dfs_session*_scan*.pkl`).
      Those arousal-pipeline scripts are not included.
13. **Code paths that need modules not in this repo** (not used for the paper):
    - `directed_benchmark_gpu.py` methods other than 14 and `directed_benchmark_cpu.py` method 26
      import `cits_plus_v1_rcit_gpu`, `cits_plus_v_lag_informed_*`, `cits_plus_v_opt_rcit_nosp_gpu`,
      `tau_max_pacf`, `tpc_rcit_gpu` or `cits_plus_v_optimal`.
    - The self-test functions in `shared/gpu_cits_lag_rcit.py` import `cits_plus_v_opt_rcit_nosp_gpu`.
    - The validation helpers in `shared/cits_pc_skeleton_{numba,optimized}.py` load the missing
      `/tmp/cits_pc_skeleton_prototype.py`.
    - `shared/_pc_raw.py` always imports `shared/cits_plus_pc_contemporaneous_test.py`, an
      arousal-pipeline test harness, for its non-cuPC backend.
14. **Minor.**
    - `fig_motif_depression_self` is intentionally identical to `fig_motif_convergence_self`;
      the script docstring still mentions a depressing-synapse marker.
    - `_fig2_cits_noise50.csv` holds only the linear paradigms, although the script loops over
      all four; the non-linear rows come from `_fig2_cits_noise50_nlng.csv`.
    - `citsproject/script_cits_neuropixels.py` is empty and was not copied.

## Not included

- The CITS package itself (`~/repos/cits`) and cuPC sources: dependencies, see Environment.
- High-dimensional / AISTATS work (`analysis/causalrivers_benchmark`, `causeme_benchmark`,
  `dream4_benchmark`, `fmri_benchmark`, RFF / batched-RCIT scaling, `highdim_*` notes). No
  figure or table in `main_nn.tex` uses them.
- MICrONS arousal-paper scripts, except the EM and data-preparation steps Fig 3 needs.
- Superseded or unused CITS-paper scripts:
  - `_grid_cpu.py`, `_grid_gpu*.py`, `_runtime_certified*.py`, `_sample_complexity_*.py`,
    `_make_{faceted,matrix,fixedp}_figure.py`
  - Fig 2 α-sweep (`_fig2_*_alpha.py`, `_supp_alpha_grid.py`), `_fig2_full_assemble.py`,
    `_fig2_regen_plot.py`, `_fig2_edgeweights.py`, `_fig2_ew_nonlin.py`, `_fig2_cits_noise_gpu2.py`
  - CTRNN, hidden-node, persistent-latent, high-dimensional and runtime-p100 benchmarks
    (`_ctrnn_*`, `directed_benchmark_{ctrnn_highdim,hidden_*,runtime_*,pcmci_nonlinear}.py`)
  - the failed rpy2 HSIC rerun (`rerun_cits_hsic_nonlin_ctrnn.py`)
  - the glasso / `val_gc` montage column (`_montage_cc_baselines.py`)
  - earlier Neuropixels regenerations (`compute_cits_neuropixels_v2.py`, `regen_neuropixels_grid.py`,
    `composite_grid_v2.py`, `plot_cfc_stimtypes_v2.py`, `_cits_stability_compute.py`,
    `_neuropixels_versionB_compute.py`, `_stimtypes_{slide,unsmoothed}_compute.py`)
  - `plot_motif_examples_bars_v2.py`, `build_latent_robustness_supp.py`
  - behaviour-control analyses
  - the other rescued scratchpad scripts

---

## Appendix: file provenance

Each file in this repository was copied from the location below on 2026-10-01; the originals
were not moved or modified. `~` is `/home/rbiswas1`; `<SCRATCHPAD>` is the ephemeral Claude
session directory `/tmp/claude-1004/-home-rbiswas1-microns/48b8216b-5c45-4c8f-923d-dc312e0dbb46/scratchpad`.
Unless noted, the copy is byte-identical to the original as of 2026-10-01; on branch
`paths-config` the paths inside the scripts were then changed (see `CHANGES_paths.md`).

<details>
<summary>Show the full list</summary>

**exploratory/**

- `exploratory/hsic_gpu/gpu_cits_ci.py` ← `~/microns/analysis/functional_circuitry/gpu_cits_ci.py`
- `exploratory/hsic_gpu/gpu_cits_ci_fast.py` ← `~/microns/analysis/functional_circuitry/gpu_cits_ci_fast.py`
- `exploratory/hsic_gpu/gpu_hsic.py` ← `~/microns/analysis/functional_circuitry/gpu_hsic.py`
- `exploratory/hsic_gpu/hsic_validate_conditional.py` ← `~/microns/analysis/functional_circuitry/hsic_validate_conditional.py`
- `exploratory/hsic_gpu/hsic_validate_conditional_fast.py` ← `~/microns/analysis/functional_circuitry/hsic_validate_conditional_fast.py`
- `exploratory/hsic_gpu/hsic_validate_gpu_vs_kpcalg.py` ← `~/microns/analysis/functional_circuitry/hsic_validate_gpu_vs_kpcalg.py`
- `exploratory/hsic_gpu/rerun_cits_hsic_gpu.py` ← `~/microns/analysis/functional_circuitry/rerun_cits_hsic_gpu.py`
- `exploratory/hsic_gpu/rerun_cits_hsic_gpu_fast.py` ← `~/microns/analysis/functional_circuitry/rerun_cits_hsic_gpu_fast.py`
- `exploratory/latent_confounders/_cits_fci.py` ← `~/microns/analysis/functional_circuitry/_cits_fci.py`
- `exploratory/latent_confounders/_latent_testbed.py` ← `~/microns/analysis/functional_circuitry/_latent_testbed.py`
- `exploratory/latent_confounders/latent_confounder_test.py` ← `~/microns/CITS_manuscript/rescued_scratchpad_scripts_2026-10-01/latent_confounder_test.py`

**fig1_scaling/**

- `fig1_scaling/_cell_cpu.py` ← `~/microns/analysis/functional_circuitry/_cell_cpu.py`
- `fig1_scaling/_cell_gpu.py` ← `~/microns/analysis/functional_circuitry/_cell_gpu.py`
- `fig1_scaling/_grid_ext.py` ← `~/microns/analysis/functional_circuitry/_grid_ext.py`
- `fig1_scaling/_grid_seeds_ext.py` ← `~/microns/analysis/functional_circuitry/_grid_seeds_ext.py`
- `fig1_scaling/_grid_v3.py` ← `~/microns/analysis/functional_circuitry/_grid_v3.py`
- `fig1_scaling/_make_scaling_figure.py` ← `~/microns/analysis/functional_circuitry/_make_scaling_figure.py`
- `fig1_scaling/scaling_benchmark_lg.py` ← `~/microns/analysis/functional_circuitry/scaling_benchmark_lg.py`
- `fig1_scaling/source_data/cits_scaling_aggregated.csv` ← `~/microns/analysis/functional_circuitry/cits_scaling_aggregated.csv`
- `fig1_scaling/source_data/grid_ext.csv` ← `~/microns/analysis/functional_circuitry/grid_ext.csv`
- `fig1_scaling/source_data/grid_seeds_ext.csv` ← `~/microns/analysis/functional_circuitry/grid_seeds_ext.csv`
- `fig1_scaling/source_data/grid_v3.csv` ← `~/microns/analysis/functional_circuitry/grid_v3.csv`
- `fig1_scaling/source_data/grid_v3_aggregated.csv` ← `~/microns/analysis/functional_circuitry/grid_v3_aggregated.csv`
- `fig1_scaling/supporting/_baseline_clean_runtime.py` ← `~/microns/analysis/functional_circuitry/_baseline_clean_runtime.py`
- `fig1_scaling/supporting/_baseline_wall_budget.py` ← `~/microns/analysis/functional_circuitry/_baseline_wall_budget.py`
- `fig1_scaling/supporting/_extreme_scale.py` ← `~/microns/analysis/functional_circuitry/_extreme_scale.py`

**fig2_simulations/**

- `fig2_simulations/_fig2_baselines_noise.py` ← `~/microns/analysis/functional_circuitry/_fig2_baselines_noise.py`
- `fig2_simulations/_fig2_cits_noise_nlng.py` ← `~/microns/analysis/functional_circuitry/_fig2_cits_noise_nlng.py`
- `fig2_simulations/_fig2_cits_noise_perregime.py` ← `~/microns/analysis/functional_circuitry/_fig2_cits_noise_perregime.py`
- `fig2_simulations/_fig2_edgedir.py` ← `~/microns/analysis/functional_circuitry/_fig2_edgedir.py`
- `fig2_simulations/_fig2_edgesign.py` ← `~/microns/analysis/functional_circuitry/_fig2_edgesign.py`
- `fig2_simulations/_fig2_kgc_noise.py` ← `~/microns/analysis/functional_circuitry/_fig2_kgc_noise.py`
- `fig2_simulations/_fig2_orig_assemble.py` ← `~/microns/analysis/functional_circuitry/_fig2_orig_assemble.py`
- `fig2_simulations/_fig2_pcmci_noise.py` ← `~/microns/analysis/functional_circuitry/_fig2_pcmci_noise.py`
- `fig2_simulations/source_data/_fig2_cits_noise50.csv` ← `~/microns/analysis/functional_circuitry/_fig2_cits_noise50.csv`
- `fig2_simulations/source_data/_fig2_cits_noise50_nlng.csv` ← `~/microns/analysis/functional_circuitry/_fig2_cits_noise50_nlng.csv`
- `fig2_simulations/source_data/_fig2_kgc_noise50.csv` ← `~/microns/analysis/functional_circuitry/_fig2_kgc_noise50.csv`
- `fig2_simulations/source_data/_fig2_pcmci_noise50.csv` ← `~/microns/analysis/functional_circuitry/_fig2_pcmci_noise50.csv`

**fig3_microns_enrichment/**

- `fig3_microns_enrichment/_plot_arousal_style.py` ← `~/microns/analysis/stimulus_fc/_plot_arousal_style.py`
- `fig3_microns_enrichment/aggregate_and_plot.py` ← `~/microns/analysis/stimulus_fc/aggregate_and_plot.py`
- `fig3_microns_enrichment/bootstrap_em_areapair_synapse_versionBsafe.py` ← `~/microns/analysis/functional_circuitry/bootstrap_em_areapair_synapse_versionBsafe.py`
- `fig3_microns_enrichment/bootstrap_em_within_vs_between_versionBsafe.py` ← `~/microns/analysis/functional_circuitry/bootstrap_em_within_vs_between_versionBsafe.py`
- `fig3_microns_enrichment/corr_l0l1_final.py` ← `~/microns/CITS_manuscript/rescued_scratchpad_scripts_2026-10-01/corr_l0l1_final.py`
- `fig3_microns_enrichment/data_prep/_convert_merged.py` ← `~/microns/analysis/stimulus_fc/_convert_merged.py`
- `fig3_microns_enrichment/data_prep/_extract_idfield.py` ← `~/microns/analysis/stimulus_fc/_extract_idfield.py`
- `fig3_microns_enrichment/data_prep/_extract_unionids.py` ← `~/microns/analysis/stimulus_fc/_extract_unionids.py`
- `fig3_microns_enrichment/data_prep/build_matched_df_v1718.py` ← `~/microns/analysis/functional_circuitry/build_matched_df_v1718.py`
- `fig3_microns_enrichment/data_prep/build_synapses_pkl_v1718.py` ← `~/microns/analysis/functional_circuitry/build_synapses_pkl_v1718.py`
- `fig3_microns_enrichment/data_prep/cits_finalize.ipynb` ← `~/citsproject/cits_finalize.ipynb` (copy modified: notebook outputs stripped, credentials redacted)
- `fig3_microns_enrichment/data_prep/fetch_all_areas.py` ← `~/microns/analysis/functional_circuitry/fetch_all_areas.py` (copy modified: credentials redacted)
- `fig3_microns_enrichment/data_prep/fetch_all_coords.py` ← `~/microns/analysis/functional_circuitry/fetch_all_coords.py` (copy modified: credentials redacted)
- `fig3_microns_enrichment/fig3A_enrichment_stimulus.py` ← `~/microns/CITS_manuscript/figures/fig3A_enrichment_stimulus.py`
- `fig3_microns_enrichment/panelA_fc_em_enrichment_versionB.py` ← `~/microns/analysis/stimulus_fc/panelA_fc_em_enrichment_versionB.py`
- `fig3_microns_enrichment/panelA_perfield.py` ← `~/microns/CITS_manuscript/rescued_scratchpad_scripts_2026-10-01/panelA_perfield.py`
- `fig3_microns_enrichment/plot_em_areapair_synapse_GRAY.py` ← `~/microns/analysis/stimulus_fc/plot_em_areapair_synapse_GRAY.py`
- `fig3_microns_enrichment/plot_fig3B_cfc_spatial_clip.py` ← `~/microns/CITS_manuscript/rescued_scratchpad_scripts_2026-10-01/plot_fig3B_cfc_spatial_clip.py`
- `fig3_microns_enrichment/plot_sc_within_vs_between_GRAY.py` ← `~/microns/analysis/stimulus_fc/plot_sc_within_vs_between_GRAY.py`
- `fig3_microns_enrichment/source_data/em_areapair_synapse_versionBsafe.csv` ← `~/microns/arousal_paper_overleaf/figures/2026-06-01_versionBsafe/fig2/em_areapair_synapse_versionBsafe.csv`
- `fig3_microns_enrichment/source_data/panelA_fc_em_enrichment_versionB.csv` ← `~/microns/analysis/stimulus_fc/panelA_fc_em_enrichment_versionB.csv`
- `fig3_microns_enrichment/source_data/panelA_perfield_counts.csv` ← `<SCRATCHPAD>/panelA_perfield_counts.csv`
- `fig3_microns_enrichment/source_data/sc_within_vs_between_perfield_versionBsafe.csv` ← `~/microns/arousal_paper_overleaf/figures/2026-06-01_versionBsafe/exp_analysis/fig1_sc/sc_within_vs_between_perfield_versionBsafe.csv`
- `fig3_microns_enrichment/source_data/sc_within_vs_between_synapse_versionBsafe.csv` ← `~/microns/arousal_paper_overleaf/figures/2026-06-01_versionBsafe/exp_analysis/fig1_sc/sc_within_vs_between_synapse_versionBsafe.csv`
- `fig3_microns_enrichment/source_data/stim_baseline_perfield_granger.csv` ← `<SCRATCHPAD>/stim_baseline_perfield_granger.csv`
- `fig3_microns_enrichment/source_data/stim_baseline_perfield_lagged01.csv` ← `<SCRATCHPAD>/stim_baseline_perfield_lagged01.csv`
- `fig3_microns_enrichment/source_data/stim_baseline_perfield_lagged1.csv` ← `<SCRATCHPAD>/stim_baseline_perfield_lagged1.csv`
- `fig3_microns_enrichment/stim_aggregate.py` ← `~/microns/analysis/stimulus_fc/stim_aggregate.py`
- `fig3_microns_enrichment/stim_baseline_enrichment.py` ← `~/microns/CITS_manuscript/rescued_scratchpad_scripts_2026-10-01/stim_baseline_enrichment.py`
- `fig3_microns_enrichment/stim_fc_pipeline.py` ← `~/microns/analysis/stimulus_fc/stim_fc_pipeline.py`
- `fig3_microns_enrichment/stim_run.py` ← `~/microns/analysis/stimulus_fc/stim_run.py`
- `fig3_microns_enrichment/task6.py` ← `~/microns/CITS_manuscript/rescued_scratchpad_scripts_2026-10-01/task6.py`
- `fig3_microns_enrichment/task9.py` ← `~/microns/CITS_manuscript/rescued_scratchpad_scripts_2026-10-01/task9.py`

**fig4_motifs/**

- `fig4_motifs/legacy/run_lagged_search.py` ← `~/citsproject/run_lagged_search.py`
- `fig4_motifs/legacy/script copy.ipynb` ← `~/citsproject/script copy.ipynb` (copy modified: notebook outputs stripped)
- `fig4_motifs/plot_motif_examples_scatter_v2.py` ← `~/microns/CITS_manuscript/figures/plot_motif_examples_scatter_v2.py`
- `fig4_motifs/plot_motif_population_v2.py` ← `~/microns/CITS_manuscript/figures/plot_motif_population_v2.py`
- `fig4_motifs/plot_motif_population_v2_split.py` ← `~/microns/CITS_manuscript/figures/plot_motif_population_v2_split.py`
- `fig4_motifs/run_motif_population_v2.py` ← `~/microns/CITS_manuscript/figures/run_motif_population_v2.py`

**fig5_neuropixels/**

- `fig5_neuropixels/_montage_cc_cits.py` ← `~/microns/CITS_manuscript/figures/_montage_cc_cits.py`
- `fig5_neuropixels/_montage_cc_gc2.py` ← `~/microns/CITS_manuscript/figures/_montage_cc_gc2.py`
- `fig5_neuropixels/_montage_sig_magnitude.py` ← `~/microns/CITS_manuscript/figures/_montage_sig_magnitude.py`
- `fig5_neuropixels/_neuropixels_versionB_pooled.py` ← `~/microns/CITS_manuscript/figures/_neuropixels_versionB_pooled.py`
- `fig5_neuropixels/_stimtypes_90swin_compute.py` ← `~/microns/CITS_manuscript/figures/_stimtypes_90swin_compute.py`
- `fig5_neuropixels/_verify_fig5_stats.py` ← `~/microns/CITS_manuscript/figures/_verify_fig5_stats.py`
- `fig5_neuropixels/legacy/scc_clustering.ipynb` ← `~/citsproject/scc_clustering.ipynb` (copy modified: notebook outputs stripped)
- `fig5_neuropixels/notebooks/neuropixels_testresults.py` ← `~/citsproject/neuropixels_testresults.py`
- `fig5_neuropixels/notebooks/script.ipynb` ← `~/citsproject/script.ipynb` (copy modified: notebook outputs stripped)
- `fig5_neuropixels/notebooks/script_matchbarplot.ipynb` ← `~/citsproject/script_matchbarplot.ipynb` (copy modified: notebook outputs stripped)
- `fig5_neuropixels/regen_cfc_stimtypes.py` ← `~/microns/CITS_manuscript/figures/regen_cfc_stimtypes.py`
- `fig5_neuropixels/render_fine_ccsig.py` ← `~/microns/CITS_manuscript/rescued_scratchpad_scripts_2026-10-01/render_fine_ccsig.py`
- `fig5_neuropixels/task7.py` ← `~/microns/CITS_manuscript/rescued_scratchpad_scripts_2026-10-01/task7.py`

**shared/**

- `shared/_cupc_wrapper.py` ← `~/microns/analysis/functional_circuitry/_cupc_wrapper.py`
- `shared/_glm_suite_baselines.py` ← `~/microns/analysis/functional_circuitry/_glm_suite_baselines.py`
- `shared/_lscm_refit.py` ← `~/microns/analysis/functional_circuitry/_lscm_refit.py`
- `shared/_pc_orientation.py` ← `~/microns/analysis/functional_circuitry/_pc_orientation.py`
- `shared/_pc_raw.py` ← `~/microns/analysis/functional_circuitry/_pc_raw.py`
- `shared/_pc_raw_v2.py` ← `~/microns/analysis/functional_circuitry/_pc_raw_v2.py`
- `shared/_pc_raw_v_latest.py` ← `~/microns/analysis/functional_circuitry/_pc_raw_v_latest.py`
- `shared/_union_cpdag.py` ← `~/microns/analysis/functional_circuitry/_union_cpdag.py`
- `shared/cits_lag_rcit.py` ← `~/microns/analysis/functional_circuitry/cits_lag_rcit.py`
- `shared/cits_optimal.py` ← `~/microns/analysis/functional_circuitry/cits_optimal.py`
- `shared/cits_pc_skeleton_numba.py` ← `~/microns/analysis/functional_circuitry/cits_pc_skeleton_numba.py`
- `shared/cits_pc_skeleton_optimized.py` ← `~/microns/analysis/functional_circuitry/cits_pc_skeleton_optimized.py`
- `shared/cits_plus_pc_contemporaneous_test.py` ← `~/microns/analysis/functional_circuitry/cits_plus_pc_contemporaneous_test.py`
- `shared/cits_plus_v_latest.py` ← `~/microns/analysis/functional_circuitry/cits_plus_v_latest.py`
- `shared/directed_metrics.py` ← `~/microns/analysis/functional_circuitry/directed_metrics.py`
- `shared/glm_spiking_sim.py` ← `~/microns/analysis/functional_circuitry/glm_spiking_sim.py`
- `shared/gpu_cits_lag_cupc.py` ← `~/microns/analysis/functional_circuitry/gpu_cits_lag_cupc.py`
- `shared/gpu_cits_lag_cupc_faithful.py` ← `~/microns/analysis/functional_circuitry/gpu_cits_lag_cupc_faithful.py`
- `shared/gpu_cits_lag_rcit.py` ← `~/microns/analysis/functional_circuitry/gpu_cits_lag_rcit.py`
- `shared/gpu_pc_skeleton_rcit.py` ← `~/microns/analysis/functional_circuitry/gpu_pc_skeleton_rcit.py`
- `shared/gpu_rcit.py` ← `~/microns/analysis/functional_circuitry/gpu_rcit.py`
- `shared/kernel_granger_baseline.py` ← `~/microns/analysis/functional_circuitry/kernel_granger_baseline.py`
- `shared/lpcmci_baseline.py` ← `~/microns/analysis/functional_circuitry/lpcmci_baseline.py`
- `shared/pc_skeleton_rcit.py` ← `~/microns/analysis/functional_circuitry/pc_skeleton_rcit.py`
- `shared/pcmci_plus_baseline.py` ← `~/microns/analysis/functional_circuitry/pcmci_plus_baseline.py`
- `shared/rcit.py` ← `~/microns/analysis/functional_circuitry/rcit.py`
- `shared/sim_scm.py` ← `~/microns/analysis/functional_circuitry/sim_scm.py`
- `shared/simulation_benchmark_fc_methods.py` ← `~/microns/analysis/functional_circuitry/simulation_benchmark_fc_methods.py`
- `shared/simulation_benchmark_fc_methods_v3.py` ← `~/microns/analysis/functional_circuitry/simulation_benchmark_fc_methods_v3.py`

**supplement/**

- `supplement/_make_supp_figure.py` ← `~/microns/analysis/functional_circuitry/_make_supp_figure.py`
- `supplement/source_data/simulation_results_directed_tau_sensitivity_gpu.csv` ← `~/microns/arousal_paper_overleaf/figures/2026-06-06_directed_cs/simulation_results_directed_tau_sensitivity_gpu.csv`
- `supplement/tau_sensitivity_gpu.py` ← `~/microns/analysis/functional_circuitry/tau_sensitivity_gpu.py`

**table1_baselines/**

- `table1_baselines/_sd_ar_lingauss_cupc.py` ← `~/microns/analysis/functional_circuitry/_sd_ar_lingauss_cupc.py`
- `table1_baselines/_sd_assemble.py` ← `~/microns/analysis/functional_circuitry/_sd_assemble.py`
- `table1_baselines/_sd_gpu2_rcit.py` ← `~/microns/analysis/functional_circuitry/_sd_gpu2_rcit.py`
- `table1_baselines/_sd_local.py` ← `~/microns/analysis/functional_circuitry/_sd_local.py`
- `table1_baselines/_sd_spk_cits_cupc_convdia.py` ← `~/microns/analysis/functional_circuitry/_sd_spk_cits_cupc_convdia.py`
- `table1_baselines/_sd_tigra.py` ← `~/microns/analysis/functional_circuitry/_sd_tigra.py`
- `table1_baselines/_spiking_pmatched_cits.py` ← `~/microns/analysis/functional_circuitry/_spiking_pmatched_cits.py`
- `table1_baselines/_spiking_pmatched_cpu.py` ← `~/microns/analysis/functional_circuitry/_spiking_pmatched_cpu.py`
- `table1_baselines/_spiking_pmatched_pcmci.py` ← `~/microns/analysis/functional_circuitry/_spiking_pmatched_pcmci.py`
- `table1_baselines/directed_benchmark_cpu.py` ← `~/microns/analysis/functional_circuitry/directed_benchmark_cpu.py`
- `table1_baselines/directed_benchmark_gpu.py` ← `~/microns/analysis/functional_circuitry/directed_benchmark_gpu.py`
- `table1_baselines/directed_benchmark_kernel_gc.py` ← `~/microns/analysis/functional_circuitry/directed_benchmark_kernel_gc.py`
- `table1_baselines/directed_benchmark_lpcmci.py` ← `~/microns/analysis/functional_circuitry/directed_benchmark_lpcmci.py`
- `table1_baselines/plot_spiking_motifs.py` ← `~/microns/CITS_manuscript/figures/final_figures_2026-08-31/plot_spiking_motifs.py`
- `table1_baselines/source_data/_sd_assembled.csv` ← `~/microns/analysis/functional_circuitry/_sd_assembled.csv`

</details>
