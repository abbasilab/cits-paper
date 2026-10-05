# Path configuration changes (branch `paths-config`)

Goal: an outside reader can run every script without editing it. All input and output roots now
come from `shared/paths.py` (environment variables with in-repository defaults; see the README
section "Environment variables"). Only paths and imports were changed. No computation, seed,
parameter or numeric constant was touched. Every edited `.py` file passes `python -m py_compile`;
notebooks were checked cell by cell with `ast.parse` (same parse result as before), and their
outputs remain stripped. No analysis was run. Quick checks only: `paths` and the shared modules
import, and the re-plot scripts `_make_scaling_figure.py`, `_make_supp_figure.py`,
`make_tab_cs_supp.py`, `fig3A_enrichment_stimulus.py` and `plot_sc_within_vs_between_GRAY.py`
ran from the committed `source_data/` into a temporary `CITS_PAPER_OUT`. The regenerated
`grid_v3_aggregated.csv` is byte-identical to `fig1_scaling/source_data/grid_v3_aggregated.csv`.

## Rules applied

- Writes: every write goes to `$CITS_PAPER_OUT/<figure folder>/<original file name>` (default
  `outputs/`, now git-ignored). Original sub-structure is kept where it was structural
  (`out/fc/`, `out/`, `results/`, `data_prep/calcium_npy/`, `legacy/save/`). Dated manuscript
  sub-folders (`final_figures_2026-08-31/`, `2026-06-06_directed_cs/`, ...) were folded into the
  figure folder. `_make_supp_figure.py`, which wrote identical copies to two folders, now writes
  one copy.
- Reads of large inputs: `$MICRONS_SAVES`, `$MICRONS_META`, `$AROUSAL_FIGS`, `$NEUROPIXELS_DATA`,
  `$NEUROPIXELS_CACHE`.
- Reads of intermediates produced by another script of this repo: the producer's output path
  (`paths.result`), falling back to the committed `source_data/` copy (or, where noted, to the
  author's original location under a data root).
- Session-scratchpad paths (`/tmp/claude-1004/...`) are gone. Staged contemporaneous-CITS FC is read from
  `stim_run.py`'s output `<OUT>/fig3_microns_enrichment/out/fc/`; staged calcium metadata from
  `$MICRONS_SAVES/calcium_npy/`; per-field CSVs from the producing scripts' outputs (fallback
  `source_data/`); scratch PNGs of `regen_cfc_stimtypes.py` go to `<OUT>/fig5_neuropixels/panels/`.
- `sys.path.insert` of `/home/rbiswas1/repos/cits` was removed from all 36 files that had it; the
  installed `cits` package is used. Inserts of `analysis/functional_circuitry` were replaced by the
  repository's `shared/`. Inserts of `repos/cupc` were dropped (no Python module there; cuPC is
  located through `CUPC_DIR`). Inserts of `CITS_manuscript/figures` / `analysis/stimulus_fc` (used
  to import a sibling script) now point at the script's own folder. Scripts that need `shared/`
  modules add `shared/` themselves, so `PYTHONPATH` is no longer required.
- `os.environ.setdefault('CUPC_DIR', '/home/rbiswas1/repos/cupc')` became
  `os.environ.setdefault('CUPC_DIR', paths.CUPC_DIR)` (default `~/repos/cupc`, the same default the
  `cits` package probes); `shared/_cupc_wrapper.py` reads `CUPC_DIR` with that default.
- The output helper is imported as `_out` / `_outdir` to avoid clashing with local variables named
  `out` used by several scripts.

## Files added

- `shared/paths.py` (roots and helpers `out`, `outdir`, `data`, `microns_saves`, `microns_meta`,
  `arousal_figs`, `neuropixels`, `result`)
- `CHANGES_paths.md` (this file)

## Files edited

Repository: `.gitignore` (adds `outputs/`), `README.md` ("Paths to edit" replaced by "Environment
variables"; figure/table map now lists `<OUT>/...` outputs and the data roots; running notes,
cuPC note, appendix preamble updated).

fig1_scaling: `_cell_cpu.py`, `_cell_gpu.py`, `_grid_ext.py`, `_grid_seeds_ext.py` (also its
temporary per-cell logs now go to `<OUT>/fig1_scaling/tmp/`), `_grid_v3.py`,
`_make_scaling_figure.py`, `scaling_benchmark_lg.py`, `supporting/_baseline_clean_runtime.py`,
`supporting/_baseline_wall_budget.py`, `supporting/_extreme_scale.py` (the three `supporting/`
scripts also gained the parent folder on `sys.path`, since they import `scaling_benchmark_lg`).

fig2_simulations: `_fig2_baselines_noise.py`, `_fig2_cits_noise_nlng.py`,
`_fig2_cits_noise_perregime.py`, `_fig2_edgedir.py`, `_fig2_edgesign.py`, `_fig2_kgc_noise.py`,
`_fig2_orig_assemble.py`, `_fig2_pcmci_noise.py`.

table1_baselines: `_sd_ar_lingauss_cupc.py`, `_sd_assemble.py`, `_sd_gpu2_rcit.py`, `_sd_local.py`,
`_sd_spk_cits_cupc_convdia.py`, `_sd_tigra.py`, `_spiking_pmatched_cits.py`,
`_spiking_pmatched_cpu.py`, `_spiking_pmatched_pcmci.py`, `directed_benchmark_cpu.py`,
`directed_benchmark_gpu.py`, `directed_benchmark_kernel_gc.py`, `directed_benchmark_lpcmci.py`,
`make_cits_rcit_table1.py` (now writes to `<OUT>/table1_baselines/` instead of overwriting
`source_data/`), `plot_spiking_motifs.py`.

fig3_microns_enrichment: `_plot_arousal_style.py`, `aggregate_and_plot.py`,
`bootstrap_em_areapair_synapse.py`, `bootstrap_em_within_vs_between.py`,
`corr_l0l1_final.py`, `fig3A_enrichment_stimulus.py`, `panelA_fc_em_enrichment.py`,
`panelA_perfield.py`, `plot_em_areapair_synapse_GRAY.py`, `plot_fig3B_cfc_spatial_clip.py`,
`plot_sc_within_vs_between_GRAY.py`, `stim_aggregate.py`, `stim_baseline_enrichment.py`,
`stim_fc_pipeline.py`, `stim_run.py`, `task6.py`, `task9.py`; `data_prep/_convert_merged.py`,
`data_prep/_extract_idfield.py`, `data_prep/_extract_unionids.py`,
`data_prep/build_matched_df_v1718.py`, `data_prep/build_synapses_pkl_v1718.py`,
`data_prep/cits_finalize.ipynb`, `data_prep/fetch_all_areas.py`, `data_prep/fetch_all_coords.py`.

fig4_motifs: `plot_motif_examples_scatter_v2.py`, `plot_motif_population_v2.py`,
`plot_motif_population_v2_split.py`, `run_motif_population_v2.py`, `legacy/run_lagged_search.py`
(also reads `script copy.ipynb` relative to its own folder), `legacy/script copy.ipynb`.

fig5_neuropixels: `_montage_cc_cits.py`, `_montage_cc_gc2.py`, `_montage_sig_magnitude.py`,
`_neuropixels_contemp_pooled.py`, `_stimtypes_90swin_compute.py`, `_verify_fig5_stats.py`,
`regen_cfc_stimtypes.py`, `render_fine_ccsig.py`, `task7.py`, `notebooks/neuropixels_testresults.py`
(CRLF line endings kept), `notebooks/script.ipynb`, `notebooks/script_matchbarplot.ipynb`,
`legacy/scc_clustering.ipynb`.

supplement: `_make_supp_figure.py`, `tau_sensitivity_gpu.py`.

shared: `_cupc_wrapper.py`, `_glm_suite_baselines.py` (usage string only),
`cits_plus_pc_contemporaneous_test.py`, `simulation_benchmark_fc_methods.py`,
`simulation_benchmark_fc_methods_v3.py`.

exploratory: `hsic_gpu/hsic_validate_conditional.py`, `hsic_gpu/hsic_validate_conditional_fast.py`,
`hsic_gpu/hsic_validate_gpu_vs_kpcalg.py`, `hsic_gpu/rerun_cits_hsic_gpu.py`,
`hsic_gpu/rerun_cits_hsic_gpu_fast.py`, `latent_confounders/_latent_testbed.py`,
`latent_confounders/latent_confounder_test.py`.

Not edited (no hard-coded paths): the remaining `shared/` modules, `supplement/make_tab_cs_supp.py`
(reads `fig1_scaling/source_data/` relative to itself), `exploratory/hsic_gpu/gpu_*.py`,
`exploratory/latent_confounders/_cits_fci.py`.

## Paths that could not be fully resolved

1. `/tmp/cits_pc_skeleton_prototype.py` in `shared/cits_pc_skeleton_numba.py` (line 419) and
   `shared/cits_pc_skeleton_optimized.py` (line 411 and docstring). A serial reference module that
   exists nowhere (not in this repo, not on the author's machine). Only the `cond_dep_hsic`
   fallback branch loads it; left unchanged.
2. Intermediates with no producing script in this repo, now expected at a fixed place:
   - `<OUT>/fig1_scaling/_feasible_cells.csv` (unsaved inline step; input of `_grid_seeds_ext.py`).
   - `<OUT>/fig3_microns_enrichment/results/stimulus_fc_combined.csv` (unsaved concatenation of the
     `stim_run.py` shards).
   - `cits_v2_union_{units,labels}.npy` (unsaved, see README gaps): read from
     `<OUT>/fig5_neuropixels/` if present, else `$NEUROPIXELS_DATA`.
   - `P_raw_{stim}.npy` (unsaved NumPy copies of the `_P.p` pickles): read from `$NEUROPIXELS_DATA`.
   - Legacy CITS results (`citsproject/save/`): expected in `<OUT>/fig4_motifs/legacy/save/` and
     `<OUT>/fig5_neuropixels/notebooks/save/`.
   - `graphs/` (AFC edge lists) read by `notebooks/script_matchbarplot.ipynb` cell 3: mapped to
     `<OUT>/fig5_neuropixels/notebooks/graphs/`; no script produces it.
3. Arousal-pipeline inputs (not produced here) are read from `$AROUSAL_FIGS`: the EM field keys
   `2026-04-29/fig4/bootstrap_sf_correlation_13sess.npz`; `task6.py`'s
   `2026-06-01_versionBsafe/exp_analysis/fig1_sc/sc_enrichment_perfield_{versionBsafe,granger}.csv`;
   the inputs of `shared/cits_plus_pc_contemporaneous_test.py` (`2026-05-09/...json`,
   `2026-04-29/fig2/...npz`, `2026-05-16_cits/...npz`, `2026-05-20/...csv`). `_sd_assemble.py`
   falls back to `$AROUSAL_FIGS/2026-06-06_directed_cs/` for per-seed baseline CSVs.
4. Assumption: the calcium metadata that Fig 3B/3C read from the scratchpad
   (`panelA_stage/npy/{ids,fields,unionids}_*.npy`) are the `calcium_npy/` files built by
   `data_prep/` (staged from gpu-2); they are now read from `$MICRONS_SAVES/calcium_npy/`.
5. Producer/consumer split for data preparation. To obey "all writes under `CITS_PAPER_OUT`",
   `fig3_microns_enrichment/data_prep/*` writes to `<OUT>/fig3_microns_enrichment/data_prep/`, and
   the Neuropixels notebooks write their prepared data to `<OUT>/fig5_neuropixels/notebooks/data/`.
   The analysis scripts read `$MICRONS_SAVES` / `$MICRONS_META` / `$NEUROPIXELS_DATA`. To chain a
   fresh data preparation into the analyses, point those variables at the output folders (README).
6. Comments were not rewritten: commented-out Windows paths (`D:\OneDrive - UW\...`), `/homes/...`
   and `/home/naomi/...` remain inside comments of the notebooks and of
   `notebooks/neuropixels_testresults.py`. They are not executed.
7. `shared/_glm_suite_baselines.py`: the `tigra` interpreter path in the usage string was replaced
   by `<tigra env python>`; nothing executable referenced it.

## Modules referenced but missing from the repo

Code paths not used for the paper (already listed in README "Known gaps", item 13); they lived in
`~/microns/analysis/functional_circuitry/` and were not copied:

- `cits_plus_v1_rcit_gpu`, `cits_plus_v_lag_informed_{chi,direct,full_direct,ges}_rcit_gpu`,
  `cits_plus_v_opt_rcit_nosp_gpu`, `tau_max_pacf`, `tpc_rcit_gpu`: imported by
  `table1_baselines/directed_benchmark_gpu.py` for methods other than 14.
- `cits_plus_v_optimal`: `table1_baselines/directed_benchmark_cpu.py` method 26.
- `cits_plus_v_opt_rcit_nosp_gpu`: self-tests in `shared/gpu_cits_lag_rcit.py`.
- `/tmp/cits_pc_skeleton_prototype.py`: see above (exists nowhere).

External packages (not repo modules, needed only in the data environment): `allensdk`,
`microns_phase3`, `caveclient`, `datajoint`, `rpy2`; `ace_tools` (not installable; one notebook cell).

## Verification

`grep -rn "/home/rbiswas1\|/data1/\|/tmp/claude" --include=*.py --include=*.ipynb .` returns no
matches. `shared/paths.py` holds no author path (its defaults are repository-relative, plus
`~/repos/cupc` for cuPC). The author's original locations appear only in `README.md` (the
"reproduce exactly" example, the old-to-new path table and the provenance appendix) and in this
file.
