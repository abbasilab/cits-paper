"""Central path configuration for the cits-paper scripts.

Every script reads its large inputs and writes its outputs through this module,
so the repository runs on any machine without editing the scripts. Each root is
read from an environment variable; the default lives inside the repository.

Roots
-----
CITS_PAPER_OUT    output root. Default: <repo>/outputs.
                  Each script writes to <CITS_PAPER_OUT>/<figure folder>/...,
                  e.g. outputs/fig1_scaling/grid_v3.csv.
CITS_PAPER_DATA   root for large external inputs. Default: <repo>/data.
MICRONS_SAVES     MICrONS calcium arrays, EM tables and arousal-pipeline caches
                  (calcium_npy/, merged_df_*.pkl, matched_df*.pkl,
                  synapses_matcheddf_frompre_v1718.pkl, statement_dfs_*.pkl,
                  {stim}_timepoints_*.pkl, filtered_*_neurons_*.pkl, ...).
                  Default: <CITS_PAPER_DATA>/microns_saves.
MICRONS_META      MICrONS unit metadata all_unit_areas.csv and
                  all_unit_coords.pkl. Default: <CITS_PAPER_DATA>/microns_meta.
AROUSAL_FIGS      outputs of the companion arousal-paper pipeline that some
                  Fig 3 scripts read (EM field keys in
                  2026-04-29/fig4/bootstrap_sf_correlation_13sess.npz).
                  Default: <CITS_PAPER_DATA>/arousal_figures.
NEUROPIXELS_DATA  Allen Neuropixels intermediate data for session 791319847
                  (ID791319847_*_bin_0.01_P.p, *_units2use_stim_*.p,
                  *_X_idx-*.p, P_raw_*.npy, cits_v2_union_*.npy).
                  Default: <CITS_PAPER_DATA>/neuropixels.
NEUROPIXELS_CACHE AllenSDK EcephysProjectCache directory used by the
                  data-preparation notebook. Default: <CITS_PAPER_DATA>/allen_cache.
CUPC_DIR          directory holding the compiled cuPC library Skeleton.so.
                  Default: ~/repos/cupc (the same default the cits package probes).

Helpers
-------
out(subdir, filename)            path under CITS_PAPER_OUT; creates the directory.
outdir(subdir)                   directory under CITS_PAPER_OUT; creates it.
data(*parts)                     path under CITS_PAPER_DATA.
microns_saves(*parts), microns_meta(*parts), arousal_figs(*parts),
neuropixels(*parts)              paths under the corresponding root.
result(subdir, filename, fallback=None)
                                 path of an intermediate produced by another
                                 script of this repository: the file under
                                 CITS_PAPER_OUT if it exists, else `fallback`
                                 (default: the committed copy in
                                 <repo>/<subdir>/source_data/<filename>) if that
                                 exists, else the CITS_PAPER_OUT path.
"""
import os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SHARED = os.path.join(REPO, 'shared')


def _env(name, default):
    v = os.environ.get(name)
    return os.path.abspath(os.path.expanduser(v)) if v else default


OUT_ROOT = _env('CITS_PAPER_OUT', os.path.join(REPO, 'outputs'))
DATA_ROOT = _env('CITS_PAPER_DATA', os.path.join(REPO, 'data'))
MICRONS_SAVES = _env('MICRONS_SAVES', os.path.join(DATA_ROOT, 'microns_saves'))
MICRONS_META = _env('MICRONS_META', os.path.join(DATA_ROOT, 'microns_meta'))
AROUSAL_FIGS = _env('AROUSAL_FIGS', os.path.join(DATA_ROOT, 'arousal_figures'))
NEUROPIXELS_DATA = _env('NEUROPIXELS_DATA', os.path.join(DATA_ROOT, 'neuropixels'))
NEUROPIXELS_CACHE = _env('NEUROPIXELS_CACHE', os.path.join(DATA_ROOT, 'allen_cache'))
CUPC_DIR = _env('CUPC_DIR', os.path.expanduser('~/repos/cupc'))


def outdir(subdir):
    """Directory <CITS_PAPER_OUT>/<subdir>, created if missing."""
    d = os.path.join(OUT_ROOT, subdir)
    os.makedirs(d, exist_ok=True)
    return d


def out(subdir, filename):
    """Path <CITS_PAPER_OUT>/<subdir>/<filename>; parent directories are created."""
    p = os.path.join(OUT_ROOT, subdir, filename)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    return p


def data(*parts):
    """Path under CITS_PAPER_DATA."""
    return os.path.join(DATA_ROOT, *parts)


def microns_saves(*parts):
    return os.path.join(MICRONS_SAVES, *parts)


def microns_meta(*parts):
    return os.path.join(MICRONS_META, *parts)


def arousal_figs(*parts):
    return os.path.join(AROUSAL_FIGS, *parts)


def neuropixels(*parts):
    return os.path.join(NEUROPIXELS_DATA, *parts)


def result(subdir, filename, fallback=None):
    """Intermediate produced by a script of this repo (see module docstring)."""
    p = os.path.join(OUT_ROOT, subdir, filename)
    if os.path.exists(p):
        return p
    if fallback is None:
        fallback = os.path.join(REPO, subdir, 'source_data', filename)
    if os.path.exists(fallback):
        return fallback
    return p
