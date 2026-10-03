#!/usr/bin/env python3
"""
Script to run the lagged correlation triple search.
This extracts and runs the code from the new notebook cells.
"""

import sys
import json
import numpy as np
import pickle as pkl
from scipy.stats import pearsonr
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'shared')))
from paths import out as _out, outdir as _outdir, neuropixels

# Load notebook to get context
print("Loading notebook context...")
with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'script copy.ipynb'), 'r') as f:
    nb = json.load(f)

# Extract sess_id and stim_label from notebook
sess_id = 791319847  # Default
stim_label = 'natural_scenes'  # Default

for cell in nb['cells']:
    if 'source' in cell:
        source = ''.join(cell['source'])
        # Look for sess_id assignments
        if 'sess_id =' in source:
            try:
                # Try to extract the number
                parts = source.split('sess_id =')
                if len(parts) > 1:
                    val = parts[1].split('\n')[0].strip()
                    sess_id = int(val)
            except:
                pass
        # Look for stim_label assignments (prefer literal strings)
        if "stim_label='natural_scenes'" in source or "stim_label='static_gratings'" in source or "stim_label='gabors'" in source or "stim_label='flashes'" in source:
            try:
                if "stim_label='natural_scenes'" in source:
                    stim_label = 'natural_scenes'
                elif "stim_label='static_gratings'" in source:
                    stim_label = 'static_gratings'
                elif "stim_label='gabors'" in source:
                    stim_label = 'gabors'
                elif "stim_label='flashes'" in source:
                    stim_label = 'flashes'
            except:
                pass

print(f"Using sess_id={sess_id}, stim_label={stim_label}")

# Define A_Config class (from notebook)
class A_Config:
    def __init__(self, save_dir, sess_id, stimulus):
        self.save_dir = save_dir
        self.sess_id = sess_id
        self.stimulus = stimulus
        self.name = "{}/ID{}_{}".format(self.save_dir, self.sess_id, self.stimulus)

# Load CFC results (causal effect matrix)
print("\nLoading CFC results...")
results_dir = os.path.join(_outdir('fig4_motifs/legacy/save'), '')   # was save/ (CITS results of the legacy notebook)
allencon_save = A_Config(results_dir, sess_id, stim_label)
bin_size = 0.01
alpha = 0.05  # Default alpha
idx = 0
niter = 50
results_file = allencon_save.name + '_bin{}_alpha{}_idx{}_niter{}.p'.format(bin_size, alpha, idx, niter)

ce = None
if os.path.exists(results_file):
    print(f"Loading CFC from: {results_file}")
    try:
        results = pkl.load(open(results_file, 'rb'))
        if isinstance(results, dict) and 'ce' in results:
            ce = results['ce']
        elif isinstance(results, np.ndarray):
            ce = results
        else:
            # Try to get from 'out' dict if that's how it's stored
            ce = results.get('ce', None) if isinstance(results, dict) else None
    except Exception as e:
        print(f"Warning: Could not load CFC from file: {e}")

# If not loaded, try to load from notebook's 'out' variable or compute
if ce is None:
    # Try alternative path or check if we need to compute
    print("CFC results file not found. Checking if 'out' variable exists in notebook context...")
    # For now, we'll require the CFC to be available
    print("ERROR: CFC results (ce matrix) required but not found.")
    print(f"Please ensure calc_cits has been run and results are saved, or")
    print(f"run the notebook cell that computes 'out = calc_cits(0)' first.")
    sys.exit(1)

print(f"CFC matrix shape: {ce.shape}")
print(f"CFC matrix: {np.sum(ce != 0)} non-zero edges out of {ce.size} total")

# Now run the search code
print("\n" + "="*80)
print("Starting lagged correlation triple search with CFC constraints...")
print("="*80 + "\n")
print("Constraints:")
print("  - CFC: x→y edge exists (ce[x,y] != 0)")
print("  - CFC: x→z edge exists (ce[x,z] != 0)")
print("  - CFC: y and z NOT connected (ce[y,z] == 0)")
print("  - Lagged correlations (x(t)↔y(t+1), etc.) >= threshold")
print("  - Synchronous correlations (x(t)↔y(t), etc.) <= max")
print("  - Conditional correlation y(t+1)↔z(t+1)|x(t) <= max")
print()

# Load time series data for all units
idx_for_corr = 0
bin_size = 0.01
corr_data_dir = os.path.join(neuropixels(), '')   # was data/
allencon = A_Config(corr_data_dir, sess_id, stim_label)
name = allencon.name + '_bin_{}_X_idx-{}.p'.format(bin_size, idx_for_corr)

if not os.path.exists(name):
    print(f"ERROR: Data file not found: {name}")
    sys.exit(1)

raw = pkl.load(open(name, 'rb'))
mask_obj = pkl.load(open(allencon.name + '_units2use_stim_{}.p'.format(stim_label), 'rb'))
mask_arr = np.asarray(mask_obj)

if mask_arr.dtype == bool:
    units_idx_array = np.where(mask_arr)[0]
    data_std = raw[:, mask_arr]
else:
    units_idx_array = mask_arr
    data_std = raw[:, units_idx_array]

# Standardize
data_std = data_std.astype(float)
col_mean = data_std.mean(axis=0)
col_std = data_std.std(axis=0)
col_std[col_std == 0] = 1.0
data_std = (data_std - col_mean) / col_std

print(f"Loaded data shape: {data_std.shape}")
print(f"Number of units: {data_std.shape[1]}")

# Parameters for search
lag = 1
lagged_corr_threshold = 0.25  # Minimum absolute lagged correlation (lowered from 0.3)
sync_corr_max = 0.5  # Maximum absolute synchronous correlation (should be weak, raised from 0.4)
cond_corr_max = 0.3  # Maximum absolute conditional correlation (for independence, raised from 0.2)
window_width = 0.3  # For conditional correlation windows
min_points_per_window = 50  # Minimum points needed in each window (lowered from 60)

def compute_lagged_corr(series_a, series_b, lag=1):
    """Compute correlation between series_a(t) and series_b(t+lag)"""
    if lag >= len(series_a) or lag >= len(series_b):
        return np.nan, 0
    a_trimmed = series_a[:-lag]
    b_shifted = series_b[lag:]
    if len(a_trimmed) < 2 or np.std(a_trimmed) == 0 or np.std(b_shifted) == 0:
        return np.nan, 0
    r, _ = pearsonr(a_trimmed, b_shifted)
    return r, len(a_trimmed)

def compute_sync_corr(series_a, series_b):
    """Compute synchronous correlation between series_a(t) and series_b(t)"""
    if len(series_a) != len(series_b) or len(series_a) < 2:
        return np.nan
    if np.std(series_a) == 0 or np.std(series_b) == 0:
        return np.nan
    r, _ = pearsonr(series_a, series_b)
    return r

def compute_conditional_corr(series_x, series_y_future, series_z_future, window_width=0.3, min_points=60, lag=1, n_intervals=3):
    """
    Compute y(t+1) vs z(t+1) correlation conditioned on x(t) in narrow windows.
    Returns the maximum absolute correlation from the top n_intervals with smallest correlations.
    This checks conditional independence: if y(t+1) ⊥ z(t+1) | x(t), correlations should be small.
    """
    if len(series_x) < 2 or len(series_y_future) < 2 or len(series_z_future) < 2:
        return np.nan, []
    
    # x(t) should align with y(t+1) and z(t+1), so trim x
    x_trimmed = series_x[:-lag] if len(series_x) > lag else series_x
    min_len = min(len(x_trimmed), len(series_y_future), len(series_z_future))
    x_trimmed = x_trimmed[:min_len]
    y_future = series_y_future[:min_len]
    z_future = series_z_future[:min_len]
    
    if len(x_trimmed) < min_points:
        return np.nan, []
    
    step = window_width / 2
    centers = np.arange(x_trimmed.min(), x_trimmed.max() + step, step)
    interval_stats = []
    
    for center in centers:
        low = center - window_width / 2
        high = center + window_width / 2
        mask = (x_trimmed >= low) & (x_trimmed < high)
        count = mask.sum()
        
        if count < min_points:
            continue
            
        y_slice = y_future[mask]
        z_slice = z_future[mask]
        
        if len(y_slice) < 2 or np.std(y_slice) == 0 or np.std(z_slice) == 0:
            continue
            
        r, _ = pearsonr(y_slice, z_slice)
        
        interval_stats.append({
            'low': low,
            'high': high,
            'count': count,
            'r': r,
        })
    
    if len(interval_stats) < n_intervals:
        return np.nan, interval_stats
    
    # Sort by absolute correlation (smallest first) to find intervals showing independence
    interval_stats_sorted = sorted(interval_stats, key=lambda rec: abs(rec['r']))
    
    # Take the top n_intervals with smallest correlations
    top_intervals = interval_stats_sorted[:n_intervals]
    
    # Return the maximum absolute correlation from these top intervals
    # (if independence holds, even the "worst" of the top intervals should be small)
    max_abs_corr = max(abs(rec['r']) for rec in top_intervals)
    
    return max_abs_corr, interval_stats

# Search through all triples
n_units = data_std.shape[1]
lagged_triples = []

# Diagnostic counters
diag_stats = {
    'total_triples_checked': 0,
    'passed_sync_xy': 0,
    'passed_lagged_xy': 0,
    'passed_sync_all': 0,
    'passed_lagged_all': 0,
    'passed_conditional': 0,
}

print(f"\nSearching for triples with:")
print(f"  - Lagged correlations (x(t)↔y(t+1), etc.) >= {lagged_corr_threshold}")
print(f"  - Synchronous correlations (x(t)↔y(t), etc.) <= {sync_corr_max}")
print(f"  - Conditional correlation y(t+1)↔z(t+1)|x(t) <= {cond_corr_max}")
print(f"\nSearching through {n_units} units...")

for x in range(n_units):
    if (x + 1) % 10 == 0:
        print(f"  Progress: {x+1}/{n_units} units checked...")
    
    series_x = data_std[:, x]
    
    for y in range(n_units):
        if x == y:
            continue
        
        # CFC constraint: x→y edge must exist
        if ce is not None and np.isclose(ce[x, y], 0.0):
            continue
        
        series_y = data_std[:, y]
        
        # Check synchronous correlation x(t) vs y(t) - should be weak
        sync_xy = compute_sync_corr(series_x, series_y)
        if np.isnan(sync_xy) or abs(sync_xy) > sync_corr_max:
            continue
        diag_stats['passed_sync_xy'] += 1
        
        # Check lagged correlation x(t) vs y(t+1) - should be strong
        lagged_xy, n_xy = compute_lagged_corr(series_x, series_y, lag)
        if np.isnan(lagged_xy) or abs(lagged_xy) < lagged_corr_threshold:
            continue
        diag_stats['passed_lagged_xy'] += 1
        
        for z in range(n_units):
            if z == x or z == y:
                continue
            
            # CFC constraints: x→z edge must exist, and y-z must NOT be connected (either direction)
            if ce is not None:
                if np.isclose(ce[x, z], 0.0):  # x→z edge must exist
                    continue
                # y and z must NOT be connected in either direction
                if not np.isclose(ce[y, z], 0.0) or not np.isclose(ce[z, y], 0.0):
                    continue
            
            diag_stats['total_triples_checked'] += 1
            series_z = data_std[:, z]
            
            # Check synchronous correlations - should all be weak
            sync_xz = compute_sync_corr(series_x, series_z)
            sync_yz = compute_sync_corr(series_y, series_z)
            if (np.isnan(sync_xz) or abs(sync_xz) > sync_corr_max or
                np.isnan(sync_yz) or abs(sync_yz) > sync_corr_max):
                continue
            diag_stats['passed_sync_all'] += 1
            
            # Check lagged correlations - should all be strong
            lagged_yz, n_yz = compute_lagged_corr(series_y, series_z, lag)
            lagged_zx, n_zx = compute_lagged_corr(series_z, series_x, lag)
            if (np.isnan(lagged_yz) or abs(lagged_yz) < lagged_corr_threshold or
                np.isnan(lagged_zx) or abs(lagged_zx) < lagged_corr_threshold):
                continue
            diag_stats['passed_lagged_all'] += 1
            
            # Check conditional independence: y(t+1) ⊥ z(t+1) | x(t)
            # Need y(t+1) and z(t+1) series
            y_future = series_y[lag:] if len(series_y) > lag else series_y
            z_future = series_z[lag:] if len(series_z) > lag else series_z
            
            max_cond_corr, interval_stats = compute_conditional_corr(
                series_x, y_future, z_future, window_width, min_points_per_window, lag, n_intervals=3
            )
            
            if np.isnan(max_cond_corr) or max_cond_corr > cond_corr_max:
                continue
            diag_stats['passed_conditional'] += 1
            
            # All criteria satisfied!
            lagged_triples.append({
                'x': x, 'y': y, 'z': z,
                'lagged_xy': lagged_xy,
                'lagged_yz': lagged_yz,
                'lagged_zx': lagged_zx,
                'sync_xy': sync_xy,
                'sync_xz': sync_xz,
                'sync_yz': sync_yz,
                'max_cond_corr': max_cond_corr,
                'interval_stats': interval_stats,
                'global_x': units_idx_array[x],
                'global_y': units_idx_array[y],
                'global_z': units_idx_array[z],
            })

# Sort by average lagged correlation strength
lagged_triples_sorted = sorted(
    lagged_triples,
    key=lambda t: (abs(t['lagged_xy']) + abs(t['lagged_yz']) + abs(t['lagged_zx'])) / 3,
    reverse=True
)

print(f"\n{'='*80}")
print(f"Found {len(lagged_triples_sorted)} triples satisfying all criteria!")
print(f"{'='*80}\n")

if lagged_triples_sorted:
    print("Top triples (sorted by average lagged correlation strength):\n")
    for i, t in enumerate(lagged_triples_sorted[:20], 1):
        print(f"{i}. (x,y,z) = ({t['x']},{t['y']},{t['z']})")
        print(f"   Global IDs: x->{t['global_x']}, y->{t['global_y']}, z->{t['global_z']}")
        print(f"   Lagged correlations:")
        print(f"     x(t)↔y(t+1): {t['lagged_xy']:.3f}")
        print(f"     y(t)↔z(t+1): {t['lagged_yz']:.3f}")
        print(f"     z(t)↔x(t+1): {t['lagged_zx']:.3f}")
        print(f"   Synchronous correlations (weak):")
        print(f"     x(t)↔y(t): {t['sync_xy']:.3f}")
        print(f"     x(t)↔z(t): {t['sync_xz']:.3f}")
        print(f"     y(t)↔z(t): {t['sync_yz']:.3f}")
        print(f"   Conditional correlation y(t+1)↔z(t+1)|x(t): {t['max_cond_corr']:.3f}")
        print()
else:
    print("No triples found. Diagnostic statistics:")
    print(f"  Total triples checked: {diag_stats['total_triples_checked']}")
    print(f"  Passed sync_xy filter: {diag_stats['passed_sync_xy']}")
    print(f"  Passed lagged_xy filter: {diag_stats['passed_lagged_xy']}")
    print(f"  Passed all sync filters: {diag_stats['passed_sync_all']}")
    print(f"  Passed all lagged filters: {diag_stats['passed_lagged_all']}")
    print(f"  Passed conditional filter: {diag_stats['passed_conditional']}")
    print("\nTry adjusting thresholds:")
    print(f"  - Lower lagged_corr_threshold (currently {lagged_corr_threshold})")
    print(f"  - Raise sync_corr_max (currently {sync_corr_max})")
    print(f"  - Raise cond_corr_max (currently {cond_corr_max})")

# Save results
if lagged_triples_sorted:
    output_file = _out('fig4_motifs/legacy', f'lagged_triples_results_{sess_id}_{stim_label}.pkl')
    with open(output_file, 'wb') as f:
        pkl.dump(lagged_triples_sorted, f)
    print(f"\nResults saved to: {output_file}")

