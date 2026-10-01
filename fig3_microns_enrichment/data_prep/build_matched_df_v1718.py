"""
Build matched_df_v1718.pkl by re-mapping supervoxel IDs to v1718 root IDs.

The matched_df from MICrONS coregistration carries pt_supervoxel_id (fixed) and
pt_root_id (v1181 — now expired). Supervoxels are stable across CAVE versions;
root IDs are not. We use the chunkedgraph to look up the current v1718 root for
each supervoxel.

Reports counts of unchanged, re-rooted, and invalidated roots.

Outputs: /data1/rb1/microns/saves/matched_df_v1718.pkl
Log:     /tmp/cave_migration_logs/build_matched_df_v1718.log
"""
import os
import sys
import pickle
import time
import numpy as np
import pandas as pd
from caveclient import CAVEclient

LOG = '/tmp/cave_migration_logs/build_matched_df_v1718.log'
os.makedirs(os.path.dirname(LOG), exist_ok=True)

logf = open(LOG, 'w')


def log(msg):
    print(msg, flush=True)
    logf.write(msg + '\n')
    logf.flush()


SRC = '/data1/rb1/microns/saves/matched_df.pkl'
DST = '/data1/rb1/microns/saves/matched_df_v1718.pkl'
TARGET_VERSION = 1718

log(f'Loading {SRC} ...')
matched_df = pickle.load(open(SRC, 'rb'))
log(f'  shape: {matched_df.shape}')
log(f'  columns: {list(matched_df.columns)}')

# Save the v1181 root as a separate column for diagnostics
matched_df = matched_df.copy()
matched_df['pt_root_id_v1181'] = matched_df['pt_root_id']

# Connect to CAVE and pull v1718 timestamp
log('Connecting to CAVE ...')
client = CAVEclient('minnie65_public')
v_meta = client.materialize.get_versions_metadata()
v1718_ts = [v['time_stamp'] for v in v_meta if v['version'] == TARGET_VERSION][0]
log(f'  v1718 timestamp: {v1718_ts}')

# Look up the v1718 root for each supervoxel
svids = matched_df['pt_supervoxel_id'].astype('int64').values
log(f'  Unique supervoxel IDs to query: {len(np.unique(svids[svids > 0]))}')

CHUNK = 5000
new_roots = np.zeros(len(svids), dtype=np.int64)
t0 = time.time()
n_chunks = (len(svids) + CHUNK - 1) // CHUNK
for i in range(0, len(svids), CHUNK):
    chunk = svids[i:i + CHUNK]
    # CAVE's chunkedgraph.get_roots returns 0 for invalid/deleted supervoxels?
    # Let's set stop_layer to default and pass timestamp.
    try:
        result = client.chunkedgraph.get_roots(chunk, timestamp=v1718_ts)
        new_roots[i:i + CHUNK] = result
    except Exception as exc:
        log(f'  ERROR at chunk {i // CHUNK}: {exc}')
        new_roots[i:i + CHUNK] = 0
    chunk_idx = i // CHUNK + 1
    if chunk_idx % 1 == 0:
        elapsed = time.time() - t0
        log(f'  chunk {chunk_idx}/{n_chunks} done  '
            f'elapsed={elapsed:.1f}s  per_chunk={elapsed / chunk_idx:.1f}s')

elapsed = time.time() - t0
log(f'\nAll chunks done in {elapsed:.1f}s')

matched_df['pt_root_id_v1718'] = new_roots
# Replace pt_root_id with the v1718 root.
matched_df['pt_root_id'] = new_roots

# Compute stats
orig_roots = matched_df['pt_root_id_v1181'].astype('int64').values
new_roots64 = new_roots.astype('int64')
mask_orig_valid = (orig_roots > 0) & (~pd.isna(matched_df['pt_root_id_v1181']))
n_total = int(mask_orig_valid.sum())
n_unchanged = int(((new_roots64 == orig_roots) & mask_orig_valid).sum())
n_rerooted = int(((new_roots64 != orig_roots) & (new_roots64 > 0) & mask_orig_valid).sum())
n_invalid = int(((new_roots64 == 0) & mask_orig_valid).sum())

log('\n--- Re-rooting stats (entries with valid v1181 root) ---')
log(f'  total entries (v1181 valid):  {n_total}')
log(f'  unchanged (same root):        {n_unchanged}  ({100.0 * n_unchanged / n_total:.2f}%)')
log(f'  re-rooted (different root):   {n_rerooted}  ({100.0 * n_rerooted / n_total:.2f}%)')
log(f'  invalidated (root=0/missing): {n_invalid}  ({100.0 * n_invalid / n_total:.2f}%)')

# unique cells (by supervoxel basically equivalent)
unique_v1181_roots = matched_df.dropna(subset=['pt_root_id_v1181'])
unique_v1181_roots = unique_v1181_roots[unique_v1181_roots['pt_root_id_v1181'].astype('int64') > 0]
n_unique_v1181 = int(unique_v1181_roots['pt_root_id_v1181'].astype('int64').nunique())
n_unique_v1718 = int(matched_df[matched_df['pt_root_id_v1718'] > 0]['pt_root_id_v1718'].nunique())
log(f'\n  unique v1181 root IDs: {n_unique_v1181}')
log(f'  unique v1718 root IDs: {n_unique_v1718}')

# Total entries with any valid pt_root_id_v1718
n_v1718_valid = int((matched_df['pt_root_id_v1718'] > 0).sum())
log(f'\n  matched_df_v1718 entries with valid root: {n_v1718_valid} / {len(matched_df)}')

# HALT check: more than 20% loss
loss_pct = 100.0 * n_invalid / n_total
log(f'\n  Loss fraction: {loss_pct:.2f}%')
if loss_pct > 20.0:
    log('  HALT: loss > 20%, aborting')
    logf.close()
    sys.exit(2)

# Save the new pkl
log(f'\nSaving {DST} ...')
with open(DST, 'wb') as fh:
    pickle.dump(matched_df, fh)
log('Done.')

# Print stats summary as JSON
import json
stats = {
    'total_entries_v1181_valid': n_total,
    'unchanged': n_unchanged,
    'rerooted': n_rerooted,
    'invalidated': n_invalid,
    'unique_v1181_roots': n_unique_v1181,
    'unique_v1718_roots': n_unique_v1718,
    'n_entries_v1718_valid': n_v1718_valid,
    'n_total_rows': int(len(matched_df)),
    'loss_pct': float(loss_pct),
}
log('\nSTATS_JSON: ' + json.dumps(stats))

logf.close()
