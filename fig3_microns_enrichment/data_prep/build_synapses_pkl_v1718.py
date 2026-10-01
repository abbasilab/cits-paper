"""
Build /data1/rb1/microns/saves/synapses_matcheddf_frompre_v1718.pkl

For every pre_root in matched_df_v1718, query synapses_pni_2 via query_table
(NOT synapse_query) at materialize.version=1718 so autapses come through.

Saves progress every chunk to handle CAVE flakiness; if killed midway can be
resumed by re-running (it skips keys already in the dict).
"""
import os
import sys
import time
import pickle
import json
import pandas as pd
import numpy as np
from caveclient import CAVEclient

LOG = '/tmp/cave_migration_logs/build_pkl_v1718.log'
os.makedirs(os.path.dirname(LOG), exist_ok=True)

logf = open(LOG, 'w')


def log(msg):
    print(msg, flush=True)
    logf.write(msg + '\n')
    logf.flush()


SRC_MATCHED = '/data1/rb1/microns/saves/matched_df_v1718.pkl'
DST_PKL = '/data1/rb1/microns/saves/synapses_matcheddf_frompre_v1718.pkl'

CHUNK_SIZE = 200

log('Loading matched_df_v1718 ...')
matched_df = pickle.load(open(SRC_MATCHED, 'rb'))
all_roots = matched_df['pt_root_id_v1718'].astype('int64')
all_roots = all_roots[all_roots > 0]
unique_pre = sorted(set(all_roots.tolist()))
log(f'  unique pre_root_ids (v1718): {len(unique_pre)}')

log('Connecting to CAVE ...')
client = CAVEclient('minnie65_public')
client.materialize.version = 1718
log(f'  client.materialize.version = {client.materialize.version}')

# Resume support
syn_dict = {}
if os.path.exists(DST_PKL):
    log(f'Loading existing pkl from {DST_PKL} ...')
    with open(DST_PKL, 'rb') as f:
        syn_dict = pickle.load(f)
    log(f'  existing keys: {len(syn_dict)}')

remaining = [r for r in unique_pre if r not in syn_dict]
log(f'  remaining to query: {len(remaining)}')

t0 = time.time()
n_chunks = (len(remaining) + CHUNK_SIZE - 1) // CHUNK_SIZE

for ci in range(n_chunks):
    chunk = remaining[ci * CHUNK_SIZE:(ci + 1) * CHUNK_SIZE]
    if not chunk:
        break
    chunk_t0 = time.time()
    try:
        df = client.materialize.query_table(
            'synapses_pni_2',
            filter_in_dict={'pre_pt_root_id': chunk},
        )
    except Exception as exc:
        log(f'  chunk {ci+1}/{n_chunks} query FAILED: {exc}')
        # Save what we have and continue with a smaller chunk strategy
        for r in chunk:
            try:
                df_sub = client.materialize.query_table(
                    'synapses_pni_2',
                    filter_in_dict={'pre_pt_root_id': [r]},
                )
                syn_dict[int(r)] = df_sub
            except Exception as exc2:
                log(f'    single-root {r} also failed: {exc2}')
                syn_dict[int(r)] = pd.DataFrame()
        with open(DST_PKL, 'wb') as f:
            pickle.dump(syn_dict, f)
        continue

    # Partition by pre_root_id
    if len(df) > 0:
        groups = df.groupby('pre_pt_root_id')
        seen = set()
        for pre_id, g in groups:
            syn_dict[int(pre_id)] = g.reset_index(drop=True)
            seen.add(int(pre_id))
        for r in chunk:
            if int(r) not in seen:
                syn_dict[int(r)] = pd.DataFrame()
    else:
        for r in chunk:
            syn_dict[int(r)] = pd.DataFrame()

    chunk_elapsed = time.time() - chunk_t0
    total_elapsed = time.time() - t0
    log(f'  chunk {ci+1}/{n_chunks} ({len(chunk)} roots) ok  '
        f'rows={len(df)}  t_chunk={chunk_elapsed:.1f}s  total={total_elapsed:.1f}s')

    # Periodic save
    if (ci + 1) % 5 == 0 or ci == n_chunks - 1:
        with open(DST_PKL, 'wb') as f:
            pickle.dump(syn_dict, f)
        log(f'    -> saved checkpoint ({len(syn_dict)} keys)')

# Final save
with open(DST_PKL, 'wb') as f:
    pickle.dump(syn_dict, f)
log(f'\nFINAL saved {DST_PKL}  total keys: {len(syn_dict)}')

# Stats
n_total = 0
n_autapse = 0
n_empty = 0
n_keys_with_syn = 0
for k, df in syn_dict.items():
    if df is None or len(df) == 0:
        n_empty += 1
        continue
    n_keys_with_syn += 1
    n_total += len(df)
    pre = df['pre_pt_root_id'].astype('int64').values
    post = df['post_pt_root_id'].astype('int64').values
    n_autapse += int((pre == post).sum())

log(f'\nSTATS:')
log(f'  total pre_roots in pkl:     {len(syn_dict)}')
log(f'  pre_roots with >=1 synapse: {n_keys_with_syn}')
log(f'  pre_roots empty:            {n_empty}')
log(f'  total synapse rows:         {n_total:,}')
log(f'  autapses (pre==post):       {n_autapse:,}')

stats = {
    'n_pre_roots': len(syn_dict),
    'n_pre_roots_with_synapses': n_keys_with_syn,
    'n_synapses_total': n_total,
    'n_autapses': n_autapse,
}
log('STATS_JSON: ' + json.dumps(stats))

logf.close()
