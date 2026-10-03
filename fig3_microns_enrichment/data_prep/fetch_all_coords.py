"""
Fetch all unit coordinates from DataJoint and cache them.
This is much faster than querying for each session individually.
"""
import os  # added for the credential redaction (DJ_USER / DJ_PASS)
import datajoint as dj
import pandas as pd
import pickle

# Connect to DataJoint
dj.config['database.host'] = '127.0.0.1:3306'
dj.config['database.user'] = os.environ['DJ_USER']  # REDACTED: was a hard-coded user name
dj.config['database.password'] = os.environ['DJ_PASS']  # REDACTED: was a hard-coded password
dj.config['database.use_tls'] = False

from microns_phase3 import nda
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'shared')))
from paths import out as _out

print("Fetching all unit coordinates from DataJoint...")
# Fetch all units at once
units = nda.ScanUnit.fetch('session', 'scan_idx', 'unit_id', 'um_x', 'um_y', 'um_z', as_dict=True)

print(f"Fetched {len(units)} units.")

# Convert to DataFrame
df = pd.DataFrame(units)

# Save to pickle
output_file = _out('fig3_microns_enrichment/data_prep', 'all_unit_coords.pkl')   # was <MICRONS_META>/all_unit_coords.pkl
df.to_pickle(output_file)
print(f"Saved to {output_file}")
print(df.head())
