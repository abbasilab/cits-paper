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

print("Fetching all unit coordinates from DataJoint...")
# Fetch all units at once
units = nda.ScanUnit.fetch('session', 'scan_idx', 'unit_id', 'um_x', 'um_y', 'um_z', as_dict=True)

print(f"Fetched {len(units)} units.")

# Convert to DataFrame
df = pd.DataFrame(units)

# Save to pickle
output_file = '/home/rbiswas1/microns/all_unit_coords.pkl'
df.to_pickle(output_file)
print(f"Saved to {output_file}")
print(df.head())
