import datajoint as dj
import pandas as pd
import os

# Configure DataJoint
dj.config['database.host'] = '127.0.0.1:3306'
dj.config['database.user'] = os.environ['DJ_USER']  # REDACTED: was a hard-coded user name
dj.config['database.password'] = os.environ['DJ_PASS']  # REDACTED: was a hard-coded password
dj.config['database.use_tls'] = False

from microns_phase3 import nda

def fetch_areas():
    print("Fetching AreaMembership...")
    # Fetch all columns: session, scan_idx, unit_id, brain_area
    df = pd.DataFrame((nda.AreaMembership()).fetch())
    print(f"Fetched {len(df)} records.")
    
    output_path = '/home/rbiswas1/microns/all_unit_areas.csv'
    df.to_csv(output_path, index=False)
    print(f"Saved to {output_path}")
    print(df.head())

if __name__ == "__main__":
    fetch_areas()
