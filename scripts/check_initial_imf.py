#!/usr/bin/env python3
"""Check the student example's far-upstream initial IMF in physical units."""
import argparse
import json
from pathlib import Path
import h5py
import numpy as np

p=argparse.ArgumentParser(description=__doc__)
p.add_argument("run",type=Path)
a=p.parse_args()
fields=[]
for face in range(6):
    with h5py.File(a.run/f"analysis_grid_p{face}.h5") as f:
        xyz=f["coordinates"][:]
    with h5py.File(a.run/f"analysis_p{face}_000000.h5") as f:
        assert float(f["time_seconds"][0])==0, "Expected t=0"
        norm=f["normalization"][:]
        b=f["state"][5:8].astype(float)*norm[6]*1e9
    # Far beyond both the front transition and cut-dipole support.
    mask=(xyz[0]<-60)
    if mask.any():
        fields.append(b[:,mask])
assert fields, "No far-upstream cells selected"
b=np.concatenate(fields,axis=1)
error=np.max(np.abs(b-np.array([0,0,-5])[:,None]),axis=1)
report=dict(cells=b.shape[1],target_nt=[0,0,-5],
            max_absolute_error_nt=error.tolist(),tolerance_nt=0.001)
print(json.dumps(report,indent=2))
assert np.isfinite(b).all() and (error<=0.001).all(), "Initial upstream IMF mismatch"
print("INITIAL_IMF_PASS")
