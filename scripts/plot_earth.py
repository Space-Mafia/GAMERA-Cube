#!/usr/bin/env python3
"""Native-cell XY/XZ slices: polygon intersections, no spatial interpolation."""
import argparse
from pathlib import Path
import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.colors import LogNorm, Normalize

CORNERS = [(i,j,k) for i in (0,1) for j in (0,1) for k in (0,1)]
EDGES = [(a,b) for a in range(8) for b in range(a+1,8)
         if sum(x!=y for x,y in zip(CORNERS[a],CORNERS[b]))==1]

def polygons(vertices, normal, extent):
    """Intersect each hexahedron with the zero plane, retaining its cell value."""
    axes = [a for a in range(3) if a != normal]
    shape = tuple(n-1 for n in vertices.shape[1:])
    corners = np.stack([vertices[:,i:i+shape[0],j:j+shape[1],k:k+shape[2]]
                        for i,j,k in CORNERS], axis=0)
    height = corners[:,normal]
    # Strict upper bound avoids double-counting a plane on a shared face.
    mask = (height.min(axis=0)<=0) & (height.max(axis=0)>0)
    for a in axes:
        mask &= (corners[:,a].min(axis=0)<extent) & (corners[:,a].max(axis=0)>-extent)
    shapes, indices = [], []
    for idx in zip(*np.nonzero(mask)):
        cell = corners[(slice(None),slice(None),*idx)]
        pts = []
        for a,b in EDGES:
            ha,hb = cell[a,normal],cell[b,normal]
            if ha==0: pts.append(cell[a,axes])
            if hb==0: pts.append(cell[b,axes])
            if ha*hb<0:
                pts.append((cell[a]+ha/(ha-hb)*(cell[b]-cell[a]))[axes])
        if len(pts)<3: continue
        pts = np.unique(np.round(pts,12),axis=0)
        if len(pts)<3: continue
        center = pts.mean(axis=0)
        order = np.argsort(np.arctan2(pts[:,1]-center[1],pts[:,0]-center[0]))
        shapes.append(pts[order]); indices.append(idx)
    return shapes, indices

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("run",type=Path)
    p.add_argument("--sequence",type=int,default=None)
    p.add_argument("--extent",type=float,default=30)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    if a.sequence is None:
        seqs=sorted(int(f.stem.rsplit("_",1)[1]) for f in a.run.glob("analysis_p0_*.h5"))
        if not seqs: p.error("No snapshots found")
        a.sequence=seqs[-1]
    fig, axs=plt.subplots(2,4,figsize=(16,8),layout="constrained")
    norms=[LogNorm(.1,50),LogNorm(.001,10),Normalize(0,600),Normalize(-50,50)]
    cmaps=["viridis","plasma","viridis","RdBu_r"]
    labels=["Density [protons cm$^{-3}$ equivalent]","Pressure [nPa]",
            "Speed [km/s]","Total $B_z$ [nT]"]
    frame_time=None
    for face in range(6):
        with h5py.File(a.run/f"analysis_grid_p{face}.h5") as f:
            vertices=f["vertices"][:]
        with h5py.File(a.run/f"analysis_p{face}_{a.sequence:06d}.h5") as f:
            s=f["state"][:].astype(float); norm=f["normalization"][:]
            t=float(f["time_seconds"][0])
        if frame_time is not None and abs(t-frame_time)>1e-7:
            raise ValueError("Incomplete/mismatched snapshot times")
        frame_time=t
        values=[s[0]*norm[4]/1.67262192369e-21, s[1]*norm[5]*1e9,
                np.linalg.norm(s[2:5],axis=0)*norm[2]/1000,s[7]*norm[6]*1e9]
        for row,normal in enumerate((2,1)):
            poly,idx=polygons(vertices,normal,a.extent)
            if not idx: continue
            select=tuple(np.array(idx).T)
            for col in range(4):
                pc=PolyCollection(poly,array=values[col][select],cmap=cmaps[col],
                                  norm=norms[col],edgecolors="none",rasterized=True)
                axs[row,col].add_collection(pc)
    for row in range(2):
        for col in range(4):
            ax=axs[row,col]; ax.set(xlim=(-a.extent,a.extent),ylim=(-a.extent,a.extent),
                                   xlabel="X [$R_E$]",ylabel=("Y" if row==0 else "Z")+" [$R_E$]")
            ax.set_aspect("equal")
            ax.add_patch(plt.Circle((0,0),2.5,color=".6"))
            fig.colorbar(plt.cm.ScalarMappable(norm=norms[col],cmap=cmaps[col]),
                         ax=ax,label=labels[col],extend="both",shrink=.8)
    fig.suptitle(f"GAMERA-Cube Earth — t={frame_time:.3f} s — XY / XZ\n"
                 "Solar wind +X (left to right); grey disk: excluded r<2.5 RE; native cells")
    a.output.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(a.output,dpi=150)
    plt.close(fig)

if __name__=="__main__":
    main()
