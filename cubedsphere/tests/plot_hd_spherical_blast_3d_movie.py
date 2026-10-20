#!/usr/bin/env python3
"""Render every native-cell angular-blast snapshot from a 45-degree view.

The output is a numbered PNG sequence for ffmpeg. Frames are the solver's
stored snapshots, without temporal or spatial interpolation of pressure.
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import numpy as np

from plot_hd_spherical_blast_3d import plot_view


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path)
    parser.add_argument("frames_dir", type=Path)
    parser.add_argument("--label", default="HD", choices=("HD", "MHD"))
    args = parser.parse_args()
    data = np.genfromtxt(args.csv, delimiter=",", names=True)
    times = np.unique(data["time"])
    n = int(np.max(data["j"])) + 1
    if data.size != len(times) * 6 * n * n or len(times) < 2:
        raise ValueError("incomplete six-face snapshot sequence")
    camera = (np.cos(np.pi / 4) * np.cos(np.pi / 6),
              np.cos(np.pi / 4) * np.sin(np.pi / 6),
              np.sin(np.pi / 4))
    cmap = plt.get_cmap("turbo")
    norm = Normalize(0, 1)
    args.frames_dir.mkdir(parents=True, exist_ok=True)

    for index, time in enumerate(times):
        frame = data[data["time"] == time]
        pressure = np.empty((6, n, n), dtype=float)
        for face in range(6):
            rows = frame[frame["face"] == face]
            pressure[face, rows["j"].astype(int), rows["k"].astype(int)] = rows["pressure"]
        if not np.isfinite(pressure).all() or np.min(pressure) <= 0:
            raise ValueError(f"nonfinite or nonpositive pressure at t={time}")
        pmax = float(np.max(pressure))
        enhancement = np.maximum(pressure - 1.0, 0.0) / (pmax - 1.0)

        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_axes((0.04, 0.19, 0.92, 0.70), projection="3d")
        plot_view(ax, enhancement, n, camera, norm, cmap)
        fig.text(0.5, 0.965, f"Cubed-sphere {args.label} blast · {n}×{n} native cells/face",
                 ha="center", va="center", fontsize=16)
        fig.text(0.5, 0.905,
                 f"t = {time:.3f}     max p = {pmax:.3f}     view: 45° above, azimuth 30°",
                 ha="center", va="center", fontsize=12)
        cax = fig.add_axes((0.18, 0.095, 0.64, 0.035))
        colorbar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap),
                                cax=cax, orientation="horizontal")
        colorbar.set_label("Normalized pressure enhancement (p − 1) / (max p − 1)")
        path = args.frames_dir / f"blast_{index:03d}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"{index:03d} t={time:.6f} max_p={pmax:.6f} {path}")


if __name__ == "__main__":
    main()
