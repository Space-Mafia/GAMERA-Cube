#!/usr/bin/env python3
"""Render a native-cell cubed-sphere blast snapshot as two 3-D globe views.

Each colored quadrilateral is one original angular cell. No field regridding
or interpolation is used; only the known gnomonic cell-corner geometry is
reconstructed for display.
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np


# Same ordered (normal, tangent_a, tangent_b) frames as cubedsphere_mesh.c.
FRAMES = np.array([
    [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
    [[-1, 0, 0], [0, -1, 0], [0, 0, 1]],
    [[0, 1, 0], [-1, 0, 0], [0, 0, 1]],
    [[0, -1, 0], [1, 0, 0], [0, 0, 1]],
    [[0, 0, 1], [1, 0, 0], [0, 1, 0]],
    [[0, 0, -1], [1, 0, 0], [0, -1, 0]],
], dtype=float)


def direction(face, a, b):
    frame = FRAMES[face]
    xyz = frame[0] + np.tan(a)[..., None] * frame[1] + np.tan(b)[..., None] * frame[2]
    return xyz / np.linalg.norm(xyz, axis=-1, keepdims=True)


def face_corners(face, n):
    angles = np.linspace(-np.pi / 4, np.pi / 4, n + 1)
    a, b = np.meshgrid(angles, angles, indexing="ij")
    return direction(face, a, b)


def plot_view(ax, pressure, n, camera, norm, cmap):
    camera = np.asarray(camera, dtype=float)
    camera /= np.linalg.norm(camera)
    quads, colors = [], []
    for face in range(6):
        corner = face_corners(face, n)
        center = 0.25 * (corner[:-1, :-1] + corner[1:, :-1]
                         + corner[1:, 1:] + corner[:-1, 1:])
        for j in range(n):
            for k in range(n):
                # Orthographic visibility. Keep native cells; never interpolate.
                if np.dot(center[j, k], camera) <= 0:
                    continue
                quads.append([corner[j, k], corner[j + 1, k],
                              corner[j + 1, k + 1], corner[j, k + 1]])
                colors.append(cmap(norm(pressure[face, j, k])))
    ax.add_collection3d(Poly3DCollection(
        quads, facecolors=colors, edgecolors="none", zsort="average",
        antialiased=False))

    # Draw only visible portions of the original twelve patch seams.
    seam_keys = set()
    samples = np.linspace(-np.pi / 4, np.pi / 4, 241)
    for face in range(6):
        for side in range(4):
            a = np.full_like(samples, (-1 if side == 0 else 1) * np.pi / 4) \
                if side < 2 else samples
            b = samples if side < 2 else np.full_like(
                samples, (-1 if side == 2 else 1) * np.pi / 4)
            points = direction(face, a, b)
            key = tuple(np.round(np.sort(points, axis=0)[[0, -1]].ravel(), 8))
            if key in seam_keys:
                continue
            seam_keys.add(key)
            visible = points @ camera > 0.01
            shown = points.copy()
            shown[~visible] = np.nan
            ax.plot(shown[:, 0] * 1.012, shown[:, 1] * 1.012,
                    shown[:, 2] * 1.012, color="0.16", linewidth=1.15,
                    zorder=100)

    ax.set(xlim=(-1.02, 1.02), ylim=(-1.02, 1.02), zlim=(-1.02, 1.02))
    ax.set_box_aspect((1, 1, 1), zoom=1.16)
    ax.set_axis_off()
    ax.set_proj_type("ortho")
    ax.view_init(elev=np.degrees(np.arcsin(camera[2])),
                 azim=np.degrees(np.arctan2(camera[1], camera[0])))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path)
    parser.add_argument("png", type=Path)
    parser.add_argument("--label", default="HD", choices=("HD", "MHD"))
    args = parser.parse_args()
    data = np.genfromtxt(args.csv, delimiter=",", names=True)
    times = np.unique(data["time"])
    n = int(np.max(data["j"])) + 1
    if n < 2 or data.size != len(times) * 6 * n * n:
        raise ValueError("incomplete six-face snapshot sequence")
    final = data[data["time"] == times[-1]]
    pressure = np.empty((6, n, n), dtype=float)
    for face in range(6):
        rows = final[final["face"] == face]
        pressure[face, rows["j"].astype(int), rows["k"].astype(int)] = rows["pressure"]
    if not np.isfinite(pressure).all() or np.min(pressure) <= 0:
        raise ValueError("nonfinite or nonpositive pressure")

    norm = Normalize(vmin=float(np.min(pressure)), vmax=float(np.max(pressure)))
    cmap = plt.get_cmap("turbo")
    fig = plt.figure(figsize=(12, 5.7))
    axes = [fig.add_axes((0.02 + 0.49 * i, 0.19, 0.47, 0.68),
                         projection="3d") for i in range(2)]
    for ax, camera, title in zip(
            axes, ((1, 1, 1), (-1, -1, -1)),
            ("Blast-side view (+X, +Y, +Z)",
             "Opposite view (−X, −Y, −Z)")):
        plot_view(ax, pressure, n, camera, norm, cmap)
        fig.text(0.255 + 0.49 * axes.index(ax), 0.87, title,
                 ha="center", va="center", fontsize=12)
    cax = fig.add_axes((0.19, 0.09, 0.62, 0.035))
    colorbar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap),
                            cax=cax, orientation="horizontal")
    colorbar.set_label("Pressure (code units), native angular cells")
    fig.suptitle(f"Cubed-sphere {args.label} blast at t={times[-1]:.3f} — {n}×{n} cells/face",
                 y=0.975)
    args.png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.png, dpi=190)
    plt.close(fig)
    print(args.png)

    indices = np.unique(np.round(np.linspace(2, len(times) - 1, 3)).astype(int))
    fig = plt.figure(figsize=(15, 5.5))
    axes = [fig.add_axes((0.02 + i * 0.325, 0.20, 0.32, 0.64),
                         projection="3d") for i in range(len(indices))]
    for i, (ax, index) in enumerate(zip(axes, indices)):
        frame = data[data["time"] == times[index]]
        field = np.empty((6, n, n), dtype=float)
        for face in range(6):
            rows = frame[frame["face"] == face]
            field[face, rows["j"].astype(int), rows["k"].astype(int)] = rows["pressure"]
        max_pressure = float(np.max(field))
        enhancement = np.maximum(field - 1.0, 0.0) / (max_pressure - 1.0)
        plot_view(ax, enhancement, n, (1, 1, 1), Normalize(0, 1), cmap)
        fig.text(0.18 + i * 0.325, 0.85,
                 f"t={times[index]:.3f}  ·  max p={max_pressure:.3f}",
                 ha="center", va="center", fontsize=11)
    cax = fig.add_axes((0.19, 0.09, 0.62, 0.035))
    colorbar = fig.colorbar(ScalarMappable(norm=Normalize(0, 1), cmap=cmap),
                            cax=cax, orientation="horizontal")
    colorbar.set_label("(p − 1) / (max p at this time − 1), clipped at zero")
    fig.suptitle(f"{args.label} blast wave crossing cubed-sphere seams — {n}×{n} native cells/face",
                 y=0.975)
    evolution = args.png.with_name(args.png.stem.replace("_final", "_evolution")
                                   + args.png.suffix)
    fig.savefig(evolution, dpi=190)
    plt.close(fig)
    print(evolution)


if __name__ == "__main__":
    main()
