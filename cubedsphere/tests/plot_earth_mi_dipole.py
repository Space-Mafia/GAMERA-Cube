#!/usr/bin/env python3
"""Plot the native aligned-dipole MI grid; no image-grid interpolation."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.colors import TwoSlopeNorm
import numpy as np


def load(path):
    table = np.genfromtxt(path, delimiter=",", names=True,
                          dtype=None, encoding="utf-8")
    theta = np.unique(table["theta_deg"])
    phi = np.unique(table["phi_deg"])
    if len(table) != len(theta) * len(phi):
        raise ValueError(f"incomplete MI grid: {path}")
    if not all(np.isfinite(table[name]).all() for name in
               ("time_s", "theta_deg", "phi_deg", "fac_A_m2", "potential_V")):
        raise ValueError(f"nonfinite MI grid: {path}")
    return table, theta, phi


def cells(theta, phi):
    theta_edges = np.empty(len(theta) + 1)
    theta_edges[0] = 0.0
    theta_edges[-1] = theta[-1]
    theta_edges[1:-1] = 0.5 * (theta[:-1] + theta[1:])
    dphi = 360.0 / len(phi)
    phi_edges = np.arange(len(phi) + 1) * dphi - 0.5 * dphi
    polygons = []
    for t in range(len(theta)):
        for p in range(len(phi)):
            vertices = []
            for radius, angle in ((theta_edges[t], phi_edges[p]),
                                  (theta_edges[t + 1], phi_edges[p]),
                                  (theta_edges[t + 1], phi_edges[p + 1]),
                                  (theta_edges[t], phi_edges[p + 1])):
                radians = np.deg2rad(angle)
                vertices.append((radius * np.sin(radians),
                                 -radius * np.cos(radians)))
            polygons.append(vertices)
    return polygons


def render_axis(ax, polygons, values, maximum, outer, label, cmap):
    collection = PolyCollection(polygons, array=values, cmap=cmap,
                                norm=TwoSlopeNorm(vmin=-maximum, vcenter=0,
                                                  vmax=maximum),
                                edgecolors="none", rasterized=True)
    ax.add_collection(collection)
    for latitude in (80, 70, 60, 50):
        radius = 90 - latitude
        if radius < outer:
            ax.add_patch(plt.Circle((0, 0), radius, fill=False,
                                    linewidth=0.5, color="0.68"))
            ax.text(0.2, radius + 0.3, f"{latitude}°", fontsize=8,
                    color="0.42")
    for angle, text in ((0, "00"), (90, "06"), (180, "12"), (270, "18")):
        radians = np.deg2rad(angle)
        ax.text((outer + 2.2) * np.sin(radians),
                -(outer + 2.2) * np.cos(radians), text,
                ha="center", va="center", fontsize=10)
    ax.set_xlim(-outer - 4, outer + 4)
    ax.set_ylim(-outer - 4, outer + 4)
    ax.set_aspect("equal")
    ax.axis("off")
    plt.colorbar(collection, ax=ax, orientation="horizontal", pad=0.03,
                 fraction=0.045, label=label)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("north", type=Path)
    parser.add_argument("south", type=Path)
    parser.add_argument("png", type=Path)
    args = parser.parse_args()
    north, theta_n, phi_n = load(args.north)
    south, theta_s, phi_s = load(args.south)
    if not np.array_equal(theta_n, theta_s) or not np.array_equal(phi_n, phi_s):
        raise ValueError("north/south MI grids differ")
    polygons = cells(theta_n, phi_n)
    fields = (
        (north["fac_A_m2"] * 1e6, "FAC [μA m⁻²]", "RdBu_r"),
        (north["potential_V"] * 1e-3, "Potential [kV]", "RdBu_r"),
        (south["fac_A_m2"] * 1e6, "FAC [μA m⁻²]", "RdBu_r"),
        (south["potential_V"] * 1e-3, "Potential [kV]", "RdBu_r"),
    )
    maxima = [max(np.percentile(np.abs(fields[q][0]), 99),
                  np.percentile(np.abs(fields[q + 2][0]), 99), 1e-12)
              for q in (0, 1)]
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 10), dpi=150)
    for index, ax in enumerate(axes.flat):
        values, label, cmap = fields[index]
        render_axis(ax, polygons, values, maxima[index % 2], theta_n[-1],
                    label, cmap)
        ax.set_title(("North" if index < 2 else "South") +
                     " — " + ("FAC" if index % 2 == 0 else "potential"),
                     fontsize=12)
    epoch = float(north["time_s"][0])
    fig.suptitle(f"Aligned-dipole cubed-sphere MI · t={epoch:.1f} s · "
                 f"{len(phi_n)}×{len(theta_n)} per hemisphere", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    args.png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.png, bbox_inches="tight")
    print(args.png)


if __name__ == "__main__":
    main()
