#!/usr/bin/env python3
"""Plot vertices exported by export_cubedsphere_grid (no resampling)."""

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def draw_faces(ax, mesh, faces, elevation, azimuth, title):
    colors = {0: "#4488b6", 1: "#4488b6",
              2: "#59a77d", 3: "#59a77d",
              4: "#d39a50", 5: "#d39a50"}
    names = {0: "+X", 1: "−X", 2: "+Y", 3: "−Y",
             4: "+Z", 5: "−Z"}
    for face in faces:
        xyz = mesh[face]
        ax.plot_surface(xyz[:, :, 0], xyz[:, :, 1], xyz[:, :, 2],
                        color=colors[face], alpha=0.14,
                        linewidth=0, shade=False)
        for index in range(0, 33, 2):
            u = xyz[index, :, :]
            v = xyz[:, index, :]
            ax.plot(*u.T, color=colors[face], linewidth=0.48, alpha=0.83)
            ax.plot(*v.T, color=colors[face], linewidth=0.48, alpha=0.83)
        for edge in (xyz[0], xyz[-1], xyz[:, 0], xyz[:, -1]):
            ax.plot(*edge.T, color="#263744", linewidth=1.25, alpha=0.86)
        center = xyz[16, 16] * 1.11
        ax.text(*center, names[face], fontsize=10, fontweight="bold",
                ha="center", va="center", color="#263744")
    ax.set(xlim=(-15.5, 15.5), ylim=(-15.5, 15.5),
           zlim=(-15.5, 15.5))
    ax.set_box_aspect((1, 1, 1), zoom=0.84)
    ax.view_init(elev=elevation, azim=azimuth)
    ax.set_axis_off()
    ax.set_title(title, fontsize=12, pad=3)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("vertices_csv")
    parser.add_argument("output_png")
    args = parser.parse_args()
    data = np.genfromtxt(args.vertices_csv, delimiter=",", names=True,
                         dtype=None, encoding="utf-8")
    surface = data[data["kind"] == "surface"]
    radial = data[data["kind"] == "radial"]
    if len(surface) != 6 * 33 * 33 or len(radial) != 73:
        raise ValueError("Unexpected vertex count")
    mesh = np.empty((6, 33, 33, 3), dtype=float)
    for row in surface:
        mesh[int(row["face"]), int(row["j"]), int(row["k"])] = [
            row["x"], row["y"], row["z"]]
    radius = np.empty(73)
    for row in radial:
        radius[int(row["i"])] = row["r"]
    if not np.isfinite(mesh).all() or not np.isfinite(radius).all():
        raise ValueError("Non-finite exported grid vertex")
    assert np.all(np.diff(radius) > 0)
    assert np.isclose(radius[0], 2.5)
    assert np.isclose(radius[23], 14.0)
    assert np.isclose(radius[-1], 200.0)

    fig = plt.figure(figsize=(15.4, 6.5), layout="constrained")
    gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 0.94])
    front = fig.add_subplot(gs[:, 0], projection="3d")
    back = fig.add_subplot(gs[:, 1], projection="3d")
    outer = fig.add_subplot(gs[0, 2])
    spacing = fig.add_subplot(gs[1, 2], sharex=outer)
    draw_faces(front, mesh, (0, 2, 4), 27, 43,
               "Three visible faces • r = 14 RE")
    draw_faces(back, mesh, (1, 3, 5), -27, 223,
               "Opposite three faces • r = 14 RE")
    i = np.arange(73)
    outer.plot(i, radius, color="#286994", linewidth=1.8)
    outer.scatter([0, 23, 72], [2.5, 14, 200], color="#c47632", s=22,
                  zorder=3)
    outer.axvline(23, color="#777777", linewidth=0.8, linestyle="--")
    outer.set(xlim=(0, 72), ylim=(0, 205), ylabel="radius / RE",
              title="Earth radial map v4 • 72 cells")
    outer.text(23.7, 44, "23 cells at 0.5 RE", fontsize=9, color="#3f494f")
    outer.grid(alpha=0.18)
    dr = np.diff(radius)
    spacing.plot(np.arange(72) + 0.5, dr, color="#59a77d", linewidth=1.8)
    spacing.axvline(23, color="#777777", linewidth=0.8, linestyle="--")
    spacing.set(xlabel="radial face index", ylabel="cell width / RE",
                yscale="log")
    spacing.grid(alpha=0.18)
    fig.suptitle("GAMERA cube-sphere mesh • 6 conforming patches, no overlap",
                 fontsize=15)
    fig.savefig(args.output_png, dpi=185, bbox_inches="tight")


if __name__ == "__main__":
    main()
