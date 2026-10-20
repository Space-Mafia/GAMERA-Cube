#!/usr/bin/env python3
"""Overlay mapped cube-face seams on the native MI FAC output."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.colors import TwoSlopeNorm
import numpy as np

from plot_earth_mi_dipole import cells, load


def sample_radius(mhd_csv: Path, shell: int) -> float:
    mhd = np.genfromtxt(mhd_csv, delimiter=",", names=True)
    selected = mhd["i"] == shell
    if not np.any(selected):
        raise ValueError(f"sample shell {shell} missing from {mhd_csv}")
    radii = np.sqrt(sum(mhd[axis][selected] ** 2 for axis in ("x", "y", "z")))
    if not np.isfinite(radii).all():
        raise ValueError("nonfinite sample-shell coordinates")
    return float(radii.mean())


def mapped_seam(phi_deg: np.ndarray, sample_radius_re: float,
                ionosphere_radius_re: float) -> np.ndarray:
    phi = np.deg2rad(phi_deg)
    largest_horizontal = np.maximum(np.abs(np.cos(phi)), np.abs(np.sin(phi)))
    theta_mhd = np.arctan(1.0 / largest_horizontal)
    return np.rad2deg(np.arcsin(np.sin(theta_mhd) *
                                np.sqrt(ionosphere_radius_re / sample_radius_re)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("north", type=Path)
    parser.add_argument("south", type=Path)
    parser.add_argument("mhd", type=Path, help="same-time MHD cell CSV")
    parser.add_argument("png", type=Path)
    parser.add_argument("--shell", type=int, default=1)
    parser.add_argument("--ionosphere-radius", type=float, default=1.0)
    parser.add_argument("--baseline-north", type=Path)
    parser.add_argument("--baseline-south", type=Path)
    args = parser.parse_args()

    north, theta_n, phi_n = load(args.north)
    south, theta_s, phi_s = load(args.south)
    if not np.array_equal(theta_n, theta_s) or not np.array_equal(phi_n, phi_s):
        raise ValueError("north/south MI grids differ")
    radius = sample_radius(args.mhd, args.shell)
    phi_line = np.linspace(0.0, 360.0, 1441)
    theta_line = mapped_seam(phi_line, radius, args.ionosphere_radius)
    angle = np.deg2rad(phi_line)
    x_seam, y_seam = theta_line * np.sin(angle), -theta_line * np.cos(angle)
    polygons = cells(theta_n, phi_n)
    if (args.baseline_north is None) != (args.baseline_south is None):
        raise ValueError("provide both baseline hemisphere files")
    panels = [("North", north["fac_A_m2"] * 1e6),
              ("South", south["fac_A_m2"] * 1e6)]
    if args.baseline_north is not None:
        before_n, theta_bn, phi_bn = load(args.baseline_north)
        before_s, theta_bs, phi_bs = load(args.baseline_south)
        if any(not np.array_equal(a, b) for a, b in
               ((theta_n, theta_bn), (phi_n, phi_bn),
                (theta_n, theta_bs), (phi_n, phi_bs))):
            raise ValueError("baseline and corrected MI grids differ")
        panels = [("Before · North", before_n["fac_A_m2"] * 1e6),
                  ("Before · South", before_s["fac_A_m2"] * 1e6),
                  ("Fixed · North", panels[0][1]),
                  ("Fixed · South", panels[1][1])]
    maximum = max(np.percentile(np.abs(values), 99) for _, values in panels)
    rows = len(panels) // 2
    fig, axes = plt.subplots(rows, 2, figsize=(11, 5.7 * rows), dpi=160)
    for ax, (label, values) in zip(np.ravel(axes), panels):
        collection = PolyCollection(polygons, array=values, cmap="RdBu_r",
                                    norm=TwoSlopeNorm(vmin=-maximum, vcenter=0,
                                                      vmax=maximum),
                                    edgecolors="none", rasterized=True)
        ax.add_collection(collection)
        ax.plot(x_seam, y_seam, color="black", lw=1.2, ls="--",
                label="mapped cube-face seam")
        corners = np.array([45.0, 135.0, 225.0, 315.0])
        corner_theta = mapped_seam(corners, radius, args.ionosphere_radius)
        corner_angle = np.deg2rad(corners)
        ax.scatter(corner_theta * np.sin(corner_angle),
                   -corner_theta * np.cos(corner_angle), s=18,
                   facecolor="yellow", edgecolor="black", zorder=5)
        for latitude in (80, 70, 60, 50):
            theta = 90.0 - latitude
            if theta < theta_n[-1]:
                ax.add_patch(plt.Circle((0, 0), theta, fill=False,
                                        color="0.65", lw=0.5))
        for phi, text in ((0, "00"), (90, "06"), (180, "12"), (270, "18")):
            phi_r = np.deg2rad(phi)
            ax.text((theta_n[-1] + 2.0) * np.sin(phi_r),
                    -(theta_n[-1] + 2.0) * np.cos(phi_r), text,
                    ha="center", va="center")
        ax.set(xlim=(-43, 43), ylim=(-43, 43), aspect="equal", title=label)
        ax.axis("off")
        fig.colorbar(collection, ax=ax, orientation="horizontal", pad=0.02,
                     fraction=0.05, label="FAC [µA m⁻²]")
    corner_latitude = 90.0 - mapped_seam(
        np.array([135.0]), radius, args.ionosphere_radius)[0]
    fig.suptitle(f"t={north['time_s'][0]:.1f} s · r={radius:.3f} RE"
                 f" · cube corners at 09/15 MLT map to {corner_latitude:.1f}° MLAT",
                 y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    args.png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.png, bbox_inches="tight")
    print(f"{args.png}\nseam MLAT at 09/15={corner_latitude:.3f}°")


if __name__ == "__main__":
    main()
