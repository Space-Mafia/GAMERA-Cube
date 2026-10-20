#!/usr/bin/env python3
"""Plot native MHD-shell parallel current with physical cube-face seams."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("shell_csv", type=Path)
    parser.add_argument("png", type=Path)
    args = parser.parse_args()
    data = np.genfromtxt(args.shell_csv, delimiter=",", names=True)
    if not np.isfinite(data["jparallel_A_m2"]).all():
        raise ValueError("nonfinite shell current")
    phi = np.deg2rad(data["phi_deg"])
    theta = 90.0 - np.abs(data["mlat_deg"])
    x, y = theta * np.sin(phi), -theta * np.cos(phi)
    values = data["jparallel_A_m2"] * 1e6
    maximum = max(np.percentile(np.abs(values), 99), 1e-12)
    seam_phi = np.linspace(0, 2 * np.pi, 1441)
    horizontal = np.maximum(np.abs(np.cos(seam_phi)), np.abs(np.sin(seam_phi)))
    seam_theta = np.rad2deg(np.arctan(1.0 / horizontal))
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.8), dpi=160)
    for ax, hemisphere, mask in ((axes[0], "North", data["mlat_deg"] > 0),
                                  (axes[1], "South", data["mlat_deg"] < 0)):
        dots = ax.scatter(x[mask], y[mask], c=values[mask], s=25,
                          cmap="RdBu_r", edgecolors="none",
                          norm=TwoSlopeNorm(vmin=-maximum, vcenter=0,
                                            vmax=maximum), rasterized=True)
        ax.plot(seam_theta * np.sin(seam_phi),
                -seam_theta * np.cos(seam_phi), "k--", lw=1.2)
        for latitude in (60, 45, 30):
            ax.add_patch(plt.Circle((0, 0), 90-latitude, fill=False,
                                    color="0.7", lw=0.5))
        for phi_deg, text in ((0, "00"), (90, "06"), (180, "12"), (270, "18")):
            angle = np.deg2rad(phi_deg)
            ax.text(73*np.sin(angle), -73*np.cos(angle), text,
                    ha="center", va="center")
        ax.set(xlim=(-77, 77), ylim=(-77, 77), aspect="equal",
               title=f"{hemisphere}: native MHD-shell J∥")
        ax.axis("off")
        fig.colorbar(dots, ax=ax, orientation="horizontal", pad=0.02,
                     fraction=0.05, label="J∥ [µA m⁻²]")
    fig.suptitle("Physical cube-face seams (dashed); no plot interpolation",
                 y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    args.png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.png, bbox_inches="tight")
    print(args.png)


if __name__ == "__main__":
    main()
