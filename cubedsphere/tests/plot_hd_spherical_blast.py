#!/usr/bin/env python3
"""Plot native-cell six-face HD blast snapshots, without regridding."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree


FACE_NAMES = ("+X", "−X", "+Y", "−Y", "+Z", "−Z")
ROTATED_FACE = (2, 3, 4, 5, 0, 1)


def panel(ax, frame, face, n, vmin, vmax, field="pressure"):
    rows = frame[frame["face"] == face]
    array = np.full((n, n), np.nan)
    array[rows["j"].astype(int), rows["k"].astype(int)] = rows[field]
    if not np.isfinite(array).all() or (array <= 0).any():
        raise ValueError(f"missing/nonpositive cells on face {face}")
    artist = ax.imshow(
        np.log10(array), origin="lower", interpolation="nearest",
        cmap="turbo", vmin=vmin, vmax=vmax,
    )
    ax.set_title(FACE_NAMES[face])
    ax.set_xticks((0, n // 2, n - 1))
    ax.set_yticks((0, n // 2, n - 1))
    ax.set_xlabel("local k cell")
    ax.set_ylabel("local j cell")
    return artist


def cyclic_symmetry(frame):
    errors = []
    distances = []
    for face in range(6):
        source = frame[frame["face"] == face]
        target = frame[frame["face"] == ROTATED_FACE[face]]
        source_xyz = np.stack([source["x"], source["y"], source["z"]], axis=1)
        target_xyz = np.stack([target["x"], target["y"], target["z"]], axis=1)
        rotated = source_xyz[:, [2, 0, 1]]
        distance, index = cKDTree(target_xyz).query(rotated)
        distances.extend(distance.tolist())
        errors.extend(np.abs(source["pressure"] - target["pressure"][index]))
    return {
        "max_matching_distance": float(np.max(distances)),
        "max_pressure_difference": float(np.max(errors)),
        "rms_pressure_difference": float(np.sqrt(np.mean(np.square(errors)))),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data = np.genfromtxt(args.csv, delimiter=",", names=True)
    if data.size == 0 or not np.isfinite(data["pressure"]).all():
        raise ValueError("empty or nonfinite blast CSV")
    times = np.unique(data["time"])
    n = int(np.max(data["j"])) + 1
    if len(times) < 2 or data.size != len(times) * 6 * n * n:
        raise ValueError("incomplete six-face snapshot sequence")
    vmin, vmax = 0.0, np.log10(20.0)
    final = data[data["time"] == times[-1]]

    fig, axes = plt.subplots(2, 3, figsize=(13, 8), constrained_layout=True)
    final_pmin = float(np.log10(np.min(final["pressure"])))
    final_pmax = float(np.log10(np.max(final["pressure"])))
    for face, ax in enumerate(axes.flat):
        artist = panel(ax, final, face, n, final_pmin, final_pmax)
    fig.colorbar(artist, ax=axes, shrink=0.8, label="log₁₀ pressure")
    fig.suptitle(f"Cubed-sphere HD angular blast — six native faces, t={times[-1]:.3f}")
    final_path = args.output_dir / "hd_blast_six_faces_final.png"
    fig.savefig(final_path, dpi=170)
    plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(13, 8), constrained_layout=True)
    final_rhomin = float(np.log10(np.min(final["density"])))
    final_rhomax = float(np.log10(np.max(final["density"])))
    for face, ax in enumerate(axes.flat):
        artist = panel(ax, final, face, n, final_rhomin, final_rhomax,
                       field="density")
    fig.colorbar(artist, ax=axes, shrink=0.8, label="log₁₀ density")
    fig.suptitle(f"Cubed-sphere HD angular blast — density, t={times[-1]:.3f}")
    density_path = args.output_dir / "hd_blast_six_faces_density_final.png"
    fig.savefig(density_path, dpi=170)
    plt.close(fig)

    selected = np.unique(np.round(np.linspace(0, len(times) - 1, 5)).astype(int))
    fig, axes = plt.subplots(len(selected), 6, figsize=(18, 3 * len(selected)),
                             constrained_layout=True)
    for row, index in enumerate(selected):
        frame = data[data["time"] == times[index]]
        for face in range(6):
            artist = panel(axes[row, face], frame, face, n, vmin, vmax)
            if face == 0:
                axes[row, face].annotate(f"t={times[index]:.3f}",
                                          (-0.4, 0.5), xycoords="axes fraction",
                                          rotation=90, va="center")
    fig.colorbar(artist, ax=axes, shrink=0.5, label="log₁₀ pressure")
    fig.suptitle("Angular blast propagation across all six cube faces (native cells)")
    evolution_path = args.output_dir / "hd_blast_six_faces_evolution.png"
    fig.savefig(evolution_path, dpi=150)
    plt.close(fig)

    summary = {"resolution": n, "times": times.tolist(),
               "final_cyclic_symmetry": cyclic_symmetry(final)}
    summary_path = args.output_dir / "hd_blast_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"final_png": str(final_path),
                      "density_png": str(density_path),
                      "evolution_png": str(evolution_path),
                      "summary": summary}, indent=2))


if __name__ == "__main__":
    main()
