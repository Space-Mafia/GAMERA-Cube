#!/usr/bin/env python3
"""Validate complete cubed-sphere Earth drive snapshots without NumPy."""

import argparse
import csv
import math
from pathlib import Path
from typing import Optional


def finite_csv(path: Path, expected_rows: Optional[int], text_columns=()):
    with path.open(newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader, None)
        if not header:
            raise ValueError(f"empty CSV: {path}")
        count = 0
        largest = [-1, -1, -1, -1]
        for line, row in enumerate(reader, 2):
            if len(row) != len(header):
                raise ValueError(f"wrong column count: {path}:{line}")
            for column, value in enumerate(row):
                if column not in text_columns and not math.isfinite(float(value)):
                    raise ValueError(f"nonfinite value: {path}:{line}")
            if header[:5] == ["face", "i", "j", "k", "x"]:
                for axis in range(4):
                    value = int(row[axis])
                    if value < 0:
                        raise ValueError(f"negative grid index: {path}:{line}")
                    largest[axis] = max(largest[axis], value)
                if float(row[7]) <= 0 or float(row[8]) <= 0:
                    raise ValueError(f"nonpositive density/pressure: {path}:{line}")
            count += 1
    if expected_rows is not None and count != expected_rows:
        raise ValueError(f"{path}: expected {expected_rows} rows, got {count}")
    if count == 0:
        raise ValueError(f"no data rows: {path}")
    shape = tuple(value + 1 for value in largest) if largest[0] >= 0 else None
    if shape is not None and count != math.prod(shape):
        raise ValueError(f"incomplete or repeated grid rows: {path}: {shape}")
    return count, shape


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("index", type=Path)
    parser.add_argument("--radial", type=int)
    parser.add_argument("--angular", type=int)
    parser.add_argument("--j-cells", type=int)
    parser.add_argument("--k-cells", type=int)
    parser.add_argument("--stop-seconds", type=float, required=True)
    parser.add_argument("--output-seconds", type=float, required=True)
    args = parser.parse_args()
    if any(value is not None and value <= 0 for value in
           (args.radial, args.angular, args.j_cells, args.k_cells)):
        raise ValueError("grid dimensions must be positive")
    j_cells = args.j_cells if args.j_cells is not None else args.angular
    k_cells = args.k_cells if args.k_cells is not None else args.angular
    if (args.angular is not None and
            ((j_cells is not None and j_cells != args.angular) or
             (k_cells is not None and k_cells != args.angular))):
        raise ValueError("--angular conflicts with --j-cells or --k-cells")

    with args.index.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["frame", "step", "simulation_seconds", "mhd_csv"]:
            raise ValueError("unexpected drive index columns")
        frames = list(reader)
    expected_frames = math.ceil(args.stop_seconds / args.output_seconds) + 1
    if len(frames) != expected_frames:
        raise ValueError(f"expected {expected_frames} frames, got {len(frames)}")

    previous_time = -1.0
    expected_shape = None
    for ordinal, row in enumerate(frames):
        time = float(row["simulation_seconds"])
        if int(row["frame"]) != ordinal or not math.isfinite(time) or time <= previous_time:
            raise ValueError(f"invalid frame/time at index row {ordinal}")
        path = Path(row["mhd_csv"])
        _, shape = finite_csv(path, None)
        if shape is None or shape[0] != 6:
            raise ValueError(f"invalid six-face grid shape: {path}: {shape}")
        if expected_shape is None:
            expected_shape = shape
            if ((args.radial is not None and shape[1] != args.radial) or
                    (j_cells is not None and shape[2] != j_cells) or
                    (k_cells is not None and shape[3] != k_cells)):
                raise ValueError(f"grid shape differs from requested assertions: {shape}")
        elif shape != expected_shape:
            raise ValueError(f"grid shape changed between frames: {shape}")
        finite_csv(Path(str(path) + ".mi_north.csv"), None, (1,))
        finite_csv(Path(str(path) + ".mi_south.csv"), None, (1,))
        finite_csv(Path(str(path) + ".shell_j.csv"), None)
        previous_time = time
    if frames[0]["step"] != "0" or float(frames[0]["simulation_seconds"]) != 0.0:
        raise ValueError("missing initial snapshot")
    if previous_time < args.stop_seconds:
        raise ValueError("drive did not reach requested stop time")
    print(f"EARTH_DRIVE_OUTPUT_PASS frames={len(frames)} final_s={previous_time:.9g}")


if __name__ == "__main__":
    main()
