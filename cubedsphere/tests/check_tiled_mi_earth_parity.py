#!/usr/bin/env python3
"""Physical Earth/MI parity: six-face vs true MPI tiles."""

import os
from pathlib import Path
import subprocess
import sys
import tempfile
import math
import re


def run(command, environment):
    result = subprocess.run(
        command, env=environment, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        timeout=240, check=False,
    )
    if result.returncode:
        raise RuntimeError(
            f"command failed ({result.returncode}): {command}\n"
            f"{result.stdout[-5000:]}"
        )
    return result.stdout


def main():
    if len(sys.argv) < 5:
        raise SystemExit(
            "usage: check_tiled_mi_earth_parity.py SERIAL TILED "
            "MPIEXEC [MPIEXEC_FLAGS...] MPI_N_FLAG"
        )
    serial, tiled, launcher = sys.argv[1:4]
    launcher_args = sys.argv[4:]
    steps = int(os.environ.get("GAMERA_CS_PARITY_STEPS", "2"))
    radial_tiles = int(os.environ.get("GAMERA_CS_PARITY_RADIAL_TILES", "2"))
    if radial_tiles not in (2, 4):
        raise ValueError("Earth parity supports two or four radial tiles")
    if steps < 1:
        raise ValueError("GAMERA_CS_PARITY_STEPS must be positive")
    environment = os.environ.copy()
    environment.update({
        "GAMERA_CS_EARTH_RADIAL_CELLS": "72",
        "GAMERA_CS_EARTH_ANGULAR_CELLS": "16",
        "GAMERA_CS_EARTH_FRONT_X": "-30",
        "GAMERA_CS_EARTH_STRICT_INNER_GAS_WALL": "1",
        "OMP_NUM_THREADS": "1",
    })
    with tempfile.TemporaryDirectory(prefix="gamera-cs-tiled-mi-") as tmp:
        reference = Path(tmp) / "reference.csv"
        pieces = Path(tmp) / "tiled.csv"
        run([serial, "earth-mi-local", str(steps), str(reference)], environment)
        tiled_env = dict(environment, GAMERA_CS_TILE_TEST_MI="1")
        tiled_log = run([launcher, *launcher_args, str(6*radial_tiles), tiled,
             str(radial_tiles), "1", "1", "72", "16", "16",
             str(pieces), str(steps)], tiled_env)
        match = re.search(r"steps=(\d+) time=([0-9.eE+-]+)", tiled_log)
        if match is None or int(match.group(1)) != steps:
            raise AssertionError(f"missing tiled time/step receipt: {tiled_log[-1000:]}")
        physical_seconds = float(match.group(2)) * 6371000.0 / 100000.0
        if steps >= 64 and physical_seconds < 10.0:
            raise AssertionError(
                f"requested cadence crossing but reached only {physical_seconds}s"
            )
        expected = reference.read_text().splitlines()
        actual = []
        for piece in sorted(Path(tmp).glob("tiled.csv.rank*.csv")):
            actual.extend(piece.read_text().splitlines()[1:])
        def keyed(rows):
            result = {}
            for row in rows:
                fields = row.split(",")
                key = tuple(map(int, fields[:4]))
                if key in result:
                    raise AssertionError(f"duplicate cell: {key}")
                result[key] = tuple(map(float, fields[4:]))
            return result

        expected_cells = keyed(expected[1:])
        actual_cells = keyed(actual)
        if expected_cells.keys() != actual_cells.keys():
            raise AssertionError(
                f"physical tiled Earth cell-key mismatch: "
                f"{len(actual_cells)}/{len(expected_cells)}"
            )
        names = expected[0].split(",")[4:]
        maxima = [0.0] * len(names)
        worst = [None] * len(names)
        nonexact = 0
        for key, reference_fields in expected_cells.items():
            observed = actual_cells[key]
            if observed != reference_fields:
                nonexact += 1
            for column, (a, b) in enumerate(zip(reference_fields, observed)):
                difference = abs(a - b)
                scaled = difference / max(1.0, abs(a))
                if not math.isfinite(scaled):
                    raise AssertionError(f"non-finite difference {key} {names[column]}")
                if scaled > maxima[column]:
                    maxima[column] = scaled
                    worst[column] = (key, a, b, difference)
        if nonexact:
            print(f"MHD nonexact cells={nonexact}/{len(expected_cells)}")
            for name, maximum, location in zip(names, maxima, worst):
                if maximum:
                    print(f"  {name}: max_scaled={maximum:.6e} at {location}")
        if max(maxima) > 1e-12:
            raise AssertionError("physical tiled Earth MHD exceeds roundoff threshold")
        for suffix in (".mi_north.csv", ".mi_south.csv", ".shell_j.csv"):
            a = Path(str(reference) + suffix).read_text().splitlines()
            b = Path(str(pieces) + suffix).read_text().splitlines()
            if a != b:
                if len(a) != len(b) or a[0] != b[0]:
                    raise AssertionError(f"physical tiled Earth MI shape differs: {suffix}")
                columns = a[0].split(",")
                largest = [0.0] * len(columns)
                worst = [None] * len(columns)
                nonexact_rows = 0
                for row_number, (line_a, line_b) in enumerate(zip(a[1:], b[1:])):
                    if line_a != line_b:
                        nonexact_rows += 1
                    for column, (va, vb) in enumerate(zip(
                            line_a.split(","), line_b.split(","))):
                        if columns[column] == "hemisphere":
                            if va != vb:
                                raise AssertionError(f"MI hemisphere mismatch: {suffix}")
                            continue
                        number_a, number_b = float(va), float(vb)
                        scaled = abs(number_a - number_b) / max(1.0, abs(number_a))
                        if not math.isfinite(scaled):
                            raise AssertionError(f"non-finite MI difference: {suffix}")
                        if scaled > largest[column]:
                            largest[column] = scaled
                            worst[column] = (row_number, number_a, number_b)
                print(f"{suffix}: nonexact rows {nonexact_rows}/{len(a)-1}")
                for column, maximum, location in zip(columns, largest, worst):
                    if maximum:
                        print(f"  {column}: max_scaled={maximum:.6e} at {location}")
                if max(largest) > 1e-12:
                    raise AssertionError(f"physical tiled Earth MI differs: {suffix}")
    print(f"physical tiled Earth {steps}-step parity to "
          f"{physical_seconds:.6g}s: MHD and MI/FAC "
          "roundoff <=1e-12 scaled")


if __name__ == "__main__":
    main()
