#!/usr/bin/env python3
"""Check the aligned-dipole zero-FAC limit against the no-MI MHD step."""

import csv
import itertools
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile


def run(binary: Path, mode: str, output: Path) -> None:
    environment = os.environ.copy()
    environment.setdefault("OMP_NUM_THREADS", "4")
    result = subprocess.run(
        [str(binary), mode, "1", str(output)],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )
    if result.returncode:
        raise RuntimeError(f"{mode} failed:\n{result.stdout}\n{result.stderr}")


def compare(off_path: Path, on_path: Path) -> None:
    tolerances = {"rho": 1e-12, "p": 1e-12,
                  "vx": 1e-12, "vy": 1e-12, "vz": 1e-12,
                  "bx": 1e-10, "by": 1e-10, "bz": 1e-10}
    maximum = {name: 0.0 for name in tolerances}
    count = 0
    with off_path.open(newline="") as off_file, on_path.open(newline="") as on_file:
        off = csv.DictReader(off_file)
        on = csv.DictReader(on_file)
        for off_row, on_row in itertools.zip_longest(off, on):
            if off_row is None or on_row is None:
                raise AssertionError("MI-on/off CSVs have different row counts")
            if any(off_row[name] != on_row[name]
                   for name in ("face", "i", "j", "k")):
                raise AssertionError(f"cell indices differ at row {count}")
            for name, tolerance in tolerances.items():
                left, right = float(off_row[name]), float(on_row[name])
                if not math.isfinite(left) or not math.isfinite(right):
                    raise AssertionError(f"nonfinite {name} at row {count}")
                difference = abs(left - right)
                maximum[name] = max(maximum[name], difference)
                if difference > tolerance:
                    raise AssertionError(
                        f"MI-on/off {name} differs by {difference:g} at row {count}"
                    )
            count += 1
    radial_cells = int(os.environ.get("GAMERA_CS_EARTH_RADIAL_CELLS", "72"))
    if count != radial_cells * 16 * 16 * 6:
        raise AssertionError(f"unexpected number of active cells: {count}")
    print(f"zero-FAC MI reduction PASS: {count} cells; max differences {maximum}")


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: check_earth_mi_reduction.py TEST_BINARY")
    binary = Path(sys.argv[1]).resolve()
    with tempfile.TemporaryDirectory(prefix="cubedsphere-mi-reduction-") as directory:
        root = Path(directory)
        off_path = root / "mi_off.csv"
        on_path = root / "mi_on.csv"
        run(binary, "earth-local", off_path)
        run(binary, "earth-mi-local", on_path)
        compare(off_path, on_path)


if __name__ == "__main__":
    main()
