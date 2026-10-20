#!/usr/bin/env python3
"""Exact 20-step six-rank vs serial Earth/MI regression gate."""

import os
from pathlib import Path
import subprocess
import sys
import tempfile


def run(command, environment):
    result = subprocess.run(
        command, env=environment, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        timeout=180, check=False,
    )
    if result.returncode:
        raise RuntimeError(
            f"command failed ({result.returncode}): {command}\n"
            f"{result.stdout[-5000:]}"
        )


def main():
    if len(sys.argv) < 5:
        raise SystemExit(
            "usage: check_mpi_earth_parity.py SERIAL MPI_BINARY "
            "MPIEXEC [MPIEXEC_FLAGS...] MPI_N_FLAG"
        )
    serial, distributed, launcher = sys.argv[1:4]
    launcher_args = sys.argv[4:]
    env = os.environ.copy()
    env.update({
        "GAMERA_CS_EARTH_RADIAL_CELLS": "48",
        "GAMERA_CS_EARTH_INNER_UNIFORM_CELLS": "23",
        "GAMERA_CS_EARTH_FRONT_X": "-12",
        "OMP_NUM_THREADS": "1",
    })
    with tempfile.TemporaryDirectory(prefix="gamera-cs-mpi-parity-") as tmp:
        serial_csv = Path(tmp) / "serial.csv"
        mpi_csv = Path(tmp) / "mpi.csv"
        run([serial, "earth-mi-local", "20", str(serial_csv)], env)
        run([launcher, *launcher_args, "6", distributed,
             "earth-mi-local", "20", str(mpi_csv)], env)
        for suffix in ("", ".mi_north.csv", ".mi_south.csv",
                       ".shell_j.csv"):
            a = Path(str(serial_csv) + suffix).read_bytes()
            b = Path(str(mpi_csv) + suffix).read_bytes()
            if a != b:
                raise AssertionError(f"six-rank output differs: {suffix}")
    print("six-rank Earth MHD/MI/FAC 20-step parity: BITWISE PASS")


if __name__ == "__main__":
    main()
