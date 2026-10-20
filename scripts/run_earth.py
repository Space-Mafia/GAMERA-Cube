#!/usr/bin/env python3
"""Reproducible Earth launch; standard library only. Never overwrites a run."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
PROFILES = {
    "smoke": ((72, 16, 16), (1, 1, 1), 1, 1.0, 1.0),
    "teaching": ((72, 24, 24), (2, 2, 2), 2, 10800.0, 300.0),
    "reference": ((128, 32, 32), (4, 2, 2), 2, 10800.0, 300.0),
}
FLAGS = ("GAMERA_CS_HALO_COMPACT", "GAMERA_CS_HALO_INDEX_CACHE",
         "GAMERA_CS_HALO_BLOCK_ACCESS", "GAMERA_CS_HALO_PARALLEL",
         "GAMERA_NO_EMF_FACE_CACHE", "GAMERA_NO_PRIMITIVE_CACHE")

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--profile", choices=PROFILES, default="reference")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--build", type=Path, default=ROOT / "build")
    p.add_argument("--launcher", choices=("mpiexec", "srun"), default="mpiexec")
    p.add_argument("--stop", type=float)
    p.add_argument("--interval", type=float)
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()
    grid, topology, threads, stop, interval = PROFILES[a.profile]
    stop = stop if a.stop is None else a.stop
    interval = interval if a.interval is None else a.interval
    if not all(math.isfinite(x) and x > 0 for x in (stop, interval)):
        p.error("stop and interval must be finite and positive")
    if interval > stop or abs(stop / interval - round(stop / interval)) > 1e-9:
        p.error("stop must be an integer multiple of interval")
    ranks = 6 * math.prod(topology)
    out = a.output.resolve()
    binary = a.build.resolve() / "cubedsphere_earth_mpi_drive"
    validator = a.build.resolve() / "check_cubedsphere_earth_hdf5"
    env = {k:v for k,v in os.environ.items()
           if not k.startswith(("GAMERA_CS_", "GAMERA_NO_", "IO_", "MARS_"))}
    env.update({k:"1" for k in FLAGS})
    env.update(OMP_NUM_THREADS=str(threads), OMP_DYNAMIC="FALSE",
               OMP_PROC_BIND="close", OMP_PLACES="cores",
               GAMERA_CS_TILE_DRIVE="1",
               GAMERA_CS_TILE_PRODUCTION_PRESSURE_CONTROL="1",
               GAMERA_CS_EARTH_CONSTANT_BZ_NT="-5",
               GAMERA_CS_EARTH_STOP_SECONDS=str(stop),
               GAMERA_CS_EARTH_OUTPUT_SECONDS=str(interval))
    launch = [a.launcher, "-n", str(ranks)]
    if a.launcher == "srun":
        launch += ["--cpu-bind=cores", "--kill-on-bad-exit=1"]
        if int(env.get("SLURM_NTASKS", "0")) != ranks or int(env.get("SLURM_CPUS_PER_TASK", "0")) != threads:
            p.error(f"Slurm allocation must have {ranks} tasks and {threads} CPUs/task")
    cmd = launch + [str(binary), *map(str, topology), *map(str, grid),
                    str(out / "frame"), "1000000"]
    receipt = dict(profile=a.profile, grid_per_face=grid, topology_per_face=topology,
                   mpi_ranks=ranks, omp_threads=threads, stop_seconds=stop,
                   output_seconds=interval, expected_frames=round(stop/interval)+1,
                   imf_bz_nt=-5, command=cmd,
                   environment={k:v for k,v in env.items() if k.startswith(("GAMERA_", "OMP_"))})
    if a.dry_run:
        print(json.dumps(receipt, indent=2)); return
    if not binary.is_file() or not validator.is_file():
        p.error("Missing production executables: build with MPI and PARALLEL HDF5")
    if out.exists():
        p.error(f"Output exists; choose a NEW directory: {out}")
    out.mkdir(parents=True)
    receipt["binary_sha256"] = hashlib.sha256(binary.read_bytes()).hexdigest()
    receipt["git_commit"] = subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    receipt["git_status"] = subprocess.run(
        ["git", "-C", str(ROOT), "status", "--porcelain"], capture_output=True, text=True).stdout
    (out/"config_receipt.json").write_text(json.dumps(receipt, indent=2)+"\n")
    print(f"Launching {ranks} MPI x {threads} OMP. Log: {out/'run.log'}", flush=True)
    start = time.monotonic()
    with (out/"run.log").open("w") as log:
        subprocess.run(cmd, env=env, cwd=out, stdout=log, stderr=subprocess.STDOUT, check=True)
    receipt["wall_seconds"] = time.monotonic()-start
    (out/"config_receipt.json").write_text(json.dumps(receipt, indent=2)+"\n")
    with (out/"validation.log").open("w") as log:
        subprocess.run([str(validator), str(out), "--radial", str(grid[0]),
                        "--j-cells", str(grid[1]), "--k-cells", str(grid[2]),
                        "--stop-seconds", str(stop), "--output-seconds", str(interval)],
                       env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    (out/"run_complete.txt").write_text("Simulation and HDF5 validator passed.\n")
    print((out/"validation.log").read_text())

if __name__ == "__main__":
    main()
