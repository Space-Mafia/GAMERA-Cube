#!/usr/bin/env python3
"""A drive validator must infer grid shape, not assume a named resolution."""

import contextlib
import io
from pathlib import Path
import sys
import tempfile

import check_earth_drive_outputs as validator


def write_frame(path: Path, radial: int, angular: int):
    with path.open("w") as stream:
        stream.write("face,i,j,k,x,y,z,rho,p,vx,vy,vz,bx,by,bz\n")
        for face in range(6):
            for i in range(radial):
                for j in range(angular):
                    for k in range(angular):
                        stream.write(f"{face},{i},{j},{k},0,0,0,1,1,0,0,0,0,0,0\n")
    for suffix in (".mi_north.csv", ".mi_south.csv"):
        Path(str(path) + suffix).write_text("value,label\n0,finite\n")
    Path(str(path) + ".shell_j.csv").write_text("value\n0\n")


def main():
    with tempfile.TemporaryDirectory(prefix="cube-drive-shape-") as temp:
        root = Path(temp)
        paths = [root / "frame_000000.csv", root / "frame_000001.csv"]
        for path in paths:
            write_frame(path, radial=2, angular=3)
        index = root / "frame.index.csv"
        index.write_text(
            "frame,step,simulation_seconds,mhd_csv\n"
            f"0,0,0,{paths[0]}\n"
            f"1,1,1,{paths[1]}\n"
        )
        sys.argv = ["check_earth_drive_outputs.py", str(index),
                    "--stop-seconds", "1", "--output-seconds", "1"]
        with contextlib.redirect_stdout(io.StringIO()):
            validator.main()
        sys.argv.extend(["--angular", "4"])
        try:
            validator.main()
        except ValueError:
            pass
        else:
            raise AssertionError("incorrect asserted angular size was accepted")
    print("Earth drive shape inference PASS")


if __name__ == "__main__":
    main()
