#!/usr/bin/env python3
"""Fast student launcher contract tests; no simulation or dependencies."""
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest

SCRIPT=Path(__file__).with_name("run_earth.py")

class Launcher(unittest.TestCase):
    def run_case(self,*args):
        env=dict(os.environ, GAMERA_CS_EARTH_CONSTANT_BZ_NT="999",
                 IO_NEUTRAL_NU0_S="1", GAMERA_CS_PROFILE_TILE="1")
        return subprocess.run([sys.executable,str(SCRIPT),"--output","runs/test-dry",
                               "--dry-run",*args],env=env,capture_output=True,text=True)
    def test_profiles(self):
        for profile,ranks,threads,frames in (("smoke",6,1,2),("teaching",48,2,37),("reference",96,2,37)):
            r=self.run_case("--profile",profile)
            self.assertEqual(r.returncode,0,r.stderr)
            data=json.loads(r.stdout)
            self.assertEqual((data["mpi_ranks"],data["omp_threads"],data["expected_frames"]),
                             (ranks,threads,frames))
            self.assertEqual(data["environment"]["GAMERA_CS_EARTH_CONSTANT_BZ_NT"],"-5")
            self.assertNotIn("GAMERA_CS_PROFILE_TILE",data["environment"])
    def test_bad_times(self):
        for value in ("nan","inf","0","-1","1.5"):
            self.assertNotEqual(self.run_case("--stop",value).returncode,0)

if __name__=="__main__":
    unittest.main()
