#!/usr/bin/env python3
"""Check smooth-wave evolution with time-dependent analytic radial ghosts."""

import math
import re
import subprocess
import sys


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit("usage: check_alfven_evolution_order.py TEST_EXECUTABLE")
    errors = []
    for n in (9, 18, 36):
        result = subprocess.run(
            [sys.argv[1], str(n), "20", "0.002", "analytic"],
            check=True,
            capture_output=True,
            text=True,
        )
        match = re.search(r"RMS ([0-9.eE+-]+).*max divB ([0-9.eE+-]+)", result.stdout)
        if match is None:
            raise RuntimeError(f"unrecognized Alfvén output: {result.stdout}")
        error, divergence = map(float, match.groups())
        if not math.isfinite(error) or divergence > 1e-9:
            raise RuntimeError(f"invalid result at n={n}: {result.stdout}")
        errors.append(error)
        print(result.stdout.strip(), flush=True)
    orders = [math.log(errors[i] / errors[i + 1], 2.0) for i in (0, 1)]
    print(f"smooth-wave observed orders: {orders[0]:.4f}, {orders[1]:.4f}")
    if not all(1.7 <= order <= 2.2 for order in orders):
        raise RuntimeError("boundary-compatible wave order fell below second-order gate")
    return 0


if __name__ == "__main__":
    sys.exit(main())
