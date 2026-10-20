#!/usr/bin/env python3
"""Measure the discrete CT/center-field initialization order, not MHD order."""

import math
import re
import subprocess
import sys


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit("usage: check_alfven_initial_order.py TEST_EXECUTABLE")
    errors = []
    for n in (9, 18, 36):
        result = subprocess.run(
            [sys.argv[1], str(n), "1", "0.000001"],
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
    print(f"CT/center-field initial observed orders: {orders[0]:.4f}, {orders[1]:.4f}")
    if not all(1.8 <= order <= 2.2 for order in orders):
        raise RuntimeError("initial field does not exhibit expected second-order convergence")
    return 0


if __name__ == "__main__":
    sys.exit(main())
