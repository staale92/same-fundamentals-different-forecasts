#!/usr/bin/env python3
"""Reproduce numerical results and figures from frozen panels and model outputs."""

from pathlib import Path
import argparse
import os
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--all",
        action="store_true",
        help="Recompute manuscript analyses and additional diagnostics from saved model outputs",
    )
    parser.add_argument(
        "--primary",
        action="store_true",
        help="Recompute the primary estimates, sensitivities, validation and benchmarks",
    )
    args = parser.parse_args()
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT / "src")
    env["MPLCONFIGDIR"] = str(ROOT / ".cache/matplotlib")
    env["XDG_CACHE_HOME"] = str(ROOT / ".cache")
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    env.setdefault("OMP_NUM_THREADS", "1")
    jobs = []
    if args.all:
        jobs = [
            ("analyze_pilot.py",),
            ("analyze_annual.py",),
            ("analyze_quarterly.py",),
            ("analyze_quarterly.py", "--confirmation"),
            ("analyze_forecasts.py",),
            ("analyze_planted.py",),
            ("analyze_controls.py",),
            ("analyze_primary.py",),
            ("analyze_presentation.py",),
            ("analyze_diagnostics.py",),
            ("analyze_audit_checks.py",),
            ("analyze_identification.py",),
            ("analyze_sensitivity.py",),
            ("analyze_elicitation.py",),
        ]
    elif args.primary:
        jobs = [
            ("analyze_primary.py",),
            ("analyze_presentation.py",),
            ("analyze_diagnostics.py",),
            ("analyze_audit_checks.py",),
            ("analyze_identification.py",),
            ("analyze_sensitivity.py",),
            ("analyze_elicitation.py",),
        ]
    jobs += [("export_results.py",), ("verify.py", "--results-only")]
    for script, *options in jobs:
        start = time.monotonic()
        print(f'Running {script} {" ".join(options)}', flush=True)
        subprocess.run(
            [sys.executable, str(ROOT / "scripts" / script), *options],
            cwd=ROOT,
            env=env,
            check=True,
        )
        print(f"Completed in {time.monotonic()-start:.1f} seconds", flush=True)


if __name__ == "__main__":
    main()
