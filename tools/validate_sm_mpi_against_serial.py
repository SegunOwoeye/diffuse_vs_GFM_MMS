#!/usr/bin/env python3
"""Disabled legacy MPI validation entry point."""

from __future__ import annotations


def main() -> int:
    print("tools/validate_sm_mpi_against_serial.py is deprecated.")
    print("MPI validation is disabled for the Report 2 workflow.")
    print("Use tools/quant_suite with --scaling openmp_threads for current performance checks.")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
