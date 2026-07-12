#!/usr/bin/env python3
"""Run a small SM serial/MPI comparison and report field differences."""

from __future__ import annotations

import argparse
import csv
import math
import shutil
import subprocess
from pathlib import Path

from merge_mpi_rank_csv import merge_rank_csv


def resolution_values(text: str) -> list[int]:
    return [int(part) for part in text.replace(",", "x").split("x") if part]


def render_resolution(values: list[int]) -> str:
    return "[" + ", ".join(str(value) for value in values) + "]"


def resolution_suffix(values: list[int]) -> str:
    return "".join(f"_N{value}" for value in values)


def rewrite_config(source: Path, target: Path, resolution: list[int], output_dir: Path, output_prefix: str) -> None:
    wrote_n = False
    wrote_output_dir = False
    wrote_output_prefix = False
    lines: list[str] = []

    for raw_line in source.read_text().splitlines():
        stripped = raw_line.strip()
        key = stripped.split("=", 1)[0].strip() if "=" in stripped else ""
        if key == "N":
            if not wrote_n:
                lines.append(f"N = {render_resolution(resolution)}")
                wrote_n = True
            continue
        if key == "output_dir":
            lines.append(f"output_dir = {output_dir.as_posix()}")
            wrote_output_dir = True
            continue
        if key == "output_prefix":
            lines.append(f"output_prefix = {output_prefix}")
            wrote_output_prefix = True
            continue
        lines.append(raw_line)

    if not wrote_n:
        lines.append(f"N = {render_resolution(resolution)}")
    if not wrote_output_dir:
        lines.append(f"output_dir = {output_dir.as_posix()}")
    if not wrote_output_prefix:
        lines.append(f"output_prefix = {output_prefix}")

    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(lines) + "\n")


def run_command(command: list[str], cwd: Path) -> None:
    subprocess.run(command, cwd=cwd, check=True)


def read_rows(path: Path) -> list[dict[str, float]]:
    with path.open(newline="") as handle:
        return [
            {key: float(value) for key, value in row.items() if value != ""}
            for row in csv.DictReader(handle)
        ]


def compare_csv(serial_csv: Path, mpi_csv: Path, columns: list[str]) -> dict[str, float]:
    serial_rows = read_rows(serial_csv)
    mpi_rows = read_rows(mpi_csv)
    if len(serial_rows) != len(mpi_rows):
        raise RuntimeError(f"row count mismatch: serial={len(serial_rows)} mpi={len(mpi_rows)}")

    max_abs: dict[str, float] = {column: 0.0 for column in columns}
    for serial_row, mpi_row in zip(serial_rows, mpi_rows):
        for column in columns:
            if column not in serial_row or column not in mpi_row:
                continue
            diff = abs(serial_row[column] - mpi_row[column])
            if math.isfinite(diff):
                max_abs[column] = max(max_abs[column], diff)
    return max_abs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dimension", type=int, required=True, choices=[1, 2, 3])
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--resolution", required=True)
    parser.add_argument("--mpi-ranks", type=int, default=2)
    parser.add_argument("--omp-threads", type=int, default=1)
    parser.add_argument("--work-dir", type=Path, default=Path("tmp/sm_mpi_validation"))
    args = parser.parse_args()

    repo = Path.cwd()
    resolution = resolution_values(args.resolution)
    if len(resolution) != args.dimension:
        raise RuntimeError("resolution dimension does not match --dimension")

    work_dir = args.work_dir
    if work_dir.exists():
        shutil.rmtree(work_dir)
    serial_dir = work_dir / "serial_raw"
    mpi_dir = work_dir / "mpi_raw"
    config_dir = work_dir / "configs"

    serial_prefix = f"serial_sm_{args.dimension}d"
    mpi_prefix = f"mpi_sm_{args.dimension}d"
    serial_config = config_dir / "serial.txt"
    mpi_config = config_dir / "mpi.txt"

    rewrite_config(args.config, serial_config, resolution, serial_dir, serial_prefix)
    rewrite_config(args.config, mpi_config, resolution, mpi_dir, mpi_prefix)

    run_command([
        "g++", "-std=c++17", "-O3", "-march=native", "-fopenmp", "-I.",
        f"-DAPP_DIM={args.dimension}", "src/app/sm_main.cpp", "-o", f"sm_main_{args.dimension}d"
    ], repo)
    run_command([
        "mpic++", "-std=c++17", "-O3", "-march=native", "-fopenmp", "-I.",
        f"-DAPP_DIM={args.dimension}", "src/app/sm_mpi_main.cpp", "-o", f"sm_mpi_main_{args.dimension}d"
    ], repo)

    run_command([f"./sm_main_{args.dimension}d", serial_config.as_posix()], repo)
    run_command([
        "mpirun", "-np", str(args.mpi_ranks), f"./sm_mpi_main_{args.dimension}d",
        mpi_config.as_posix(), "--output-dir", (mpi_dir / mpi_prefix).as_posix()
    ], repo)

    suffix = resolution_suffix(resolution)
    serial_csv = serial_dir / serial_prefix / f"{serial_prefix}{suffix}.csv"
    merged_mpi_csv = mpi_dir / mpi_prefix / f"{mpi_prefix}{suffix}_merged.csv"
    merge_rank_csv(mpi_dir / mpi_prefix, merged_mpi_csv, mpi_prefix)

    columns = ["rho", "p", "e"] + [f"u{d}" for d in range(args.dimension)]
    errors = compare_csv(serial_csv, merged_mpi_csv, columns)

    print("column,max_abs_error")
    for column in columns:
        print(f"{column},{errors[column]:.16e}")
    print(f"serial_csv,{serial_csv}")
    print(f"mpi_csv,{merged_mpi_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
