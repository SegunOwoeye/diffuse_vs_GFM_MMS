#!/usr/bin/env python3
"""Merge rank-local MPI solver CSV files into one solver-style CSV."""

from __future__ import annotations

import argparse
import re
from pathlib import Path


def rank_key(path: Path) -> int:
    match = re.match(r"rank_(\d+)_", path.name)
    if not match:
        return 10**9
    return int(match.group(1))


def merge_rank_csv(input_dir: Path, output: Path, prefix: str | None) -> None:
    if prefix:
        rank_files = sorted(
            input_dir.glob(f"rank_*_{prefix}.csv"),
            key=rank_key,
        )
    else:
        rank_files = sorted(input_dir.glob("rank_*.csv"), key=rank_key)

    if not rank_files:
        raise FileNotFoundError(f"No rank CSV files found in {input_dir}")

    output.parent.mkdir(parents=True, exist_ok=True)
    wrote_header = False
    with output.open("w", newline="") as merged:
        for rank_file in rank_files:
            with rank_file.open(newline="") as handle:
                for line_number, line in enumerate(handle):
                    if line_number == 0:
                        if not wrote_header:
                            merged.write(line)
                            wrote_header = True
                        continue
                    merged.write(line)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--prefix", default=None)
    args = parser.parse_args()

    merge_rank_csv(args.input_dir, args.output, args.prefix)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
