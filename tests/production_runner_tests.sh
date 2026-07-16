#!/usr/bin/env bash
set -euo pipefail

# [0] Repository
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

# [1] Shell Syntax
bash -n scripts/run_report2_local.sh
bash -n scripts/run_report2_csc.sh
bash -n scripts/run_quant_suite.sh

# [2] Production Catalogue
suite_list="$(scripts/run_report2_local.sh --list)"
grep -Fq "Fedkiw D2 1D" <<< "$suite_list"
grep -Fq "325x45x45" <<< "$suite_list"
grep -Fq "120x100x100" <<< "$suite_list"

# [3] Method Isolation
performance_dry_run="$(
    scripts/run_report2_local.sh \
        --suite performance \
        --methods SIM \
        --scaling-threads 1,3 \
        --benchmark-warmups 0 \
        --benchmark-repeats 1 \
        --dry-run
)"
grep -Fq -- "--methods SIM" <<< "$performance_dry_run"
if grep -Fq -- "--methods SIM,DIM" <<< "$performance_dry_run"; then
    echo "production runner test failed: SIM-only dry run includes DIM" >&2
    exit 1
fi

sensitivity_dry_run="$(
    scripts/run_report2_local.sh \
        --suite sensitivity \
        --methods SIM \
        --dry-run
)"
grep -Fq "sim_reinit_bubble" <<< "$sensitivity_dry_run"
if grep -Fq "dim_alpha_bubble" <<< "$sensitivity_dry_run"; then
    echo "production runner test failed: SIM-only sensitivity includes DIM" >&2
    exit 1
fi

# [4] CSC Delegation And Resource Validation
csc_dry_run="$(
    SLURM_JOB_ID=test SLURM_CPUS_PER_TASK=32 \
        scripts/run_report2_csc.sh \
        --suite performance \
        --methods SIM \
        --scaling-threads 1,2,4,8,16,32 \
        --benchmark-warmups 0 \
        --benchmark-repeats 1 \
        --dry-run
)"
grep -Fq "cpus_per_task=32" <<< "$csc_dry_run"
grep -Fq -- "--methods SIM" <<< "$csc_dry_run"
if SLURM_JOB_ID=test SLURM_CPUS_PER_TASK=32 \
    scripts/run_report2_csc.sh \
    --suite performance \
    --methods SIM \
    --scaling-threads 1,64 \
    --dry-run >/dev/null 2>&1; then
    echo "production runner test failed: CSC accepted more threads than allocated" >&2
    exit 1
fi

# [5] Quantitative Registry
build_dir="$(mktemp -d)"
trap 'rm -rf "$build_dir"' EXIT
g++ -std=c++17 -O0 -fopenmp -I. tests/quant_runner_tests.cpp -o "$build_dir/quant_runner_tests"
"$build_dir/quant_runner_tests"

echo "production runner tests passed"
