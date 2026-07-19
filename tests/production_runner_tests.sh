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

# [3] One-Dimensional Reference Commands
base_dry_run="$(
    scripts/run_report2_local.sh \
        --suite base \
        --methods SIM,DIM \
        --generate-exact-references \
        --dry-run
)"
grep -Fq -- "tools/generate_fedkiw_exact_references.py" <<< "$base_dry_run"
grep -Fq -- "--resolutions 100\\,200\\,400\\,800" <<< "$base_dry_run"

three_material_dry_run="$(
    scripts/run_report2_local.sh \
        --suite three-material \
        --methods SIM,DIM \
        --dry-run
)"
grep -Fq -- "--resolutions 100\\,200\\,400\\,800\\,2000" <<< "$three_material_dry_run"

# [4] Method Isolation
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
grep -Fq -- "--sensitivity sim_weno2_reinit_interval_bubble" <<< "$sensitivity_dry_run"
grep -Fq -- "report_selected_sim_weno2_reinit_interval_sensitivity" <<< "$sensitivity_dry_run"
if grep -Fq "dim_interface_thickness_bubble" <<< "$sensitivity_dry_run" ||
   grep -Fq "dim_alpha_bubble" <<< "$sensitivity_dry_run" ||
   grep -Fq "sim_reinit_bubble" <<< "$sensitivity_dry_run" ||
   grep -Fq "sim_redistance_interval_bubble" <<< "$sensitivity_dry_run"; then
    echo "production runner test failed: SIM sensitivity selected the wrong study" >&2
    exit 1
fi

dim_sensitivity_dry_run="$(
    scripts/run_report2_local.sh \
        --suite sensitivity \
        --methods DIM \
        --dry-run
)"
grep -Fq -- "--sensitivity dim_interface_thickness_bubble" <<< "$dim_sensitivity_dry_run"
grep -Fq -- "report_selected_dim_interface_thickness_sensitivity" <<< "$dim_sensitivity_dry_run"
if grep -Fq "dim_alpha_bubble" <<< "$dim_sensitivity_dry_run" ||
   grep -Fq "sim_reinit_bubble" <<< "$dim_sensitivity_dry_run" ||
   grep -Fq "sim_redistance_interval_bubble" <<< "$dim_sensitivity_dry_run" ||
   grep -Fq "sim_weno2_reinit_interval_bubble" <<< "$dim_sensitivity_dry_run"; then
    echo "production runner test failed: DIM thickness dry run selected the wrong study" >&2
    exit 1
fi

# [5] CSC Delegation And Resource Validation
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

# [6] Exact Reference Generation
python_bin=".venv/bin/python"
if [[ ! -x "$python_bin" ]]; then
    python_bin="python3"
fi
"$python_bin" tools/generate_fedkiw_exact_references.py \
    --tests test5 \
    --samples 2000

# [7] Quantitative Registry
build_dir="$(mktemp -d)"
trap 'rm -rf "$build_dir"' EXIT
g++ -std=c++17 -O0 -fopenmp -I. tests/quant_runner_tests.cpp -o "$build_dir/quant_runner_tests"
"$build_dir/quant_runner_tests"

# [8] Report Plot References
"$python_bin" tests/report2_plot_tests.py

# [9] Zero-Preserving Redistance
g++ -std=c++17 -O0 -fopenmp -I. \
    tests/level_set_redistance_tests.cpp \
    -o "$build_dir/level_set_redistance_tests"
"$build_dir/level_set_redistance_tests"

# [10] WENO2 And SSP-RK2 Level Set
g++ -std=c++17 -O0 -fopenmp -I. \
    tests/level_set_weno2_tests.cpp \
    -o "$build_dir/level_set_weno2_tests"
"$build_dir/level_set_weno2_tests"

echo "production runner tests passed"
