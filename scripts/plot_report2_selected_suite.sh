#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

result_root_base="results/quantitative"
output_dir=""

print_help() {
    cat <<'EOF'
Generate PNG figures for the selected Report 2 quantitative suite.

Usage:
  scripts/plot_report2_selected_suite.sh [options]

Options:
  --result-root-base PATH    Root containing the selected report result directories.
  --output-dir PATH          Directory for the collected report PNGs.
  --help, -h                 Show this help.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --result-root-base)
            if [[ $# -lt 2 ]]; then
                echo "plot_report2_selected_suite.sh: --result-root-base requires a path" >&2
                exit 2
            fi
            result_root_base="$2"
            shift 2
            ;;
        --output-dir)
            if [[ $# -lt 2 ]]; then
                echo "plot_report2_selected_suite.sh: --output-dir requires a path" >&2
                exit 2
            fi
            output_dir="$2"
            shift 2
            ;;
        --help|-h)
            print_help
            exit 0
            ;;
        *)
            echo "plot_report2_selected_suite.sh: unknown argument '$1'" >&2
            print_help >&2
            exit 2
            ;;
    esac
done

python_bin=".venv/bin/python"
if [[ ! -x "$python_bin" ]]; then
    python_bin="python3"
fi

args=("--result-root-base" "$result_root_base")
if [[ -n "$output_dir" ]]; then
    args+=("--output-dir" "$output_dir")
fi

"$python_bin" src/graphing/plot_report2_selected_suite.py "${args[@]}"
"$python_bin" scripts/update_report2_run_tracker.py
