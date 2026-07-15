#!/usr/bin/env bash
set -euo pipefail

echo "scripts/run_sm_mpi_probe.sh is deprecated."
echo "MPI probe runs are disabled for the Report 2 workflow."
echo "Use scripts/run_quant_suite.sh --scaling openmp_threads --case bubble --methods SIM,DIM instead."
exit 2
