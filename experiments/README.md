# Quantitative Assessment Suite

This suite runs existing DIM, SIM/rGFM, and common Euler cases without changing solver physics. The quantitative runner is C++ (`tools/quant_suite.cpp`) and is launched through a thin Bash wrapper. Python is optional and only used for plotting.

## Quick Start

```bash
scripts/run_quant_suite.sh --preset quick --dry-run
scripts/run_quant_suite.sh --preset quick
scripts/run_quant_suite.sh --preset full
```

The wrapper compiles `tools/quant_suite.cpp`; the C++ runner compiles solver executables with `-fopenmp` when needed and runs them with `OMP_NUM_THREADS=6` by default. Override this with `--omp-threads N`.

Runs also use a per-case timeout so one unstable case cannot trap the whole suite:

```bash
scripts/run_quant_suite.sh --cases fedkiw --timeout-seconds 900
scripts/run_quant_suite.sh --cases fedkiw --conservation-interval 25
```

Set `--timeout-seconds 0` to disable the timeout.

## Useful Filters

```bash
scripts/run_quant_suite.sh --cases fedkiw --methods DIM SIM --resolutions 100 200 400 800
scripts/run_quant_suite.sh --cases shock_bubble --methods DIM SIM
scripts/run_quant_suite.sh --cases explosion3d --resolutions 100
scripts/run_quant_suite.sh --sensitivity dim_epsilon
scripts/run_quant_suite.sh --sensitivity sim_reinit
scripts/run_quant_suite.sh --scaling openmp_threads
scripts/run_quant_suite.sh --preset quick --omp-threads 6
```

`DIM epsilon_alpha` is implemented through the existing solver config key `interface_thickness`. Metadata and summary CSVs report it as `epsilon_alpha`.

## Outputs

Each run creates a timestamped directory under:

```text
results/quantitative/<timestamp>/
```

Important files:

```text
manifest.json
summaries/summary.csv
summaries/error_summary.csv
summaries/convergence_summary.csv
summaries/error_compact_summary.csv
summaries/conservation_summary.csv
summaries/conservation_balance_compact.csv
summaries/performance_summary.csv
summaries/interface_summary.csv
summaries/bubble_feature_summary.csv
summaries/bubble_feature_positions.csv
summaries/bubble_feature_timeseries.csv
summaries/bubble_feature_velocity_fits.csv
summaries/bubble_feature_velocity_summary.csv
summaries/sensitivity_summary.csv
summaries/scaling_summary.csv
report/error_report_summary.csv
report/error_summary.csv
report/convergence_summary.csv
report/convergence_report_summary.csv
report/error_linf_appendix_summary.csv
report/convergence_order_report_summary.csv
report/main_results_table.csv
report/performance_report_summary.csv
report/performance_environment.csv
report/conservation_drift_report_summary.csv
report/interface_location_report_summary.csv
report/bubble_feature_velocity_report_summary.csv
figures/bubble_feature_x_t.svg
figures/bubble_feature_y_t.svg
logs/
raw/
runs/<run_id>/metadata.json
runs/<run_id>/bubble_features.json  # shock-bubble runs only
```

`summaries/` contains the full diagnostic/audit CSVs. `report/` contains compact tables intended to be copied into dissertation tables and plots.

The suite enables conservation diagnostics by setting `SOLVER_CONSERVATION=1` for solver subprocesses.

## Error Columns

`summaries/error_summary.csv` keeps both absolute and normalized error metrics:

- `raw_error` / `error`: absolute L1, L2, or Linf error.
- `reference_norm`: the matching norm of the reference data.
- `normalized_error`: `raw_error / max(reference_norm, floor)`, which is the safer cross-variable comparison metric.

Digitized Fedkiw references labelled `Entropy` are not treated as internal energy. Energy rows are only emitted when the reference column is actually an energy/internal-energy quantity.

`summaries/convergence_summary.csv` reports all available refinement pairs, including non-adjacent pairs such as 100 to 400 when a middle run is missing. `summaries/error_compact_summary.csv` gives one row per case/method/variable/norm with 100, 200, and 400 normalized errors, observed orders, a `non_monotonic_100_400` flag, and a `non_monotonic_any_refinement` flag.

Use `report/error_summary.csv` for the main normalized-error evidence table. `report/error_report_summary.csv` is kept as the same table for compatibility with older notes/scripts. These files contain normalized L1 and L2 error magnitudes only:

```text
case,method,variable,norm,normalized_error_100,normalized_error_200,normalized_error_400,non_monotonic_100_400,non_monotonic_any_refinement
```

Use `report/convergence_summary.csv` for the derived pairwise convergence evidence:

```text
case,method,variable,norm,N_coarse,N_fine,normalized_error_coarse,normalized_error_fine,observed_order
```

Use `report/convergence_report_summary.csv` when you want one compact row per case/method/variable/norm with the observed orders across 100, 200, and 400:

```text
case,method,variable,norm,order_100_200,order_200_400,order_100_400,non_monotonic_100_400,non_monotonic_any_refinement
```

Treat normalized L1 as the primary validation metric and L2 as supporting evidence. `error_linf_appendix_summary.csv` keeps normalized Linf values and compact observed orders for overshoot and wave-position checks, but Linf should not be the headline convergence metric for shock/contact problems.

Use `report/main_results_table.csv` for the compact headline convergence-order table. `report/convergence_order_report_summary.csv` is kept as the same table for compatibility:

```text
case,SIM_mean_L1_order_100_400,DIM_mean_L1_order_100_400,SIM_variable_count,DIM_variable_count
```

This table averages normalized L1 100-to-400 observed orders over pressure, density, and velocity, then adds an `Overall mean` row.

## Performance Tables

`summaries/performance_summary.csv` is the per-run diagnostic timing table. It records clean scalar `N` values such as `100` or `200x200`, repeat metadata, output/diagnostic status, wall time, throughput, and cost-per-cell-update.

Use `report/performance_report_summary.csv` for the dissertation performance table:

```text
case,resolution,SIM_cell_updates_per_second_median,DIM_cell_updates_per_second_median,SIM_over_DIM_speedup,SIM_wall_time_seconds_median,DIM_wall_time_seconds_median,SIM_cost_per_cell_update_seconds_median,DIM_cost_per_cell_update_seconds_median,SIM_repeat_count,DIM_repeat_count
```

Warmup runs are excluded from the medians. Use `cell_updates_per_second` as the main performance metric, with wall time and cost per cell update as supporting metrics.

For a controlled lab-computer benchmark run, use repeats and a warmup:

```bash
scripts/run_quant_suite.sh --cases fedkiw --methods DIM SIM --resolutions 100 200 400 --benchmark-mode clean --benchmark-warmups 1 --benchmark-repeats 5 --omp-threads 6
```

`report/performance_environment.csv` records hostname, CPU model when available, compiler/version, compiler flags, OpenMP thread count, build type, benchmark mode, repeat count, and warmup count.

## Conservation Columns

`summaries/conservation_summary.csv` preserves raw solver diagnostics and adds boundary-corrected balance diagnostics:

- `raw_max_relative_drift`: maximum of all raw `eps_*` columns, including momentum.
- `reportable_max_relative_drift`: maximum raw relative drift, excluding momentum only when the initial momentum is zero or near-zero.
- `absolute_momentum*_drift`: absolute change in total momentum.
- `trajectory_scaled_momentum*_drift`: absolute momentum drift divided by the maximum absolute sampled momentum, avoiding division by near-zero initial momentum.
- `raw_eps_momentum*_near_zero_initial_flag`: `1` when the raw relative momentum metric is unsafe to interpret directly.
- `raw_drift_*`: final domain integral minus initial domain integral.
- `boundary_flux_*`: cumulative `sum_t dt * (F_right - F_left)` domain-boundary flux.
- `balance_residual_*`: `raw_drift + boundary_flux`.
- `normalized_balance_residual_*`: absolute balance residual divided by `max(abs(Q0), characteristic_scale, floor)`.
- `interface_flux_mismatch_*`: SIM/GFM cumulative interface flux mismatch; zero for DIM.
- `near_zero_initial_*`: `1` when a quantity's initial integral is near zero relative to its observed scale.

`summaries/conservation_balance_compact.csv` gives the audit long form for boundary-corrected balances:

```text
case,method,resolution,variable,raw_drift,boundary_flux,balance_residual,normalized_balance_residual,interface_flux_mismatch,near_zero_initial_flag
```

For historical runs that were produced before boundary-flux accumulation was enabled, use `report/conservation_drift_report_summary.csv` as the report-facing conservation table:

```text
case,method,drift_100,drift_200,drift_400
```

This table uses the maximum raw relative drift over non-momentum conserved quantities, so momentum relative drift cannot dominate the report metric. Interpret it as corrected raw domain-integral drift, not as a final boundary-corrected conservation residual.

## Interface Tables

`summaries/interface_summary.csv` is the wide diagnostic output. Use the compact location table in `report/` for the main text:

- `report/interface_location_report_summary.csv`: interface positions at 100, 200, and 400 cells, the 100-to-400 shift, and SIM-DIM position differences.

Pressure/velocity oscillation and DIM thickness diagnostics remain available in `summaries/interface_summary.csv` for audit work, but they are not emitted as separate report tables.

## Case Arms

- `toro`: Toro 1D shock tubes plus 2D/3D explosions.
- `toro_no3d`: Toro 1D plus 2D explosion only.
- `toro_1d`: Toro shock tubes only.
- `explosion2d` / `explosion3d`: individual explosion arms.
- `fedkiw`: 1D Fedkiw DIM/SIM cases.
- `planar`: 2D planar multimaterial cases.
- `shock_bubble` or `bubble`: 2D helium shock-bubble case.

## Bubble Features

Run the shock-bubble arm with:

```bash
scripts/run_quant_suite.sh --cases shock_bubble --methods DIM SIM --omp-threads 6 --timeout-seconds 0 --result-root results/quantitative/bubble_full
```

For a cheaper pipeline check, pass a smaller grid such as `--resolutions 325x45`.

`summaries/bubble_feature_summary.csv` gives one compact row per method/run with the latest accepted tracked feature positions:

```text
case,case_label,method,resolution,run_id,success,snapshot_count,latest_time_code,upstream_interface_x_latest_position_mm,downstream_interface_x_latest_position_mm,jet_head_x_latest_position_mm,transverse_interface_y_latest_position_mm,transverse_triple_point_x_latest_position_mm,transverse_triple_point_y_latest_position_mm,transverse_wave_y_latest_position_mm
```

`summaries/bubble_feature_positions.csv` contains one row per tracked feature at each saved output time:

```text
case,case_label,method,resolution,run_id,feature,time_code,time_us_from_initial,position_mm,confidence_flag,accepted,jet_detected
```

For the configured `test6` helium bubble case, the incident shock starts on the right and travels toward decreasing `x`. The tracked interface definitions therefore use centerline crossings as:

- `upstream_interface_x`: rightmost centreline-band `alpha1 = 0.5` crossing, facing the incident shock.
- `downstream_interface_x`: leftmost centreline-band `alpha1 = 0.5` crossing.
- `jet_head_x`: connected centreline air-intrusion region inside the bubble, found from `alpha1 < 0.2`; if the intrusion is shorter than `2 mm`, it is reported as not detected.
- `transverse_interface_y`: maximum y-coordinate of the main connected `alpha1 = 0.5` bubble contour.
- `transverse_triple_point_x`, `transverse_triple_point_y`: red-dot/triple-point coordinate from the upper, shock-facing portion of the main material-interface contour. The contour is extracted by sub-cell interpolation of `alpha1 = 0.5` when available, `phi0 = 0` for SIM level-set output, or a density threshold only as a fallback. The selected point is where interpolated `|grad p|` is largest along that contour.
- `transverse_wave_y`: optional upper-bubble pressure-gradient ridge location; low confidence or not detected when no coherent ridge is found.

Disconnected alpha fragments are removed before contour and centreline measurements. The tracker does not assign `jet_head_x` to the upstream interface when no resolved air intrusion exists.

Use `summaries/bubble_feature_velocity_fits.csv` for final feature velocities:

```text
method,feature,start_time_us,end_time_us,start_position_mm,end_position_mm,displacement_mm,duration_us,fitted_velocity_m_per_s,r_squared,rmse_mm,n_points,confidence_flag
```

For each method and feature, the fitted-velocity summary sorts the positions by `time_us_from_initial`, converts time to seconds and position to metres, scans every contiguous window with at least six points, and fits:

```text
x_m = a + V t_s
```

The selected velocity is the slope `V` in `m/s`. Windows must move by at least `5 mm` for `x` features or `1 mm` for transverse `y` features. The selected window is the longest one with `R^2 >= 0.995` for `x` features or `R^2 >= 0.95` for transverse features. If no window passes, the best available fit is retained with `confidence_flag=low`.

Points that imply a frame-to-frame speed above `1000 m/s` are rejected before fitting and marked with `confidence_flag=speed_rejected` in the positions CSV. The fitted velocity is the positive speed magnitude.

`summaries/bubble_feature_velocity_summary.csv` is kept as a compatibility copy with both `fitted_velocity_m_per_s` and the older `fit_velocity_m_per_s` column.

Use `figures/bubble_feature_x_t.svg` and `figures/bubble_feature_y_t.svg` to visually audit the accepted tracked points over x-t and y-t diagrams.

Use `report/bubble_feature_velocity_report_summary.csv` only for the x-feature method comparison:

```text
feature,tracked_feature,rGFM_m_per_s,Allaire5eq_m_per_s,ref_comp_m_per_s,experiment_m_per_s,rGFM_error_percent,Allaire5eq_error_percent,rGFM_confidence,Allaire5eq_confidence
```

Feature mapping:

- `V_ui`: `upstream_interface_x`
- `V_j`: `jet_head_x`
- `V_di`: `downstream_interface_x`
- `V_T`: `transverse_triple_point_x`

The time conversion uses the shock-bubble setup scaling from the config comments: `123` code-time units correspond to `427 microseconds`.
