#pragma once

// Header-only implementation units for the quantitative validation runner.
// Orchestrates raw artifact collection, then delegates result-specific table building.
#include "report_conservation.hpp"
#include "report_errors.hpp"
#include "report_interfaces.hpp"
#include "report_performance.hpp"
#include "report_bubble.hpp"

namespace quant {

inline bool uses_bubble_feature_diagnostics(const RunSpec& run)
{
    return run.case_def.group == "shock_bubble_2d" ||
           run.case_def.group == "shock_bubble_2d_zero_velocity" ||
           run.case_def.group == "shock_bubble_2d_zero_velocity_physical_flow" ||
           run.case_def.group == "shock_bubble_2d_zero_velocity_input_mean_star" ||
           run.case_def.group == "shock_bubble_2d_zero_velocity_zero_star" ||
           run.case_def.group == "shock_bubble_2d_static_equilibrium";
}

void collect(const fs::path& result_root, const std::vector<RunSpec>& runs, const std::vector<bool>& successes)
{
    std::vector<std::map<std::string, std::string>> summary;
    std::vector<std::map<std::string, std::string>> performance;
    std::vector<std::map<std::string, std::string>> conservation_rows;
    std::vector<std::map<std::string, std::string>> conservation_compact;
    std::vector<std::map<std::string, std::string>> interface_rows;
    std::vector<std::map<std::string, std::string>> sensitivity_rows;
    std::vector<std::map<std::string, std::string>> scaling_rows;
    std::vector<std::map<std::string, std::string>> bubble_rows;
    std::vector<std::map<std::string, std::string>> bubble_timeseries_rows;
    std::vector<std::map<std::string, std::string>> error_rows;

    for (std::size_t i = 0; i < runs.size(); ++i) {
        const auto& run = runs[i];
        auto row = base_row(run, successes[i]);
        const fs::path case_dir = result_root / "raw" / run.run_id / run.output_prefix;
        const fs::path solution = case_dir / (run.output_prefix + resolution_suffix(run.resolution) + ".csv");
        const fs::path runtime = case_dir / (run.output_prefix + resolution_suffix(run.resolution) + "_runtime.txt");
        const fs::path conservation = case_dir / (run.output_prefix + resolution_suffix(run.resolution) + "_conservation.csv");

        const auto perf_values = read_key_values(runtime);
        auto perf_row = row;
        for (const auto& item : perf_values) {
            const std::string value =
                (item.first == "N") ? normalise_runtime_n(item.second) : item.second;
            perf_row[item.first] = value;
            row["performance_" + item.first] = value;
        }
        perf_row["N"] = resolution_label(run.resolution);
        perf_row["benchmark_mode"] = run.benchmark_mode;
        perf_row["repeat_id"] = std::to_string(run.repeat_id);
        perf_row["warmup"] = run.warmup ? "true" : "false";
        perf_row["output_enabled"] = "true";
        perf_row["diagnostics_enabled"] = "true";
        perf_row["conservation_logging_enabled"] = "true";
        perf_row["plotting_enabled"] = "false";
        if (!perf_values.empty()) {
            const double cells = std::stod(perf_values.at("cells"));
            const double steps = std::stod(perf_values.at("steps"));
            const double final_time = std::stod(perf_values.at("final_time"));
            const double wall = std::stod(perf_values.at("wall_time_seconds"));
            if (final_time > 0.0) {
                perf_row["cost_per_simulated_time"] = double_text(wall / final_time);
                row["performance_cost_per_simulated_time"] = perf_row["cost_per_simulated_time"];
            }
            if (wall > 0.0 && cells > 0.0 && steps > 0.0) {
                perf_row["cell_updates_per_second"] = double_text(cells * steps / wall);
                row["performance_cell_updates_per_second"] = perf_row["cell_updates_per_second"];
            }
            performance.push_back(perf_row);
        }

        const auto cons = final_conservation(conservation);
        if (!cons.empty()) {
            auto cons_row = row;
            for (const auto& item : cons) {
                cons_row[item.first] = double_text(item.second);
                row["conservation_" + item.first] = double_text(item.second);
            }
            conservation_rows.push_back(cons_row);
            const auto compact = conservation_compact_rows(run, cons);
            conservation_compact.insert(
                conservation_compact.end(),
                compact.begin(),
                compact.end()
            );
        }

        const auto iface = interface_metrics(solution, run.case_def.method);
        if (!iface.empty()) {
            auto iface_row = row;
            for (const auto& item : iface) {
                iface_row[item.first] = double_text(item.second);
                row["interface_" + item.first] = double_text(item.second);
            }
            const double dx_interface = [&]() {
                const auto cols = read_csv_columns(solution);
                return cols.count("x0") ? cell_width(cols.at("x0")) : std::nan("");
            }();
            if (iface.count("pressure_oscillation")) {
                const double scale = max_abs_reference_or_solution(run, solution, case_dir, "p", 1.0);
                iface_row["normalized_pressure_oscillation"] =
                    double_text(iface.at("pressure_oscillation") / scale);
                row["interface_normalized_pressure_oscillation"] =
                    iface_row["normalized_pressure_oscillation"];
            }
            if (iface.count("velocity_oscillation")) {
                const double scale = max_abs_reference_or_solution(run, solution, case_dir, "u0", 1.0e-12);
                iface_row["normalized_velocity_oscillation"] =
                    double_text(iface.at("velocity_oscillation") / scale);
                row["interface_normalized_velocity_oscillation"] =
                    iface_row["normalized_velocity_oscillation"];
            }
            if (iface.count("interface_thickness") && std::isfinite(dx_interface) && dx_interface > 0.0) {
                iface_row["interface_thickness_physical"] =
                    double_text(iface.at("interface_thickness"));
                iface_row["interface_thickness_cells"] =
                    double_text(iface.at("interface_thickness") / dx_interface);
                row["interface_interface_thickness_physical"] =
                    iface_row["interface_thickness_physical"];
                row["interface_interface_thickness_cells"] =
                    iface_row["interface_thickness_cells"];
            }
            interface_rows.push_back(iface_row);
        }

        if (!run.scaling.empty()) {
            scaling_rows.push_back(row);
        }
        if (!run.sensitivity.empty()) {
            sensitivity_rows.push_back(row);
        }
        if (uses_bubble_feature_diagnostics(run) && successes[i]) {
            const auto features = bubble_features(result_root / "raw" / run.run_id, run.case_def.method, result_root / "runs" / run.run_id / "bubble_features.json");
            if (!features.summary.empty()) {
                std::map<std::string, std::string> bubble_row = {
                    {"case", run.case_def.name},
                    {"case_label", run.case_def.label},
                    {"method", run.case_def.method},
                    {"resolution", resolution_label(run.resolution)},
                    {"run_id", run.run_id},
                    {"success", successes[i] ? "true" : "false"},
                };
                for (const auto& item : features.summary) {
                    bubble_row[item.first] = double_text(item.second);
                    row["bubble_" + item.first] = double_text(item.second);
                }
                bubble_rows.push_back(bubble_row);
            }
            for (const auto& item : features.timeseries_rows) {
                std::map<std::string, std::string> bubble_time_row = {
                    {"case", run.case_def.name},
                    {"case_label", run.case_def.label},
                    {"method", run.case_def.method},
                    {"resolution", resolution_label(run.resolution)},
                    {"run_id", run.run_id},
                    {"success", successes[i] ? "true" : "false"},
                };
                for (const auto& field : item) {
                    bubble_time_row[field.first] = field.second;
                }
                bubble_timeseries_rows.push_back(bubble_time_row);
            }
        }
        const auto errors = error_rows_for(run, solution, case_dir);
        error_rows.insert(error_rows.end(), errors.begin(), errors.end());
        summary.push_back(row);
    }
    const auto error_report = build_error_reports(error_rows);
    const auto interface_report = build_interface_reports(interface_rows);
    const auto conservation_report = build_conservation_reports(conservation_rows);
    const auto performance_report = build_performance_reports(performance, runs);
    const auto bubble_velocity_fit_rows =
        build_bubble_velocity_fit_rows(bubble_timeseries_rows, bubble_rows);
    const auto bubble_velocity_comparison_rows =
        build_bubble_velocity_comparison_rows(bubble_velocity_fit_rows);

    const fs::path summaries_dir = result_root / "summaries";
    const fs::path report_dir = result_root / "report";
    const fs::path figures_dir = result_root / "figures";

    fs::remove(report_dir / "conservation_balance_compact.csv");
    fs::remove(report_dir / "interface_oscillation_report_summary.csv");
    fs::remove(report_dir / "interface_thickness_report_summary.csv");

    write_csv(summaries_dir / "summary.csv", summary);
    write_csv(summaries_dir / "performance_summary.csv", performance);
    write_csv_ordered(
        report_dir / "performance_report_summary.csv",
        performance_report.performance_rows,
        {
            "case",
            "resolution",
            "SIM_cell_updates_per_second_median",
            "DIM_cell_updates_per_second_median",
            "SIM_over_DIM_speedup",
            "SIM_wall_time_seconds_median",
            "DIM_wall_time_seconds_median",
            "SIM_cost_per_cell_update_seconds_median",
            "DIM_cost_per_cell_update_seconds_median",
            "SIM_repeat_count",
            "DIM_repeat_count",
        }
    );
    write_csv_ordered(
        report_dir / "performance_environment.csv",
        performance_report.environment_rows,
        {
            "hostname",
            "cpu_model",
            "compiler",
            "compiler_flags",
            "build_type",
            "omp_threads",
            "benchmark_mode",
            "benchmark_repeats_non_warmup",
            "benchmark_warmups",
        }
    );
    write_csv(summaries_dir / "conservation_summary.csv", conservation_rows);
    write_csv_ordered(
        report_dir / "conservation_drift_report_summary.csv",
        conservation_report.drift_rows,
        {
            "case",
            "method",
            "drift_100",
            "drift_200",
            "drift_400",
        }
    );
    write_csv(summaries_dir / "conservation_balance_compact.csv", conservation_compact);
    write_csv(summaries_dir / "interface_summary.csv", interface_rows);
    write_csv_ordered(
        report_dir / "interface_location_report_summary.csv",
        interface_report.location_rows,
        {
            "case",
            "method",
            "x_interface_100",
            "x_interface_200",
            "x_interface_400",
            "shift_100_400",
            "sim_dim_difference_100",
            "sim_dim_difference_200",
            "sim_dim_difference_400",
        }
    );
    write_csv(summaries_dir / "sensitivity_summary.csv", sensitivity_rows);
    write_csv(summaries_dir / "scaling_summary.csv", scaling_rows);
    write_csv_ordered(
        summaries_dir / "bubble_feature_summary.csv",
        bubble_rows,
        {
            "case",
            "case_label",
            "method",
            "resolution",
            "run_id",
            "success",
            "snapshot_count",
            "latest_time_code",
            "latest_bubble_area_mm2",
            "latest_tracking_confidence",
            "latest_jet_detected",
            "upstream_interface_x_latest_position_mm",
            "downstream_interface_x_latest_position_mm",
            "jet_head_x_latest_position_mm",
            "transverse_interface_y_latest_position_mm",
            "transverse_triple_point_x_latest_position_mm",
            "transverse_triple_point_y_latest_position_mm",
            "transverse_wave_y_latest_position_mm",
        }
    );
    write_csv_ordered(
        summaries_dir / "bubble_feature_positions.csv",
        bubble_timeseries_rows,
        {
            "case",
            "case_label",
            "method",
            "resolution",
            "run_id",
            "feature",
            "time_code",
            "time_us_from_initial",
            "position_mm",
            "confidence_flag",
            "accepted",
            "jet_detected",
        }
    );
    write_csv_ordered(
        summaries_dir / "bubble_feature_timeseries.csv",
        bubble_timeseries_rows,
        {
            "case",
            "case_label",
            "method",
            "resolution",
            "run_id",
            "feature",
            "time_code",
            "time_us_from_initial",
            "position_mm",
            "confidence_flag",
            "accepted",
            "jet_detected",
        }
    );
    write_csv_ordered(
        summaries_dir / "bubble_feature_velocity_fits.csv",
        bubble_velocity_fit_rows,
        {
            "method",
            "feature",
            "start_time_us",
            "end_time_us",
            "start_position_mm",
            "end_position_mm",
            "displacement_mm",
            "duration_us",
            "fitted_velocity_m_per_s",
            "r_squared",
            "rmse_mm",
            "n_points",
            "confidence_flag",
        }
    );
    write_csv_ordered(
        summaries_dir / "bubble_feature_velocity_summary.csv",
        bubble_velocity_fit_rows,
        {
            "method",
            "feature",
            "start_time_us",
            "end_time_us",
            "start_position_mm",
            "end_position_mm",
            "displacement_mm",
            "duration_us",
            "fitted_velocity_m_per_s",
            "fit_velocity_m_per_s",
            "r_squared",
            "rmse_mm",
            "n_points",
            "confidence_flag",
        }
    );
    write_bubble_feature_plot(
        figures_dir / "bubble_feature_x_t.svg",
        bubble_timeseries_rows,
        "_x",
        "Helium Bubble Feature x-t Tracking"
    );
    write_bubble_feature_plot(
        figures_dir / "bubble_feature_y_t.svg",
        bubble_timeseries_rows,
        "_y",
        "Helium Bubble Feature y-t Tracking"
    );
    write_csv_ordered(
        report_dir / "bubble_feature_velocity_report_summary.csv",
        bubble_velocity_comparison_rows,
        {
            "feature",
            "tracked_feature",
            "rGFM_m_per_s",
            "Allaire5eq_m_per_s",
            "ref_comp_m_per_s",
            "experiment_m_per_s",
            "rGFM_error_percent",
            "Allaire5eq_error_percent",
            "rGFM_confidence",
            "Allaire5eq_confidence",
        }
    );
    write_csv(summaries_dir / "error_summary.csv", error_rows);
    write_csv(summaries_dir / "convergence_summary.csv", error_report.convergence_rows);
    write_csv(summaries_dir / "error_compact_summary.csv", error_report.compact_error_rows);
    write_csv_ordered(
        report_dir / "error_report_summary.csv",
        error_report.report_error_rows,
        {
            "case",
            "method",
            "variable",
            "norm",
            "normalized_error_100",
            "normalized_error_200",
            "normalized_error_400",
            "non_monotonic_100_400",
            "non_monotonic_any_refinement",
        }
    );
    write_csv_ordered(
        report_dir / "error_summary.csv",
        error_report.report_error_rows,
        {
            "case",
            "method",
            "variable",
            "norm",
            "normalized_error_100",
            "normalized_error_200",
            "normalized_error_400",
            "non_monotonic_100_400",
            "non_monotonic_any_refinement",
        }
    );
    write_csv_ordered(
        report_dir / "convergence_summary.csv",
        error_report.report_convergence_rows,
        {
            "case",
            "method",
            "variable",
            "norm",
            "N_coarse",
            "N_fine",
            "normalized_error_coarse",
            "normalized_error_fine",
            "observed_order",
        }
    );
    write_csv_ordered(
        report_dir / "convergence_report_summary.csv",
        error_report.compact_convergence_rows,
        {
            "case",
            "method",
            "variable",
            "norm",
            "order_100_200",
            "order_200_400",
            "order_100_400",
            "non_monotonic_100_400",
            "non_monotonic_any_refinement",
        }
    );
    write_csv_ordered(
        report_dir / "error_linf_appendix_summary.csv",
        error_report.linf_appendix_rows,
        {
            "case",
            "method",
            "variable",
            "norm",
            "normalized_error_100",
            "normalized_error_200",
            "normalized_error_400",
            "order_100_200",
            "order_200_400",
            "order_100_400",
            "non_monotonic_100_400",
            "non_monotonic_any_refinement",
        }
    );
    write_csv_ordered(
        report_dir / "convergence_order_report_summary.csv",
        error_report.convergence_order_report_rows,
        {
            "case",
            "SIM_mean_L1_order_100_400",
            "DIM_mean_L1_order_100_400",
            "SIM_variable_count",
            "DIM_variable_count",
        }
    );
    write_csv_ordered(
        report_dir / "main_results_table.csv",
        error_report.convergence_order_report_rows,
        {
            "case",
            "SIM_mean_L1_order_100_400",
            "DIM_mean_L1_order_100_400",
            "SIM_variable_count",
            "DIM_variable_count",
        }
    );
}


} // namespace quant
