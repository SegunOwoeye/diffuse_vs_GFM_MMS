#pragma once

// Header-only implementation units for the quantitative validation runner.
// Orchestrates raw artifact collection, then delegates result-specific table building.
#include "report_conservation.hpp"
#include "report_errors.hpp"
#include "report_interfaces.hpp"
#include "report_performance.hpp"
#include "report_bubble.hpp"
#include "gorsse_tc9_metrics.hpp"

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

inline bool uses_gorsse_tc9_diagnostics(const RunSpec& run)
{
    return run.case_def.group == "gorsse_tc9_water_air_bubble_2d" ||
           run.case_def.group == "gorsse_tc9_water_air_bubble_2d_lowres";
}

inline fs::path run_case_dir(const fs::path& result_root, const RunSpec& run)
{
    return result_root / "raw" / run.run_id / run.output_prefix;
}

inline fs::path run_solution_path(const fs::path& result_root, const RunSpec& run)
{
    const fs::path case_dir = run_case_dir(result_root, run);
    const fs::path direct = case_dir / (run.output_prefix + resolution_suffix(run.resolution) + ".csv");
    if (fs::exists(direct)) {
        return direct;
    }

    std::vector<fs::path> candidates;
    const std::string suffix = resolution_suffix(run.resolution) + ".csv";
    if (fs::exists(case_dir)) {
        for (const auto& entry : fs::directory_iterator(case_dir)) {
            if (!entry.is_regular_file()) {
                continue;
            }
            const fs::path path = entry.path();
            const std::string name = path.filename().string();
            if (path.extension() != ".csv" ||
                name.find(run.output_prefix + "_t") != 0 ||
                name.size() < suffix.size() ||
                name.substr(name.size() - suffix.size()) != suffix ||
                name.find("conservation") != std::string::npos ||
                name.find("diagnostic") != std::string::npos ||
                name.find("exact") != std::string::npos) {
                continue;
            }
            candidates.push_back(path);
        }
    }

    if (!candidates.empty()) {
        std::sort(candidates.begin(), candidates.end());
        return candidates.back();
    }

    return direct;
}

inline std::vector<std::map<std::string, std::string>> he2023_self_reference_error_rows(
    const fs::path& result_root,
    const std::vector<RunSpec>& runs,
    const std::vector<bool>& successes
)
{
    struct Candidate {
        std::size_t index = 0;
        int n = 0;
        fs::path solution;
    };

    std::map<std::string, std::vector<Candidate>> groups;
    for (std::size_t i = 0; i < runs.size() && i < successes.size(); ++i) {
        const auto& run = runs[i];
        if (!successes[i] ||
            run.case_def.group != "he2023_three_material_1d" ||
            run.case_def.dimension != 1) {
            continue;
        }

        const fs::path solution = run_solution_path(result_root, run);
        if (!fs::exists(solution)) {
            continue;
        }

        const std::string key = run.case_def.label + "|" + run.case_def.method;
        groups[key].push_back({
            i,
            resolution_n(resolution_label(run.resolution)),
            solution
        });
    }

    std::vector<std::map<std::string, std::string>> rows;
    const std::map<std::string, std::string> variables = {
        {"rho", "density"},
        {"u0", "velocity"},
        {"p", "pressure"},
        {"e", "energy"},
    };

    for (auto& item : groups) {
        auto& candidates = item.second;
        if (candidates.size() < 2) {
            continue;
        }
        std::sort(
            candidates.begin(),
            candidates.end(),
            [](const Candidate& lhs, const Candidate& rhs) {
                return lhs.n < rhs.n;
            }
        );

        const Candidate& reference_candidate = candidates.back();
        const auto ref_cols = read_csv_columns(reference_candidate.solution);
        if (!ref_cols.count("x0")) {
            continue;
        }

        for (const auto& candidate : candidates) {
            if (candidate.index == reference_candidate.index) {
                continue;
            }

            const auto& run = runs[candidate.index];
            const auto solution_cols = read_csv_columns(candidate.solution);
            if (!solution_cols.count("x0")) {
                continue;
            }

            const double dx = cell_width(solution_cols.at("x0"));
            for (const auto& variable : variables) {
                const std::string& column = variable.first;
                if (!solution_cols.count(column) || !ref_cols.count(column)) {
                    continue;
                }

                double l1 = 0.0;
                double l2sum = 0.0;
                double linf = 0.0;
                double ref_l1 = 0.0;
                double ref_l2sum = 0.0;
                double ref_linf = 0.0;
                const auto& xs = solution_cols.at("x0");
                const auto& values = solution_cols.at(column);

                for (std::size_t row_index = 0; row_index < xs.size() && row_index < values.size(); ++row_index) {
                    const double ref_value = interp(ref_cols.at("x0"), ref_cols.at(column), xs[row_index]);
                    if (!std::isfinite(values[row_index]) || !std::isfinite(ref_value)) {
                        continue;
                    }
                    const double diff = values[row_index] - ref_value;
                    l1 += std::abs(diff);
                    l2sum += diff * diff;
                    linf = std::max(linf, std::abs(diff));
                    ref_l1 += std::abs(ref_value);
                    ref_l2sum += ref_value * ref_value;
                    ref_linf = std::max(ref_linf, std::abs(ref_value));
                }

                for (const auto& norm_error : {
                    std::pair<std::string, std::pair<double, double>>{"L1", {dx * l1, dx * ref_l1}},
                    std::pair<std::string, std::pair<double, double>>{"L2", {std::sqrt(dx * l2sum), std::sqrt(dx * ref_l2sum)}},
                    std::pair<std::string, std::pair<double, double>>{"Linf", {linf, ref_linf}},
                }) {
                    const double raw_error = norm_error.second.first;
                    const double ref_norm = norm_error.second.second;
                    const double denominator = std::max(ref_norm, reference_floor(column, norm_error.first));
                    auto row = base_row(run, true);
                    row["variable"] = variable.second;
                    row["norm"] = norm_error.first;
                    row["error"] = double_text(raw_error);
                    row["raw_error"] = double_text(raw_error);
                    row["reference_norm"] = double_text(ref_norm);
                    row["normalized_error"] = double_text(raw_error / denominator);
                    row["reference"] = "finest_self:" + reference_candidate.solution.string();
                    rows.push_back(row);
                }
            }
        }
    }

    return rows;
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
    std::vector<std::map<std::string, std::string>> gorsse_tc9_rows;
    std::vector<std::map<std::string, std::string>> gorsse_tc9_reference_rows;
    std::vector<std::map<std::string, std::string>> gorsse_tc9_feature_rows;
    std::vector<std::map<std::string, std::string>> gorsse_tc9_feature_reference_rows;
    std::vector<std::map<std::string, std::string>> error_rows;

    for (std::size_t i = 0; i < runs.size(); ++i) {
        const auto& run = runs[i];
        auto row = base_row(run, successes[i]);
        const fs::path case_dir = run_case_dir(result_root, run);
        const fs::path solution = run_solution_path(result_root, run);
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
        perf_row["output_enabled"] = run.timing_only ? "false" : "true";
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

        const auto cons = run.timing_only
            ? std::map<std::string, double>{}
            : final_conservation(conservation);
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

        const auto iface = run.timing_only
            ? std::map<std::string, double>{}
            : interface_metrics(solution, run.case_def.method);
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
        if (!run.timing_only && uses_bubble_feature_diagnostics(run) && successes[i]) {
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
        if (!run.timing_only && uses_gorsse_tc9_diagnostics(run) && successes[i]) {
            auto tc9_rows = tc9_model_summary_rows(
                result_root / "raw" / run.run_id,
                run.case_def.method
            );
            for (auto& tc9_row : tc9_rows) {
                tc9_row["case"] = run.case_def.name;
                tc9_row["case_label"] = run.case_def.label;
                tc9_row["method"] = run.case_def.method;
                tc9_row["resolution"] = resolution_label(run.resolution);
                tc9_row["run_id"] = run.run_id;
                tc9_row["success"] = successes[i] ? "true" : "false";
            }
            const fs::path reference_summary =
                fs::path("data") / "bubble_collapse_validation" /
                "gorsse_2014_tc9" / "gorsse_tc9_reference_interface_summary.csv";
            auto comparison_rows = tc9_reference_comparison_rows(
                tc9_rows,
                reference_summary
            );
            for (auto& comparison_row : comparison_rows) {
                comparison_row["case"] = run.case_def.name;
                comparison_row["case_label"] = run.case_def.label;
                comparison_row["method"] = run.case_def.method;
                comparison_row["resolution"] = resolution_label(run.resolution);
                comparison_row["run_id"] = run.run_id;
                comparison_row["success"] = successes[i] ? "true" : "false";
            }
            gorsse_tc9_rows.insert(
                gorsse_tc9_rows.end(),
                tc9_rows.begin(),
                tc9_rows.end()
            );
            gorsse_tc9_reference_rows.insert(
                gorsse_tc9_reference_rows.end(),
                comparison_rows.begin(),
                comparison_rows.end()
            );
            const auto feature_rows = tc9_feature_timeseries_rows(tc9_rows);
            const auto feature_reference_rows =
                tc9_feature_reference_comparison_rows(comparison_rows);
            gorsse_tc9_feature_rows.insert(
                gorsse_tc9_feature_rows.end(),
                feature_rows.begin(),
                feature_rows.end()
            );
            gorsse_tc9_feature_reference_rows.insert(
                gorsse_tc9_feature_reference_rows.end(),
                feature_reference_rows.begin(),
                feature_reference_rows.end()
            );
        }
        if (!run.timing_only) {
            const auto errors = error_rows_for(run, solution, case_dir);
            error_rows.insert(error_rows.end(), errors.begin(), errors.end());
        }
        summary.push_back(row);
    }
    const auto self_reference_errors =
        he2023_self_reference_error_rows(result_root, runs, successes);
    error_rows.insert(
        error_rows.end(),
        self_reference_errors.begin(),
        self_reference_errors.end()
    );
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
            "common_cell_updates_per_second_median",
            "SM_MPI_cell_updates_per_second_median",
            "SIM_cell_updates_per_second_median",
            "SIM_MPI_cell_updates_per_second_median",
            "DIM_cell_updates_per_second_median",
            "DIM_MPI_cell_updates_per_second_median",
            "SM_MPI_over_common_speedup",
            "SIM_MPI_over_SIM_speedup",
            "DIM_MPI_over_DIM_speedup",
            "SIM_over_DIM_speedup",
            "common_wall_time_seconds_median",
            "SM_MPI_wall_time_seconds_median",
            "SIM_wall_time_seconds_median",
            "SIM_MPI_wall_time_seconds_median",
            "DIM_wall_time_seconds_median",
            "DIM_MPI_wall_time_seconds_median",
            "SIM_cost_per_cell_update_seconds_median",
            "DIM_cost_per_cell_update_seconds_median",
            "common_repeat_count",
            "SM_MPI_repeat_count",
            "SIM_repeat_count",
            "SIM_MPI_repeat_count",
            "DIM_repeat_count",
            "DIM_MPI_repeat_count",
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
            "mpi_ranks",
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
            "finest_resolution",
            "finest_reportable_drift",
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
        summaries_dir / "gorsse_tc9_interface_summary.csv",
        gorsse_tc9_rows,
        {
            "case",
            "case_label",
            "method",
            "resolution",
            "run_id",
            "success",
            "time_s",
            "time_us",
            "interface_label",
            "extraction_mode",
            "contour_point_count",
            "x0_min_m",
            "x0_max_m",
            "x1_min_m",
            "x1_max_m",
            "centroid_x0_m",
            "centroid_x1_m",
            "csv_file",
        }
    );
    write_csv_ordered(
        summaries_dir / "gorsse_tc9_reference_comparison.csv",
        gorsse_tc9_reference_rows,
        {
            "case",
            "case_label",
            "method",
            "resolution",
            "run_id",
            "success",
            "time_us",
            "reference_time_us",
            "time_delta_us",
            "reference_x_offset_m",
            "interface_label",
            "extraction_mode",
            "x0_min_m",
            "reference_x0_min_m",
            "x0_min_m_error_m",
            "x0_max_m",
            "reference_x0_max_m",
            "x0_max_m_error_m",
            "x1_min_m",
            "reference_x1_min_m",
            "x1_min_m_error_m",
            "x1_max_m",
            "reference_x1_max_m",
            "x1_max_m_error_m",
            "centroid_x0_m",
            "reference_centroid_x0_m",
            "centroid_x0_m_error_m",
            "centroid_x1_m",
            "reference_centroid_x1_m",
            "centroid_x1_m_error_m",
            "csv_file",
        }
    );
    write_csv_ordered(
        report_dir / "gorsse_tc9_reference_comparison_summary.csv",
        gorsse_tc9_reference_rows,
        {
            "case_label",
            "method",
            "resolution",
            "time_us",
            "reference_time_us",
            "reference_x_offset_m",
            "extraction_mode",
            "x0_min_m_error_m",
            "x0_max_m_error_m",
            "x1_min_m_error_m",
            "x1_max_m_error_m",
            "centroid_x0_m_error_m",
            "centroid_x1_m_error_m",
        }
    );
    write_csv_ordered(
        summaries_dir / "gorsse_tc9_feature_timeseries.csv",
        gorsse_tc9_feature_rows,
        {
            "case",
            "case_label",
            "method",
            "resolution",
            "run_id",
            "success",
            "time_us",
            "feature",
            "feature_label",
            "value",
            "unit",
            "previous_time_us",
            "finite_difference_rate",
            "rate_unit",
            "interface_label",
            "extraction_mode",
            "csv_file",
        }
    );
    write_csv_ordered(
        summaries_dir / "gorsse_tc9_feature_reference_comparison.csv",
        gorsse_tc9_feature_reference_rows,
        {
            "case",
            "case_label",
            "method",
            "resolution",
            "run_id",
            "success",
            "time_us",
            "reference_time_us",
            "time_delta_us",
            "feature",
            "feature_label",
            "value",
            "reference_value",
            "error",
            "relative_error",
            "unit",
            "reference_x_offset_m",
            "interface_label",
            "extraction_mode",
            "csv_file",
        }
    );
    write_csv_ordered(
        report_dir / "gorsse_tc9_feature_reference_comparison_summary.csv",
        gorsse_tc9_feature_reference_rows,
        {
            "case_label",
            "method",
            "resolution",
            "time_us",
            "reference_time_us",
            "feature",
            "feature_label",
            "value",
            "reference_value",
            "error",
            "relative_error",
            "unit",
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
            "normalized_error_800",
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
            "normalized_error_800",
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
            "order_400_800",
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
            "normalized_error_800",
            "order_100_200",
            "order_200_400",
            "order_400_800",
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
