#pragma once

// Header-only implementation units for the quantitative validation runner.
// Builds median timing summaries and hardware/compiler metadata rows.

#include "metrics.hpp"

namespace quant {

struct PerformanceReportTables {
    std::vector<std::map<std::string, std::string>> performance_rows;
    std::vector<std::map<std::string, std::string>> environment_rows;
};

inline PerformanceReportTables build_performance_reports(
    const std::vector<std::map<std::string, std::string>>& performance,
    const std::vector<RunSpec>& runs
)
{
    struct PerformanceStats {
    std::string case_label;
    std::string resolution;
    std::map<std::string, std::vector<double>> throughput_by_method;
    std::map<std::string, std::vector<double>> wall_by_method;
    std::map<std::string, std::vector<double>> cost_per_cell_by_method;
    std::map<std::string, int> repeat_count_by_method;
    };

    std::map<std::string, PerformanceStats> performance_groups;
    for (const auto& row : performance) {
    if (row.count("success") && row.at("success") != "true") continue;
    if (row.count("warmup") && row.at("warmup") == "true") continue;
    if (!row.count("case_label") || !row.count("method") || !row.count("resolution")) continue;

    const std::string key = row.at("case_label") + "|" + row.at("resolution");
    auto& stats = performance_groups[key];
    stats.case_label = row.at("case_label");
    stats.resolution = row.at("resolution");
    const std::string method = row.at("method");

    if (const auto value = parse_double_field(row, "cell_updates_per_second")) {
        stats.throughput_by_method[method].push_back(value.value());
    }
    if (const auto value = parse_double_field(row, "wall_time_seconds")) {
        stats.wall_by_method[method].push_back(value.value());
    }
    if (const auto value = parse_double_field(row, "cost_per_cell_update_seconds")) {
        stats.cost_per_cell_by_method[method].push_back(value.value());
    }
    stats.repeat_count_by_method[method] += 1;
    }

    std::vector<std::map<std::string, std::string>> performance_report_rows;
    for (const auto& item : performance_groups) {
    const auto& stats = item.second;
    const std::string sim_throughput = median_text(
        stats.throughput_by_method.count("SIM") ? stats.throughput_by_method.at("SIM") : std::vector<double>{}
    );
    const std::string dim_throughput = median_text(
        stats.throughput_by_method.count("DIM") ? stats.throughput_by_method.at("DIM") : std::vector<double>{}
    );
    std::string speedup;
    if (!sim_throughput.empty() && !dim_throughput.empty() && std::stod(dim_throughput) > 0.0) {
        speedup = double_text(std::stod(sim_throughput) / std::stod(dim_throughput));
    }

    performance_report_rows.push_back({
        {"case", stats.case_label},
        {"resolution", stats.resolution},
        {"SIM_cell_updates_per_second_median", sim_throughput},
        {"DIM_cell_updates_per_second_median", dim_throughput},
        {"SIM_over_DIM_speedup", speedup},
        {"SIM_wall_time_seconds_median", median_text(
            stats.wall_by_method.count("SIM") ? stats.wall_by_method.at("SIM") : std::vector<double>{}
        )},
        {"DIM_wall_time_seconds_median", median_text(
            stats.wall_by_method.count("DIM") ? stats.wall_by_method.at("DIM") : std::vector<double>{}
        )},
        {"SIM_cost_per_cell_update_seconds_median", median_text(
            stats.cost_per_cell_by_method.count("SIM") ? stats.cost_per_cell_by_method.at("SIM") : std::vector<double>{}
        )},
        {"DIM_cost_per_cell_update_seconds_median", median_text(
            stats.cost_per_cell_by_method.count("DIM") ? stats.cost_per_cell_by_method.at("DIM") : std::vector<double>{}
        )},
        {"SIM_repeat_count", std::to_string(
            stats.repeat_count_by_method.count("SIM") ? stats.repeat_count_by_method.at("SIM") : 0
        )},
        {"DIM_repeat_count", std::to_string(
            stats.repeat_count_by_method.count("DIM") ? stats.repeat_count_by_method.at("DIM") : 0
        )},
    });
    }

    int planned_warmups = 0;
    int planned_non_warmups = 0;
    if (!runs.empty()) {
    const std::string first_case_key =
        runs.front().case_def.label + "|" + runs.front().case_def.method + "|" +
        resolution_label(runs.front().resolution);
    for (const auto& run : runs) {
        const std::string key =
            run.case_def.label + "|" + run.case_def.method + "|" +
            resolution_label(run.resolution);
        if (key != first_case_key) continue;
        if (run.warmup) ++planned_warmups;
        else ++planned_non_warmups;
    }
    }

    std::vector<std::map<std::string, std::string>> performance_environment_rows = {{
    {"hostname", read_hostname()},
    {"cpu_model", read_first_cpu_model()},
    {"compiler", std::string("g++ ") + __VERSION__},
    {"compiler_flags", "-std=c++17 -O2 -fopenmp -I."},
    {"build_type", "Release"},
    {"omp_threads", std::to_string(runs.empty() ? 0 : runs.front().omp_threads)},
    {"benchmark_mode", runs.empty() ? std::string{} : runs.front().benchmark_mode},
    {"benchmark_repeats_non_warmup", std::to_string(planned_non_warmups)},
    {"benchmark_warmups", std::to_string(planned_warmups)},
    }};


    return {performance_report_rows, performance_environment_rows};
}

} // namespace quant
