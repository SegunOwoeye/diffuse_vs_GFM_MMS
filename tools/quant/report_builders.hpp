#pragma once

// Header-only implementation units for the quantitative validation runner.
// Orchestrates raw artifact collection, then delegates result-specific table building.
#include "report_conservation.hpp"
#include "report_errors.hpp"
#include "report_interfaces.hpp"
#include "report_performance.hpp"

namespace quant {

struct BubbleFitSample {
    double time_us = std::nan("");
    double position_mm = std::nan("");
};

struct BubbleVelocityFit {
    std::size_t start = 0;
    std::size_t end = 0;
    double velocity_m_per_s = std::nan("");
    double r_squared = -1.0;
    double rmse_mm = std::nan("");
    bool high_confidence = false;
    bool valid = false;
};

BubbleVelocityFit fit_bubble_window(
    const std::vector<BubbleFitSample>& samples,
    std::size_t start,
    std::size_t end
)
{
    BubbleVelocityFit fit;
    fit.start = start;
    fit.end = end;
    const std::size_t n = end - start + 1;
    if (n < 2) return fit;

    double sum_t = 0.0;
    double sum_x = 0.0;
    double sum_tt = 0.0;
    double sum_tx = 0.0;
    for (std::size_t i = start; i <= end; ++i) {
        const double t = samples[i].time_us * 1.0e-6;
        const double x = samples[i].position_mm * 1.0e-3;
        sum_t += t;
        sum_x += x;
        sum_tt += t * t;
        sum_tx += t * x;
    }

    const double denom = static_cast<double>(n) * sum_tt - sum_t * sum_t;
    if (denom == 0.0) return fit;
    const double velocity = (static_cast<double>(n) * sum_tx - sum_t * sum_x) / denom;
    const double intercept = (sum_x - velocity * sum_t) / static_cast<double>(n);
    const double mean_x = sum_x / static_cast<double>(n);

    double ss_res_m2 = 0.0;
    double ss_tot_m2 = 0.0;
    for (std::size_t i = start; i <= end; ++i) {
        const double t = samples[i].time_us * 1.0e-6;
        const double x = samples[i].position_mm * 1.0e-3;
        const double residual = x - (intercept + velocity * t);
        ss_res_m2 += residual * residual;
        const double centred = x - mean_x;
        ss_tot_m2 += centred * centred;
    }

    fit.velocity_m_per_s = velocity;
    fit.r_squared = (ss_tot_m2 == 0.0) ? 1.0 : 1.0 - ss_res_m2 / ss_tot_m2;
    fit.rmse_mm = std::sqrt(ss_res_m2 / static_cast<double>(n)) * 1.0e3;
    fit.valid = true;
    return fit;
}

std::vector<std::map<std::string, std::string>> build_bubble_velocity_fit_rows(
    const std::vector<std::map<std::string, std::string>>& timeseries_rows,
    const std::vector<std::map<std::string, std::string>>& bubble_rows
)
{
    std::map<std::string, std::vector<BubbleFitSample>> groups;
    std::map<std::string, std::map<std::string, std::string>> metadata;
    for (const auto& row : timeseries_rows) {
        const auto time = parse_double_field(row, "time_us_from_initial");
        const auto position = parse_double_field(row, "position_mm");
        if (!time || !position || !row.count("method") || !row.count("feature")) continue;
        const std::string key = row.at("method") + "|" + row.at("feature");
        groups[key].push_back({time.value(), position.value()});
        metadata[key] = row;
    }

    std::vector<std::map<std::string, std::string>> out;
    std::map<std::string, bool> emitted;
    for (auto& item : groups) {
        auto& samples = item.second;
        std::sort(samples.begin(), samples.end(), [](const auto& a, const auto& b) {
            return a.time_us < b.time_us;
        });

        const auto& meta = metadata.at(item.first);
        const std::string feature = meta.at("feature");
        if (samples.size() < 6) {
            out.push_back({
                {"method", meta.at("method")},
                {"feature", feature},
                {"n_points", std::to_string(samples.size())},
                {"confidence_flag", "insufficient_samples"},
            });
            emitted[item.first] = true;
            continue;
        }
        const bool transverse = feature.find("_y") != std::string::npos;
        const double min_displacement_mm = transverse ? 1.0 : 5.0;
        const double r2_threshold = transverse ? 0.95 : 0.995;

        BubbleVelocityFit best_passing;
        BubbleVelocityFit best_any;
        BubbleVelocityFit best_any_without_displacement_filter;
        for (std::size_t start = 0; start < samples.size(); ++start) {
            for (std::size_t end = start + 5; end < samples.size(); ++end) {
                const double displacement_mm = samples[end].position_mm - samples[start].position_mm;
                BubbleVelocityFit unfiltered_fit = fit_bubble_window(samples, start, end);
                if (unfiltered_fit.valid) {
                    const std::size_t n = end - start + 1;
                    if (!best_any_without_displacement_filter.valid ||
                        unfiltered_fit.r_squared > best_any_without_displacement_filter.r_squared ||
                        (unfiltered_fit.r_squared == best_any_without_displacement_filter.r_squared &&
                         n > best_any_without_displacement_filter.end -
                             best_any_without_displacement_filter.start + 1)) {
                        best_any_without_displacement_filter = unfiltered_fit;
                    }
                }
                if (std::abs(displacement_mm) < min_displacement_mm) continue;
                BubbleVelocityFit fit = unfiltered_fit;
                if (!fit.valid) continue;
                const std::size_t n = end - start + 1;
                if (!best_any.valid ||
                    fit.r_squared > best_any.r_squared ||
                    (fit.r_squared == best_any.r_squared && n > best_any.end - best_any.start + 1)) {
                    best_any = fit;
                }
                if (fit.r_squared >= r2_threshold) {
                    if (!best_passing.valid ||
                        n > best_passing.end - best_passing.start + 1 ||
                        (n == best_passing.end - best_passing.start + 1 &&
                         fit.r_squared > best_passing.r_squared)) {
                        best_passing = fit;
                    }
                }
            }
        }

        BubbleVelocityFit selected = best_passing.valid ? best_passing : best_any;
        if (!selected.valid) selected = best_any_without_displacement_filter;
        if (!selected.valid) continue;
        selected.high_confidence = best_passing.valid;
        const double displacement_mm =
            samples[selected.end].position_mm - samples[selected.start].position_mm;
        const double duration_us =
            samples[selected.end].time_us - samples[selected.start].time_us;

        out.push_back({
            {"method", meta.at("method")},
            {"feature", feature},
            {"start_time_us", double_text(samples[selected.start].time_us)},
            {"end_time_us", double_text(samples[selected.end].time_us)},
            {"start_position_mm", double_text(samples[selected.start].position_mm)},
            {"end_position_mm", double_text(samples[selected.end].position_mm)},
            {"displacement_mm", double_text(displacement_mm)},
            {"duration_us", double_text(duration_us)},
            {"fitted_velocity_m_per_s", double_text(std::abs(selected.velocity_m_per_s))},
            {"fit_velocity_m_per_s", double_text(std::abs(selected.velocity_m_per_s))},
            {"r_squared", double_text(selected.r_squared)},
            {"rmse_mm", double_text(selected.rmse_mm)},
            {"n_points", std::to_string(selected.end - selected.start + 1)},
            {"confidence_flag", selected.high_confidence ? "high" : "low"},
        });
        emitted[item.first] = true;
    }

    const std::vector<std::string> expected_features = {
        "upstream_interface_x",
        "downstream_interface_x",
        "jet_head_x",
        "transverse_interface_y",
        "transverse_triple_point_x",
        "transverse_triple_point_y",
        "transverse_wave_y",
    };
    for (const auto& row : bubble_rows) {
        if (!row.count("method")) continue;
        for (const auto& feature : expected_features) {
            const std::string key = row.at("method") + "|" + feature;
            if (emitted[key]) continue;
            out.push_back({
                {"method", row.at("method")},
                {"feature", feature},
                {"n_points", "0"},
                {"confidence_flag", "not_detected"},
            });
            emitted[key] = true;
        }
    }
    return out;
}

std::vector<std::map<std::string, std::string>> build_bubble_velocity_comparison_rows(
    const std::vector<std::map<std::string, std::string>>& fit_rows
)
{
    struct Reference {
        std::string label;
        std::string feature;
        double ref_comp = std::nan("");
        double experiment = std::nan("");
    };
    const std::vector<Reference> references = {
        {"V_ui", "upstream_interface_x", 178.0, 170.0},
        {"V_j", "jet_head_x", 227.0, 230.0},
        {"V_di", "downstream_interface_x", 146.0, 145.0},
        {"V_T", "transverse_triple_point_x", 377.0, 393.0},
    };

    std::map<std::string, std::map<std::string, std::string>> by_key;
    for (const auto& row : fit_rows) {
        if (!row.count("method") || !row.count("feature")) continue;
        by_key[row.at("method") + "|" + row.at("feature")] = row;
    }

    auto value = [&](const std::string& method, const std::string& feature) -> std::optional<double> {
        const std::string key = method + "|" + feature;
        if (!by_key.count(key)) return std::nullopt;
        auto fitted = parse_double_field(by_key.at(key), "fitted_velocity_m_per_s");
        if (fitted.has_value()) return fitted;
        return parse_double_field(by_key.at(key), "fit_velocity_m_per_s");
    };
    auto confidence = [&](const std::string& method, const std::string& feature) -> std::string {
        const std::string key = method + "|" + feature;
        if (!by_key.count(key) || !by_key.at(key).count("confidence_flag")) return {};
        return by_key.at(key).at("confidence_flag");
    };
    auto error_percent = [](std::optional<double> method_value, double experiment) -> std::string {
        if (!method_value.has_value() || experiment == 0.0) return {};
        return double_text(100.0 * (method_value.value() - experiment) / experiment);
    };

    std::vector<std::map<std::string, std::string>> out;
    for (const auto& ref : references) {
        const auto sim = value("SIM", ref.feature);
        const auto dim = value("DIM", ref.feature);
        out.push_back({
            {"feature", ref.label},
            {"tracked_feature", ref.feature},
            {"rGFM_m_per_s", sim.has_value() ? double_text(sim.value()) : std::string{}},
            {"Allaire5eq_m_per_s", dim.has_value() ? double_text(dim.value()) : std::string{}},
            {"ref_comp_m_per_s", double_text(ref.ref_comp)},
            {"experiment_m_per_s", double_text(ref.experiment)},
            {"rGFM_error_percent", error_percent(sim, ref.experiment)},
            {"Allaire5eq_error_percent", error_percent(dim, ref.experiment)},
            {"rGFM_confidence", confidence("SIM", ref.feature)},
            {"Allaire5eq_confidence", confidence("DIM", ref.feature)},
        });
    }
    return out;
}

void write_bubble_feature_plot(
    const fs::path& path,
    const std::vector<std::map<std::string, std::string>>& rows,
    const std::string& axis_suffix,
    const std::string& title
)
{
    struct Point {
        double time_us = std::nan("");
        double position_mm = std::nan("");
        std::string label;
        std::string confidence;
    };
    std::vector<Point> points;
    double min_time = std::numeric_limits<double>::infinity();
    double max_time = -std::numeric_limits<double>::infinity();
    double min_position = std::numeric_limits<double>::infinity();
    double max_position = -std::numeric_limits<double>::infinity();
    for (const auto& row : rows) {
        if (!row.count("feature")) continue;
        const std::string feature = row.at("feature");
        if (feature.size() < axis_suffix.size() ||
            feature.substr(feature.size() - axis_suffix.size()) != axis_suffix) {
            continue;
        }
        const auto time = parse_double_field(row, "time_us_from_initial");
        const auto position = parse_double_field(row, "position_mm");
        if (!time || !position) continue;
        const std::string method = row.count("method") ? row.at("method") : "";
        const std::string label = method + " " + feature;
        const std::string confidence = row.count("confidence_flag") ? row.at("confidence_flag") : "";
        points.push_back({time.value(), position.value(), label, confidence});
        min_time = std::min(min_time, time.value());
        max_time = std::max(max_time, time.value());
        min_position = std::min(min_position, position.value());
        max_position = std::max(max_position, position.value());
    }
    if (points.empty()) return;
    if (min_time == max_time) max_time = min_time + 1.0;
    if (min_position == max_position) max_position = min_position + 1.0;

    const double width = 920.0;
    const double height = 620.0;
    const double left = 80.0;
    const double right = 260.0;
    const double top = 45.0;
    const double bottom = 70.0;
    const double plot_w = width - left - right;
    const double plot_h = height - top - bottom;
    auto sx = [&](double value) {
        return left + (value - min_position) / (max_position - min_position) * plot_w;
    };
    auto sy = [&](double value) {
        return top + (value - min_time) / (max_time - min_time) * plot_h;
    };
    const std::vector<std::string> palette = {
        "#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#17becf",
        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22"
    };
    std::map<std::string, std::string> color_by_label;
    for (const auto& point : points) {
        if (!color_by_label.count(point.label)) {
            color_by_label[point.label] =
                palette[color_by_label.size() % palette.size()];
        }
    }

    std::ostringstream svg;
    svg << "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"" << width
        << "\" height=\"" << height << "\" viewBox=\"0 0 " << width << " " << height << "\">\n";
    svg << "<rect width=\"100%\" height=\"100%\" fill=\"white\"/>\n";
    svg << "<text x=\"" << left << "\" y=\"28\" font-family=\"Arial\" font-size=\"18\">"
        << title << "</text>\n";
    svg << "<rect x=\"" << left << "\" y=\"" << top << "\" width=\"" << plot_w
        << "\" height=\"" << plot_h << "\" fill=\"#fafafa\" stroke=\"#333\"/>\n";
    for (int i = 0; i <= 5; ++i) {
        const double tx = left + plot_w * i / 5.0;
        const double ty = top + plot_h * i / 5.0;
        svg << "<line x1=\"" << tx << "\" y1=\"" << top << "\" x2=\"" << tx
            << "\" y2=\"" << top + plot_h << "\" stroke=\"#e5e5e5\"/>\n";
        svg << "<line x1=\"" << left << "\" y1=\"" << ty << "\" x2=\"" << left + plot_w
            << "\" y2=\"" << ty << "\" stroke=\"#e5e5e5\"/>\n";
    }
    svg << "<text x=\"" << left + plot_w / 2.0 << "\" y=\"" << height - 22
        << "\" font-family=\"Arial\" font-size=\"14\" text-anchor=\"middle\">position [mm]</text>\n";
    svg << "<text transform=\"translate(22 " << top + plot_h / 2.0
        << ") rotate(-90)\" font-family=\"Arial\" font-size=\"14\" text-anchor=\"middle\">time [us]</text>\n";

    for (const auto& item : color_by_label) {
        std::vector<Point> series;
        for (const auto& point : points) {
            if (point.label == item.first) series.push_back(point);
        }
        std::sort(series.begin(), series.end(), [](const Point& a, const Point& b) {
            return a.time_us < b.time_us;
        });
        svg << "<polyline fill=\"none\" stroke=\"" << item.second
            << "\" stroke-width=\"1.5\" points=\"";
        for (const auto& point : series) {
            svg << sx(point.position_mm) << "," << sy(point.time_us) << " ";
        }
        svg << "\"/>\n";
        for (const auto& point : series) {
            const double radius = point.confidence == "high" ? 3.0 : 4.5;
            const std::string fill = point.confidence == "high" ? item.second : "white";
            svg << "<circle cx=\"" << sx(point.position_mm) << "\" cy=\"" << sy(point.time_us)
                << "\" r=\"" << radius << "\" fill=\"" << fill << "\" stroke=\""
                << item.second << "\" stroke-width=\"1.5\"/>\n";
        }
    }
    double legend_y = top + 18.0;
    for (const auto& item : color_by_label) {
        svg << "<line x1=\"" << width - right + 35 << "\" y1=\"" << legend_y
            << "\" x2=\"" << width - right + 55 << "\" y2=\"" << legend_y
            << "\" stroke=\"" << item.second << "\" stroke-width=\"2\"/>\n";
        svg << "<text x=\"" << width - right + 62 << "\" y=\"" << legend_y + 5
            << "\" font-family=\"Arial\" font-size=\"12\">" << item.first << "</text>\n";
        legend_y += 18.0;
    }
    svg << "</svg>\n";
    write_file(path, svg.str());
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
        if (run.case_def.group == "shock_bubble_2d" && successes[i]) {
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
