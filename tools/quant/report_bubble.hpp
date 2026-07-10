#pragma once

/*
    Header-only implementation units for the quantitative validation runner.
    Builds shock-bubble report velocity tables and lightweight feature plots.

    The feature extractor emits positions at each saved output time. This layer
    converts those trajectories into report-facing velocities and comparisons against the Haas-Sturtevant/Fedkiw reference values.
*/

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
    // Fit x(t) over a contiguous time window. Window selection happens outside this routine so the same linear fit can be used for all feature types.
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
        // Transverse features move less than axial features, so use a smaller displacement threshold but a slightly looser R^2 cutoff.
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



} // namespace quant


