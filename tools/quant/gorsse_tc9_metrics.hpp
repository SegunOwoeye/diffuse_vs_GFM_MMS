#pragma once

// Header-only implementation units for the quantitative validation runner.
// Gorsse et al. TC9 water-air shock-bubble interface feature metrics.

#include "csv_io.hpp"

namespace quant {

struct Tc9Grid {
    std::vector<double> x0;
    std::vector<double> x1;
    std::vector<double> field;
    int nx = 0;
    int ny = 0;
};

struct Tc9FeatureRow {
    std::map<std::string, std::string> values;
};

double tc9_time_from_name(const fs::path& path)
{
    const std::string name = path.filename().string();
    const auto npos = name.rfind("_N");
    if (npos == std::string::npos) return std::nan("");
    std::size_t tpos = std::string::npos;
    std::size_t search = 0;
    while (search < npos) {
        const auto candidate = name.find("_t", search);
        if (candidate == std::string::npos || candidate >= npos) break;
        if (candidate + 2 < name.size() &&
            std::isdigit(static_cast<unsigned char>(name[candidate + 2]))) {
            tpos = candidate;
        }
        search = candidate + 2;
    }
    if (tpos == std::string::npos) return std::nan("");
    std::string tag = name.substr(tpos + 2, npos - (tpos + 2));
    const auto epos = tag.find('e');
    if (epos == std::string::npos) return std::nan("");
    std::string mantissa = tag.substr(0, epos);
    std::string exponent = tag.substr(epos + 1);
    std::replace(mantissa.begin(), mantissa.end(), 'p', '.');
    std::replace(mantissa.begin(), mantissa.end(), 'm', '-');
    const char sign = (!exponent.empty() && exponent[0] == 'm') ? '-' : '+';
    const std::string text = mantissa + "e" + sign + exponent.substr(1);
    try {
        return std::stod(text);
    }
    catch (...) {
        return std::nan("");
    }
}

std::vector<double> sorted_unique(std::vector<double> values)
{
    std::sort(values.begin(), values.end());
    values.erase(std::unique(values.begin(), values.end()), values.end());
    return values;
}

int nearest_tc9_index(const std::vector<double>& values, double value)
{
    if (values.empty()) return -1;
    auto it = std::lower_bound(values.begin(), values.end(), value);
    if (it == values.begin()) return 0;
    if (it == values.end()) return static_cast<int>(values.size() - 1);
    const int hi = static_cast<int>(it - values.begin());
    const int lo = hi - 1;
    return std::abs(values[hi] - value) < std::abs(values[lo] - value) ? hi : lo;
}

Tc9Grid build_tc9_grid(
    const std::map<std::string, std::vector<double>>& columns,
    const std::string& field_name,
    double offset = 0.0
)
{
    Tc9Grid grid;
    if (!columns.count("x0") || !columns.count("x1") || !columns.count(field_name)) {
        return grid;
    }
    grid.x0 = sorted_unique(columns.at("x0"));
    grid.x1 = sorted_unique(columns.at("x1"));
    grid.nx = static_cast<int>(grid.x0.size());
    grid.ny = static_cast<int>(grid.x1.size());
    grid.field.assign(static_cast<std::size_t>(grid.nx * grid.ny), std::nan(""));

    for (std::size_t row = 0; row < columns.at("x0").size(); ++row) {
        const int ix = nearest_tc9_index(grid.x0, columns.at("x0")[row]);
        const int iy = nearest_tc9_index(grid.x1, columns.at("x1")[row]);
        if (ix < 0 || iy < 0) continue;
        grid.field[static_cast<std::size_t>(iy * grid.nx + ix)] =
            columns.at(field_name)[row] + offset;
    }
    return grid;
}

std::vector<std::pair<double, double>> tc9_zero_crossing_points(const Tc9Grid& grid)
{
    std::vector<std::pair<double, double>> points;
    if (grid.nx < 2 || grid.ny < 2) return points;
    auto id = [&](int iy, int ix) {
        return static_cast<std::size_t>(iy * grid.nx + ix);
    };
    auto crosses = [](double a, double b) {
        if (!std::isfinite(a) || !std::isfinite(b)) return false;
        return a == 0.0 || b == 0.0 || a * b < 0.0;
    };
    auto weight = [](double a, double b) {
        if (a == b) return 0.5;
        return std::max(0.0, std::min(1.0, (0.0 - a) / (b - a)));
    };

    for (int iy = 0; iy < grid.ny; ++iy) {
        for (int ix = 0; ix + 1 < grid.nx; ++ix) {
            const double a = grid.field[id(iy, ix)];
            const double b = grid.field[id(iy, ix + 1)];
            if (!crosses(a, b)) continue;
            const double w = weight(a, b);
            const double x0 = grid.x0[ix] + w * (grid.x0[ix + 1] - grid.x0[ix]);
            points.push_back({x0, grid.x1[iy]});
        }
    }
    for (int iy = 0; iy + 1 < grid.ny; ++iy) {
        for (int ix = 0; ix < grid.nx; ++ix) {
            const double a = grid.field[id(iy, ix)];
            const double b = grid.field[id(iy + 1, ix)];
            if (!crosses(a, b)) continue;
            const double w = weight(a, b);
            const double x1 = grid.x1[iy] + w * (grid.x1[iy + 1] - grid.x1[iy]);
            points.push_back({grid.x0[ix], x1});
        }
    }
    return points;
}

std::vector<unsigned char> largest_tc9_component(
    const Tc9Grid& grid,
    double threshold
)
{
    std::vector<unsigned char> selected(static_cast<std::size_t>(grid.nx * grid.ny), 0);
    if (grid.nx == 0 || grid.ny == 0) return selected;
    std::vector<unsigned char> mask(selected.size(), 0);
    for (std::size_t i = 0; i < grid.field.size(); ++i) {
        mask[i] = std::isfinite(grid.field[i]) && grid.field[i] >= threshold ? 1 : 0;
    }

    std::vector<unsigned char> visited(mask.size(), 0);
    std::vector<int> best;
    for (int iy = 0; iy < grid.ny; ++iy) {
        for (int ix = 0; ix < grid.nx; ++ix) {
            const int start = iy * grid.nx + ix;
            if (!mask[static_cast<std::size_t>(start)] || visited[static_cast<std::size_t>(start)]) {
                continue;
            }
            std::vector<int> component;
            std::vector<int> stack = {start};
            visited[static_cast<std::size_t>(start)] = 1;
            while (!stack.empty()) {
                const int cell = stack.back();
                stack.pop_back();
                component.push_back(cell);
                const int cx = cell % grid.nx;
                const int cy = cell / grid.nx;
                const int nx4[4] = {cx - 1, cx + 1, cx, cx};
                const int ny4[4] = {cy, cy, cy - 1, cy + 1};
                for (int k = 0; k < 4; ++k) {
                    if (nx4[k] < 0 || nx4[k] >= grid.nx || ny4[k] < 0 || ny4[k] >= grid.ny) {
                        continue;
                    }
                    const int next = ny4[k] * grid.nx + nx4[k];
                    if (!mask[static_cast<std::size_t>(next)] ||
                        visited[static_cast<std::size_t>(next)]) {
                        continue;
                    }
                    visited[static_cast<std::size_t>(next)] = 1;
                    stack.push_back(next);
                }
            }
            if (component.size() > best.size()) {
                best = std::move(component);
            }
        }
    }
    for (int cell : best) {
        selected[static_cast<std::size_t>(cell)] = 1;
    }
    return selected;
}

std::map<std::string, std::string> summarize_tc9_points(
    const std::vector<std::pair<double, double>>& points,
    const fs::path& csv_path,
    const std::string& method,
    const std::string& interface_label,
    const std::string& extraction_mode
)
{
    std::map<std::string, std::string> row;
    row["method"] = method;
    row["csv_file"] = csv_path.string();
    const double time_s = tc9_time_from_name(csv_path);
    row["time_s"] = double_text(time_s);
    row["time_us"] = double_text(time_s * 1.0e6);
    row["interface_label"] = interface_label;
    row["extraction_mode"] = extraction_mode;
    row["contour_point_count"] = std::to_string(points.size());

    if (points.empty()) {
        return row;
    }

    double x0_min = std::numeric_limits<double>::infinity();
    double x0_max = -std::numeric_limits<double>::infinity();
    double x1_min = std::numeric_limits<double>::infinity();
    double x1_max = -std::numeric_limits<double>::infinity();
    double x0_sum = 0.0;
    double x1_sum = 0.0;
    for (const auto& point : points) {
        x0_min = std::min(x0_min, point.first);
        x0_max = std::max(x0_max, point.first);
        x1_min = std::min(x1_min, point.second);
        x1_max = std::max(x1_max, point.second);
        x0_sum += point.first;
        x1_sum += point.second;
    }
    row["x0_min_m"] = double_text(x0_min);
    row["x0_max_m"] = double_text(x0_max);
    row["x1_min_m"] = double_text(x1_min);
    row["x1_max_m"] = double_text(x1_max);
    row["centroid_x0_m"] = double_text(x0_sum / static_cast<double>(points.size()));
    row["centroid_x1_m"] = double_text(x1_sum / static_cast<double>(points.size()));
    return row;
}

std::map<std::string, std::string> summarize_tc9_support(
    const Tc9Grid& grid,
    const fs::path& csv_path,
    const std::string& method,
    double threshold
)
{
    const auto component = largest_tc9_component(grid, threshold);
    std::vector<std::pair<double, double>> points;
    for (int iy = 0; iy < grid.ny; ++iy) {
        for (int ix = 0; ix < grid.nx; ++ix) {
            const int cell = iy * grid.nx + ix;
            if (!component[static_cast<std::size_t>(cell)]) continue;
            points.push_back({grid.x0[ix], grid.x1[iy]});
        }
    }
    return summarize_tc9_points(
        points,
        csv_path,
        method,
        "alpha1>=" + double_text(threshold),
        "largest alpha1 support"
    );
}

std::map<std::string, std::string> tc9_model_summary_row(
    const fs::path& csv_path,
    const std::string& method,
    double dim_support_threshold
)
{
    const auto columns = read_csv_columns(csv_path);
    if (columns.count("phi0")) {
        const auto grid = build_tc9_grid(columns, "phi0");
        const auto points = tc9_zero_crossing_points(grid);
        return summarize_tc9_points(points, csv_path, method, "phi0=0", "zero contour");
    }
    if (columns.count("alpha1")) {
        const auto grid = build_tc9_grid(columns, "alpha1", -0.5);
        const auto points = tc9_zero_crossing_points(grid);
        if (!points.empty()) {
            return summarize_tc9_points(points, csv_path, method, "alpha1=0.5", "zero contour");
        }
        const auto support_grid = build_tc9_grid(columns, "alpha1");
        return summarize_tc9_support(support_grid, csv_path, method, dim_support_threshold);
    }
    return {
        {"method", method},
        {"csv_file", csv_path.string()},
        {"interface_label", "missing"},
        {"extraction_mode", "not_detected"},
    };
}

std::vector<std::map<std::string, std::string>> tc9_model_summary_rows(
    const fs::path& raw_dir,
    const std::string& method,
    double dim_support_threshold = 0.20
)
{
    std::vector<fs::path> files;
    if (!fs::exists(raw_dir)) return {};
    for (const auto& entry : fs::recursive_directory_iterator(raw_dir)) {
        if (!entry.is_regular_file() || entry.path().extension() != ".csv") continue;
        const std::string name = entry.path().filename().string();
        if (name.find("_N") == std::string::npos ||
            name.find("runtime") != std::string::npos ||
            name.find("conservation") != std::string::npos ||
            name.find("exact") != std::string::npos) {
            continue;
        }
        files.push_back(entry.path());
    }
    std::sort(files.begin(), files.end());

    std::vector<std::map<std::string, std::string>> rows;
    for (const auto& file : files) {
        rows.push_back(tc9_model_summary_row(file, method, dim_support_threshold));
    }
    return rows;
}

std::vector<std::map<std::string, std::string>> tc9_reference_rows(
    const fs::path& reference_summary,
    double reference_x_offset
)
{
    const auto columns = read_csv_columns(reference_summary);
    if (!columns.count("time_us") || !columns.count("component_id")) return {};
    std::vector<std::map<std::string, std::string>> rows;
    for (std::size_t i = 0; i < columns.at("time_us").size(); ++i) {
        if (static_cast<int>(columns.at("component_id")[i]) != 0) continue;
        std::map<std::string, std::string> row;
        row["time_us"] = double_text(columns.at("time_us")[i]);
        for (const auto& key : {
                 "x0_min_m",
                 "x0_max_m",
                 "x1_min_m",
                 "x1_max_m",
                 "centroid_x0_m",
                 "centroid_x1_m"}) {
            if (!columns.count(key) || i >= columns.at(key).size()) continue;
            double value = columns.at(key)[i];
            const std::string name(key);
            if (name == "x0_min_m" || name == "x0_max_m" || name == "centroid_x0_m") {
                value += reference_x_offset;
            }
            row["reference_" + name] = double_text(value);
        }
        rows.push_back(row);
    }
    return rows;
}

std::vector<std::map<std::string, std::string>> tc9_reference_comparison_rows(
    const std::vector<std::map<std::string, std::string>>& model_rows,
    const fs::path& reference_summary,
    double reference_x_offset = -0.20
)
{
    const auto references = tc9_reference_rows(reference_summary, reference_x_offset);
    if (references.empty()) return {};
    const std::vector<std::string> metrics = {
        "x0_min_m",
        "x0_max_m",
        "x1_min_m",
        "x1_max_m",
        "centroid_x0_m",
        "centroid_x1_m",
    };

    std::vector<std::map<std::string, std::string>> out;
    for (const auto& model : model_rows) {
        const auto model_time = parse_double_field(model, "time_us");
        if (!model_time) continue;
        const std::map<std::string, std::string>* best = nullptr;
        double best_delta = std::numeric_limits<double>::infinity();
        for (const auto& reference : references) {
            const auto ref_time = parse_double_field(reference, "time_us");
            if (!ref_time) continue;
            const double delta = std::abs(model_time.value() - ref_time.value());
            if (delta < best_delta) {
                best_delta = delta;
                best = &reference;
            }
        }
        if (best == nullptr) continue;
        auto row = model;
        row["reference_time_us"] = best->at("time_us");
        row["time_delta_us"] = double_text(model_time.value() - std::stod(best->at("time_us")));
        row["reference_x_offset_m"] = double_text(reference_x_offset);
        for (const auto& metric : metrics) {
            const auto model_value = parse_double_field(model, metric);
            const auto reference_value = parse_double_field(*best, "reference_" + metric);
            if (!model_value || !reference_value) continue;
            row["reference_" + metric] = double_text(reference_value.value());
            row[metric + "_error_m"] = double_text(model_value.value() - reference_value.value());
        }
        out.push_back(row);
    }
    return out;
}

std::map<std::string, double> tc9_geometry_feature_values(
    const std::map<std::string, std::string>& row,
    const std::string& prefix = ""
)
{
    std::map<std::string, double> values;
    const auto x0_min = parse_double_field(row, prefix + "x0_min_m");
    const auto x0_max = parse_double_field(row, prefix + "x0_max_m");
    const auto x1_min = parse_double_field(row, prefix + "x1_min_m");
    const auto x1_max = parse_double_field(row, prefix + "x1_max_m");
    const auto centroid_x0 = parse_double_field(row, prefix + "centroid_x0_m");
    const auto centroid_x1 = parse_double_field(row, prefix + "centroid_x1_m");

    if (x0_min) values["left_edge_x0_m"] = x0_min.value();
    if (x0_max) values["right_edge_x0_m"] = x0_max.value();
    if (x1_min) values["lower_edge_x1_m"] = x1_min.value();
    if (x1_max) values["upper_edge_x1_m"] = x1_max.value();
    if (centroid_x0) values["centroid_x0_m"] = centroid_x0.value();
    if (centroid_x1) values["centroid_x1_m"] = centroid_x1.value();
    if (x0_min && x0_max) values["axial_width_m"] = x0_max.value() - x0_min.value();
    if (x1_min && x1_max) values["transverse_height_m"] = x1_max.value() - x1_min.value();
    if (x0_min && x0_max && x1_min && x1_max) {
        const double height = x1_max.value() - x1_min.value();
        if (height != 0.0) {
            values["aspect_ratio_width_over_height"] = (x0_max.value() - x0_min.value()) / height;
        }
    }
    return values;
}

std::string tc9_feature_label(const std::string& feature)
{
    if (feature == "left_edge_x0_m") return "left interface edge";
    if (feature == "right_edge_x0_m") return "right interface edge";
    if (feature == "lower_edge_x1_m") return "lower interface edge";
    if (feature == "upper_edge_x1_m") return "upper interface edge";
    if (feature == "centroid_x0_m") return "interface centroid x";
    if (feature == "centroid_x1_m") return "interface centroid y";
    if (feature == "axial_width_m") return "axial bubble width";
    if (feature == "transverse_height_m") return "transverse bubble height";
    if (feature == "aspect_ratio_width_over_height") return "bubble aspect ratio";
    return feature;
}

std::vector<std::map<std::string, std::string>> tc9_feature_timeseries_rows(
    const std::vector<std::map<std::string, std::string>>& model_rows
)
{
    std::vector<std::map<std::string, std::string>> out;
    for (const auto& row : model_rows) {
        const auto features = tc9_geometry_feature_values(row);
        for (const auto& feature : features) {
            std::map<std::string, std::string> item;
            for (const auto& key : {
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
                     "csv_file"}) {
                if (row.count(key)) item[key] = row.at(key);
            }
            item["feature"] = feature.first;
            item["feature_label"] = tc9_feature_label(feature.first);
            item["value"] = double_text(feature.second);
            item["unit"] = feature.first.find("_m") != std::string::npos ? "m" : "dimensionless";
            out.push_back(item);
        }
    }
    std::sort(out.begin(), out.end(), [](const auto& lhs, const auto& rhs) {
        const auto key = [](const auto& row) {
            return std::make_tuple(
                row.count("method") ? row.at("method") : std::string{},
                row.count("resolution") ? row.at("resolution") : std::string{},
                row.count("run_id") ? row.at("run_id") : std::string{},
                row.count("feature") ? row.at("feature") : std::string{},
                parse_double_field(row, "time_us").value_or(std::numeric_limits<double>::infinity())
            );
        };
        return key(lhs) < key(rhs);
    });

    struct PreviousFeature {
        double time_us = std::nan("");
        double value = std::nan("");
    };
    std::map<std::string, PreviousFeature> previous;
    for (auto& row : out) {
        if (!row.count("method") || !row.count("resolution") ||
            !row.count("run_id") || !row.count("feature")) {
            continue;
        }
        const auto time_us = parse_double_field(row, "time_us");
        const auto value = parse_double_field(row, "value");
        if (!time_us || !value) continue;
        const std::string key =
            row.at("method") + "|" + row.at("resolution") + "|" +
            row.at("run_id") + "|" + row.at("feature");
        if (previous.count(key)) {
            const double dt_s = (time_us.value() - previous.at(key).time_us) * 1.0e-6;
            if (dt_s > 0.0) {
                row["previous_time_us"] = double_text(previous.at(key).time_us);
                row["finite_difference_rate"] =
                    double_text((value.value() - previous.at(key).value) / dt_s);
                row["rate_unit"] = row.count("unit") && row.at("unit") == "m"
                    ? "m/s"
                    : "1/s";
            }
        }
        previous[key] = {time_us.value(), value.value()};
    }
    return out;
}

std::vector<std::map<std::string, std::string>> tc9_feature_reference_comparison_rows(
    const std::vector<std::map<std::string, std::string>>& comparison_rows
)
{
    std::vector<std::map<std::string, std::string>> out;
    for (const auto& row : comparison_rows) {
        const auto model_features = tc9_geometry_feature_values(row);
        const auto reference_features = tc9_geometry_feature_values(row, "reference_");
        for (const auto& feature : model_features) {
            if (!reference_features.count(feature.first)) continue;
            const double reference_value = reference_features.at(feature.first);
            const double error = feature.second - reference_value;
            std::map<std::string, std::string> item;
            for (const auto& key : {
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
                     "csv_file"}) {
                if (row.count(key)) item[key] = row.at(key);
            }
            item["feature"] = feature.first;
            item["feature_label"] = tc9_feature_label(feature.first);
            item["value"] = double_text(feature.second);
            item["reference_value"] = double_text(reference_value);
            item["error"] = double_text(error);
            if (std::abs(reference_value) > 1.0e-14) {
                item["relative_error"] = double_text(error / reference_value);
            }
            item["unit"] = feature.first.find("_m") != std::string::npos ? "m" : "dimensionless";
            out.push_back(item);
        }
    }
    return out;
}

} // namespace quant
