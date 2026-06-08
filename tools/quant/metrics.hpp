#pragma once

// Header-only implementation units for the quantitative validation runner.
// Numerical post-processing for conservation, reference errors, interfaces, and bubbles.

#include "runner.hpp"

namespace quant {

std::map<std::string, std::string> read_key_values(const fs::path& path)
{
    std::map<std::string, std::string> values;
    std::ifstream file(path);
    std::string line;
    while (std::getline(file, line)) {
        const auto pos = line.find('=');
        if (pos == std::string::npos) continue;
        values[trim(line.substr(0, pos))] = trim(line.substr(pos + 1));
    }
    return values;
}

std::map<std::string, double> final_conservation(const fs::path& path)
{
    std::map<std::string, double> metrics;
    std::ifstream file(path);
    if (!file) return metrics;

    std::string header_line;
    if (!std::getline(file, header_line)) return metrics;
    const auto headers = split(header_line, ',');
    std::string line;
    std::vector<std::vector<std::string>> rows;
    while (std::getline(file, line)) {
        if (!trim(line).empty()) rows.push_back(split(line, ','));
    }
    if (rows.empty()) return metrics;

    auto column_index = [&](const std::string& name) -> std::optional<std::size_t> {
        for (std::size_t i = 0; i < headers.size(); ++i) {
            if (trim(headers[i]) == name) return i;
        }
        return std::nullopt;
    };
    auto value_at = [&](const std::vector<std::string>& row, const std::string& name) -> std::optional<double> {
        const auto index = column_index(name);
        if (!index.has_value() || index.value() >= row.size()) return std::nullopt;
        try {
            return std::stod(row[index.value()]);
        }
        catch (...) {
            return std::nullopt;
        }
    };

    const auto& first = rows.front();
    const auto& final = rows.back();

    const auto has_prefix = [](const std::string& key, const std::string& prefix) {
        return key.rfind(prefix, 0) == 0;
    };

    for (std::size_t i = 0; i < headers.size() && i < final.size(); ++i) {
        const std::string key = trim(headers[i]);
        if (has_prefix(key, "eps_") ||
            has_prefix(key, "raw_drift_") ||
            has_prefix(key, "boundary_flux_") ||
            has_prefix(key, "balance_residual_") ||
            has_prefix(key, "normalized_balance_residual_") ||
            has_prefix(key, "interface_flux_mismatch_") ||
            has_prefix(key, "near_zero_initial_")) {
            try {
                metrics[key] = std::stod(final[i]);
            }
            catch (...) {}
        }
    }

    for (int d = 0; d < 3; ++d) {
        const std::string key = "momentum" + std::to_string(d);
        const auto initial = value_at(first, key);
        const auto end = value_at(final, key);
        if (!initial.has_value() || !end.has_value()) continue;

        double max_abs_momentum = std::max(std::abs(initial.value()), std::abs(end.value()));
        for (const auto& row : rows) {
            const auto sample = value_at(row, key);
            if (sample.has_value()) {
                max_abs_momentum = std::max(max_abs_momentum, std::abs(sample.value()));
            }
        }
        const double absolute_drift = std::abs(end.value() - initial.value());
        const double scale = std::max(max_abs_momentum, 1.0e-12);
        metrics["absolute_" + key + "_drift"] = absolute_drift;
        metrics["trajectory_scaled_" + key + "_drift"] = absolute_drift / scale;
        if (std::abs(initial.value()) < 1.0e-12 && metrics.count("eps_" + key)) {
            metrics["raw_eps_" + key + "_near_zero_initial_flag"] = 1.0;
        }
    }

    double raw_max_drift = 0.0;
    double reportable_max_drift = 0.0;
    double max_normalized_balance = 0.0;
    for (const auto& item : metrics) {
        if (has_prefix(item.first, "eps_")) {
            const std::string variable = item.first.substr(4);
            const bool near_zero_momentum =
                variable.find("momentum") != std::string::npos &&
                metrics.count("raw_eps_" + variable + "_near_zero_initial_flag");
            raw_max_drift = std::max(raw_max_drift, std::abs(item.second));
            if (!near_zero_momentum) {
                reportable_max_drift = std::max(reportable_max_drift, std::abs(item.second));
            }
        }
        if (has_prefix(item.first, "normalized_balance_residual_")) {
            max_normalized_balance = std::max(max_normalized_balance, std::abs(item.second));
        }
    }
    if (!metrics.empty()) {
        metrics["raw_max_relative_drift"] = raw_max_drift;
        metrics["reportable_max_relative_drift"] = reportable_max_drift;
        metrics["max_relative_drift"] = reportable_max_drift;
        metrics["max_normalized_balance_residual"] = max_normalized_balance;
    }
    return metrics;
}

std::vector<std::map<std::string, std::string>> conservation_compact_rows(
    const RunSpec& run,
    const std::map<std::string, double>& metrics
)
{
    std::set<std::string> variables;
    const std::string prefix = "raw_drift_";
    for (const auto& item : metrics) {
        if (item.first.rfind(prefix, 0) == 0) {
            variables.insert(item.first.substr(prefix.size()));
        }
    }

    std::vector<std::map<std::string, std::string>> rows;
    for (const auto& variable : variables) {
        auto value_text = [&](const std::string& key) {
            const auto it = metrics.find(key + variable);
            if (it == metrics.end()) return std::string{};
            std::ostringstream out;
            out << std::setprecision(12) << it->second;
            return out.str();
        };

        rows.push_back({
            {"case", run.case_def.label},
            {"method", run.case_def.method},
            {"resolution", resolution_label(run.resolution)},
            {"variable", variable},
            {"raw_drift", value_text("raw_drift_")},
            {"boundary_flux", value_text("boundary_flux_")},
            {"balance_residual", value_text("balance_residual_")},
            {"normalized_balance_residual", value_text("normalized_balance_residual_")},
            {"interface_flux_mismatch", value_text("interface_flux_mismatch_")},
            {"near_zero_initial_flag", value_text("near_zero_initial_")},
        });
    }

    return rows;
}

std::map<std::string, std::vector<double>> read_csv_columns(const fs::path& path)
{
    std::map<std::string, std::vector<double>> columns;
    std::ifstream file(path);
    if (!file) return columns;
    std::string line;
    if (!std::getline(file, line)) return columns;
    auto headers = split(line, ',');
    for (auto& header : headers) {
        header = trim(header);
        if (!header.empty() && static_cast<unsigned char>(header[0]) == 0xEF) {
            header = header.substr(3);
        }
        columns[header] = {};
    }
    while (std::getline(file, line)) {
        const auto values = split(line, ',');
        for (std::size_t i = 0; i < headers.size() && i < values.size(); ++i) {
            try {
                columns[headers[i]].push_back(std::stod(values[i]));
            }
            catch (...) {}
        }
    }
    return columns;
}

std::map<std::string, std::pair<std::vector<double>, std::vector<double>>> digitized_reference(const fs::path& path)
{
    std::map<std::string, std::pair<std::vector<double>, std::vector<double>>> refs;
    std::ifstream file(path);
    if (!file) return refs;

    std::vector<std::vector<std::string>> rows;
    std::string line;
    while (std::getline(file, line)) rows.push_back(split(line, ','));
    if (rows.empty()) return refs;

    auto field_name = [](const std::string& raw) -> std::string {
        const std::string name = lower(trim(raw));
        if (name == "density" || name == "rho") return "rho";
        if (name == "velocity" || name == "u" || name == "u0") return "u0";
        if (name == "pressure" || name == "p") return "p";
        if (name == "energy" || name == "specific internal energy" || name == "internal energy") return "e";
        return "";
    };

    for (std::size_t col = 0; col + 1 < rows[0].size(); col += 2) {
        const std::string key = field_name(rows[0][col]);
        if (key.empty()) continue;
        std::vector<double> x;
        std::vector<double> y;
        for (std::size_t row = 2; row < rows.size(); ++row) {
            if (col + 1 >= rows[row].size()) continue;
            try {
                x.push_back(std::stod(rows[row][col]));
                y.push_back(std::stod(rows[row][col + 1]));
            }
            catch (...) {}
        }
        if (!x.empty()) refs[key] = {x, y};
    }
    return refs;
}

double interp(const std::vector<double>& x, const std::vector<double>& y, double value)
{
    if (x.empty()) return std::nan("");
    if (value <= x.front()) return y.front();
    if (value >= x.back()) return y.back();
    auto upper = std::upper_bound(x.begin(), x.end(), value);
    const std::size_t hi = static_cast<std::size_t>(upper - x.begin());
    const std::size_t lo = hi - 1;
    const double denom = x[hi] - x[lo];
    if (denom == 0.0) return y[lo];
    const double w = (value - x[lo]) / denom;
    return y[lo] + w * (y[hi] - y[lo]);
}

double cell_width(const std::vector<double>& x)
{
    if (x.size() < 2) return 1.0;
    std::vector<double> diffs;
    for (std::size_t i = 0; i + 1 < x.size(); ++i) diffs.push_back(std::abs(x[i + 1] - x[i]));
    std::sort(diffs.begin(), diffs.end());
    return diffs[diffs.size() / 2];
}

double reference_floor(const std::string& column, const std::string& norm)
{
    double scale = 1e-12;
    if (column == "p" || column == "e") {
        scale = 1.0;
    }
    if (norm == "L2") {
        return std::sqrt(scale * scale);
    }
    return scale;
}

int resolution_n(const std::string& label)
{
    return std::stoi(label.substr(0, label.find('x')));
}

std::optional<fs::path> exact_reference_path(const RunSpec& run, const fs::path& case_dir)
{
    if (run.case_def.group == "toro_1d") {
        for (const auto& entry : fs::directory_iterator(case_dir)) {
            const std::string name = entry.path().filename().string();
            if (name.find("_exact_N") != std::string::npos && entry.path().extension() == ".csv") {
                return entry.path();
            }
        }
    }
    if (run.case_def.group == "fedkiw_1d") {
        const std::map<std::string, std::string> names = {
            {"test1", "test1_exact.csv"},
            {"test2", "test2_exact.csv"},
            {"test3", "test3_exact.csv"},
            {"test4", "test4_exact.csv"},
            {"test5", "test5_exact.csv"},
        };
        const auto it = names.find(run.case_def.name);
        if (it != names.end()) {
            const fs::path path = fs::path("data") / "exact" / "fedkiw" / it->second;
            if (fs::exists(path)) return path;
        }
    }
    return std::nullopt;
}

std::vector<std::map<std::string, std::string>> error_rows_for(const RunSpec& run, const fs::path& solution, const fs::path& case_dir)
{
    std::vector<std::map<std::string, std::string>> rows;
    if (!fs::exists(solution)) return rows;
    const auto reference = exact_reference_path(run, case_dir);
    if (!reference.has_value()) return rows;

    const auto solution_cols = read_csv_columns(solution);
    if (!solution_cols.count("x0")) return rows;

    std::map<std::string, std::pair<std::vector<double>, std::vector<double>>> refs;
    if (reference->parent_path() == case_dir) {
        const auto ref_cols = read_csv_columns(reference.value());
        for (const auto& key : {"rho", "u0", "p", "e"}) {
            if (ref_cols.count("x0") && ref_cols.count(key)) refs[key] = {ref_cols.at("x0"), ref_cols.at(key)};
        }
    }
    else {
        refs = digitized_reference(reference.value());
    }

    const std::map<std::string, std::string> variables = {
        {"rho", "density"},
        {"u0", "velocity"},
        {"p", "pressure"},
        {"e", "energy"},
    };
    const double dx = cell_width(solution_cols.at("x0"));
    for (const auto& item : variables) {
        const std::string& column = item.first;
        if (!solution_cols.count(column) || !refs.count(column)) continue;
        double l1 = 0.0;
        double l2sum = 0.0;
        double linf = 0.0;
        double ref_l1 = 0.0;
        double ref_l2sum = 0.0;
        double ref_linf = 0.0;
        const auto& xs = solution_cols.at("x0");
        const auto& values = solution_cols.at(column);
        for (std::size_t i = 0; i < xs.size() && i < values.size(); ++i) {
            const double ref_value = interp(refs[column].first, refs[column].second, xs[i]);
            if (!std::isfinite(values[i]) || !std::isfinite(ref_value)) continue;
            const double diff = values[i] - ref_value;
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
            row["variable"] = item.second;
            row["norm"] = norm_error.first;
            row["error"] = double_text(raw_error);
            row["raw_error"] = double_text(raw_error);
            row["reference_norm"] = double_text(ref_norm);
            row["normalized_error"] = double_text(raw_error / denominator);
            row["reference"] = reference->string();
            rows.push_back(row);
        }
    }
    return rows;
}

double max_abs_reference_or_solution(
    const RunSpec& run,
    const fs::path& solution,
    const fs::path& case_dir,
    const std::string& column,
    double floor = 1.0e-12
)
{
    double scale = floor;
    const auto reference = exact_reference_path(run, case_dir);
    if (reference.has_value()) {
        std::map<std::string, std::pair<std::vector<double>, std::vector<double>>> refs;
        if (reference->parent_path() == case_dir) {
            const auto ref_cols = read_csv_columns(reference.value());
            if (ref_cols.count(column)) {
                for (double value : ref_cols.at(column)) {
                    if (std::isfinite(value)) scale = std::max(scale, std::abs(value));
                }
            }
        }
        else {
            refs = digitized_reference(reference.value());
            if (refs.count(column)) {
                for (double value : refs.at(column).second) {
                    if (std::isfinite(value)) scale = std::max(scale, std::abs(value));
                }
            }
        }
    }

    const auto solution_cols = read_csv_columns(solution);
    if (solution_cols.count(column)) {
        for (double value : solution_cols.at(column)) {
            if (std::isfinite(value)) scale = std::max(scale, std::abs(value));
        }
    }

    return scale;
}

int representative_n(const std::vector<int>& resolution)
{
    double product = 1.0;
    for (int value : resolution) product *= value;
    return static_cast<int>(std::round(std::pow(product, 1.0 / resolution.size())));
}

double crossing_position(const std::vector<double>& x, const std::vector<double>& y, double level)
{
    for (std::size_t i = 0; i + 1 < x.size() && i + 1 < y.size(); ++i) {
        const double a = y[i] - level;
        const double b = y[i + 1] - level;
        if (a == 0.0) return x[i];
        if (a * b <= 0.0) {
            const double denom = y[i + 1] - y[i];
            const double w = (denom == 0.0) ? 0.5 : (level - y[i]) / denom;
            return x[i] + w * (x[i + 1] - x[i]);
        }
    }
    return std::nan("");
}

std::map<std::string, double> interface_metrics(const fs::path& solution, const std::string& method)
{
    std::map<std::string, double> metrics;
    const auto c = read_csv_columns(solution);
    if (!c.count("x0")) return metrics;
    const auto& x = c.at("x0");
    double position = std::nan("");
    if (method == "DIM") {
        const std::string key = c.count("alpha1") ? "alpha1" : (c.count("alpha0") ? "alpha0" : "");
        if (!key.empty()) {
            position = crossing_position(x, c.at(key), 0.5);
            const double p05 = crossing_position(x, c.at(key), 0.05);
            const double p95 = crossing_position(x, c.at(key), 0.95);
            if (!std::isnan(position)) metrics["interface_position"] = position;
            if (!std::isnan(p05) && !std::isnan(p95)) metrics["interface_thickness"] = std::abs(p95 - p05);
        }
    }
    else if (method == "SIM" && c.count("phi0")) {
        position = crossing_position(x, c.at("phi0"), 0.0);
        if (!std::isnan(position)) metrics["interface_position"] = position;
    }
    if (c.count("u1")) {
        double max_abs = 0.0;
        for (double value : c.at("u1")) max_abs = std::max(max_abs, std::abs(value));
        metrics["max_abs_transverse_velocity"] = max_abs;
    }
    if (!std::isnan(position) && c.count("p") && c.count("u0") && x.size() > 1) {
        const double dx = std::abs(x[1] - x[0]);
        double pmin = 1e300, pmax = -1e300, umin = 1e300, umax = -1e300;
        bool any = false;
        for (std::size_t i = 0; i < x.size(); ++i) {
            if (std::abs(x[i] - position) <= 3.0 * dx) {
                any = true;
                pmin = std::min(pmin, c.at("p")[i]);
                pmax = std::max(pmax, c.at("p")[i]);
                umin = std::min(umin, c.at("u0")[i]);
                umax = std::max(umax, c.at("u0")[i]);
            }
        }
        if (any) {
            metrics["pressure_oscillation"] = pmax - pmin;
            metrics["velocity_oscillation"] = umax - umin;
        }
    }
    return metrics;
}

double time_from_name(const fs::path& path)
{
    const std::string name = path.filename().string();
    const auto tpos = name.find("_t");
    const auto npos = name.find("_N", tpos == std::string::npos ? 0 : tpos);
    if (tpos == std::string::npos || npos == std::string::npos) return std::nan("");
    std::string tag = name.substr(tpos + 2, npos - (tpos + 2));
    const auto epos = tag.find('e');
    if (epos == std::string::npos) return std::nan("");
    std::string mantissa = tag.substr(0, epos);
    std::string exponent = tag.substr(epos + 1);
    std::replace(mantissa.begin(), mantissa.end(), 'p', '.');
    std::replace(mantissa.begin(), mantissa.end(), 'm', '-');
    const char sign = (!exponent.empty() && exponent[0] == 'm') ? '-' : '+';
    const std::string text = mantissa + "e" + sign + exponent.substr(1);
    try { return std::stod(text); } catch (...) { return std::nan(""); }
}

struct BubbleSnapshot {
    double time = std::nan("");
    double upstream_x = std::nan("");
    double downstream_x = std::nan("");
    double jet_head_x = std::nan("");
    double transverse_interface_y = std::nan("");
    double transverse_triple_point_x = std::nan("");
    double transverse_triple_point_y = std::nan("");
    double transverse_wave_y = std::nan("");
    bool jet_detected = false;
    double bubble_area = std::nan("");
    double tracking_confidence = 0.0;
};

std::vector<double> crossing_positions(
    const std::vector<std::pair<double, double>>& samples,
    double level
)
{
    std::vector<double> crossings;
    for (std::size_t i = 0; i + 1 < samples.size(); ++i) {
        const double x0 = samples[i].first;
        const double x1 = samples[i + 1].first;
        const double s0 = samples[i].second - level;
        const double s1 = samples[i + 1].second - level;
        if (s0 == 0.0) {
            crossings.push_back(x0);
            continue;
        }
        if (s0 * s1 > 0.0) continue;
        const double denom = samples[i + 1].second - samples[i].second;
        const double w = (denom == 0.0) ? 0.5 : (level - samples[i].second) / denom;
        crossings.push_back(x0 + w * (x1 - x0));
    }
    return crossings;
}

struct BubbleGrid {
    std::vector<double> x;
    std::vector<double> y;
    std::vector<double> helium;
    std::vector<unsigned char> main_component;
    int nx = 0;
    int ny = 0;
    double dx = std::nan("");
    double dy = std::nan("");
    double area = std::nan("");
};

std::vector<double> unique_sorted_values(std::vector<double> values)
{
    std::sort(values.begin(), values.end());
    values.erase(std::unique(values.begin(), values.end()), values.end());
    return values;
}

int nearest_index(const std::vector<double>& values, double value)
{
    if (values.empty()) return -1;
    auto it = std::lower_bound(values.begin(), values.end(), value);
    if (it == values.begin()) return 0;
    if (it == values.end()) return static_cast<int>(values.size() - 1);
    const int hi = static_cast<int>(it - values.begin());
    const int lo = hi - 1;
    return (std::abs(values[hi] - value) < std::abs(values[lo] - value)) ? hi : lo;
}

double shock_bubble_helium_value(
    const std::map<std::string, std::vector<double>>& c,
    const std::string& method,
    std::size_t i
)
{
    if (c.count("alpha1")) {
        return std::max(0.0, std::min(1.0, c.at("alpha1")[i]));
    }
    if (c.count("alpha0")) {
        return std::max(0.0, std::min(1.0, 1.0 - c.at("alpha0")[i]));
    }
    if (method == "SIM" && c.count("phi0")) {
        return c.at("phi0")[i] <= 0.0 ? 1.0 : 0.0;
    }
    return std::nan("");
}

BubbleGrid build_bubble_grid(const std::map<std::string, std::vector<double>>& c, const std::string& method)
{
    BubbleGrid grid;
    grid.x = unique_sorted_values(c.at("x0"));
    grid.y = unique_sorted_values(c.at("x1"));
    grid.nx = static_cast<int>(grid.x.size());
    grid.ny = static_cast<int>(grid.y.size());
    if (grid.nx == 0 || grid.ny == 0) return grid;
    grid.dx = (grid.nx > 1) ? std::abs(grid.x[1] - grid.x[0]) : 1.0;
    grid.dy = (grid.ny > 1) ? std::abs(grid.y[1] - grid.y[0]) : 1.0;
    grid.helium.assign(static_cast<std::size_t>(grid.nx * grid.ny), 0.0);

    for (std::size_t i = 0; i < c.at("x0").size(); ++i) {
        const int ix = nearest_index(grid.x, c.at("x0")[i]);
        const int iy = nearest_index(grid.y, c.at("x1")[i]);
        if (ix < 0 || iy < 0) continue;
        const double value = shock_bubble_helium_value(c, method, i);
        if (std::isnan(value)) continue;
        grid.helium[static_cast<std::size_t>(iy * grid.nx + ix)] = value;
    }

    std::vector<unsigned char> mask(grid.helium.size(), 0);
    for (std::size_t i = 0; i < grid.helium.size(); ++i) {
        mask[i] = grid.helium[i] > 0.5 ? 1 : 0;
    }

    std::vector<unsigned char> visited(mask.size(), 0);
    std::vector<int> best_component;
    for (int iy = 0; iy < grid.ny; ++iy) {
        for (int ix = 0; ix < grid.nx; ++ix) {
            const int start = iy * grid.nx + ix;
            if (!mask[start] || visited[start]) continue;
            std::vector<int> stack = {start};
            std::vector<int> component;
            visited[start] = 1;
            while (!stack.empty()) {
                const int id = stack.back();
                stack.pop_back();
                component.push_back(id);
                const int cx = id % grid.nx;
                const int cy = id / grid.nx;
                const int nx4[4] = {cx - 1, cx + 1, cx, cx};
                const int ny4[4] = {cy, cy, cy - 1, cy + 1};
                for (int k = 0; k < 4; ++k) {
                    if (nx4[k] < 0 || nx4[k] >= grid.nx || ny4[k] < 0 || ny4[k] >= grid.ny) continue;
                    const int nid = ny4[k] * grid.nx + nx4[k];
                    if (!mask[nid] || visited[nid]) continue;
                    visited[nid] = 1;
                    stack.push_back(nid);
                }
            }
            if (component.size() > best_component.size()) {
                best_component = std::move(component);
            }
        }
    }

    grid.main_component.assign(mask.size(), 0);
    for (int id : best_component) grid.main_component[static_cast<std::size_t>(id)] = 1;
    grid.area = static_cast<double>(best_component.size()) * grid.dx * grid.dy;
    return grid;
}

std::vector<std::pair<double, double>> centerline_band_profile(const BubbleGrid& grid)
{
    std::vector<int> band_rows;
    for (int iy = 0; iy < grid.ny; ++iy) {
        if (grid.y[iy] <= 1.0 + 1.0e-12) band_rows.push_back(iy);
    }
    if (band_rows.empty() && grid.ny > 0) {
        band_rows.push_back(nearest_index(grid.y, 0.0));
    }

    std::vector<std::pair<double, double>> profile;
    for (int ix = 0; ix < grid.nx; ++ix) {
        double sum = 0.0;
        int count = 0;
        for (int iy : band_rows) {
            const int id = iy * grid.nx + ix;
            double value = grid.helium[static_cast<std::size_t>(id)];
            if (value > 0.5 && !grid.main_component[static_cast<std::size_t>(id)]) value = 0.0;
            sum += value;
            ++count;
        }
        profile.push_back({grid.x[ix], count > 0 ? sum / static_cast<double>(count) : 0.0});
    }
    return profile;
}

double transverse_extent_from_component(const BubbleGrid& grid)
{
    double ymax = std::nan("");
    for (int iy = 0; iy < grid.ny; ++iy) {
        for (int ix = 0; ix < grid.nx; ++ix) {
            const int id = iy * grid.nx + ix;
            if (!grid.main_component[static_cast<std::size_t>(id)]) continue;
            const double y_face = grid.y[iy] + 0.5 * grid.dy;
            if (std::isnan(ymax) || y_face > ymax) ymax = y_face;
        }
    }
    return ymax;
}

std::pair<double, double> transverse_triple_point_from_interface_pressure_gradient(
    const std::map<std::string, std::vector<double>>& c,
    const BubbleGrid& grid
)
{
    if (!c.count("p") || grid.nx < 3 || grid.ny < 3) {
        return {std::nan(""), std::nan("")};
    }

    std::vector<double> pressure(static_cast<std::size_t>(grid.nx * grid.ny), std::nan(""));
    std::vector<double> interface_scalar(static_cast<std::size_t>(grid.nx * grid.ny), std::nan(""));
    double interface_level = 0.5;
    for (std::size_t i = 0; i < c.at("x0").size(); ++i) {
        const int ix = nearest_index(grid.x, c.at("x0")[i]);
        const int iy = nearest_index(grid.y, c.at("x1")[i]);
        if (ix < 0 || iy < 0) continue;
        const std::size_t idx = static_cast<std::size_t>(iy * grid.nx + ix);
        pressure[idx] = c.at("p")[i];
        if (c.count("alpha1")) {
            interface_scalar[idx] = c.at("alpha1")[i];
            interface_level = 0.5;
        }
        else if (c.count("alpha0")) {
            interface_scalar[idx] = 1.0 - c.at("alpha0")[i];
            interface_level = 0.5;
        }
        else if (c.count("phi0")) {
            interface_scalar[idx] = c.at("phi0")[i];
            interface_level = 0.0;
        }
        else if (c.count("rho")) {
            interface_scalar[idx] = c.at("rho")[i];
        }
    }
    if (!c.count("alpha1") && !c.count("alpha0") && !c.count("phi0") && c.count("rho")) {
        double min_rho = std::numeric_limits<double>::infinity();
        double max_rho = -std::numeric_limits<double>::infinity();
        for (double value : c.at("rho")) {
            if (!std::isfinite(value)) continue;
            min_rho = std::min(min_rho, value);
            max_rho = std::max(max_rho, value);
        }
        if (std::isfinite(min_rho) && std::isfinite(max_rho)) {
            interface_level = 0.5 * (min_rho + max_rho);
        }
    }

    const auto id = [nx = grid.nx](int y, int x) {
        return static_cast<std::size_t>(y * nx + x);
    };
    std::vector<double> grad_mag(static_cast<std::size_t>(grid.nx * grid.ny), std::nan(""));
    for (int iy = 1; iy + 1 < grid.ny; ++iy) {
        for (int ix = 1; ix + 1 < grid.nx; ++ix) {
            const double pxm = pressure[id(iy, ix - 1)];
            const double pxp = pressure[id(iy, ix + 1)];
            const double pym = pressure[id(iy - 1, ix)];
            const double pyp = pressure[id(iy + 1, ix)];
            if (!std::isfinite(pxm) || !std::isfinite(pxp) ||
                !std::isfinite(pym) || !std::isfinite(pyp)) {
                continue;
            }
            const double dx = grid.x[ix + 1] - grid.x[ix - 1];
            const double dy = grid.y[iy + 1] - grid.y[iy - 1];
            if (dx == 0.0 || dy == 0.0) continue;
            const double gx = (pxp - pxm) / dx;
            const double gy = (pyp - pym) / dy;
            grad_mag[id(iy, ix)] = std::sqrt(gx * gx + gy * gy);
        }
    }

    auto sample_grad = [&](double x, double y) -> double {
        if (x < grid.x.front() || x > grid.x.back() || y < grid.y.front() || y > grid.y.back()) {
            return std::nan("");
        }
        auto xhi_it = std::lower_bound(grid.x.begin(), grid.x.end(), x);
        auto yhi_it = std::lower_bound(grid.y.begin(), grid.y.end(), y);
        if (xhi_it == grid.x.begin() || xhi_it == grid.x.end() ||
            yhi_it == grid.y.begin() || yhi_it == grid.y.end()) {
            const int ix = nearest_index(grid.x, x);
            const int iy = nearest_index(grid.y, y);
            if (ix < 0 || iy < 0) return std::nan("");
            return grad_mag[id(iy, ix)];
        }
        const int ix1 = static_cast<int>(xhi_it - grid.x.begin());
        const int iy1 = static_cast<int>(yhi_it - grid.y.begin());
        const int ix0 = ix1 - 1;
        const int iy0 = iy1 - 1;
        const double x0 = grid.x[ix0];
        const double x1 = grid.x[ix1];
        const double y0 = grid.y[iy0];
        const double y1 = grid.y[iy1];
        const double tx = (x1 == x0) ? 0.0 : (x - x0) / (x1 - x0);
        const double ty = (y1 == y0) ? 0.0 : (y - y0) / (y1 - y0);
        const double g00 = grad_mag[id(iy0, ix0)];
        const double g10 = grad_mag[id(iy0, ix1)];
        const double g01 = grad_mag[id(iy1, ix0)];
        const double g11 = grad_mag[id(iy1, ix1)];
        if (!std::isfinite(g00) || !std::isfinite(g10) ||
            !std::isfinite(g01) || !std::isfinite(g11)) {
            return std::nan("");
        }
        const double gx0 = (1.0 - tx) * g00 + tx * g10;
        const double gx1 = (1.0 - tx) * g01 + tx * g11;
        return (1.0 - ty) * gx0 + ty * gx1;
    };

    double ymax = std::nan("");
    double xmin = std::numeric_limits<double>::infinity();
    double xmax = -std::numeric_limits<double>::infinity();
    for (int iy = 0; iy < grid.ny; ++iy) {
        for (int ix = 0; ix < grid.nx; ++ix) {
            const int center = iy * grid.nx + ix;
            if (!grid.main_component[static_cast<std::size_t>(center)]) continue;
            ymax = std::isnan(ymax) ? grid.y[iy] : std::max(ymax, grid.y[iy]);
            xmin = std::min(xmin, grid.x[ix]);
            xmax = std::max(xmax, grid.x[ix]);
        }
    }
    if (!std::isfinite(xmin) || !std::isfinite(xmax) || std::isnan(ymax)) {
        return {std::nan(""), std::nan("")};
    }

    const double upper_interface_floor = std::max(0.35 * ymax, 5.0);
    const double shock_facing_x_floor = xmin + 0.45 * (xmax - xmin);
    double best_score = -1.0;
    double best_x = std::nan("");
    double best_y = std::nan("");

    auto consider_contour_point = [&](double x, double y, int id0, int id1) {
        if (x < shock_facing_x_floor || y < upper_interface_floor) return;
        if (!grid.main_component[static_cast<std::size_t>(id0)] &&
            !grid.main_component[static_cast<std::size_t>(id1)]) {
            return;
        }
        const double score = sample_grad(x, y);
        if (!std::isfinite(score)) return;
        if (score > best_score) {
            best_score = score;
            best_x = x;
            best_y = y;
        }
    };
    auto edge_crosses = [&](double a, double b) {
        if (!std::isfinite(a) || !std::isfinite(b)) return false;
        const double s0 = a - interface_level;
        const double s1 = b - interface_level;
        return s0 == 0.0 || s1 == 0.0 || s0 * s1 < 0.0;
    };
    auto weight = [&](double a, double b) {
        const double denom = b - a;
        if (denom == 0.0) return 0.5;
        return std::max(0.0, std::min(1.0, (interface_level - a) / denom));
    };

    for (int iy = 0; iy < grid.ny; ++iy) {
        for (int ix = 0; ix + 1 < grid.nx; ++ix) {
            const int left = iy * grid.nx + ix;
            const int right = iy * grid.nx + ix + 1;
            const double a = interface_scalar[static_cast<std::size_t>(left)];
            const double b = interface_scalar[static_cast<std::size_t>(right)];
            if (!edge_crosses(a, b)) continue;
            const double w = weight(a, b);
            consider_contour_point(
                grid.x[ix] + w * (grid.x[ix + 1] - grid.x[ix]),
                grid.y[iy],
                left,
                right
            );
        }
    }
    for (int iy = 0; iy + 1 < grid.ny; ++iy) {
        for (int ix = 0; ix < grid.nx; ++ix) {
            const int bottom = iy * grid.nx + ix;
            const int top = (iy + 1) * grid.nx + ix;
            const double a = interface_scalar[static_cast<std::size_t>(bottom)];
            const double b = interface_scalar[static_cast<std::size_t>(top)];
            if (!edge_crosses(a, b)) continue;
            const double w = weight(a, b);
            consider_contour_point(
                grid.x[ix],
                grid.y[iy] + w * (grid.y[iy + 1] - grid.y[iy]),
                bottom,
                top
            );
        }
    }
    if (!std::isfinite(best_score)) {
        return {std::nan(""), std::nan("")};
    }
    return {best_x, best_y};
}

double transverse_wave_y_from_pressure_gradient(
    const std::map<std::string, std::vector<double>>& c,
    const BubbleGrid& grid,
    double transverse_interface_y
)
{
    if (!c.count("p") || grid.nx < 3 || grid.ny < 3) return std::nan("");

    std::vector<double> pressure(static_cast<std::size_t>(grid.nx * grid.ny), std::nan(""));
    for (std::size_t i = 0; i < c.at("x0").size(); ++i) {
        const int ix = nearest_index(grid.x, c.at("x0")[i]);
        const int iy = nearest_index(grid.y, c.at("x1")[i]);
        if (ix < 0 || iy < 0) continue;
        pressure[static_cast<std::size_t>(iy * grid.nx + ix)] = c.at("p")[i];
    }

    const double y_threshold = std::isfinite(transverse_interface_y)
        ? std::max(0.5 * transverse_interface_y, 5.0)
        : 5.0;
    std::vector<double> row_scores(static_cast<std::size_t>(grid.ny), 0.0);
    std::vector<int> row_counts(static_cast<std::size_t>(grid.ny), 0);
    double global_best = 0.0;
    for (int iy = 1; iy + 1 < grid.ny; ++iy) {
        if (grid.y[iy] < y_threshold) continue;
        for (int ix = 1; ix + 1 < grid.nx; ++ix) {
            const int center = iy * grid.nx + ix;
            if (!grid.main_component[static_cast<std::size_t>(center)]) continue;
            const auto id = [nx = grid.nx](int y, int x) {
                return static_cast<std::size_t>(y * nx + x);
            };
            const double pc = pressure[id(iy, ix)];
            const double pxm = pressure[id(iy, ix - 1)];
            const double pxp = pressure[id(iy, ix + 1)];
            const double pym = pressure[id(iy - 1, ix)];
            const double pyp = pressure[id(iy + 1, ix)];
            if (!std::isfinite(pc) || !std::isfinite(pxm) || !std::isfinite(pxp) ||
                !std::isfinite(pym) || !std::isfinite(pyp)) {
                continue;
            }
            const double dx = grid.x[ix + 1] - grid.x[ix - 1];
            const double dy = grid.y[iy + 1] - grid.y[iy - 1];
            if (dx == 0.0 || dy == 0.0) continue;
            const double gx = (pxp - pxm) / dx;
            const double gy = (pyp - pym) / dy;
            const double score = std::sqrt(gx * gx + gy * gy);
            row_scores[static_cast<std::size_t>(iy)] += score;
            ++row_counts[static_cast<std::size_t>(iy)];
            global_best = std::max(global_best, score);
        }
    }
    if (global_best <= 0.0) return std::nan("");

    double best_row_score = 0.0;
    double best_y = std::nan("");
    for (int iy = 1; iy + 1 < grid.ny; ++iy) {
        const int count = row_counts[static_cast<std::size_t>(iy)];
        if (count * grid.dx < 2.0) continue;
        const double score = row_scores[static_cast<std::size_t>(iy)];
        if (score > best_row_score) {
            best_row_score = score;
            best_y = grid.y[iy];
        }
    }
    return best_y;
}

double jet_head_from_profile(
    const std::vector<std::pair<double, double>>& profile,
    double downstream_x,
    double upstream_x,
    bool& detected
)
{
    detected = false;
    if (std::isnan(downstream_x) || std::isnan(upstream_x) || downstream_x >= upstream_x) {
        return std::nan("");
    }
    bool in_air_run = false;
    double run_right = std::nan("");
    double run_left = std::nan("");
    for (auto it = profile.rbegin(); it != profile.rend(); ++it) {
        const double x = it->first;
        const double a = it->second;
        if (x >= upstream_x || x <= downstream_x) continue;
        if (a < 0.2) {
            if (!in_air_run) {
                in_air_run = true;
                run_right = x;
            }
            run_left = x;
        }
        else if (in_air_run) {
            if (run_right - run_left >= 2.0) {
                detected = true;
                return run_left;
            }
            return std::nan("");
        }
    }
    if (in_air_run && run_right - run_left >= 2.0) {
        detected = true;
        return run_left;
    }
    return std::nan("");
}

BubbleSnapshot bubble_snapshot(const fs::path& path, const std::string& method)
{
    BubbleSnapshot out;
    out.time = time_from_name(path);
    const auto c = read_csv_columns(path);
    if (!c.count("x0") || !c.count("x1")) return out;
    const BubbleGrid grid = build_bubble_grid(c, method);
    if (grid.helium.empty()) return out;
    const auto profile = centerline_band_profile(grid);
    const auto centerline_crossings = crossing_positions(profile, 0.5);
    if (!centerline_crossings.empty()) {
        // In the configured Quirk/Karni shock-bubble case the incident shock
        // starts to the right of the bubble and travels toward decreasing x.
        // Therefore the shock-facing/upstream interface is the rightmost
        // centerline helium crossing, not the leftmost crossing.
        out.upstream_x = centerline_crossings.back();
        out.downstream_x = centerline_crossings.front();
        out.jet_head_x = jet_head_from_profile(
            profile,
            out.downstream_x,
            out.upstream_x,
            out.jet_detected
        );
    }
    out.transverse_interface_y = transverse_extent_from_component(grid);
    const auto triple_point =
        transverse_triple_point_from_interface_pressure_gradient(c, grid);
    out.transverse_triple_point_x = triple_point.first;
    out.transverse_triple_point_y = triple_point.second;
    out.transverse_wave_y = transverse_wave_y_from_pressure_gradient(
        c,
        grid,
        out.transverse_interface_y
    );
    out.bubble_area = grid.area;
    out.tracking_confidence =
        (!std::isnan(out.upstream_x) &&
         !std::isnan(out.downstream_x) &&
         !std::isnan(out.transverse_interface_y))
            ? 1.0
            : 0.0;
    return out;
}

struct BubbleFeatureResult {
    std::map<std::string, double> summary;
    std::vector<std::map<std::string, std::string>> timeseries_rows;
};

double bubble_feature_value(const BubbleSnapshot& snap, const std::string& feature)
{
    if (feature == "upstream_interface_x") return snap.upstream_x;
    if (feature == "downstream_interface_x") return snap.downstream_x;
    if (feature == "jet_head_x") return snap.jet_head_x;
    if (feature == "transverse_interface_y") return snap.transverse_interface_y;
    if (feature == "transverse_triple_point_x") return snap.transverse_triple_point_x;
    if (feature == "transverse_triple_point_y") return snap.transverse_triple_point_y;
    if (feature == "transverse_wave_y") return snap.transverse_wave_y;
    return std::nan("");
}

double shock_bubble_mm_per_code_time_to_m_per_s()
{
    // Haas-Sturtevant/Fedkiw shock-bubble setup: 123 code-time units correspond
    // to 427 microseconds after shock-bubble encounter, while coordinates are mm.
    constexpr double seconds_per_code_time = 427.0e-6 / 123.0;
    return 1.0e-3 / seconds_per_code_time;
}

BubbleFeatureResult bubble_features(const fs::path& raw_dir, const std::string& method, const fs::path& out_json)
{
    std::vector<fs::path> files;
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
    std::vector<BubbleSnapshot> snaps;
    for (const auto& file : files) snaps.push_back(bubble_snapshot(file, method));
    std::sort(snaps.begin(), snaps.end(), [](const BubbleSnapshot& a, const BubbleSnapshot& b) {
        if (std::isnan(a.time)) return false;
        if (std::isnan(b.time)) return true;
        return a.time < b.time;
    });

    BubbleFeatureResult result;
    auto& metrics = result.summary;
    if (snaps.empty()) return result;
    metrics["snapshot_count"] = static_cast<double>(snaps.size());
    for (auto it = snaps.rbegin(); it != snaps.rend(); ++it) {
        if (!std::isnan(it->bubble_area)) {
            metrics["latest_bubble_area_mm2"] = it->bubble_area;
            metrics["latest_tracking_confidence"] = it->tracking_confidence;
            metrics["latest_jet_detected"] = it->jet_detected ? 1.0 : 0.0;
            break;
        }
    }

    const std::vector<std::string> features = {
        "upstream_interface_x",
        "downstream_interface_x",
        "jet_head_x",
        "transverse_interface_y",
        "transverse_triple_point_x",
        "transverse_triple_point_y",
        "transverse_wave_y",
    };

    double latest_time = std::nan("");
    for (const auto& feature : features) {
        bool have_previous = false;
        double previous_time_us = std::nan("");
        double previous_position_mm = std::nan("");
        double latest_position = std::nan("");
        for (const auto& snap : snaps) {
            if (std::isnan(snap.time)) continue;
            const double time_us = snap.time * (427.0 / 123.0);
            const double position = bubble_feature_value(snap, feature);
            std::string confidence = "high";
            std::string accepted = "true";
            std::string position_text;
            if (std::isnan(position)) {
                confidence = (feature == "jet_head_x" || feature == "transverse_wave_y")
                    ? "not_detected"
                    : "nan";
                accepted = "false";
            }
            else if (have_previous) {
                const double dt_s = (time_us - previous_time_us) * 1.0e-6;
                const double speed = dt_s > 0.0
                    ? std::abs((position - previous_position_mm) * 1.0e-3 / dt_s)
                    : std::numeric_limits<double>::infinity();
                if (speed > 1000.0) {
                    confidence = "speed_rejected";
                    accepted = "false";
                }
            }
            if (accepted == "true") {
                position_text = double_text(position);
                latest_position = position;
                latest_time = snap.time;
                have_previous = true;
                previous_time_us = time_us;
                previous_position_mm = position;
            }
            std::map<std::string, std::string> row = {
                {"feature", feature},
                {"time_code", double_text(snap.time)},
                {"time_us_from_initial", double_text(time_us)},
                {"position_mm", position_text},
                {"confidence_flag", confidence},
                {"accepted", accepted},
                {"jet_detected", snap.jet_detected ? "true" : "false"},
            };
            result.timeseries_rows.push_back(row);
        }

        if (!std::isnan(latest_position)) {
            metrics[feature + "_latest_position_mm"] = latest_position;
        }
    }
    if (!std::isnan(latest_time)) {
        metrics["latest_time_code"] = latest_time;
    }

    std::ostringstream json;
    json << "{\n";
    bool first = true;
    for (const auto& item : metrics) {
        if (!first) json << ",\n";
        first = false;
        json << "  \"" << item.first << "\": " << item.second;
    }
    json << "\n}\n";
    write_file(out_json, json.str());
    return result;
}

} // namespace quant
