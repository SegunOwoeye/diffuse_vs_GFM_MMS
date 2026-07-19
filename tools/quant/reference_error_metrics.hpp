#pragma once

// Header-only implementation units for the quantitative validation runner.
// Exact-reference lookup and L1/L2/Linf error extraction.

#include "csv_io.hpp"

namespace quant {

std::optional<fs::path> exact_reference_path(const RunSpec& run, const fs::path& case_dir)
{
    /* 
        Toro exact solutions are emitted beside the run. Fedkiw references prefer
        generated two-material exact data, falling back to digitized literature Curves when generation has not been run.
    */
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
            {"test1", "test1_exact_raw.csv"},
            {"test2", "test2_exact_raw.csv"},
            {"test3", "test3_exact_raw.csv"},
            {"test4", "test4_exact_raw.csv"},
            {"test5", "test5_exact_raw.csv"},
        };
        const auto it = names.find(run.case_def.name);
        if (it != names.end()) {
            const fs::path generated = fs::path("data") / "exact" / "generated_multimaterial" / it->second;
            if (fs::exists(generated)) return generated;
            const std::string fallback_name =
                it->second.substr(0, it->second.size() - std::string("_raw.csv").size()) +
                ".csv";
            const fs::path digitized =
                fs::path("data") / "exact" / "fedkiw" / fallback_name;
            if (fs::exists(digitized)) return digitized;
        }
    }
    return std::nullopt;
}

std::map<std::string, std::pair<std::vector<double>, std::vector<double>>>
exact_reference_fields(const fs::path& reference)
{
    const auto columns = read_csv_columns(reference);
    const std::string x_key =
        columns.count("x0") ? "x0" : (columns.count("x") ? "x" : "");
    if (!x_key.empty()) {
        const std::map<std::string, std::string> field_columns = {
            {"rho", "rho"},
            {"u0", columns.count("u0") ? "u0" : "u"},
            {"p", "p"},
            {"e", "e"},
        };
        std::map<std::string, std::pair<std::vector<double>, std::vector<double>>> fields;
        for (const auto& item : field_columns) {
            if (columns.count(item.second)) {
                fields[item.first] = {
                    columns.at(x_key),
                    columns.at(item.second)
                };
            }
        }
        if (!fields.empty()) {
            return fields;
        }
    }

    return digitized_reference(reference);
}

std::vector<std::map<std::string, std::string>> error_rows_for(const RunSpec& run, const fs::path& solution, const fs::path& case_dir)
{
    std::vector<std::map<std::string, std::string>> rows;
    if (!fs::exists(solution)) return rows;
    const auto reference = exact_reference_path(run, case_dir);
    if (!reference.has_value()) return rows;

    const auto solution_cols = read_csv_columns(solution);
    if (!solution_cols.count("x0")) return rows;

    const auto refs = exact_reference_fields(reference.value());

    // Report labels are intentionally human-readable, while the lookup keys remain the solver CSV column names.
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
            const double ref_value = interp(
                refs.at(column).first,
                refs.at(column).second,
                xs[i]
            );
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
        const auto refs = exact_reference_fields(reference.value());
        if (refs.count(column)) {
            for (double value : refs.at(column).second) {
                if (std::isfinite(value)) {
                    scale = std::max(scale, std::abs(value));
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



} // namespace quant



