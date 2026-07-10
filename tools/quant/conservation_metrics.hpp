#pragma once

// Header-only implementation units for the quantitative validation runner.
// Conservation drift and balance metric extraction.

#include "csv_io.hpp"

namespace quant {

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

    /*
        The conservation CSV stores cumulative raw drift, open-boundary fluxes,
        and boundary-corrected balance residuals. Report tables prefer residuals
        because shock-bubble cases intentionally exchange mass through boundaries.
    */
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

    // Momentum components that start near zero can have meaningless relative drift. Keep the raw flag, but also provide trajectory-scaled values.
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
    // One compact row per conserved variable makes it easy to compare raw drift, boundary flux, and rGFM interface mismatch side by side.
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

} // namespace quant



