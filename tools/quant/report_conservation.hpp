#pragma once

// Header-only implementation units for the quantitative validation runner.
// Builds the compact raw-drift conservation table used for report framing.

#include "metrics.hpp"

namespace quant {

struct ConservationReportTables {
    std::vector<std::map<std::string, std::string>> drift_rows;
};

inline ConservationReportTables build_conservation_reports(
    const std::vector<std::map<std::string, std::string>>& conservation_rows
)
{
    std::map<std::string, std::map<int, std::map<std::string, std::string>>> conservation_groups;
    for (const auto& row : conservation_rows) {
    if (!row.count("case_label") || !row.count("method") || !row.count("resolution")) continue;
    if (row.count("success") && row.at("success") != "true") continue;
    const std::string key = row.at("case_label") + "|" + row.at("method");
    conservation_groups[key][resolution_n(row.at("resolution"))] = row;
    }

    auto conservation_drift_value = [](
    const std::map<int, std::map<std::string, std::string>>& rows,
    int n
    ) {
    if (!rows.count(n)) {
        return std::string{};
    }
    double value = 0.0;
    bool found = false;
    for (const auto& item : rows.at(n)) {
        if (item.first.rfind("eps_", 0) != 0 ||
            item.first.find("momentum") != std::string::npos) {
            continue;
        }
        try {
            value = std::max(value, std::abs(std::stod(item.second)));
            found = true;
        }
        catch (...) {}
    }
    if (found) return double_text(value);
    if (rows.at(n).count("reportable_max_relative_drift")) {
        return rows.at(n).at("reportable_max_relative_drift");
    }
    return std::string{};
    };

    std::vector<std::map<std::string, std::string>> conservation_drift_report_rows;
    for (const auto& item : conservation_groups) {
    const auto& rows = item.second;
    if (rows.empty()) continue;
    const auto& first = rows.begin()->second;
    conservation_drift_report_rows.push_back({
        {"case", first.at("case_label")},
        {"method", first.at("method")},
        {"drift_100", conservation_drift_value(rows, 100)},
        {"drift_200", conservation_drift_value(rows, 200)},
        {"drift_400", conservation_drift_value(rows, 400)},
    });
    }


    return {conservation_drift_report_rows};
}

} // namespace quant
