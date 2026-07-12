#pragma once

// Header-only implementation units for the quantitative validation runner.
// Builds the report-facing interface location table.

#include "metrics.hpp"

namespace quant {

struct InterfaceReportTables {
    std::vector<std::map<std::string, std::string>> location_rows;
};

inline InterfaceReportTables build_interface_reports(
    const std::vector<std::map<std::string, std::string>>& interface_rows
)
{
    std::map<std::string, std::map<int, std::map<std::string, std::string>>> interface_groups;
    std::map<std::string, std::map<int, std::map<std::string, std::string>>> interface_case_resolution_method;
    for (const auto& row : interface_rows) {
        if (!row.count("case_label") || !row.count("method") || !row.count("resolution")) continue;
        const int n = resolution_n(row.at("resolution"));
        const std::string group_key = row.at("case_label") + "|" + row.at("method");
        interface_groups[group_key][n] = row;

        const std::string method_key = row.at("case_label") + "|" + std::to_string(n);
        const std::string family = method_family(row.at("method"));
        if (family == "SIM") {
            interface_case_resolution_method[method_key][0] = row;
        }
        else if (family == "DIM") {
            interface_case_resolution_method[method_key][1] = row;
        }
    }

    auto interface_value = [](const std::map<int, std::map<std::string, std::string>>& rows,
                              int n,
                              const std::string& key) {
        if (!rows.count(n) || !rows.at(n).count(key)) return std::string{};
        return rows.at(n).at(key);
    };

    auto shift_100_400 = [&](const std::map<int, std::map<std::string, std::string>>& rows,
                             const std::string& key) {
        if (!rows.count(100) || !rows.count(400) ||
            !rows.at(100).count(key) || !rows.at(400).count(key)) {
            return std::string{};
        }
        return double_text(std::stod(rows.at(400).at(key)) - std::stod(rows.at(100).at(key)));
    };

    auto method_difference = [&](const std::string& case_label, int n) {
        const std::string key = case_label + "|" + std::to_string(n);
        if (!interface_case_resolution_method.count(key)) return std::string{};
        const auto& by_method = interface_case_resolution_method.at(key);
        if (!by_method.count(0) || !by_method.count(1)) return std::string{};
        if (!by_method.at(0).count("interface_position") ||
            !by_method.at(1).count("interface_position")) {
            return std::string{};
        }
        return double_text(std::abs(
            std::stod(by_method.at(0).at("interface_position")) -
            std::stod(by_method.at(1).at("interface_position"))
        ));
    };

    std::vector<std::map<std::string, std::string>> interface_location_report_rows;
    for (const auto& item : interface_groups) {
        const auto& rows = item.second;
        if (rows.empty()) continue;
        const auto& first = rows.begin()->second;
        const std::string case_label = first.at("case_label");

        interface_location_report_rows.push_back({
            {"case", case_label},
            {"method", first.at("method")},
            {"x_interface_100", interface_value(rows, 100, "interface_position")},
            {"x_interface_200", interface_value(rows, 200, "interface_position")},
            {"x_interface_400", interface_value(rows, 400, "interface_position")},
            {"shift_100_400", shift_100_400(rows, "interface_position")},
            {"sim_dim_difference_100", method_difference(case_label, 100)},
            {"sim_dim_difference_200", method_difference(case_label, 200)},
            {"sim_dim_difference_400", method_difference(case_label, 400)},
        });
    }

    return {interface_location_report_rows};
}

} // namespace quant
