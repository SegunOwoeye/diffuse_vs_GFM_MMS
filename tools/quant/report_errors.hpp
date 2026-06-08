#pragma once

// Header-only implementation units for the quantitative validation runner.
// Builds normalized error evidence tables, convergence-rate tables, and the
// compact headline convergence comparison.

#include "metrics.hpp"

namespace quant {

struct ErrorReportTables {
    std::vector<std::map<std::string, std::string>> convergence_rows;
    std::vector<std::map<std::string, std::string>> compact_error_rows;
    std::vector<std::map<std::string, std::string>> report_error_rows;
    std::vector<std::map<std::string, std::string>> report_convergence_rows;
    std::vector<std::map<std::string, std::string>> compact_convergence_rows;
    std::vector<std::map<std::string, std::string>> linf_appendix_rows;
    std::vector<std::map<std::string, std::string>> convergence_order_report_rows;
};

inline bool is_report_error_variable(const std::string& variable)
{
    return variable == "pressure" || variable == "density" || variable == "velocity";
}

inline bool is_report_error_norm(const std::string& norm)
{
    return norm == "L1" || norm == "L2";
}

inline std::string compact_order(
    const std::map<int, std::map<std::string, std::string>>& rows,
    int n0,
    int n1
)
{
    if (!rows.count(n0) || !rows.count(n1)) return {};
    const double e0 = std::stod(rows.at(n0).at("normalized_error"));
    const double e1 = std::stod(rows.at(n1).at("normalized_error"));
    if (e0 <= 0.0 || e1 <= 0.0) return {};
    return double_text(std::log(e0 / e1) / std::log(static_cast<double>(n1) / n0));
}

inline std::string normalized_error_at(
    const std::map<int, std::map<std::string, std::string>>& rows,
    int n
)
{
    return rows.count(n) ? rows.at(n).at("normalized_error") : std::string{};
}

inline std::string raw_error_at(
    const std::map<int, std::map<std::string, std::string>>& rows,
    int n
)
{
    return rows.count(n) ? rows.at(n).at("raw_error") : std::string{};
}

inline std::string non_monotonic_100_400(
    const std::map<int, std::map<std::string, std::string>>& rows
)
{
    if (!rows.count(100) || !rows.count(400)) return {};
    const double e100 = std::stod(rows.at(100).at("normalized_error"));
    const double e400 = std::stod(rows.at(400).at("normalized_error"));
    return (e400 > e100) ? "true" : "false";
}

inline std::string non_monotonic_any_refinement(
    const std::map<int, std::map<std::string, std::string>>& rows
)
{
    bool has_pair = false;
    bool increased = false;
    for (const auto& pair : {std::pair<int, int>{100, 200}, std::pair<int, int>{200, 400}}) {
        if (!rows.count(pair.first) || !rows.count(pair.second)) continue;
        has_pair = true;
        const double coarse = std::stod(rows.at(pair.first).at("normalized_error"));
        const double fine = std::stod(rows.at(pair.second).at("normalized_error"));
        if (fine > coarse) increased = true;
    }
    if (!has_pair) return {};
    return increased ? "true" : "false";
}

inline ErrorReportTables build_error_reports(
    const std::vector<std::map<std::string, std::string>>& error_rows
)
{
    std::vector<std::map<std::string, std::string>> convergence_rows;
    for (std::size_t i = 0; i < error_rows.size(); ++i) {
        for (std::size_t j = 0; j < error_rows.size(); ++j) {
            const auto& coarse = error_rows[i];
            const auto& fine = error_rows[j];
            if (coarse.at("case_label") != fine.at("case_label") ||
                coarse.at("method") != fine.at("method") ||
                coarse.at("variable") != fine.at("variable") ||
                coarse.at("norm") != fine.at("norm")) {
                continue;
            }

            const int n0 = resolution_n(coarse.at("resolution"));
            const int n1 = resolution_n(fine.at("resolution"));
            if (n1 <= n0) continue;

            const double raw0 = std::stod(coarse.at("raw_error"));
            const double raw1 = std::stod(fine.at("raw_error"));
            const double e0 = std::stod(coarse.at("normalized_error"));
            const double e1 = std::stod(fine.at("normalized_error"));
            if (e0 <= 0.0 || e1 <= 0.0 || raw0 <= 0.0 || raw1 <= 0.0) continue;

            convergence_rows.push_back({
                {"case", coarse.at("case_label")},
                {"method", coarse.at("method")},
                {"variable", coarse.at("variable")},
                {"norm", coarse.at("norm")},
                {"N_coarse", std::to_string(n0)},
                {"N_fine", std::to_string(n1)},
                {"raw_error_coarse", coarse.at("raw_error")},
                {"raw_error_fine", fine.at("raw_error")},
                {"normalized_error_coarse", coarse.at("normalized_error")},
                {"normalized_error_fine", fine.at("normalized_error")},
                {"observed_order_raw", double_text(std::log(raw0 / raw1) / std::log(static_cast<double>(n1) / n0))},
                {"observed_order", double_text(std::log(e0 / e1) / std::log(static_cast<double>(n1) / n0))},
            });
        }
    }

    std::vector<std::map<std::string, std::string>> report_convergence_rows;
    for (const auto& row : convergence_rows) {
        if (!is_report_error_variable(row.at("variable")) ||
            !is_report_error_norm(row.at("norm")) ||
            (row.at("method") != "SIM" && row.at("method") != "DIM")) {
            continue;
        }

        report_convergence_rows.push_back({
            {"case", row.at("case")},
            {"method", row.at("method")},
            {"variable", row.at("variable")},
            {"norm", row.at("norm")},
            {"N_coarse", row.at("N_coarse")},
            {"N_fine", row.at("N_fine")},
            {"normalized_error_coarse", row.at("normalized_error_coarse")},
            {"normalized_error_fine", row.at("normalized_error_fine")},
            {"observed_order", row.at("observed_order")},
        });
    }

    struct OrderAccumulator {
        double sum = 0.0;
        int count = 0;

        void add(double value)
        {
            sum += value;
            ++count;
        }

        double mean() const
        {
            return (count == 0) ? 0.0 : sum / static_cast<double>(count);
        }
    };

    std::map<std::string, std::map<std::string, OrderAccumulator>> l1_order_groups;
    for (const auto& row : convergence_rows) {
        if (row.at("N_coarse") != "100" ||
            row.at("N_fine") != "400" ||
            row.at("norm") != "L1" ||
            !is_report_error_variable(row.at("variable")) ||
            (row.at("method") != "SIM" && row.at("method") != "DIM")) {
            continue;
        }

        l1_order_groups[row.at("case")][row.at("method")].add(
            std::stod(row.at("observed_order"))
        );
    }

    std::vector<std::map<std::string, std::string>> convergence_order_report_rows;
    double sim_case_mean_sum = 0.0;
    double dim_case_mean_sum = 0.0;
    int sim_case_count = 0;
    int dim_case_count = 0;
    for (const auto& item : l1_order_groups) {
        const auto sim = item.second.find("SIM");
        const auto dim = item.second.find("DIM");
        const bool has_sim = sim != item.second.end();
        const bool has_dim = dim != item.second.end();
        if (has_sim) {
            sim_case_mean_sum += sim->second.mean();
            ++sim_case_count;
        }
        if (has_dim) {
            dim_case_mean_sum += dim->second.mean();
            ++dim_case_count;
        }

        convergence_order_report_rows.push_back({
            {"case", item.first},
            {"SIM_mean_L1_order_100_400", has_sim ? double_text(sim->second.mean()) : ""},
            {"DIM_mean_L1_order_100_400", has_dim ? double_text(dim->second.mean()) : ""},
            {"SIM_variable_count", has_sim ? std::to_string(sim->second.count) : "0"},
            {"DIM_variable_count", has_dim ? std::to_string(dim->second.count) : "0"},
        });
    }

    convergence_order_report_rows.push_back({
        {"case", "Overall mean"},
        {"SIM_mean_L1_order_100_400", sim_case_count > 0 ? double_text(sim_case_mean_sum / static_cast<double>(sim_case_count)) : ""},
        {"DIM_mean_L1_order_100_400", dim_case_count > 0 ? double_text(dim_case_mean_sum / static_cast<double>(dim_case_count)) : ""},
        {"SIM_variable_count", std::to_string(sim_case_count)},
        {"DIM_variable_count", std::to_string(dim_case_count)},
    });

    std::map<std::string, std::map<int, std::map<std::string, std::string>>> compact_groups;
    for (const auto& row : error_rows) {
        const std::string key = row.at("case_label") + "|" + row.at("method") + "|" +
                                row.at("variable") + "|" + row.at("norm");
        compact_groups[key][resolution_n(row.at("resolution"))] = row;
    }

    std::vector<std::map<std::string, std::string>> compact_error_rows;
    std::vector<std::map<std::string, std::string>> report_error_rows;
    std::vector<std::map<std::string, std::string>> compact_convergence_rows;
    std::vector<std::map<std::string, std::string>> linf_appendix_rows;
    for (const auto& item : compact_groups) {
        const auto& rows = item.second;
        if (rows.empty()) continue;
        const auto& first = rows.begin()->second;

        std::map<std::string, std::string> row = {
            {"case", first.at("case_label")},
            {"method", first.at("method")},
            {"variable", first.at("variable")},
            {"norm", first.at("norm")},
            {"metric", "normalized_error"},
            {"normalized_error_100", normalized_error_at(rows, 100)},
            {"normalized_error_200", normalized_error_at(rows, 200)},
            {"normalized_error_400", normalized_error_at(rows, 400)},
            {"raw_error_100", raw_error_at(rows, 100)},
            {"raw_error_200", raw_error_at(rows, 200)},
            {"raw_error_400", raw_error_at(rows, 400)},
            {"order_100_200", compact_order(rows, 100, 200)},
            {"order_200_400", compact_order(rows, 200, 400)},
            {"order_100_400", compact_order(rows, 100, 400)},
            {"non_monotonic_100_400", non_monotonic_100_400(rows)},
            {"non_monotonic_any_refinement", non_monotonic_any_refinement(rows)},
        };
        compact_error_rows.push_back(row);

        const bool report_variable = is_report_error_variable(row.at("variable"));
        const bool report_norm = is_report_error_norm(row.at("norm"));
        if (report_variable && report_norm) {
            report_error_rows.push_back({
                {"case", row.at("case")},
                {"method", row.at("method")},
                {"variable", row.at("variable")},
                {"norm", row.at("norm")},
                {"normalized_error_100", row.at("normalized_error_100")},
                {"normalized_error_200", row.at("normalized_error_200")},
                {"normalized_error_400", row.at("normalized_error_400")},
                {"non_monotonic_100_400", row.at("non_monotonic_100_400")},
                {"non_monotonic_any_refinement", row.at("non_monotonic_any_refinement")},
            });
            compact_convergence_rows.push_back({
                {"case", row.at("case")},
                {"method", row.at("method")},
                {"variable", row.at("variable")},
                {"norm", row.at("norm")},
                {"order_100_200", row.at("order_100_200")},
                {"order_200_400", row.at("order_200_400")},
                {"order_100_400", row.at("order_100_400")},
                {"non_monotonic_100_400", row.at("non_monotonic_100_400")},
                {"non_monotonic_any_refinement", row.at("non_monotonic_any_refinement")},
            });
        }
        else if (report_variable && row.at("norm") == "Linf") {
            linf_appendix_rows.push_back({
                {"case", row.at("case")},
                {"method", row.at("method")},
                {"variable", row.at("variable")},
                {"norm", row.at("norm")},
                {"normalized_error_100", row.at("normalized_error_100")},
                {"normalized_error_200", row.at("normalized_error_200")},
                {"normalized_error_400", row.at("normalized_error_400")},
                {"order_100_200", row.at("order_100_200")},
                {"order_200_400", row.at("order_200_400")},
                {"order_100_400", row.at("order_100_400")},
                {"non_monotonic_100_400", row.at("non_monotonic_100_400")},
                {"non_monotonic_any_refinement", row.at("non_monotonic_any_refinement")},
            });
        }
    }

    return {
        convergence_rows,
        compact_error_rows,
        report_error_rows,
        report_convergence_rows,
        compact_convergence_rows,
        linf_appendix_rows,
        convergence_order_report_rows
    };
}

} // namespace quant
