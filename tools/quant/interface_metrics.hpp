#pragma once

// Header-only implementation units for the quantitative validation runner.
// Interface location, smearing, and local oscillation diagnostics.

#include "csv_io.hpp"

namespace quant {

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
    /*
        These diagnostics are intentionally cheap and centreline-like: they work
        on the flat CSV columns produced by both 1D and reduced comparison cases.
        Full shock-bubble geometry is handled separately by bubble_metrics.hpp.
    */
    std::map<std::string, double> metrics;
    const auto c = read_csv_columns(solution);
    if (!c.count("x0")) return metrics;
    const auto& x = c.at("x0");
    const std::string family = method_family(method);
    double position = std::nan("");
    if (family == "DIM") {
        // DIM exposes a smeared material fraction; the 5%-95% crossings give a grid-independent interface-thickness estimate.
        const std::string key = c.count("alpha1") ? "alpha1" : (c.count("alpha0") ? "alpha0" : "");
        if (!key.empty()) {
            position = crossing_position(x, c.at(key), 0.5);
            const double p05 = crossing_position(x, c.at(key), 0.05);
            const double p95 = crossing_position(x, c.at(key), 0.95);
            if (!std::isnan(position)) metrics["interface_position"] = position;
            if (!std::isnan(p05) && !std::isnan(p95)) metrics["interface_thickness"] = std::abs(p95 - p05);
        }
    }
    else if (family == "SIM" && c.count("phi0")) {
        // SIM/rGFM keeps a sharp level set, so only the zero crossing is used.
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

} // namespace quant
