#pragma once

// Header-only implementation units for the quantitative validation runner.
// CSV/key-value parsing and interpolation helpers shared by metric extractors.

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
        // Some spreadsheet/export paths leave a UTF-8 BOM on the first header.
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

    // Digitized literature CSVs use human labels; solver outputs use compact field names. Normalize here so the error code can stay field-agnostic.
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

} // namespace quant















