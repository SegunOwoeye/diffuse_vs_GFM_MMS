#pragma once

#include <array>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace solid {

// Boundary names are parsed in configs and interpreted by each solid solver.
struct BoundaryConditions {
    std::string left = "transmissive";
    std::string right = "transmissive";
};

namespace text {

inline std::string trim(std::string s)
{
    const auto first = s.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) {
        return "";
    }
    const auto last = s.find_last_not_of(" \t\r\n");
    return s.substr(first, last - first + 1);
}

inline std::vector<std::string> split_csv(const std::string& text)
{
    std::vector<std::string> parts;
    std::string current;
    int bracket_depth = 0;

    for (char c : text) {
        if (c == '[') {
            ++bracket_depth;
        }
        else if (c == ']') {
            --bracket_depth;
        }

        if (c == ',' && bracket_depth == 0) {
            parts.push_back(trim(current));
            current.clear();
        }
        else {
            current.push_back(c);
        }
    }

    parts.push_back(trim(current));
    return parts;
}

inline std::pair<std::string, std::string> split_key_value(const std::string& text)
{
    const auto pos = text.find('=');
    if (pos == std::string::npos) {
        throw std::runtime_error("Expected key=value in: " + text);
    }
    return {trim(text.substr(0, pos)), trim(text.substr(pos + 1))};
}

inline double parse_single_bracket_value(const std::string& text)
{
    std::string s = trim(text);
    if (!s.empty() && s.front() == '[' && s.back() == ']') {
        s = trim(s.substr(1, s.size() - 2));
    }
    return std::stod(s);
}

inline std::array<double, 2> parse_pair2d(const std::string& value)
{
    const auto parts = split_csv(value);
    if (parts.size() == 2) {
        return {parse_single_bracket_value(parts[0]), parse_single_bracket_value(parts[1])};
    }

    std::string s = trim(value);
    if (!s.empty() && s.front() == '[' && s.back() == ']') {
        s = trim(s.substr(1, s.size() - 2));
        const auto comma = s.find(',');
        if (comma == std::string::npos) {
            throw std::runtime_error("Expected two values in: " + value);
        }
        return {
            std::stod(trim(s.substr(0, comma))),
            std::stod(trim(s.substr(comma + 1)))
        };
    }

    throw std::runtime_error("Expected 2D value pair in: " + value);
}

inline std::vector<double> parse_numeric_list(const std::string& value)
{
    std::string s = trim(value);
    const bool single_bracketed_list =
        !s.empty() && s.front() == '[' && s.back() == ']' &&
        s.find("],") == std::string::npos;
    if (single_bracketed_list) {
        s = trim(s.substr(1, s.size() - 2));
    }

    std::vector<double> values;
    for (const auto& part : split_csv(s)) {
        std::string token = trim(part);
        if (!token.empty() && token.front() == '[' && token.back() == ']') {
            token = trim(token.substr(1, token.size() - 2));
        }
        if (!token.empty()) {
            values.push_back(std::stod(token));
        }
    }
    if (values.empty()) {
        values.push_back(std::stod(s));
    }
    return values;
}

} // namespace text

} // namespace solid
