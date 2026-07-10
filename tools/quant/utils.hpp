#pragma once

// Header-only implementation units for the quantitative validation runner.
// Small formatting, parsing, filesystem, and CSV helpers used across the suite.

#include "types.hpp"

namespace quant {

std::string shell_quote(const std::string& text)
{
    std::string out = "'";
    for (char c : text) {
        if (c == '\'') {
            out += "'\\''";
        }
        else {
            out.push_back(c);
        }
    }
    out += "'";
    return out;
}

std::string join_command(const std::vector<std::string>& parts)
{
    std::string command;
    for (std::size_t i = 0; i < parts.size(); ++i) {
        if (i > 0) {
            command += " ";
        }
        command += shell_quote(parts[i]);
    }
    return command;
}

std::string now_stamp()
{
    const auto now = std::chrono::system_clock::now();
    const std::time_t time = std::chrono::system_clock::to_time_t(now);
    std::tm tm{};
#ifdef _WIN32
    gmtime_s(&tm, &time);
#else
    gmtime_r(&time, &tm);
#endif
    std::ostringstream out;
    out << std::put_time(&tm, "%Y%m%d_%H%M%S");
    return out.str();
}

std::string trim(const std::string& text)
{
    const auto begin = text.find_first_not_of(" \t\r\n");
    if (begin == std::string::npos) {
        return "";
    }
    const auto end = text.find_last_not_of(" \t\r\n");
    return text.substr(begin, end - begin + 1);
}

std::vector<std::string> split(const std::string& text, char delim)
{
    std::vector<std::string> parts;
    std::stringstream stream(text);
    std::string item;
    while (std::getline(stream, item, delim)) {
        parts.push_back(item);
    }
    return parts;
}

std::string lower(std::string text)
{
    std::transform(text.begin(), text.end(), text.begin(),
        [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return text;
}

std::string resolution_label(const std::vector<int>& resolution)
{
    std::ostringstream out;
    for (std::size_t i = 0; i < resolution.size(); ++i) {
        if (i > 0) {
            out << "x";
        }
        out << resolution[i];
    }
    return out.str();
}

std::string resolution_suffix(const std::vector<int>& resolution)
{
    std::ostringstream out;
    for (int value : resolution) {
        out << "_N" << value;
    }
    return out.str();
}

std::string render_resolution(const std::vector<int>& resolution)
{
    std::ostringstream out;
    out << "[";
    for (std::size_t i = 0; i < resolution.size(); ++i) {
        if (i > 0) {
            out << ", ";
        }
        out << resolution[i];
    }
    out << "]";
    return out.str();
}

std::string normalise_runtime_n(std::string value)
{
    value = trim(value);
    if (value.size() >= 2 && value.front() == '[' && value.back() == ']') {
        value = value.substr(1, value.size() - 2);
    }
    for (char& c : value) {
        if (c == ',' || std::isspace(static_cast<unsigned char>(c))) {
            c = 'x';
        }
    }
    std::string out;
    bool last_x = false;
    for (char c : value) {
        if (c == 'x') {
            if (!last_x && !out.empty()) out.push_back(c);
            last_x = true;
        }
        else {
            out.push_back(c);
            last_x = false;
        }
    }
    if (!out.empty() && out.back() == 'x') out.pop_back();
    return out;
}

std::vector<int> parse_resolution(const std::string& value)
{
    std::vector<int> out;
    std::string normal = value;
    std::replace(normal.begin(), normal.end(), ',', 'x');
    for (const auto& part : split(normal, 'x')) {
        if (!part.empty()) {
            out.push_back(std::stoi(part));
        }
    }
    return out;
}

std::string read_file(const fs::path& path)
{
    std::ifstream file(path);
    if (!file) {
        throw std::runtime_error("Cannot open " + path.string());
    }
    std::ostringstream buffer;
    buffer << file.rdbuf();
    return buffer.str();
}

void write_file(const fs::path& path, const std::string& text)
{
    fs::create_directories(path.parent_path());
    std::ofstream file(path);
    if (!file) {
        throw std::runtime_error("Cannot write " + path.string());
    }
    file << text;
}

std::string json_escape(const std::string& text)
{
    std::string out;
    for (char c : text) {
        if (c == '\\') out += "\\\\";
        else if (c == '"') out += "\\\"";
        else if (c == '\n') out += "\\n";
        else out.push_back(c);
    }
    return out;
}

std::string method_normalized(const std::string& method)
{
    const std::string text = lower(method);
    if (text == "gfm" || text == "rgfm" || text == "sim") {
        return "SIM";
    }
    if (text == "allaire" || text == "dim") {
        return "DIM";
    }
    if (text == "sm" || text == "common") {
        return "common";
    }
    return method;
}


void write_csv(const fs::path& path, const std::vector<std::map<std::string, std::string>>& rows)
{
    fs::create_directories(path.parent_path());
    std::vector<std::string> keys;
    for (const auto& row : rows) {
        for (const auto& item : row) {
            if (std::find(keys.begin(), keys.end(), item.first) == keys.end()) {
                keys.push_back(item.first);
            }
        }
    }
    std::ofstream file(path);
    for (std::size_t i = 0; i < keys.size(); ++i) {
        if (i > 0) file << ",";
        file << keys[i];
    }
    file << "\n";
    for (const auto& row : rows) {
        for (std::size_t i = 0; i < keys.size(); ++i) {
            if (i > 0) file << ",";
            const auto it = row.find(keys[i]);
            if (it != row.end()) file << it->second;
        }
        file << "\n";
    }
}

void write_csv_ordered(
    const fs::path& path,
    const std::vector<std::map<std::string, std::string>>& rows,
    const std::vector<std::string>& keys
)
{
    fs::create_directories(path.parent_path());
    std::ofstream file(path);
    for (std::size_t i = 0; i < keys.size(); ++i) {
        if (i > 0) file << ",";
        file << keys[i];
    }
    file << "\n";
    for (const auto& row : rows) {
        for (std::size_t i = 0; i < keys.size(); ++i) {
            if (i > 0) file << ",";
            const auto it = row.find(keys[i]);
            if (it != row.end()) file << it->second;
        }
        file << "\n";
    }
}

std::string double_text(double value)
{
    std::ostringstream out;
    out << std::setprecision(12) << value;
    return out.str();
}

std::optional<double> parse_double_field(const std::map<std::string, std::string>& row, const std::string& key)
{
    const auto it = row.find(key);
    if (it == row.end() || it->second.empty()) return std::nullopt;
    try {
        return std::stod(it->second);
    }
    catch (...) {
        return std::nullopt;
    }
}

std::string median_text(std::vector<double> values)
{
    if (values.empty()) return "";
    std::sort(values.begin(), values.end());
    const std::size_t mid = values.size() / 2;
    if (values.size() % 2 == 1) return double_text(values[mid]);
    return double_text(0.5 * (values[mid - 1] + values[mid]));
}

std::string read_first_cpu_model()
{
    std::ifstream file("/proc/cpuinfo");
    std::string line;
    while (std::getline(file, line)) {
        const auto pos = line.find(':');
        if (pos == std::string::npos) continue;
        if (trim(line.substr(0, pos)) == "model name") {
            return trim(line.substr(pos + 1));
        }
    }
    return "";
}

std::string read_hostname()
{
    if (const char* env = std::getenv("HOSTNAME")) {
        if (*env != '\0') return env;
    }
    std::ifstream file("/etc/hostname");
    std::string line;
    if (std::getline(file, line)) return trim(line);
    return "";
}

} // namespace quant
