#pragma once

#include <array>
#include <algorithm>
#include <cctype>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

#include "src/io/config.hpp"


// [1] String utility functions
inline std::string trim(const std::string& s)
{
    const auto start = s.find_first_not_of(" \t\r\n");
    const auto end = s.find_last_not_of(" \t\r\n");

    if (start == std::string::npos) return "";
    return s.substr(start, end - start + 1);
}

inline std::string to_lower(std::string s)
{
    std::transform(s.begin(), s.end(), s.begin(),
        [](unsigned char c){ return std::tolower(c); });
    return s;
}



// [2] Splitting
inline std::vector<std::string> split_top_level(const std::string& s, char delim)
{
    std::vector<std::string> tokens;
    std::string current;
    int depth = 0;

    for (char c : s) {
        if (c == '[') depth++;
        else if (c == ']') depth--;

        if (c == delim && depth == 0) {
            tokens.push_back(trim(current));
            current.clear();
        } else {
            current.push_back(c);
        }
    }

    if (depth != 0) {
        throw std::runtime_error("Unmatched brackets in: " + s);
    }

    tokens.push_back(trim(current));
    return tokens;
}

inline std::pair<std::string, std::string> split_key_value(const std::string& s)
{
    int depth = 0;

    for (size_t i = 0; i < s.size(); ++i) {
        if (s[i] == '[') depth++;
        else if (s[i] == ']') depth--;
        else if (s[i] == '=' && depth == 0) {
            return {
                trim(s.substr(0, i)),
                trim(s.substr(i + 1))
            };
        }
    }

    throw std::runtime_error("Invalid key=value: " + s);
}



// [3] Basic txt parser
inline std::string strip_brackets(const std::string& s)
{
    auto t = trim(s);
    if (t.front() != '[' || t.back() != ']') {
        throw std::runtime_error("Expected [ ... ]: " + s);
    }
    return trim(t.substr(1, t.size() - 2));
}

inline bool parse_bool(const std::string& s)
{
    std::string v = to_lower(trim(s));

    if (v == "true" || v == "1" || v == "yes") return true;
    if (v == "false" || v == "0" || v == "no") return false;

    throw std::runtime_error("Invalid boolean: " + s);
}

inline std::vector<double> parse_double_list(const std::string& s)
{
    auto inner = strip_brackets(s);
    auto tokens = split_top_level(inner, ',');

    std::vector<double> out;
    out.reserve(tokens.size());

    for (const auto& token : tokens) {
        out.push_back(std::stod(token));
    }

    return out;
}



// [4] Array Parser
template<int DIM>
inline std::array<double, DIM> parse_array(const std::string& s)
{
    auto inner = strip_brackets(s);
    auto tokens = split_top_level(inner, ',');

    if ((int)tokens.size() != DIM) {
        throw std::runtime_error("Wrong array size");
    }

    std::array<double, DIM> out{};
    for (int i = 0; i < DIM; ++i) {
        out[i] = std::stod(tokens[i]);
    }
    return out;
}

template<int DIM>
inline std::array<int, DIM> parse_int_array(const std::string& s)
{
    auto inner = strip_brackets(s);
    auto tokens = split_top_level(inner, ',');

    if ((int)tokens.size() != DIM) {
        throw std::runtime_error("Wrong int array size");
    }

    std::array<int, DIM> out{};
    for (int i = 0; i < DIM; ++i) {
        out[i] = std::stoi(tokens[i]);
    }
    return out;
}

template<int DIM>
inline std::array<std::string, DIM> parse_string_array(const std::string& s)
{
    auto inner = strip_brackets(s);
    auto tokens = split_top_level(inner, ',');

    if ((int)tokens.size() != DIM) {
        throw std::runtime_error("Wrong string array size");
    }

    std::array<std::string, DIM> out{};
    for (int i = 0; i < DIM; ++i) {
        out[i] = to_lower(trim(tokens[i]));
    }
    return out;
}

template<int DIM>
inline std::vector<std::array<double, DIM>> parse_array_list(const std::string& s)
{
    auto inner = strip_brackets(s);
    auto tokens = split_top_level(inner, ',');

    std::vector<std::array<double, DIM>> out;
    out.reserve(tokens.size());

    for (const auto& token : tokens) {
        out.push_back(parse_array<DIM>(token));
    }

    return out;
}



// [5] Material
inline MaterialConfig parse_material(const std::string& value)
{
    auto tokens = split_top_level(value, ',');

    if (tokens.size() < 2) {
        throw std::runtime_error("Invalid material line");
    }

    MaterialConfig m;
    m.id = std::stoi(tokens[0]);
    m.type = tokens[1];

    for (size_t i = 2; i < tokens.size(); ++i) {
        auto kv = split_key_value(tokens[i]);
        m.params[kv.first] = std::stod(kv.second);
    }

    return m;
}



// [6] Region Extraction
template<int DIM>
inline Region<DIM> parse_region(const std::string& value)
{
    auto tokens = split_top_level(value, ',');

    if (tokens.size() < 2) {
        throw std::runtime_error("Invalid region line");
    }

    Region<DIM> r;
    r.lower = parse_array<DIM>(tokens[0]);
    r.upper = parse_array<DIM>(tokens[1]);

    for (size_t i = 2; i < tokens.size(); ++i) {
        auto kv = split_key_value(tokens[i]);

        if (kv.first == "rho") r.rho = std::stod(kv.second);
        else if (kv.first == "p") r.p = std::stod(kv.second);
        else if (kv.first == "vel") r.vel = parse_array<DIM>(kv.second);
        else if (kv.first == "material") r.material_id = std::stoi(kv.second);
        else throw std::runtime_error("Unknown region key: " + kv.first);
    }

    return r;
}


// [7] Validation of Config Settings
template<int DIM>
inline void validate_config(const Config<DIM>& cfg)
{
    if (cfg.materials.empty()) {
        throw std::runtime_error("No materials defined");
    }

    if (cfg.N_list.empty()) {
        throw std::runtime_error("No grid resolution specified");
    }

    for (double output_time : cfg.output_times) {
        if (output_time <= 0.0) {
            throw std::runtime_error("output_times must be positive");
        }
    }

    if (!std::is_sorted(cfg.output_times.begin(), cfg.output_times.end())) {
        throw std::runtime_error("output_times must be sorted in ascending order");
    }

    for (std::size_t i = 1; i < cfg.output_times.size(); ++i) {
        if (cfg.output_times[i] <= cfg.output_times[i - 1]) {
            throw std::runtime_error("output_times must be strictly increasing");
        }
    }

    if (!cfg.output_times.empty()) {
        const double last_output_time = cfg.output_times.back();

        if (cfg.tfinal <= 0.0) {
            throw std::runtime_error("tfinal must be positive when output_times is set");
        }

        if (last_output_time > cfg.tfinal + 1e-14) {
            throw std::runtime_error("output_times cannot exceed tfinal");
        }
    }

    if (cfg.interface_method != "SM" &&
        cfg.interface_method != "GFM" &&
        cfg.interface_method != "DIM") {
        throw std::runtime_error("Invalid interface_method");
    }

    if (!core::model_type_matches_interface_method(cfg.model_type, cfg.interface_method)) {
        throw std::runtime_error("model_type does not match interface_method");
    }

    if (cfg.time_update != "split" &&
        cfg.time_update != "unsplit") {
        throw std::runtime_error("time_update must be split or unsplit");
    }

    if (cfg.interface_method == "GFM" && !cfg.use_level_set) {
        throw std::runtime_error("GFM requires level set");
    }

    if (cfg.level_set_advection != "normal_speed" &&
        cfg.level_set_advection != "flow") {
        throw std::runtime_error("level_set_advection must be normal_speed or flow");
    }

    if (cfg.interface_method == "DIM" && cfg.use_level_set) {
        throw std::runtime_error("DIM should not use level set");
    }

    auto valid_bc = [](const std::string& bc) {
        return bc == "transmissive" ||
               bc == "zero_gradient" ||
               bc == "outflow" ||
               bc == "reflective" ||
               bc == "reflection";
    };

    for (int d = 0; d < DIM; ++d) {
        if (!valid_bc(cfg.bc_lo[d]) || !valid_bc(cfg.bc_hi[d])) {
            throw std::runtime_error("Invalid boundary condition");
        }
    }

    if ((cfg.initial_condition == "regions" ||
         cfg.initial_condition == "planar_regions") &&
        cfg.regions.empty()) {
        throw std::runtime_error("No regions defined");
    }

    if (cfg.initial_condition == "planar_regions") {
        double normal_norm_sq = 0.0;
        for (int d = 0; d < DIM; ++d) {
            normal_norm_sq += cfg.planar_normal[d] * cfg.planar_normal[d];
        }

        if (normal_norm_sq <= 0.0) {
            throw std::runtime_error("planar_regions requires non-zero planar_normal");
        }
    }

    if (cfg.initial_condition == "explosion" ||
        cfg.initial_condition == "double_explosion") {
        if (cfg.explosion_radius <= 0.0) {
            throw std::runtime_error("Invalid explosion radius");
        }

        if (cfg.initial_condition == "double_explosion" &&
            cfg.explosion_centers.empty()) {
            throw std::runtime_error("double_explosion requires explosion_centers");
        }
    }

    if (cfg.initial_condition == "shock_bubble") {
        if (cfg.shock_axis < 0 || cfg.shock_axis >= DIM) {
            throw std::runtime_error("Invalid shock axis");
        }

        if (cfg.bubble_radius <= 0.0) {
            throw std::runtime_error("Invalid bubble radius");
        }

        if (cfg.material_left < 0 ||
            cfg.material_right < 0 ||
            cfg.material_bubble < 0) {
            throw std::runtime_error("Invalid material id in shock_bubble IC");
        }
    }
}



// [8] Load Config
template<int DIM>
inline Config<DIM> load_config(const std::string& filename)
{
    std::ifstream file(filename);
    if (!file) {
        throw std::runtime_error("Cannot open config file");
    }

    Config<DIM> cfg;
    std::string line;
    int line_number = 0;

    while (std::getline(file, line)) {
        ++line_number;

        auto comment = line.find('#');
        if (comment != std::string::npos) {
            line = line.substr(0, comment);
        }

        line = trim(line);
        if (line.empty()) continue;

        auto pos = line.find('=');
        if (pos == std::string::npos) {
            throw std::runtime_error("Invalid line: " + line);
        }

        std::string key = trim(line.substr(0, pos));
        std::string value = trim(line.substr(pos + 1));

        if (key == "dimension") {
            if (std::stoi(value) != DIM) {
                throw std::runtime_error("Dimension mismatch");
            }
        }
        else if (key == "model_type") cfg.model_type = core::parse_model_type(value);
        else if (key == "domain_min") cfg.domain_min = parse_array<DIM>(value);
        else if (key == "domain_max") cfg.domain_max = parse_array<DIM>(value);
        else if (key == "N") cfg.N_list.push_back(parse_int_array<DIM>(value));
        else if (key == "tfinal") cfg.tfinal = std::stod(value);
        else if (key == "output_times") cfg.output_times = parse_double_list(value);
        else if (key == "cfl") cfg.cfl = std::stod(value);
        else if (key == "exact_riemann") cfg.exact_riemann = parse_bool(value);
        else if (key == "output_prefix") cfg.output_prefix = value;
        else if (key == "output_dir") cfg.output_dir = value;
        else if (key == "material") cfg.materials.push_back(parse_material(value));
        else if (key == "region") cfg.regions.push_back(parse_region<DIM>(value));
        else if (key == "initial_condition") cfg.initial_condition = value;
        else if (key == "planar_normal") cfg.planar_normal = parse_array<DIM>(value);

        else if (key == "explosion_center") cfg.explosion_center = parse_array<DIM>(value);
        else if (key == "explosion_centers") cfg.explosion_centers = parse_array_list<DIM>(value);
        else if (key == "explosion_radius") cfg.explosion_radius = std::stod(value);

        else if (key == "rho_in") cfg.rho_in = std::stod(value);
        else if (key == "vel_in") cfg.vel_in = parse_array<DIM>(value);
        else if (key == "p_in") cfg.p_in = std::stod(value);
        else if (key == "material_in") cfg.material_in = std::stoi(value);

        else if (key == "rho_out") cfg.rho_out = std::stod(value);
        else if (key == "vel_out") cfg.vel_out = parse_array<DIM>(value);
        else if (key == "p_out") cfg.p_out = std::stod(value);
        else if (key == "material_out") cfg.material_out = std::stoi(value);

        else if (key == "shock_axis") cfg.shock_axis = std::stoi(value);
        else if (key == "shock_position") cfg.shock_position = std::stod(value);

        else if (key == "rho_left") cfg.rho_left = std::stod(value);
        else if (key == "vel_left") cfg.vel_left = parse_array<DIM>(value);
        else if (key == "p_left") cfg.p_left = std::stod(value);
        else if (key == "material_left") cfg.material_left = std::stoi(value);

        else if (key == "rho_right") cfg.rho_right = std::stod(value);
        else if (key == "vel_right") cfg.vel_right = parse_array<DIM>(value);
        else if (key == "p_right") cfg.p_right = std::stod(value);
        else if (key == "material_right") cfg.material_right = std::stoi(value);

        else if (key == "bubble_center") cfg.bubble_center = parse_array<DIM>(value);
        else if (key == "bubble_radius") cfg.bubble_radius = std::stod(value);
        else if (key == "rho_bubble") cfg.rho_bubble = std::stod(value);
        else if (key == "vel_bubble") cfg.vel_bubble = parse_array<DIM>(value);
        else if (key == "p_bubble") cfg.p_bubble = std::stod(value);
        else if (key == "material_bubble") cfg.material_bubble = std::stoi(value);

        else if (key == "interface_method") cfg.interface_method = value;
        else if (key == "time_update") cfg.time_update = to_lower(value);
        else if (key == "use_level_set") cfg.use_level_set = parse_bool(value);
        else if (key == "reinit_interval") cfg.reinit_interval = std::stoi(value);
        else if (key == "level_set_advection") cfg.level_set_advection = to_lower(value);
        else if (key == "interface_thickness") cfg.interface_thickness = std::stod(value);
        else if (key == "bc_lo") cfg.bc_lo = parse_string_array<DIM>(value);
        else if (key == "bc_hi") cfg.bc_hi = parse_string_array<DIM>(value);

        else {
            throw std::runtime_error("Unknown key: " + key);
        }
    }

    if (cfg.model_type == core::ModelType::Auto) {
        cfg.model_type = core::model_type_from_interface_method(cfg.interface_method);
    }
    validate_config(cfg);
    return cfg;
}
