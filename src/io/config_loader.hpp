#pragma once

#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <stdexcept>
#include <algorithm>
#include <iostream>

#include "src/io/config.hpp"

// [1] Utils
inline std::string trim(const std::string& s) {
    const auto start = s.find_first_not_of(" \t\r\n");
    const auto end = s.find_last_not_of(" \t\r\n");
    return (start == std::string::npos) ? "" : s.substr(start, end - start + 1);
}

inline std::vector<std::string> split(const std::string& s, char delim) {
    std::vector<std::string> tokens;
    std::stringstream ss(s);
    std::string item;

    while (std::getline(ss, item, delim)) {
        tokens.push_back(trim(item));
    }

    return tokens;
}

// [2] Parse floating-point array of fixed dimension
template<int DIM>
inline std::array<double, DIM> parse_array(const std::string& s) {
    std::string clean = s;
    clean.erase(std::remove(clean.begin(), clean.end(), '['), clean.end());
    clean.erase(std::remove(clean.begin(), clean.end(), ']'), clean.end());

    auto tokens = split(clean, ',');

    if (tokens.size() != DIM) {
        throw std::runtime_error("parse_array: wrong dimension");
    }

    std::array<double, DIM> result{};
    for (int i = 0; i < DIM; ++i) {
        result[i] = std::stod(tokens[i]);
    }

    return result;
}

// [2.1] Parse integer list
inline std::vector<int> parse_int_list(const std::string& s) {
    std::vector<int> result;
    for (const auto& tok : split(s, ',')) {
        result.push_back(std::stoi(tok));
    }
    return result;
}

// [3] Parse material line
inline MaterialConfig parse_material(const std::string& value) {
    auto tokens = split(value, ',');

    if (tokens.size() < 2) {
        throw std::runtime_error("Invalid material line");
    }

    MaterialConfig mat;
    mat.id = std::stoi(tokens[0]);
    mat.type = tokens[1];

    for (size_t i = 2; i < tokens.size(); ++i) {
        auto kv = split(tokens[i], '=');
        if (kv.size() != 2) continue;

        mat.params[kv[0]] = std::stod(kv[1]);
    }

    return mat;
}

// [4] Parse rectangular region
template<int DIM>
inline Region<DIM> parse_region(const std::string& value) {
    auto tokens = split(value, ',');

    if (tokens.size() < 3) {
        throw std::runtime_error("Invalid region line");
    }

    Region<DIM> r{};

    r.lower = parse_array<DIM>(tokens[0]);
    r.upper = parse_array<DIM>(tokens[1]);

    for (size_t i = 2; i < tokens.size(); ++i) {
        auto kv = split(tokens[i], '=');
        if (kv.size() != 2) continue;

        const std::string key = kv[0];
        const std::string val = kv[1];

        if (key == "rho") r.rho = std::stod(val);
        else if (key == "p") r.p = std::stod(val);
        else if (key == "vel") r.vel = parse_array<DIM>(val);
        else if (key == "material") r.material_id = std::stoi(val);
    }

    return r;
}

// [5] Parse integer array of fixed dimension
template<int DIM>
inline std::array<int, DIM> parse_int_array(const std::string& s) {
    std::string clean = s;
    clean.erase(std::remove(clean.begin(), clean.end(), '['), clean.end());
    clean.erase(std::remove(clean.begin(), clean.end(), ']'), clean.end());

    auto tokens = split(clean, ',');

    if (tokens.size() != DIM) {
        throw std::runtime_error("parse_int_array: wrong dimension");
    }

    std::array<int, DIM> result{};
    for (int i = 0; i < DIM; ++i) {
        result[i] = std::stoi(tokens[i]);
    }

    return result;
}

// [6] Load config
template<int DIM>
inline Config<DIM> load_config(const std::string& filename) {
    std::ifstream file(filename);
    if (!file) {
        throw std::runtime_error("Failed to open config file");
    }

    Config<DIM> cfg;
    std::string line;

    while (std::getline(file, line)) {
        line = trim(line);

        if (line.empty() || line[0] == '#') continue;

        const size_t pos = line.find('=');
        if (pos == std::string::npos) {
            std::cerr << "Skipping invalid line: " << line << "\n";
            continue;
        }

        const std::string key = trim(line.substr(0, pos));
        const std::string value = trim(line.substr(pos + 1));

        if (key == "dimension") {
            if (std::stoi(value) != DIM) {
                throw std::runtime_error("Config dimension mismatch");
            }
        }
        else if (key == "domain_min") {
            cfg.domain_min = parse_array<DIM>(value);
        }
        else if (key == "domain_max") {
            cfg.domain_max = parse_array<DIM>(value);
        }
        else if (key == "N") {
            cfg.N_list.push_back(parse_int_array<DIM>(value));
        }
        else if (key == "tfinal") {
            cfg.tfinal = std::stod(value);
        }
        else if (key == "cfl") {
            cfg.cfl = std::stod(value);
        }
        else if (key == "exact_riemann") {
            cfg.exact_riemann = (value == "true");
        }
        else if (key == "output_prefix") {
            cfg.output_prefix = value;
        }
        else if (key == "output_dir") {
            cfg.output_dir = value;
        }
        else if (key == "material") {
            cfg.materials.push_back(parse_material(value));
        }
        else if (key == "region") {
            cfg.regions.push_back(parse_region<DIM>(value));
        }
        else if (key == "initial_condition") {
            cfg.initial_condition = value;
        }
        else if (key == "explosion_center") {
            cfg.explosion_center = parse_array<DIM>(value);
        }
        else if (key == "explosion_radius") {
            cfg.explosion_radius = std::stod(value);
        }
        else if (key == "rho_in") {
            cfg.rho_in = std::stod(value);
        }
        else if (key == "vel_in") {
            cfg.vel_in = parse_array<DIM>(value);
        }
        else if (key == "p_in") {
            cfg.p_in = std::stod(value);
        }
        else if (key == "material_in") {
            cfg.material_in = std::stoi(value);
        }
        else if (key == "rho_out") {
            cfg.rho_out = std::stod(value);
        }
        else if (key == "vel_out") {
            cfg.vel_out = parse_array<DIM>(value);
        }
        else if (key == "p_out") {
            cfg.p_out = std::stod(value);
        }
        else if (key == "material_out") {
            cfg.material_out = std::stoi(value);
        }
    }

    return cfg;
}