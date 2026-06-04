#pragma once

// Barton config parsing for 1D and tensor validation cases.

#include "src/solid/elastoplastic/barton/state.hpp"

// 1D plate-impact config parsing.

// Config detection and parsing for Barton 1D validation cases.

namespace solid::barton {

inline std::string detect_model(const std::string& filename)
{
    std::ifstream file(filename);
    if (!file) {
        throw std::runtime_error("Cannot open solid config: " + filename);
    }
    std::string line;
    while (std::getline(file, line)) {
        const auto comment = line.find('#');
        if (comment != std::string::npos) {
            line = line.substr(0, comment);
        }
        line = solid::text::trim(line);
        if (line.empty()) {
            continue;
        }
        const auto eq = line.find('=');
        if (eq == std::string::npos) {
            continue;
        }
        const std::string key = solid::text::trim(line.substr(0, eq));
        if (key == "model" || key == "solid_model") {
            return solid::text::trim(line.substr(eq + 1));
        }
    }
    return "";
}

inline void parse_material_value(Material& mat, const std::string& key, const std::string& value)
{
    if (key == "rho0") {
        mat.rho0 = std::stod(value);
    }
    else if (key == "G" || key == "mu" || key == "shear_modulus") {
        mat.shear_modulus = std::stod(value);
    }
    else if (key == "sigma0" || key == "yield_stress" || key == "Y") {
        mat.sigma0 = std::stod(value);
    }
    else if (key == "tau0") {
        mat.tau0 = std::stod(value);
    }
    else if (key == "n" || key == "relaxation_n") {
        mat.relaxation_n = std::stod(value);
    }
    else if (key == "p01") {
        mat.p01 = std::stod(value);
    }
    else if (key == "p02") {
        mat.p02 = std::stod(value);
    }
    else if (key == "p03") {
        mat.p03 = std::stod(value);
    }
    else if (key == "eos" || key == "pressure_model") {
        if (value != "eq49" && value != "barton_aluminium" && value != "wilkins_aluminium") {
            throw std::runtime_error("Barton 1D supports eos=eq49/barton_aluminium");
        }
    }
    else {
        throw std::runtime_error("Unknown Barton material key: " + key);
    }
}

inline Region parse_region(const std::string& value)
{
    const auto parts = solid::text::split_csv(value);
    if (parts.size() < 2) {
        throw std::runtime_error("Barton region requires lower and upper bounds");
    }
    Region region{};
    region.x_min = solid::text::parse_single_bracket_value(parts[0]);
    region.x_max = solid::text::parse_single_bracket_value(parts[1]);
    for (std::size_t i = 2; i < parts.size(); ++i) {
        const auto [key, raw] = solid::text::split_key_value(parts[i]);
        if (key == "rho") {
            region.rho = std::stod(raw);
        }
        else if (key == "u" || key == "vel") {
            region.u = std::stod(raw);
        }
        else if (key == "p" || key == "sxx") {
            continue;
        }
        else {
            throw std::runtime_error("Unknown Barton region key: " + key);
        }
    }
    return region;
}

inline Config load_config(const std::string& filename)
{
    std::ifstream file(filename);
    if (!file) {
        throw std::runtime_error("Cannot open Barton solid config: " + filename);
    }

    Config cfg{};
    std::string line;
    while (std::getline(file, line)) {
        const auto comment = line.find('#');
        if (comment != std::string::npos) {
            line = line.substr(0, comment);
        }
        line = solid::text::trim(line);
        if (line.empty()) {
            continue;
        }
        const auto [key, value] = solid::text::split_key_value(line);
        if (key == "model" || key == "solid_model") {
            if (solid::text::trim(value) != "barton") {
                throw std::runtime_error("Barton config must set model=barton");
            }
        }
        else if (key == "model_type") {
            const auto type = core::parse_model_type(value);
            if (type != core::ModelType::ElastoplasticSolid &&
                type != core::ModelType::ElasticSolid) {
                throw std::runtime_error("Barton config requires model_type=elastoplastic_solid");
            }
        }
        else if (key == "dimension") {
            if (static_cast<int>(solid::text::parse_single_bracket_value(value)) != 1) {
                throw std::runtime_error("Barton first implementation supports dimension=1");
            }
        }
        else if (key == "domain_min") {
            cfg.domain_min = solid::text::parse_single_bracket_value(value);
        }
        else if (key == "domain_max") {
            cfg.domain_max = solid::text::parse_single_bracket_value(value);
        }
        else if (key == "N" || key == "cells") {
            cfg.cells = static_cast<int>(solid::text::parse_single_bracket_value(value));
        }
        else if (key == "tfinal") {
            cfg.tfinal = std::stod(value);
        }
        else if (key == "cfl") {
            cfg.cfl = std::stod(value);
        }
        else if (key == "moving_free_surface") {
            const std::string v = solid::text::trim(value);
            cfg.moving_free_surface = (v == "true" || v == "1" || v == "yes");
        }
        else if (key == "free_surface_position") {
            cfg.free_surface_position = solid::text::parse_single_bracket_value(value);
        }
        else if (key == "output_times") {
            cfg.output_times.clear();
            for (const auto& part : solid::text::split_csv(value)) {
                cfg.output_times.push_back(solid::text::parse_single_bracket_value(part));
            }
            std::sort(cfg.output_times.begin(), cfg.output_times.end());
        }
        else if (key == "material") {
            for (const auto& part : solid::text::split_csv(value)) {
                const auto [mat_key, mat_value] = solid::text::split_key_value(part);
                parse_material_value(cfg.material, mat_key, mat_value);
            }
        }
        else if (key == "region") {
            cfg.regions.push_back(parse_region(value));
        }
        else if (key == "bc_lo") {
            cfg.bc.left = solid::text::trim(value);
        }
        else if (key == "bc_hi") {
            cfg.bc.right = solid::text::trim(value);
        }
        else if (key == "output_prefix") {
            cfg.output_prefix = value;
        }
        else if (key == "output_dir") {
            cfg.output_dir = value;
        }
        else if (key == "flattening_strength" || key == "flattening_stress_floor" ||
                 key == "release_cleanup_threshold" || key == "release_cleanup_main_stress") {
            continue;
        }
        else {
            throw std::runtime_error("Unknown Barton config key: " + key);
        }
    }

    if (cfg.cells <= 0 || cfg.domain_max <= cfg.domain_min || cfg.tfinal <= 0.0 || cfg.cfl <= 0.0) {
        throw std::runtime_error("Invalid Barton solid mesh/time controls");
    }
    if (cfg.regions.empty()) {
        throw std::runtime_error("No Barton solid regions defined");
    }
    return cfg;
}

} // namespace solid::barton

// Tensor radial-impact config parsing.

// Configuration, material constants, and EOS helpers for the Barton tensor solid solver.

namespace solid::barton {

struct TensorSolverConfig {
    std::array<double, 2> domain_min{0.0, 0.0};
    std::array<double, 2> domain_max{0.10, 0.10};
    std::array<int, 2> cells{250, 250};
    int radial_cells = 500;
    double tfinal = 10.0e-6;
    double cfl = 0.6;
    double hot_radius = 0.02;
    double hot_pressure = 10.0e9;
    double cold_temperature = 300.0;
    double hot_temperature = 600.0;
    TensorMaterial material{};
    std::string output_prefix = "barton_tensor_radial_2d";
    std::string output_dir = "data/csv/solid";
};

inline void parse_tensor_material_value(TensorMaterial& mat, const std::string& key, const std::string& value)
{
    if (key == "eos" || key == "model") {
        mat.eos = value;
    }
    else if (key == "rho0") mat.rho0 = std::stod(value);
    else if (key == "c0") mat.c0 = std::stod(value);
    else if (key == "b0") mat.b0 = std::stod(value);
    else if (key == "cv") mat.cv = std::stod(value);
    else if (key == "T0") mat.T0 = std::stod(value);
    else if (key == "alpha") mat.alpha = std::stod(value);
    else if (key == "beta") mat.beta = std::stod(value);
    else if (key == "gamma") mat.gamma = std::stod(value);
    else if (key == "sigma0" || key == "yield_stress" || key == "Y") mat.sigma0 = std::stod(value);
    else if (key == "tau0") mat.tau0 = std::stod(value);
    else if (key == "n" || key == "relaxation_n") mat.relaxation_n = std::stod(value);
    else throw std::runtime_error("Unknown Barton tensor material key: " + key);
}

inline TensorSolverConfig load_tensor_solver_config(const std::string& filename)
{
    std::ifstream file(filename);
    if (!file) throw std::runtime_error("Cannot open Barton tensor solver config: " + filename);
    TensorSolverConfig cfg{};
    std::string line;
    while (std::getline(file, line)) {
        const auto comment = line.find('#');
        if (comment != std::string::npos) line = line.substr(0, comment);
        line = solid::text::trim(line);
        if (line.empty()) continue;
        const auto [key, value] = solid::text::split_key_value(line);
        if (key == "domain_min") cfg.domain_min = solid::text::parse_pair2d(value);
        else if (key == "domain_max") cfg.domain_max = solid::text::parse_pair2d(value);
        else if (key == "N" || key == "cells") {
            const auto n = solid::text::parse_pair2d(value);
            cfg.cells = {static_cast<int>(n[0]), static_cast<int>(n[1])};
        }
        else if (key == "radial_cells") cfg.radial_cells = static_cast<int>(std::stod(value));
        else if (key == "tfinal") cfg.tfinal = std::stod(value);
        else if (key == "cfl") cfg.cfl = std::stod(value);
        else if (key == "hot_radius") cfg.hot_radius = std::stod(value);
        else if (key == "hot_pressure") cfg.hot_pressure = std::stod(value);
        else if (key == "cold_temperature") cfg.cold_temperature = std::stod(value);
        else if (key == "hot_temperature") cfg.hot_temperature = std::stod(value);
        else if (key == "output_prefix") cfg.output_prefix = value;
        else if (key == "output_dir") cfg.output_dir = value;
        else if (key == "material") {
            for (const auto& part : solid::text::split_csv(value)) {
                const auto [mat_key, mat_value] = solid::text::split_key_value(part);
                parse_tensor_material_value(cfg.material, mat_key, mat_value);
            }
        }
        else if (key == "model_type") {
            const auto type = core::parse_model_type(value);
            if (type != core::ModelType::ElastoplasticSolid &&
                type != core::ModelType::ElasticSolid) {
                throw std::runtime_error("Barton tensor solver requires model_type=elastoplastic_solid");
            }
        }
        else if (key == "model" || key == "dimension" || key == "test_case" || key == "formulation" ||
                 key == "bc_lo" || key == "bc_hi") {
            continue;
        }
        else throw std::runtime_error("Unknown Barton tensor solver config key: " + key);
    }
    if (cfg.cells[0] <= 0 || cfg.cells[1] <= 0 || cfg.radial_cells <= 0) {
        throw std::runtime_error("Barton tensor solver requires positive grid sizes");
    }
    if (cfg.domain_max[0] <= cfg.domain_min[0] || cfg.domain_max[1] <= cfg.domain_min[1]) {
        throw std::runtime_error("Barton tensor solver domain_max must exceed domain_min");
    }
    if (cfg.tfinal <= 0.0 || cfg.cfl <= 0.0 || cfg.hot_radius <= 0.0) {
        throw std::runtime_error("Invalid Barton tensor solver time/CFL/radius controls");
    }
    return cfg;
}

inline bool is_tensor_solver_config(const std::string& filename)
{
    std::ifstream file(filename);
    std::string line;
    while (std::getline(file, line)) {
        const auto comment = line.find('#');
        if (comment != std::string::npos) line = line.substr(0, comment);
        line = solid::text::trim(line);
        if (line.empty()) continue;
        const auto eq = line.find('=');
        if (eq == std::string::npos) continue;
        const std::string key = solid::text::trim(line.substr(0, eq));
        const std::string value = solid::text::trim(line.substr(eq + 1));
        if (key == "formulation") {
            return value == "tensor" || value == "tensor_2d";
        }
        if (key == "test_case") {
            return value == "radial_pressure" ||
                   value == "radial_thermal_pressure";
        }
    }
    return false;
}

} // namespace solid::barton
