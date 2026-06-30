#pragma once

#include <algorithm>
#include <array>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "src/dim/barton_dim/material.hpp"
#include "src/solid/common/config_text.hpp"

namespace dim::barton_dim {

template<int DIM>
struct Region {
    std::string shape = "box";
    std::array<double, DIM> lower{};
    std::array<double, DIM> upper{};
    std::array<double, DIM> center{};
    double radius = 0.0;
    double alpha_solid = 0.0;
    double rho_solid = 2700.0;
    double rho_fluid = 1.0;
    std::array<double, DIM> velocity{};
    bool has_solid_pressure = false;
    double p_solid = 1.0e5;
    double p_fluid = 1.0e5;
    double temperature = 300.0;
    std::array<double, 9> deformation{1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
    double equivalent_plastic_strain = 0.0;
    double damage = 0.0;
};

template<int DIM>
struct Config {
    std::array<double, DIM> domain_min{};
    std::array<double, DIM> domain_max{};
    std::array<int, DIM> cells{};
    double final_time = 0.0;
    double cfl = 0.5;
    std::vector<double> output_times{};
    Materials materials{};
    std::vector<Region<DIM>> regions{};
    std::array<std::string, DIM> bc_lo{};
    std::array<std::string, DIM> bc_hi{};
    std::string output_prefix = "barton_dim";
    std::string output_dir = "data/csv/barton_dim";
    std::string output_format = "native";
    std::string output_layout = "flat";
    int progress_interval = 0;

    Config()
    {
        bc_lo.fill("transmissive");
        bc_hi.fill("transmissive");
    }
};

inline std::vector<std::string> string_list(const std::string& value)
{
    std::string text = solid::text::trim(value);
    if (text.size() >= 2 && text.front() == '[' && text.back() == ']') {
        text = text.substr(1, text.size() - 2);
    }
    return solid::text::split_csv(text);
}

template<int DIM>
inline std::array<double, DIM> parse_vector(const std::string& value)
{
    const std::vector<double> values = solid::text::parse_numeric_list(value);
    if (static_cast<int>(values.size()) != DIM) {
        throw std::runtime_error("Barton-DIM vector has the wrong dimension: " + value);
    }
    std::array<double, DIM> result{};
    for (int d = 0; d < DIM; ++d) result[d] = values[d];
    return result;
}

template<int DIM>
inline std::array<int, DIM> parse_cells(const std::string& value)
{
    const std::vector<double> values = solid::text::parse_numeric_list(value);
    if (static_cast<int>(values.size()) != DIM) {
        throw std::runtime_error("Barton-DIM cell count has the wrong dimension: " + value);
    }
    std::array<int, DIM> result{};
    for (int d = 0; d < DIM; ++d) result[d] = static_cast<int>(values[d]);
    return result;
}

inline void parse_fluid(Materials& materials, const std::string& value)
{
    for (const std::string& entry : solid::text::split_csv(value)) {
        const auto [key, raw] = solid::text::split_key_value(entry);
        if (key == "eos") materials.fluid.kind = eos_kind_from_string(raw);
        else if (key == "gamma") materials.fluid.gamma = std::stod(raw);
        else if (key == "p_inf") materials.fluid.p_inf = std::stod(raw);
        else throw std::runtime_error("Unknown Barton-DIM fluid key: " + key);
    }
    if (materials.fluid.kind != EOSKind::ideal_gas && materials.fluid.kind != EOSKind::stiffened_gas) {
        throw std::runtime_error("Barton-DIM currently supports ideal_gas or stiffened_gas fluid EOS");
    }
}

inline void parse_solid(Materials& materials, const std::string& value)
{
    solid::barton::TensorMaterial& material = materials.solid;
    for (const std::string& entry : solid::text::split_csv(value)) {
        const auto [key, raw] = solid::text::split_key_value(entry);
        if (key == "rho0") material.rho0 = std::stod(raw);
        else if (key == "c0") material.c0 = std::stod(raw);
        else if (key == "b0") material.b0 = std::stod(raw);
        else if (key == "cv") material.cv = std::stod(raw);
        else if (key == "T0") material.T0 = std::stod(raw);
        else if (key == "alpha") material.alpha = std::stod(raw);
        else if (key == "beta") material.beta = std::stod(raw);
        else if (key == "gamma") material.gamma = std::stod(raw);
        else if (key == "sigma0") material.sigma0 = std::stod(raw);
        else if (key == "tau0") material.tau0 = std::stod(raw);
        else if (key == "n") material.relaxation_n = std::stod(raw);
        else throw std::runtime_error("Unknown Barton-DIM solid key: " + key);
    }
}

template<int DIM>
inline Region<DIM> parse_region(const std::string& value)
{
    Region<DIM> region{};
    for (const std::string& entry : solid::text::split_csv(value)) {
        const auto [key, raw] = solid::text::split_key_value(entry);
        if (key == "shape") region.shape = raw;
        else if (key == "lower") region.lower = parse_vector<DIM>(raw);
        else if (key == "upper") region.upper = parse_vector<DIM>(raw);
        else if (key == "center") region.center = parse_vector<DIM>(raw);
        else if (key == "radius") region.radius = std::stod(raw);
        else if (key == "alpha_solid") region.alpha_solid = std::stod(raw);
        else if (key == "rho_solid") region.rho_solid = std::stod(raw);
        else if (key == "rho_fluid") region.rho_fluid = std::stod(raw);
        else if (key == "vel" || key == "velocity") region.velocity = parse_vector<DIM>(raw);
        else if (key == "p_solid") {
            region.has_solid_pressure = true;
            region.p_solid = std::stod(raw);
        }
        else if (key == "p_fluid") region.p_fluid = std::stod(raw);
        else if (key == "temperature") region.temperature = std::stod(raw);
        else if (key == "F" || key == "deformation") {
            const auto values = solid::text::parse_numeric_list(raw);
            if (values.size() != 9) throw std::runtime_error("Barton-DIM deformation must contain 9 values");
            for (int q = 0; q < 9; ++q) region.deformation[q] = values[q];
        }
        else if (key == "eqps") region.equivalent_plastic_strain = std::stod(raw);
        else if (key == "damage") region.damage = std::stod(raw);
        else throw std::runtime_error("Unknown Barton-DIM region key: " + key);
    }
    return region;
}

template<int DIM>
inline Config<DIM> load_config(const std::string& filename)
{
    std::ifstream file(filename);
    if (!file) throw std::runtime_error("Cannot open Barton-DIM config: " + filename);
    Config<DIM> config{};
    std::string line;
    while (std::getline(file, line)) {
        const std::size_t comment = line.find('#');
        if (comment != std::string::npos) line.erase(comment);
        line = solid::text::trim(line);
        if (line.empty()) continue;
        const auto [key, value] = solid::text::split_key_value(line);
        if (key == "model") {
            if (value != "barton_dim") throw std::runtime_error("Barton-DIM config must set model=barton_dim");
        }
        else if (key == "dimension") {
            if (static_cast<int>(solid::text::parse_single_bracket_value(value)) != DIM) {
                throw std::runtime_error("Barton-DIM config dimension does not match APP_DIM");
            }
        }
        else if (key == "domain_min") config.domain_min = parse_vector<DIM>(value);
        else if (key == "domain_max") config.domain_max = parse_vector<DIM>(value);
        else if (key == "N" || key == "cells") config.cells = parse_cells<DIM>(value);
        else if (key == "tfinal" || key == "final_time") config.final_time = std::stod(value);
        else if (key == "cfl") config.cfl = std::stod(value);
        else if (key == "output_times") config.output_times = solid::text::parse_numeric_list(value);
        else if (key == "fluid") parse_fluid(config.materials, value);
        else if (key == "solid") parse_solid(config.materials, value);
        else if (key == "region") config.regions.push_back(parse_region<DIM>(value));
        else if (key == "bc_lo") {
            const auto values = string_list(value);
            if (static_cast<int>(values.size()) != DIM) throw std::runtime_error("Barton-DIM bc_lo has the wrong dimension");
            for (int d = 0; d < DIM; ++d) config.bc_lo[d] = values[d];
        }
        else if (key == "bc_hi") {
            const auto values = string_list(value);
            if (static_cast<int>(values.size()) != DIM) throw std::runtime_error("Barton-DIM bc_hi has the wrong dimension");
            for (int d = 0; d < DIM; ++d) config.bc_hi[d] = values[d];
        }
        else if (key == "output_prefix") config.output_prefix = value;
        else if (key == "output_dir") config.output_dir = value;
        else if (key == "output_format") config.output_format = value;
        else if (key == "output_layout") config.output_layout = value;
        else if (key == "progress_interval") config.progress_interval = std::stoi(value);
        else throw std::runtime_error("Unknown Barton-DIM config key: " + key);
    }
    for (int d = 0; d < DIM; ++d) {
        if (config.cells[d] <= 0 || config.domain_max[d] <= config.domain_min[d]) {
            throw std::runtime_error("Invalid Barton-DIM domain or mesh");
        }
    }
    if (config.final_time <= 0.0 || config.cfl <= 0.0 || config.cfl > 1.0 || config.regions.empty()) {
        throw std::runtime_error("Invalid Barton-DIM time controls or no initial regions");
    }
    if (config.output_times.empty()) config.output_times.push_back(config.final_time);
    if (config.output_format != "native" && config.output_format != "rgfm_compatible") {
        throw std::runtime_error("Barton-DIM output_format must be native or rgfm_compatible");
    }
    if (config.output_layout != "flat" && config.output_layout != "rgfm") {
        throw std::runtime_error("Barton-DIM output_layout must be flat or rgfm");
    }
    if (config.progress_interval < 0) {
        throw std::runtime_error("Barton-DIM progress_interval must be non-negative");
    }
    std::sort(config.output_times.begin(), config.output_times.end());
    return config;
}

} // namespace dim::barton_dim
