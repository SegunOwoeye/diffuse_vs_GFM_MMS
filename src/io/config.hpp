#pragma once

#include <array>
#include <string>
#include <unordered_map>
#include <vector>

// [1] Material
struct MaterialConfig {
    int id;
    std::string type;
    std::unordered_map<std::string, double> params;
};

// [2] Region (dimension-agnostic)
template<int DIM>
struct Region {
    std::array<double, DIM> lower;
    std::array<double, DIM> upper;

    double rho;
    std::array<double, DIM> vel;
    double p;

    int material_id;
};

// [3] Config
template<int DIM>
struct Config {
    int dimension = DIM;

    std::array<double, DIM> domain_min{};
    std::array<double, DIM> domain_max{};

    std::vector<std::array<int, DIM>> N_list;

    double tfinal = 0.0;
    double cfl = 0.0;

    bool exact_riemann = false;

    std::string output_prefix = "output";
    std::string output_dir = "output";

    std::vector<MaterialConfig> materials;
    std::vector<Region<DIM>> regions;

    // [3.1] Initial condition mode
    std::string initial_condition = "regions";

    // [3.2] Explosion / implosion style circular initial condition
    std::array<double, DIM> explosion_center{};
    double explosion_radius = 0.0;

    double rho_in = 0.0;
    std::array<double, DIM> vel_in{};
    double p_in = 0.0;
    int material_in = 0;

    double rho_out = 0.0;
    std::array<double, DIM> vel_out{};
    double p_out = 0.0;
    int material_out = 0;
};