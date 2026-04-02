#pragma once

#include <array>
#include <string>
#include <unordered_map>
#include <vector>

// [1] Material
struct MaterialConfig {
    int id = -1;
    std::string type{};
    std::unordered_map<std::string, double> params{};
};


// [2] Region (dimension-agnostic)
template<int DIM>
struct Region {
    std::array<double, DIM> lower{};
    std::array<double, DIM> upper{};

    double rho = 0.0;
    std::array<double, DIM> vel{};
    double p = 0.0;

    int material_id = -1;
};


// [3] Config
template<int DIM>
struct Config {
    int dimension = DIM;

    std::array<double, DIM> domain_min{};
    std::array<double, DIM> domain_max{};

    // Keep for convergence studies
    std::vector<std::array<int, DIM>> N_list{};

    double tfinal = 0.0;
    double cfl = 0.0;

    bool exact_riemann = false;

    std::string output_prefix = "output";
    std::string output_dir = "output";

    std::vector<MaterialConfig> materials{};
    std::vector<Region<DIM>> regions{};

    // [3.1] Initial condition mode
    std::string initial_condition = "regions"; // "regions" or "explosion"

    // [3.2] Explosion IC
    std::array<double, DIM> explosion_center{};
    double explosion_radius = 0.0;

    double rho_in = 0.0;
    std::array<double, DIM> vel_in{};
    double p_in = 0.0;
    int material_in = -1;

    double rho_out = 0.0;
    std::array<double, DIM> vel_out{};
    double p_out = 0.0;
    int material_out = -1;

    // [3.3] Interface method
    std::string interface_method = "SM"; // SM, GFM, DIM

    // [3.4] Level set (required for GFM)
    bool use_level_set = false;
    int reinit_interval = 0;

    // [3.5] Diffuse interface (DIM only)
    double interface_thickness = 0.0;
};


