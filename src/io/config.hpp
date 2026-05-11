#pragma once

#include <array>
#include <string>
#include <unordered_map>
#include <vector>

#include "src/euler/primitives.hpp"

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
    std::string initial_condition = "regions"; // "regions", "explosion", or "shock_bubble"

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

    // [3.3] Shock-bubble IC
    int shock_axis = 0;
    double shock_position = 0.0;

    double rho_left = 0.0;
    std::array<double, DIM> vel_left{};
    double p_left = 0.0;
    int material_left = -1;

    double rho_right = 0.0;
    std::array<double, DIM> vel_right{};
    double p_right = 0.0;
    int material_right = -1;

    std::array<double, DIM> bubble_center{};
    double bubble_radius = 0.0;
    double rho_bubble = 0.0;
    std::array<double, DIM> vel_bubble{};
    double p_bubble = 0.0;
    int material_bubble = -1;

    // [3.4] Interface method
    std::string interface_method = "SM"; // SM, GFM, DIM

    // [3.5] Level set (required for GFM)
    bool use_level_set = false;
    int reinit_interval = 0;

    // [3.6] Diffuse interface (DIM only)
    double interface_thickness = 0.0;

    // [3.7] Physical boundary conditions
    std::array<std::string, DIM> bc_lo{};
    std::array<std::string, DIM> bc_hi{};

    Config()
    {
        bc_lo.fill("transmissive");
        bc_hi.fill("transmissive");
    }
};

