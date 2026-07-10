#pragma once

#include <array>
#include <string>
#include <unordered_map>
#include <vector>

#include "src/core/model_type.hpp"
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
    core::ModelType model_type = core::ModelType::Auto;

    std::array<double, DIM> domain_min{};
    std::array<double, DIM> domain_max{};

    // Keep for convergence studies
    std::vector<std::array<int, DIM>> N_list{};

    double tfinal = 0.0;
    std::vector<double> output_times{};
    double cfl = 0.0;

    bool exact_riemann = false;

    std::string output_prefix = "output";
    std::string output_dir = "output";

    std::vector<MaterialConfig> materials{};
    std::vector<Region<DIM>> regions{};

    // [3.1] Initial condition mode
    std::string initial_condition = "regions"; // "regions", "planar_regions", "explosion", "double_explosion", "shock_bubble", or "coated_shock_bubble"

    // [3.1.1] Oblique planar region IC
    std::array<double, DIM> planar_normal{};

    // [3.2] Explosion IC
    std::array<double, DIM> explosion_center{};
    std::vector<std::array<double, DIM>> explosion_centers{};
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

    double film_radius = 0.0;
    double rho_film = 0.0;
    std::array<double, DIM> vel_film{};
    double p_film = 0.0;
    int material_film = -1;

    // [3.4] Interface method
    std::string interface_method = "SM"; // SM, GFM, DIM
    std::string time_update = "split"; // split, unsplit

    // [3.5] Level set (required for GFM)
    bool use_level_set = false;
    int reinit_interval = 0;
    int reinit_iterations = 10;
    std::string level_set_reinit_method = "sussman";
    std::string level_set_advection = "normal_speed";
    std::string level_set_spatial_derivative = "tvd";
    std::string level_set_component_policy = "largest_overlap";
    bool rgfm_diagnostics = false;
    int rgfm_diagnostics_interval = 1;
    std::string rgfm_star_velocity_mode = "input_mean";

    // [3.6] Diffuse interface (DIM only)
    double interface_thickness = 0.0;
    double interface_sharpness_alpha = 2.0;
    double dim_alpha_source_floor = 0.0;
    std::string dim_lambda_model = "kapila";
    int barton_solid_material = -1;
    double barton_temperature = 300.0;

    // [3.7] Physical boundary conditions
    std::array<std::string, DIM> bc_lo{};
    std::array<std::string, DIM> bc_hi{};

    Config()
    {
        bc_lo.fill("transmissive");
        bc_hi.fill("transmissive");
    }
};
