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
    struct PlateState {
        double rho = 2700.0;
        std::array<double, 3> vel{0.0, 0.0, 0.0};
        double temperature = 300.0;
        bool has_pressure = false;
        double pressure = 1.0e5;
    };

    int dimension = 2;
    std::string test_case = "radial_pressure";
    std::array<double, 3> domain_min{0.0, 0.0, 0.0};
    std::array<double, 3> domain_max{0.10, 0.10, 0.10};
    std::array<int, 3> cells{250, 250, 1};
    int radial_cells = 500;
    double tfinal = 10.0e-6;
    double cfl = 0.6;
    double hot_radius = 0.02;
    double hot_pressure = 10.0e9;
    double cold_temperature = 300.0;
    double hot_temperature = 600.0;
    std::vector<double> output_times{};
    BoundaryConditions bc{};
    TensorMaterial material{};
    TensorMaterial left_material{};
    TensorMaterial right_material{};
    bool has_left_material = false;
    bool has_right_material = false;
    double interface_position = 0.25;
    PlateState left_state{};
    PlateState right_state{};
    bool material_points = false;
    int material_point_stride = 2;
    int material_point_output_interval = 0;
    std::string output_prefix = "barton_tensor_radial_2d";
    std::string output_dir = "data/csv/solid";
};

inline std::array<double, 3> parse_tensor_vector3(const std::string& value, double z_fallback)
{
    const auto values = solid::text::parse_numeric_list(value);
    if (values.size() == 1) {
        return {values[0], values[0], z_fallback};
    }
    if (values.size() == 2) {
        return {values[0], values[1], z_fallback};
    }
    if (values.size() == 3) {
        return {values[0], values[1], values[2]};
    }
    throw std::runtime_error("Expected one, two, or three tensor config values in: " + value);
}

inline std::array<int, 3> parse_tensor_cells3(const std::string& value, int z_fallback)
{
    const auto values = solid::text::parse_numeric_list(value);
    if (values.size() == 1) {
        const int n = static_cast<int>(values[0]);
        return {n, n, z_fallback};
    }
    if (values.size() == 2) {
        return {static_cast<int>(values[0]), static_cast<int>(values[1]), z_fallback};
    }
    if (values.size() == 3) {
        return {static_cast<int>(values[0]), static_cast<int>(values[1]), static_cast<int>(values[2])};
    }
    throw std::runtime_error("Expected one, two, or three tensor cell counts in: " + value);
}

inline bool parse_bool_value(const std::string& value)
{
    const std::string v = solid::text::trim(value);
    return v == "true" || v == "1" || v == "yes" || v == "on";
}

inline void parse_tensor_material_value(TensorMaterial& mat, const std::string& key, const std::string& value)
{
    if (key == "eos" || key == "model") {
        mat.eos = value;
    }
    else if (key == "damage_model") {
        const std::string model = solid::text::trim(value);
        if (model == "none" || model == "off") {
            mat.damage_enabled = false;
        }
        else if (model == "johnson_cook" || model == "johnson-cook" || model == "JC") {
            mat.damage_enabled = true;
        }
        else {
            throw std::runtime_error("Unknown Barton tensor damage_model: " + value);
        }
    }
    else if (key == "damage_enabled") mat.damage_enabled = parse_bool_value(value);
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
    else if (key == "D1" || key == "jc_D1") mat.jc_D1 = std::stod(value);
    else if (key == "D2" || key == "jc_D2") mat.jc_D2 = std::stod(value);
    else if (key == "D3" || key == "jc_D3") mat.jc_D3 = std::stod(value);
    else if (key == "D4" || key == "jc_D4") mat.jc_D4 = std::stod(value);
    else if (key == "D5" || key == "jc_D5") mat.jc_D5 = std::stod(value);
    else if (key == "epsdot0" || key == "reference_plastic_strain_rate") {
        mat.reference_plastic_strain_rate = std::stod(value);
    }
    else if (key == "Tm" || key == "melt_temperature") mat.melt_temperature = std::stod(value);
    else if (key == "failed_damage") mat.failed_damage = std::stod(value);
    else if (key == "residual_strength_fraction") mat.residual_strength_fraction = std::stod(value);
    else throw std::runtime_error("Unknown Barton tensor material key: " + key);
}

inline void parse_johnson_cook_damage_value(TensorMaterial& mat, const std::string& value)
{
    mat.damage_enabled = true;
    for (const auto& part : solid::text::split_csv(value)) {
        const auto [key, raw] = solid::text::split_key_value(part);
        parse_tensor_material_value(mat, key, raw);
    }
}

inline TensorSolverConfig::PlateState parse_tensor_plate_state(const std::string& value)
{
    TensorSolverConfig::PlateState state{};
    for (const auto& part : solid::text::split_csv(value)) {
        const auto [key, raw] = solid::text::split_key_value(part);
        if (key == "rho") state.rho = std::stod(raw);
        else if (key == "u" || key == "ux") state.vel[0] = std::stod(raw);
        else if (key == "v" || key == "uy") state.vel[1] = std::stod(raw);
        else if (key == "w" || key == "uz") state.vel[2] = std::stod(raw);
        else if (key == "T" || key == "temperature") state.temperature = std::stod(raw);
        else if (key == "p" || key == "pressure") {
            state.has_pressure = true;
            state.pressure = std::stod(raw);
        }
        else throw std::runtime_error("Unknown Barton tensor plate state key: " + key);
    }
    return state;
}

inline bool is_tensor_bimaterial_1d_test_case(const std::string& test_case)
{
    return test_case == "plate_impact_1d" ||
           test_case == "solid_solid_shear_1d" ||
           test_case == "solid_fluid_1d";
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
        if (key == "domain_min") cfg.domain_min = parse_tensor_vector3(value, 0.0);
        else if (key == "domain_max") cfg.domain_max = parse_tensor_vector3(value, cfg.domain_max[2]);
        else if (key == "N" || key == "cells") {
            cfg.cells = parse_tensor_cells3(value, cfg.cells[2]);
        }
        else if (key == "radial_cells") cfg.radial_cells = static_cast<int>(std::stod(value));
        else if (key == "interface" || key == "interface_x") {
            cfg.interface_position = solid::text::parse_single_bracket_value(value);
        }
        else if (key == "tfinal") cfg.tfinal = std::stod(value);
        else if (key == "cfl") cfg.cfl = std::stod(value);
        else if (key == "hot_radius") cfg.hot_radius = std::stod(value);
        else if (key == "hot_pressure") cfg.hot_pressure = std::stod(value);
        else if (key == "cold_temperature") cfg.cold_temperature = std::stod(value);
        else if (key == "hot_temperature") cfg.hot_temperature = std::stod(value);
        else if (key == "output_times") {
            cfg.output_times.clear();
            for (const auto& part : solid::text::split_csv(value)) {
                cfg.output_times.push_back(solid::text::parse_single_bracket_value(part));
            }
            std::sort(cfg.output_times.begin(), cfg.output_times.end());
        }
        else if (key == "damage_model") {
            parse_tensor_material_value(cfg.material, key, value);
        }
        else if (key == "damage_enabled") {
            cfg.material.damage_enabled = parse_bool_value(value);
        }
        else if (key == "johnson_cook_damage" || key == "damage") {
            parse_johnson_cook_damage_value(cfg.material, value);
        }
        else if (key == "jc_D1" || key == "D1" ||
                 key == "jc_D2" || key == "D2" ||
                 key == "jc_D3" || key == "D3" ||
                 key == "jc_D4" || key == "D4" ||
                 key == "jc_D5" || key == "D5" ||
                 key == "epsdot0" || key == "reference_plastic_strain_rate" ||
                 key == "Tm" || key == "melt_temperature" ||
                 key == "failed_damage" || key == "residual_strength_fraction") {
            parse_tensor_material_value(cfg.material, key, value);
        }
        else if (key == "material_points" || key == "tracers") cfg.material_points = parse_bool_value(value);
        else if (key == "material_point_stride" || key == "tracer_stride") {
            cfg.material_point_stride = static_cast<int>(std::stod(value));
        }
        else if (key == "material_point_output_interval" || key == "tracer_output_interval") {
            cfg.material_point_output_interval = static_cast<int>(std::stod(value));
        }
        else if (key == "output_prefix") cfg.output_prefix = value;
        else if (key == "output_dir") cfg.output_dir = value;
        else if (key == "left_material" || key == "material_left") {
            cfg.left_material = TensorMaterial{};
            for (const auto& part : solid::text::split_csv(value)) {
                const auto [mat_key, mat_value] = solid::text::split_key_value(part);
                parse_tensor_material_value(cfg.left_material, mat_key, mat_value);
            }
            cfg.has_left_material = true;
        }
        else if (key == "right_material" || key == "material_right") {
            cfg.right_material = TensorMaterial{};
            for (const auto& part : solid::text::split_csv(value)) {
                const auto [mat_key, mat_value] = solid::text::split_key_value(part);
                parse_tensor_material_value(cfg.right_material, mat_key, mat_value);
            }
            cfg.has_right_material = true;
        }
        else if (key == "left_state") cfg.left_state = parse_tensor_plate_state(value);
        else if (key == "right_state") cfg.right_state = parse_tensor_plate_state(value);
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
        else if (key == "dimension") {
            cfg.dimension = static_cast<int>(solid::text::parse_single_bracket_value(value));
            if (cfg.dimension == 2) {
                cfg.cells[2] = 1;
            }
        }
        else if (key == "test_case") cfg.test_case = solid::text::trim(value);
        else if (key == "bc_lo") cfg.bc.left = solid::text::trim(value);
        else if (key == "bc_hi") cfg.bc.right = solid::text::trim(value);
        else if (key == "model" || key == "formulation") {
            continue;
        }
        else throw std::runtime_error("Unknown Barton tensor solver config key: " + key);
    }
    if (is_tensor_bimaterial_1d_test_case(cfg.test_case)) {
        cfg.dimension = 1;
        cfg.cells[1] = 1;
        cfg.cells[2] = 1;
        if (!cfg.has_left_material) cfg.left_material = cfg.material;
        if (!cfg.has_right_material) cfg.right_material = cfg.material;
    }
    if (cfg.dimension != 1 && cfg.dimension != 2 && cfg.dimension != 3) {
        throw std::runtime_error("Barton tensor solver supports dimension=1, dimension=2, or dimension=3");
    }
    if (cfg.dimension == 2) {
        cfg.cells[2] = 1;
    }
    if (cfg.cells[0] <= 0 || cfg.cells[1] <= 0 || cfg.cells[2] <= 0 || cfg.radial_cells <= 0) {
        throw std::runtime_error("Barton tensor solver requires positive grid sizes");
    }
    if (cfg.domain_max[0] <= cfg.domain_min[0] ||
        (cfg.dimension >= 2 && cfg.domain_max[1] <= cfg.domain_min[1]) ||
        (cfg.dimension == 3 && cfg.domain_max[2] <= cfg.domain_min[2])) {
        throw std::runtime_error("Barton tensor solver domain_max must exceed domain_min");
    }
    if (is_tensor_bimaterial_1d_test_case(cfg.test_case) &&
        (cfg.interface_position <= cfg.domain_min[0] || cfg.interface_position >= cfg.domain_max[0])) {
        throw std::runtime_error("Barton tensor plate-impact interface must lie inside the domain");
    }
    if (cfg.tfinal <= 0.0 || cfg.cfl <= 0.0 || cfg.hot_radius <= 0.0) {
        throw std::runtime_error("Invalid Barton tensor solver time/CFL/radius controls");
    }
    if (cfg.material_point_stride <= 0 || cfg.material_point_output_interval < 0) {
        throw std::runtime_error("Invalid Barton material-point controls");
    }
    if (cfg.material.damage_enabled) {
        if (cfg.material.reference_plastic_strain_rate <= 0.0) {
            throw std::runtime_error("Johnson-Cook damage requires positive reference_plastic_strain_rate");
        }
        if (cfg.material.melt_temperature <= cfg.material.T0) {
            throw std::runtime_error("Johnson-Cook damage requires melt_temperature > T0");
        }
        if (cfg.material.failed_damage <= 0.0) {
            throw std::runtime_error("Johnson-Cook damage requires positive failed_damage");
        }
        cfg.material.residual_strength_fraction =
            std::clamp(cfg.material.residual_strength_fraction, 0.0, 1.0);
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
            return value == "tensor" || value == "tensor_2d" || value == "tensor_3d";
        }
        if (key == "test_case") {
            return value == "radial_pressure" ||
                   value == "radial_thermal_pressure" ||
                   value == "spherical_pressure" ||
                   value == "spherical_thermal_pressure" ||
                   is_tensor_bimaterial_1d_test_case(value);
        }
    }
    return false;
}

} // namespace solid::barton
