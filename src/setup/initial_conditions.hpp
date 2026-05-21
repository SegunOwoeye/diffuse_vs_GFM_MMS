#pragma once

#include <vector>
#include <array>
#include <stdexcept>
#include <cmath>

#include "src/io/config.hpp"
#include "src/euler/state.hpp"
#include "src/euler/primitives.hpp"
#include "src/euler/eos.hpp"
#include "src/euler/eos_params.hpp"


// [0] Material lookup 
inline const MaterialConfig& get_material_by_id(
    const std::vector<MaterialConfig>& materials,
    int id
)
{
    for (const auto& m : materials) {
        if (m.id == id) return m;
    }
    throw std::runtime_error("Material id not found: " + std::to_string(id));
}


// [1] Check if point is inside rectangular region
template<int DIM>
inline bool point_in_region(
    const std::array<double, DIM>& x,
    const Region<DIM>& region
)
{
    for (int d = 0; d < DIM; ++d) {
        if (x[d] < region.lower[d] || x[d] >= region.upper[d]) {
            return false;
        }
    }
    return true;
}


// [2] Compute cell centre
template<int DIM>
inline std::array<double, DIM> compute_cell_center(
    const std::array<int, DIM>& idx,
    const std::array<double, DIM>& domain_min,
    const std::array<double, DIM>& dx
)
{
    std::array<double, DIM> x{};
    for (int d = 0; d < DIM; ++d) {
        x[d] = domain_min[d] + (idx[d] + 0.5) * dx[d];
    }
    return x;
}

template<int DIM>
inline std::array<int, DIM> unflatten_raw_index(
    int linear,
    const std::array<int, DIM>& N
)
{
    std::array<int, DIM> idx{};
    int stride = 1;

    for (int d = 0; d < DIM; ++d) {
        idx[d] = (linear / stride) % N[d];
        stride *= N[d];
    }

    return idx;
}


// [3] EOS builder 
inline EOSParams build_eos_params_from_material(
    const std::vector<MaterialConfig>& materials,
    int mat_id
)
{
    const auto& mat = get_material_by_id(materials, mat_id);

    EOSParams params{};

    if (mat.type == "ideal_gas" || mat.type == "stiffened_gas") {
        if (mat.params.count("gamma") == 0) {
            throw std::runtime_error("Missing gamma in material id=" + std::to_string(mat_id));
        }
        params.gamma = mat.params.at("gamma");

        if (mat.type == "stiffened_gas") {
            if (mat.params.count("p_inf") == 0) {
                throw std::runtime_error("Missing p_inf in material id=" + std::to_string(mat_id));
            }
            params.p_inf = mat.params.at("p_inf");
        }
    }
    else {
        throw std::runtime_error("Unsupported EOS type: " + mat.type);
    }

    return params;
}

// [4] Explosion region
template<int DIM>
inline bool point_in_explosion_region(
    const std::array<double, DIM>& x,
    const std::array<double, DIM>& center,
    double radius
)
{
    double r2 = 0.0;
    for (int d = 0; d < DIM; ++d) {
        const double diff = x[d] - center[d];
        r2 += diff * diff;
    }
    return (r2 < radius * radius);
}


// [5] Initialise
template<int DIM, typename EOS>
inline void initialise_from_config(
    std::vector<Conserved<DIM>>& U,
    std::vector<int>& material_id,
    const Config<DIM>& cfg,
    const std::array<int, DIM>& N
)
{
    // total cells
    int total_cells = 1;
    for (int d = 0; d < DIM; ++d) {
        if (N[d] <= 0) {
            throw std::runtime_error("Invalid grid size");
        }
        total_cells *= N[d];
    }

    U.resize(total_cells);
    material_id.resize(total_cells);

    // dx
    std::array<double, DIM> dx{};
    for (int d = 0; d < DIM; ++d) {
        dx[d] = (cfg.domain_max[d] - cfg.domain_min[d]) / N[d];
    }

    std::array<int, DIM> idx{};

    
    // [5.1] Region Initialisation    
    if (cfg.initial_condition == "regions") {

        if (cfg.regions.empty()) {
            throw std::runtime_error("No regions defined");
        }

        for (int linear = 0; linear < total_cells; ++linear) {

            idx = unflatten_raw_index<DIM>(linear, N);

            const auto x = compute_cell_center<DIM>(idx, cfg.domain_min, dx);

            bool found = false;

            for (const auto& region : cfg.regions) {

                if (region.material_id < 0) {
                    throw std::runtime_error("Region has invalid material_id");
                }

                if (point_in_region<DIM>(x, region)) {

                    Primitive<DIM> P{};
                    P.rho = region.rho;
                    P.p = region.p;
                    P.vel = region.vel;

                    const int mat_id = region.material_id;

                    const EOSParams params = build_eos_params_from_material(cfg.materials, mat_id);

                    U[linear] = prim_to_cons<DIM, EOS>(P, params);
                    material_id[linear] = mat_id;

                    found = true;
                    break;
                }
            }

            if (!found) {
                throw std::runtime_error("Cell not covered by any region");
            }
        }

        return;
    }

    
    // [5.2] Explosion Initialisation
    if (cfg.initial_condition == "explosion" ||
        cfg.initial_condition == "double_explosion") {

        if (cfg.explosion_radius <= 0.0) {
            throw std::runtime_error("Explosion radius must be positive");
        }

        const bool use_multiple_centers = (cfg.initial_condition == "double_explosion");
        if (use_multiple_centers && cfg.explosion_centers.empty()) {
            throw std::runtime_error("double_explosion requires explosion_centers");
        }

        for (int linear = 0; linear < total_cells; ++linear) {

            idx = unflatten_raw_index<DIM>(linear, N);

            const auto x = compute_cell_center<DIM>(idx, cfg.domain_min, dx);

            bool inside = false;
            if (use_multiple_centers) {
                for (const auto& center : cfg.explosion_centers) {
                    if (point_in_explosion_region<DIM>(x, center, cfg.explosion_radius)) {
                        inside = true;
                        break;
                    }
                }
            }
            else {
                inside = point_in_explosion_region<DIM>(
                    x,
                    cfg.explosion_center,
                    cfg.explosion_radius
                );
            }

            Primitive<DIM> P{};
            int mat_id = -1;

            if (inside) {
                P.rho = cfg.rho_in;
                P.vel = cfg.vel_in;
                P.p = cfg.p_in;
                mat_id = cfg.material_in;
            }
            else {
                P.rho = cfg.rho_out;
                P.vel = cfg.vel_out;
                P.p = cfg.p_out;
                mat_id = cfg.material_out;
            }

            if (mat_id < 0) {
                throw std::runtime_error("Invalid material id in explosion IC");
            }

            const EOSParams params = build_eos_params_from_material(cfg.materials, mat_id);

            U[linear] = prim_to_cons<DIM, EOS>(P, params);
            material_id[linear] = mat_id;
        }

        return;
    }

    // [5.3] Shock-bubble Initialisation
    if (cfg.initial_condition == "shock_bubble") {

        if (cfg.shock_axis < 0 || cfg.shock_axis >= DIM) {
            throw std::runtime_error("Invalid shock axis");
        }

        if (cfg.bubble_radius <= 0.0) {
            throw std::runtime_error("Bubble radius must be positive");
        }

        for (int linear = 0; linear < total_cells; ++linear) {

            idx = unflatten_raw_index<DIM>(linear, N);

            const auto x = compute_cell_center<DIM>(idx, cfg.domain_min, dx);

            Primitive<DIM> P{};
            int mat_id = -1;

            const bool inside_bubble = point_in_explosion_region<DIM>(
                x,
                cfg.bubble_center,
                cfg.bubble_radius
            );

            if (inside_bubble) {
                P.rho = cfg.rho_bubble;
                P.vel = cfg.vel_bubble;
                P.p = cfg.p_bubble;
                mat_id = cfg.material_bubble;
            }
            else if (x[cfg.shock_axis] < cfg.shock_position) {
                P.rho = cfg.rho_left;
                P.vel = cfg.vel_left;
                P.p = cfg.p_left;
                mat_id = cfg.material_left;
            }
            else {
                P.rho = cfg.rho_right;
                P.vel = cfg.vel_right;
                P.p = cfg.p_right;
                mat_id = cfg.material_right;
            }

            if (mat_id < 0) {
                throw std::runtime_error("Invalid material id in shock_bubble IC");
            }

            const EOSParams params = build_eos_params_from_material(cfg.materials, mat_id);

            U[linear] = prim_to_cons<DIM, EOS>(P, params);
            material_id[linear] = mat_id;
        }

        return;
    }

    
    // Failure
    
    throw std::runtime_error("Unsupported initial_condition: " + cfg.initial_condition);
}
