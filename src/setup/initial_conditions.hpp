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

// [0] Check if point is inside rectangular region
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

// [1] Compute cell centre coordinate
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

// [2] Build EOS params from material config
inline EOSParams build_eos_params_from_material(
    const std::vector<MaterialConfig>& materials,
    int mat_id
)
{
    if (mat_id < 0 || mat_id >= static_cast<int>(materials.size())) {
        throw std::runtime_error("Invalid material id");
    }

    const auto& mat = materials[mat_id];

    EOSParams params{};

    if (mat.type == "ideal_gas") {
        if (mat.params.count("gamma") == 0) {
            throw std::runtime_error("Missing gamma in material");
        }
        params.gamma = mat.params.at("gamma");
    }
    else {
        throw std::runtime_error("Unsupported EOS type in material");
    }

    return params;
}

// [3] Check if point is inside explosion circle / sphere
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

// [4] Initialise data from config
template<int DIM, typename EOS>
inline void initialise_from_config(
    std::vector<Conserved<DIM>>& U,
    std::vector<int>& material_id,
    const Config<DIM>& cfg,
    const std::array<int, DIM>& N
)
{
    // total number of cells
    int total_cells = 1;
    for (int d = 0; d < DIM; ++d) {
        if (N[d] <= 0) {
            throw std::runtime_error("Invalid grid size");
        }
        total_cells *= N[d];
    }

    U.resize(total_cells);
    material_id.resize(total_cells);

    // compute dx
    std::array<double, DIM> dx{};
    for (int d = 0; d < DIM; ++d) {
        dx[d] = (cfg.domain_max[d] - cfg.domain_min[d]) / N[d];
    }

    // [4.1] Rectangular region-based initialisation
    if (cfg.initial_condition == "regions") {

        if (cfg.regions.empty()) {
            throw std::runtime_error("No regions defined in config");
        }

        std::array<int, DIM> idx{};

        for (int linear = 0; linear < total_cells; ++linear) {

            int tmp = linear;
            for (int d = DIM - 1; d >= 0; --d) {
                idx[d] = tmp % N[d];
                tmp /= N[d];
            }

            const auto x = compute_cell_center<DIM>(idx, cfg.domain_min, dx);

            bool found = false;

            for (const auto& region : cfg.regions) {
                if (point_in_region<DIM>(x, region)) {

                    Primitive<DIM> P{};
                    P.rho = region.rho;
                    P.p = region.p;
                    P.vel = region.vel;

                    const int mat_id = region.material_id;
                    const EOSParams params =
                        build_eos_params_from_material(cfg.materials, mat_id);

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

    // [4.2] Circular / spherical explosion initialisation
    if (cfg.initial_condition == "explosion") {

        if (cfg.explosion_radius <= 0.0) {
            throw std::runtime_error("Explosion radius must be positive");
        }

        std::array<int, DIM> idx{};

        for (int linear = 0; linear < total_cells; ++linear) {

            int tmp = linear;
            for (int d = DIM - 1; d >= 0; --d) {
                idx[d] = tmp % N[d];
                tmp /= N[d];
            }

            const auto x = compute_cell_center<DIM>(idx, cfg.domain_min, dx);

            const bool inside = point_in_explosion_region<DIM>(
                x,
                cfg.explosion_center,
                cfg.explosion_radius
            );

            Primitive<DIM> P{};
            int mat_id = 0;

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

            const EOSParams params = build_eos_params_from_material(cfg.materials, mat_id);

            U[linear] = prim_to_cons<DIM, EOS>(P, params);
            material_id[linear] = mat_id;
        }

        return;
    }

    throw std::runtime_error("Unsupported initial_condition type: " + cfg.initial_condition);
}


