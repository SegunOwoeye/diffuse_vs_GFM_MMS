#pragma once

#include <array>
#include <cmath>
#include <stdexcept>
#include <vector>

#include "src/io/config.hpp"
#include "src/dim/state.hpp"
#include "src/dim/eos_params.hpp"


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


// [1] Compute cell centre
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


// [2] Check if point is inside explosion region
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


// [3] Map config material id to DIM material slot
template<int DIM>
inline int material_slot_from_id(
    const Config<DIM>& cfg,
    int material_id
)
{
    for (int k = 0; k < static_cast<int>(cfg.materials.size()); ++k) {
        if (cfg.materials[k].id == material_id) {
            return k;
        }
    }

    throw std::runtime_error(
        "material_slot_from_id: material id not found"
    );
}


// [4] Pure-material DIM state from primitive inputs
template<int DIM, int NMAT>
inline Conserved<DIM, NMAT> make_pure_material_cell(
    double rho,
    const std::array<double, DIM>& vel,
    double p,
    int mat_slot,
    const EOSParams<NMAT>& params
)
{
    if (mat_slot < 0 || mat_slot >= NMAT) {
        throw std::runtime_error("make_pure_material_cell: invalid material slot");
    }

    if (rho <= 0.0) {
        throw std::runtime_error("make_pure_material_cell: rho must be positive");
    }

    if (p <= 0.0) {
        throw std::runtime_error("make_pure_material_cell: p must be positive");
    }

    Conserved<DIM, NMAT> U{};

    // [4.1] Volume fractions
    for (int k = 0; k < NMAT; ++k) {
        U.alpha[k] = 0.0;
        U.arho[k] = 0.0;
    }

    U.alpha[mat_slot] = 1.0;
    U.arho[mat_slot] = rho;

    // [4.2] Momentum
    double v2 = 0.0;

    for (int d = 0; d < DIM; ++d) {
        U.mom[d] = rho * vel[d];
        v2 += vel[d] * vel[d];
    }

    // [4.3] Total energy
    const double gamma = params.gamma[mat_slot];
    const double e = p / ((gamma - 1.0) * rho);
    const double kinetic = 0.5 * rho * v2;

    U.E = rho * e + kinetic;

    return U;
}


// [5] Initialise from region-based config
template<int DIM, int NMAT>
inline void initialise_dim_from_regions(
    std::vector<Conserved<DIM, NMAT>>& U,
    const Config<DIM>& cfg,
    const std::array<int, DIM>& N,
    const EOSParams<NMAT>& params
)
{
    if (cfg.regions.empty()) {
        throw std::runtime_error("initialise_dim_from_regions: no regions defined");
    }

    int total_cells = 1;

    for (int d = 0; d < DIM; ++d) {
        if (N[d] <= 0) {
            throw std::runtime_error("initialise_dim_from_regions: invalid grid size");
        }

        total_cells *= N[d];
    }

    U.resize(total_cells);

    std::array<double, DIM> dx{};

    for (int d = 0; d < DIM; ++d) {
        dx[d] = (cfg.domain_max[d] - cfg.domain_min[d])
            / static_cast<double>(N[d]);
    }

    std::array<int, DIM> idx{};

    for (int linear = 0; linear < total_cells; ++linear) {
        int tmp = linear;

        for (int d = DIM - 1; d >= 0; --d) {
            idx[d] = tmp % N[d];
            tmp /= N[d];
        }

        const std::array<double, DIM> x =
            compute_cell_center<DIM>(idx, cfg.domain_min, dx);

        bool found = false;

        for (const auto& region : cfg.regions) {
            if (region.material_id < 0) {
                throw std::runtime_error(
                    "initialise_dim_from_regions: invalid region material_id"
                );
            }

            if (point_in_region<DIM>(x, region)) {
                const int mat_slot =
                    material_slot_from_id<DIM>(cfg, region.material_id);

                U[linear] = make_pure_material_cell<DIM, NMAT>(
                    region.rho,
                    region.vel,
                    region.p,
                    mat_slot,
                    params
                );

                found = true;
                break;
            }
        }

        if (!found) {
            throw std::runtime_error(
                "initialise_dim_from_regions: cell not covered by any region"
            );
        }
    }
}


// [6] Initialise from explosion config
template<int DIM, int NMAT>
inline void initialise_dim_from_explosion(
    std::vector<Conserved<DIM, NMAT>>& U,
    const Config<DIM>& cfg,
    const std::array<int, DIM>& N,
    const EOSParams<NMAT>& params
)
{
    if (cfg.explosion_radius <= 0.0) {
        throw std::runtime_error(
            "initialise_dim_from_explosion: explosion radius must be positive"
        );
    }

    int total_cells = 1;

    for (int d = 0; d < DIM; ++d) {
        if (N[d] <= 0) {
            throw std::runtime_error("initialise_dim_from_explosion: invalid grid size");
        }

        total_cells *= N[d];
    }

    U.resize(total_cells);

    std::array<double, DIM> dx{};

    for (int d = 0; d < DIM; ++d) {
        dx[d] = (cfg.domain_max[d] - cfg.domain_min[d])
            / static_cast<double>(N[d]);
    }

    std::array<int, DIM> idx{};

    for (int linear = 0; linear < total_cells; ++linear) {
        int tmp = linear;

        for (int d = DIM - 1; d >= 0; --d) {
            idx[d] = tmp % N[d];
            tmp /= N[d];
        }

        const std::array<double, DIM> x =
            compute_cell_center<DIM>(idx, cfg.domain_min, dx);

        const bool inside = point_in_explosion_region<DIM>(
            x,
            cfg.explosion_center,
            cfg.explosion_radius
        );

        double rho = 0.0;
        std::array<double, DIM> vel{};
        double p = 0.0;
        int mat_id = -1;

        if (inside) {
            rho = cfg.rho_in;
            vel = cfg.vel_in;
            p = cfg.p_in;
            mat_id = cfg.material_in;
        } else {
            rho = cfg.rho_out;
            vel = cfg.vel_out;
            p = cfg.p_out;
            mat_id = cfg.material_out;
        }

        if (mat_id < 0) {
            throw std::runtime_error(
                "initialise_dim_from_explosion: invalid material id"
            );
        }

        const int mat_slot = material_slot_from_id<DIM>(cfg, mat_id);

        U[linear] = make_pure_material_cell<DIM, NMAT>(
            rho,
            vel,
            p,
            mat_slot,
            params
        );
    }
}


// [7] Main DIM initialisation entry point
template<int DIM, int NMAT>
inline void initialise_dim_from_config(
    std::vector<Conserved<DIM, NMAT>>& U,
    const Config<DIM>& cfg,
    const std::array<int, DIM>& N,
    const EOSParams<NMAT>& params
)
{
    if (static_cast<int>(cfg.materials.size()) != NMAT) {
        throw std::runtime_error(
            "initialise_dim_from_config: cfg.materials size does not match NMAT"
        );
    }

    if (cfg.initial_condition == "regions") {
        initialise_dim_from_regions<DIM, NMAT>(
            U,
            cfg,
            N,
            params
        );
        return;
    }

    if (cfg.initial_condition == "explosion") {
        initialise_dim_from_explosion<DIM, NMAT>(
            U,
            cfg,
            N,
            params
        );
        return;
    }

    throw std::runtime_error(
        "initialise_dim_from_config: unsupported initial_condition"
    );
}