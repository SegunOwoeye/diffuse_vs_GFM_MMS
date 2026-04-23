#pragma once

#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

#include "src/euler/gfm/tracked_interface.hpp"
#include "src/euler/level_set/level_set_core.hpp"

// [1] Flatten index from raw N array (solver-side helper)
template<int DIM>
inline int flatten_index(
    const std::array<int, DIM>& idx,
    const std::array<int, DIM>& N
)
{
    int id = idx[0];
    int stride = N[0];

    for (int d = 1; d < DIM; ++d) {
        if (idx[d] < 0 || idx[d] >= N[d]) {
            throw std::runtime_error("flatten_index(raw N): index out of bounds");
        }

        id += idx[d] * stride;
        stride *= N[d];
    }

    if (idx[0] < 0 || idx[0] >= N[0]) {
        throw std::runtime_error("flatten_index(raw N): index out of bounds");
    }

    return id;
}

/*
    [2] Assign material IDs from multiple level sets
    
    One level set is associated with one non-background material.
    Rule:
        - if one or more phi_k < 0, choose the most negative phi_k
        - otherwise assign background material
*/
template<int DIM>
inline void assign_material_ids_from_phi(
    const std::vector<std::vector<double>>& phi_list,
    const std::vector<TrackedInterface>& tracked_interfaces,
    int background_material_id,
    std::vector<int>& material_id,
    const LevelSetGrid<DIM>& grid
)
{
    const int Ntot = static_cast<int>(total_cells(grid));

    if (phi_list.empty()) {
        material_id.assign(Ntot, background_material_id);
        return;
    }

    if (static_cast<int>(tracked_interfaces.size()) != static_cast<int>(phi_list.size())) {
        throw std::runtime_error("assign_material_ids_from_phi: tracked_interfaces mismatch");
    }

    for (int k = 0; k < static_cast<int>(phi_list.size()); ++k) {
        if (static_cast<int>(phi_list[k].size()) != Ntot) {
            throw std::runtime_error("assign_material_ids_from_phi: phi_list entry size mismatch");
        }
    }

    if (static_cast<int>(material_id.size()) != Ntot) {
        material_id.assign(Ntot, background_material_id);
    }

    double min_dx = grid.dx[0];
    for (int d = 1; d < DIM; ++d) {
        min_dx = std::min(min_dx, grid.dx[d]);
    }

    const double phi_tol = 1e-3 * min_dx;

    for (int id = 0; id < Ntot; ++id) {
        bool found = false;
        bool near_interface = false;
        double best_phi = 0.0;
        int best_mat = background_material_id;

        for (int k = 0; k < static_cast<int>(phi_list.size()); ++k) {
            const double phi = phi_list[k][id];

            if (phi < -phi_tol) {
                if (!found || phi < best_phi) {
                    found = true;
                    best_phi = phi;
                    best_mat = tracked_interfaces[k].negative_material_id;
                }
            }
            else if (std::abs(phi) <= phi_tol) {
                near_interface = true;
            }
        }

        if (found) {
            material_id[id] = best_mat;
            continue;
        }

        if (near_interface) {
            continue;
        }

        material_id[id] = background_material_id;
    }
}


