#pragma once

#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

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
    const std::vector<int>& phi_material_ids,
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

    if (static_cast<int>(phi_material_ids.size()) != static_cast<int>(phi_list.size())) {
        throw std::runtime_error("assign_material_ids_from_phi: phi_material_ids mismatch");
    }

    for (int k = 0; k < static_cast<int>(phi_list.size()); ++k) {
        if (static_cast<int>(phi_list[k].size()) != Ntot) {
            throw std::runtime_error("assign_material_ids_from_phi: phi_list entry size mismatch");
        }
    }

    material_id.assign(Ntot, background_material_id);

    for (int id = 0; id < Ntot; ++id) {
        bool found = false;
        double best_phi = 0.0;
        int best_mat = background_material_id;

        for (int k = 0; k < static_cast<int>(phi_list.size()); ++k) {
            const double phi = phi_list[k][id];

            if (phi < 0.0) {
                if (!found || phi < best_phi) {
                    found = true;
                    best_phi = phi;
                    best_mat = phi_material_ids[k];
                }
            }
        }

        material_id[id] = found ? best_mat : background_material_id;
    }
}


