#pragma once

#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>
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

    const bool has_previous_material_map =
        static_cast<int>(material_id.size()) == Ntot;

    const std::vector<int> previous_material_id = has_previous_material_map
        ? material_id
        : std::vector<int>{};

    double min_dx = grid.dx[0];
    for (int d = 1; d < DIM; ++d) {
        min_dx = std::min(min_dx, grid.dx[d]);
    }

    const double phi_tol = 1e-3 * min_dx;
    std::vector<std::vector<char>> accepted(phi_list.size());

    for (int k = 0; k < static_cast<int>(phi_list.size()); ++k) {
        const int tracked_mat = tracked_interfaces[k].negative_material_id;
        std::vector<char> visited(Ntot, 0);
        accepted[k].assign(Ntot, 0);
        std::vector<int> best_component_cells;
        int best_overlap_count = -1;
        int best_component_size = -1;

        for (int seed_id = 0; seed_id < Ntot; ++seed_id) {
            if (visited[seed_id] != 0 || phi_list[k][seed_id] >= -phi_tol) {
                continue;
            }

            bool overlaps_previous_component = !has_previous_material_map;
            int overlap_count = 0;
            std::vector<int> queue;
            std::vector<int> component_cells;

            queue.push_back(seed_id);
            visited[seed_id] = 1;

            for (int head = 0; head < static_cast<int>(queue.size()); ++head) {
                const int id = queue[head];
                component_cells.push_back(id);

                if (has_previous_material_map &&
                    previous_material_id[id] == tracked_mat) {
                    overlaps_previous_component = true;
                    overlap_count += 1;
                }

                const std::array<int, DIM> idx =
                    unflatten_index<DIM>(id, grid);

                for (int dir = 0; dir < DIM; ++dir) {
                    for (const int step : {-1, 1}) {
                        std::array<int, DIM> nb_idx{};

                        if (!try_offset_index<DIM>(idx, dir, step, grid, nb_idx)) {
                            continue;
                        }

                        const int nb_id = flatten_index<DIM>(nb_idx, grid);

                        if (visited[nb_id] != 0 ||
                            phi_list[k][nb_id] >= -phi_tol) {
                            continue;
                        }

                        visited[nb_id] = 1;
                        queue.push_back(nb_id);
                    }
                }
            }

            if (!overlaps_previous_component) {
                continue;
            }

            const int component_size = static_cast<int>(component_cells.size());

            if (overlap_count > best_overlap_count ||
                (overlap_count == best_overlap_count &&
                 component_size > best_component_size)) {
                best_overlap_count = overlap_count;
                best_component_size = component_size;
                best_component_cells = std::move(component_cells);
            }
        }

        for (const int id : best_component_cells) {
            accepted[k][id] = 1;
        }
    }

    material_id.assign(Ntot, background_material_id);

    for (int id = 0; id < Ntot; ++id) {
        bool found = false;
        bool near_interface = false;
        double best_phi = 0.0;
        int best_mat = background_material_id;

        for (int k = 0; k < static_cast<int>(phi_list.size()); ++k) {
            const double phi = phi_list[k][id];

            if (accepted[k][id] != 0) {
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

        if (near_interface && has_previous_material_map) {
            const int previous_mat = previous_material_id[id];
            bool preserve_previous = false;

            for (int k = 0; k < static_cast<int>(phi_list.size()); ++k) {
                if (previous_mat != tracked_interfaces[k].negative_material_id ||
                    std::abs(phi_list[k][id]) > phi_tol) {
                    continue;
                }

                const std::array<int, DIM> idx = unflatten_index<DIM>(id, grid);

                for (int dir = 0; dir < DIM && !preserve_previous; ++dir) {
                    for (const int step : {-1, 1}) {
                        std::array<int, DIM> nb_idx{};

                        if (!try_offset_index<DIM>(idx, dir, step, grid, nb_idx)) {
                            continue;
                        }

                        const int nb_id = flatten_index<DIM>(nb_idx, grid);

                        if (accepted[k][nb_id] != 0) {
                            preserve_previous = true;
                            break;
                        }
                    }
                }

                if (preserve_previous) {
                    break;
                }
            }

            if (preserve_previous) {
                material_id[id] = previous_mat;
            }
        }
    }
}


