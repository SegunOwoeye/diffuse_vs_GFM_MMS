#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

#include "src/sim/gfm/tracked_interface.hpp"
#include "src/sim/level_set/level_set_core.hpp"
#include "src/math/numerical_safety.hpp"

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
        - if one or more phi_k < 0, choose the accepted field closest to its interface
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

    const double phi_tol = geometry_tolerance(min_dx);
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
                    tracked_interface_contains_negative_material(
                        tracked_interfaces[k],
                        previous_material_id[id])) {
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
        double best_abs_phi = std::numeric_limits<double>::max();
        int best_mat = background_material_id;

        for (int k = 0; k < static_cast<int>(phi_list.size()); ++k) {
            const double phi = phi_list[k][id];

            if (accepted[k][id] != 0) {
                const double abs_phi = std::abs(phi);
                if (!found || abs_phi < best_abs_phi) {
                    found = true;
                    best_abs_phi = abs_phi;
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
                if (!tracked_interface_contains_negative_material(
                        tracked_interfaces[k],
                        previous_mat) ||
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


template<int DIM>
inline int fill_small_enclosed_background_cavities(
    std::vector<int>& material_id,
    const std::vector<TrackedInterface>& tracked_interfaces,
    int background_material_id,
    const LevelSetGrid<DIM>& grid,
    int max_cavity_cells = -1
)
{
    const int Ntot = static_cast<int>(total_cells(grid));

    if (static_cast<int>(material_id.size()) != Ntot) {
        throw std::runtime_error(
            "fill_small_enclosed_background_cavities: material size mismatch"
        );
    }

    if (tracked_interfaces.empty()) {
        return 0;
    }

    if (max_cavity_cells < 0) {
        if constexpr (DIM == 2) {
            const double cell_area =
                std::max(grid.dx[0] * grid.dx[1], std::numeric_limits<double>::min());
            constexpr double max_cavity_area = 9.0;
            max_cavity_cells = std::max(
                16,
                static_cast<int>(std::ceil(max_cavity_area / cell_area))
            );
        }
        else {
            max_cavity_cells = 16;
        }
    }

    if (max_cavity_cells <= 0) {
        return 0;
    }

    std::vector<char> is_tracked_material(Ntot > 0 ? 1 : 0, 0);
    int max_material_id = background_material_id;

    for (const auto& tracked : tracked_interfaces) {
        max_material_id = std::max(max_material_id, tracked.negative_material_id);
    }

    is_tracked_material.assign(max_material_id + 1, 0);

    for (const auto& tracked : tracked_interfaces) {
        if (tracked.negative_material_id >= 0 &&
            tracked.negative_material_id < static_cast<int>(is_tracked_material.size())) {
            is_tracked_material[tracked.negative_material_id] = 1;
        }
    }

    std::vector<char> visited(Ntot, 0);
    int filled_cells = 0;

    for (int seed_id = 0; seed_id < Ntot; ++seed_id) {
        if (visited[seed_id] != 0 ||
            material_id[seed_id] != background_material_id) {
            continue;
        }

        std::vector<int> queue;
        std::vector<int> component_cells;
        bool touches_domain_boundary = false;
        int enclosing_material = -1;
        bool mixed_boundary = false;

        queue.push_back(seed_id);
        visited[seed_id] = 1;

        for (int head = 0; head < static_cast<int>(queue.size()); ++head) {
            const int id = queue[head];
            component_cells.push_back(id);
            const std::array<int, DIM> idx = unflatten_index<DIM>(id, grid);

            if (is_boundary_cell<DIM>(idx, grid)) {
                touches_domain_boundary = true;
            }

            for (int dir = 0; dir < DIM; ++dir) {
                for (const int step : {-1, 1}) {
                    std::array<int, DIM> nb_idx{};

                    if (!try_offset_index<DIM>(idx, dir, step, grid, nb_idx)) {
                        touches_domain_boundary = true;
                        continue;
                    }

                    const int nb_id = flatten_index<DIM>(nb_idx, grid);
                    const int nb_mat = material_id[nb_id];

                    if (nb_mat == background_material_id) {
                        if (visited[nb_id] == 0) {
                            visited[nb_id] = 1;
                            queue.push_back(nb_id);
                        }
                        continue;
                    }

                    const bool is_tracked =
                        nb_mat >= 0 &&
                        nb_mat < static_cast<int>(is_tracked_material.size()) &&
                        is_tracked_material[nb_mat] != 0;

                    if (!is_tracked) {
                        mixed_boundary = true;
                    }
                    else if (enclosing_material < 0) {
                        enclosing_material = nb_mat;
                    }
                    else if (enclosing_material != nb_mat) {
                        mixed_boundary = true;
                    }
                }
            }
        }

        if (touches_domain_boundary || mixed_boundary || enclosing_material < 0 ||
            static_cast<int>(component_cells.size()) > max_cavity_cells) {
            continue;
        }

        for (const int id : component_cells) {
            material_id[id] = enclosing_material;
        }

        filled_cells += static_cast<int>(component_cells.size());
    }

    return filled_cells;
}

