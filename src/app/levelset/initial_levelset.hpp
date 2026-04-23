#pragma once

#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

#include "src/euler/gfm/tracked_interface.hpp"
#include "src/euler/grid/grid_utils.hpp"
#include "src/io/config.hpp"


// Initial level set container
template<int DIM>
struct InitialLevelSetData {
    // One signed-distance field per tracked material component.
    // Convention: phi < 0 means the cell belongs to tracked_interfaces[k].
    std::vector<std::vector<double>> phi_list;

    // Keeps each phi field tied to its negative-side material.
    std::vector<TrackedInterface> tracked_interfaces;

    // Material used when no tracked phi field claims a cell.
    int background_material_id = 0;
};

// [0] Helper Functions for the level set
namespace initial_level_set_detail {
    /*
    [0.1] Axis-aligned portion of a material boundary
        - The signed-distance builder measures each cell centre to these faces
    */
    template<int DIM>
    struct BoundaryFace {
        std::array<double, DIM> lower{};
        std::array<double, DIM> upper{};
    };


    template<int DIM>
    inline int total_cells_from_shape(
        const std::array<int, DIM>& N
    )
    {
        int total_cells = 1;

        for (int d = 0; d < DIM; ++d) {
            if (N[d] <= 0) {
                throw std::runtime_error("initialise_phi_data_from_regions: invalid grid size");
            }

            total_cells *= N[d];
        }

        return total_cells;
    }


    template<int DIM>
    inline std::array<double, DIM> compute_dx(
        const Config<DIM>& cfg,
        const std::array<int, DIM>& N
    )
    {
        std::array<double, DIM> dx{};

        for (int d = 0; d < DIM; ++d) {
            dx[d] = (cfg.domain_max[d] - cfg.domain_min[d]) / static_cast<double>(N[d]);
        }

        return dx;
    }


    template<int DIM>
    inline std::array<double, DIM> cell_lower_corner(
        const std::array<int, DIM>& idx,
        const std::array<double, DIM>& domain_min,
        const std::array<double, DIM>& dx
    )
    {
        std::array<double, DIM> x{};

        for (int d = 0; d < DIM; ++d) {
            x[d] = domain_min[d] + static_cast<double>(idx[d]) * dx[d];
        }

        return x;
    }


    template<int DIM>
    inline std::array<double, DIM> cell_center(
        const std::array<int, DIM>& idx,
        const std::array<double, DIM>& domain_min,
        const std::array<double, DIM>& dx
    )
    {
        std::array<double, DIM> x{};

        for (int d = 0; d < DIM; ++d) {
            x[d] = domain_min[d] + (static_cast<double>(idx[d]) + 0.5) * dx[d];
        }

        return x;
    }

    /*
    [0.2] Choose the dominant material from the realised cell map, not from
        raw config regions. 
            - This avoids treating same-material IC subregions as separate GFM phases.
    */
    template<int DIM>
    inline int choose_background_material_id(
        const std::vector<int>& material_id,
        int num_materials
    )
    {
        
        if (material_id.empty()) {
            throw std::runtime_error("initialise_phi_data_from_regions: empty material map");
        }

        std::vector<int> counts(num_materials, 0);

        for (const int mat : material_id) {
            if (mat < 0 || mat >= num_materials) {
                throw std::runtime_error("initialise_phi_data_from_regions: invalid material id in map");
            }

            counts[mat] += 1;
        }

        int best_mat = 0;
        int best_count = counts[0];

        for (int mat = 1; mat < num_materials; ++mat) {
            if (counts[mat] > best_count) {
                best_count = counts[mat];
                best_mat = mat;
            }
        }

        return best_mat;
    }

    /*
    [0.3] Flood-fill one connected component of a non-background material
        - Each component gets its own phi field, so disconnected regions of the
          same material are tracked independently.
    */
    template<int DIM>
    inline std::vector<int> extract_component_cells(
        int seed_id,
        int target_material,
        const std::vector<int>& material_id,
        const LevelSetGrid<DIM>& grid,
        std::vector<char>& visited
    )
    {
        
        std::vector<int> queue;
        std::vector<int> component_cells;

        queue.push_back(seed_id);
        visited[seed_id] = 1;

        for (int head = 0; head < static_cast<int>(queue.size()); ++head) {
            const int id = queue[head];
            component_cells.push_back(id);

            const std::array<int, DIM> idx = unflatten_index<DIM>(id, grid);

            for (int dir = 0; dir < DIM; ++dir) {
                for (int step : {-1, 1}) {
                    std::array<int, DIM> nb_idx{};

                    if (!try_offset_index<DIM>(idx, dir, step, grid, nb_idx)) {
                        continue;
                    }

                    const int nb_id = flatten_index<DIM>(nb_idx, grid);

                    if (visited[nb_id] || material_id[nb_id] != target_material) {
                        continue;
                    }

                    visited[nb_id] = 1;
                    queue.push_back(nb_id);
                }
            }
        }

        return component_cells;
    }

    /*
    [0.4] A boundary face is emitted only where the component touches another
        material inside the domain. 
        - Physical domain boundaries are skipped so they do not become artificial material interfaces.
    */
    template<int DIM>
    inline std::vector<BoundaryFace<DIM>> build_component_boundary_faces(
        const std::vector<int>& component_cells,
        const std::vector<char>& component_mask,
        const LevelSetGrid<DIM>& grid,
        const std::array<double, DIM>& domain_min,
        const std::array<double, DIM>& dx
    )
    {
        std::vector<BoundaryFace<DIM>> faces;

        for (const int id : component_cells) {
            const std::array<int, DIM> idx = unflatten_index<DIM>(id, grid);
            const std::array<double, DIM> lower_corner = cell_lower_corner<DIM>(idx, domain_min, dx);

            for (int dir = 0; dir < DIM; ++dir) {
                for (int side = 0; side < 2; ++side) {
                    const int step = (side == 0) ? -1 : 1;
                    std::array<int, DIM> nb_idx{};

                    if (!try_offset_index<DIM>(idx, dir, step, grid, nb_idx)) {
                        continue;
                    }

                    const int nb_id = flatten_index<DIM>(nb_idx, grid);

                    if (component_mask[nb_id] != 0) {
                        continue;
                    }

                    BoundaryFace<DIM> face{};

                    for (int d = 0; d < DIM; ++d) {
                        if (d == dir) {
                            const double coord = lower_corner[d] + (side == 0 ? 0.0 : dx[d]);
                            face.lower[d] = coord;
                            face.upper[d] = coord;
                        }
                        else {
                            face.lower[d] = lower_corner[d];
                            face.upper[d] = lower_corner[d] + dx[d];
                        }
                    }

                    faces.push_back(face);
                }
            }
        }

        return faces;
    }

    /*
    [0.5] Point to axis-aligned face distance 
        - In 1D this reduces to distance from a point to an interface location
        - In 2D/3D it also handles face extents.
    */
    template<int DIM>
    inline double distance_to_face(
        const std::array<double, DIM>& x,
        const BoundaryFace<DIM>& face
    )
    {
        double dist_sq = 0.0;

        for (int d = 0; d < DIM; ++d) {
            double delta = 0.0;

            if (x[d] < face.lower[d]) {
                delta = face.lower[d] - x[d];
            }
            else if (x[d] > face.upper[d]) {
                delta = x[d] - face.upper[d];
            }

            dist_sq += delta * delta;
        }

        return std::sqrt(dist_sq);
    }

    /*
    [0.6] Build phi from material topology 
        This is the key for GFM used downstream:
           - phi < 0 inside the tracked component
           - phi > 0 outside the tracked component
    */
    template<int DIM>
    inline std::vector<double> build_signed_distance_field(
        const std::vector<char>& component_mask,
        const std::vector<BoundaryFace<DIM>>& faces,
        const std::array<int, DIM>& N,
        const std::array<double, DIM>& domain_min,
        const std::array<double, DIM>& dx
    )
    {
        const int total_cells = total_cells_from_shape<DIM>(N);
        std::vector<double> phi(total_cells, 0.0);

        if (faces.empty()) {
            throw std::runtime_error("initialise_phi_data_from_regions: tracked component has no material boundary");
        }

        for (int id = 0; id < total_cells; ++id) {
            const std::array<int, DIM> idx = unflatten_index<DIM>(
                id,
                make_level_set_grid<DIM>(N, dx)
            );
            const std::array<double, DIM> x = cell_center<DIM>(idx, domain_min, dx);

            double best_dist = std::numeric_limits<double>::max();

            for (const auto& face : faces) {
                best_dist = std::min(best_dist, distance_to_face<DIM>(x, face));
            }

            phi[id] = (component_mask[id] != 0) ? -best_dist : best_dist;
        }

        return phi;
    }

} 

/*
[1] Build initial level sets from region ICs for MM GFM.

    Regions still initialise the conserved state, but GFM interfaces are derived
    from the resulting material map. This prevents internal jumps between two
    regions of the same material from becoming fake tracked interfaces.
*/
template<int DIM>
inline InitialLevelSetData<DIM> initialise_phi_data_from_regions(
    const Config<DIM>& cfg,
    const std::array<int, DIM>& N,
    const std::vector<int>& material_id
)
{
    InitialLevelSetData<DIM> out{};

    if (cfg.interface_method != "GFM") {
        return out;
    }

    if (!cfg.use_level_set) {
        throw std::runtime_error("initialise_phi_data_from_regions: GFM requires use_level_set = true");
    }

    if (cfg.initial_condition != "regions") {
        throw std::runtime_error(
            "initialise_phi_data_from_regions: current multimaterial GFM initialisation only supports region ICs"
        );
    }

    if (cfg.regions.size() < 2) {
        throw std::runtime_error(
            "initialise_phi_data_from_regions: GFM requires at least 2 regions/materials"
        );
    }

    const int total_cells = initial_level_set_detail::total_cells_from_shape<DIM>(N);

    if (static_cast<int>(material_id.size()) != total_cells) {
        throw std::runtime_error("initialise_phi_data_from_regions: material map size mismatch");
    }

    const std::array<double, DIM> dx = initial_level_set_detail::compute_dx<DIM>(cfg, N);
    const LevelSetGrid<DIM> grid = make_level_set_grid<DIM>(N, dx);

    out.background_material_id = initial_level_set_detail::choose_background_material_id<DIM>(
        material_id,
        static_cast<int>(cfg.materials.size())
    );

    // Visit each non-background connected component exactly once
    std::vector<char> visited(total_cells, 0);
    int next_component_id = 0;

    for (int seed_id = 0; seed_id < total_cells; ++seed_id) {
        const int mat = material_id[seed_id];

        // Background cells are represented implicitly by positive phi values
        if (mat == out.background_material_id || visited[seed_id] != 0) {
            continue;
        }

        const std::vector<int> component_cells =
            initial_level_set_detail::extract_component_cells<DIM>(
                seed_id,
                mat,
                material_id,
                grid,
                visited
            );

        std::vector<char> component_mask(total_cells, 0);

        for (const int id : component_cells) {
            component_mask[id] = 1;
        }

        const auto faces = initial_level_set_detail::build_component_boundary_faces<DIM>(
            component_cells,
            component_mask,
            grid,
            cfg.domain_min,
            dx
        );

        if (faces.empty()) {
            continue;
        }

        /* Store the signed-distance field and the data needed later by
        material reassignment and mixed-material face selection
        */
        out.phi_list.push_back(
            initial_level_set_detail::build_signed_distance_field<DIM>(
                component_mask,
                faces,
                N,
                cfg.domain_min,
                dx
            )
        );

        out.tracked_interfaces.push_back(TrackedInterface{mat, next_component_id});

        ++next_component_id;
    }

    return out;
}





