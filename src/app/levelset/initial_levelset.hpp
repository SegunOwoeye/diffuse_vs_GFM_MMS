#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

#include "src/sim/gfm/tracked_interface.hpp"
#include "src/sim/grid/grid_utils.hpp"
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

    template<int DIM>
    inline std::array<double, DIM> normalised_planar_normal(
        const Config<DIM>& cfg
    )
    {
        double norm_sq = 0.0;
        for (int d = 0; d < DIM; ++d) {
            norm_sq += cfg.planar_normal[d] * cfg.planar_normal[d];
        }

        if (norm_sq <= 0.0) {
            throw std::runtime_error("initialise_phi_data_from_regions: zero planar normal");
        }

        const double inv_norm = 1.0 / std::sqrt(norm_sq);
        std::array<double, DIM> normal{};

        for (int d = 0; d < DIM; ++d) {
            normal[d] = cfg.planar_normal[d] * inv_norm;
        }

        return normal;
    }

    template<int DIM>
    inline double planar_coordinate(
        const std::array<double, DIM>& x,
        const std::array<double, DIM>& normal
    )
    {
        double s = 0.0;
        for (int d = 0; d < DIM; ++d) {
            s += normal[d] * x[d];
        }

        return s;
    }

    template<int DIM>
    inline double distance_to_planar_interval(
        double s,
        const Region<DIM>& region
    )
    {
        if (s < region.lower[0]) {
            return region.lower[0] - s;
        }

        if (s > region.upper[0]) {
            return s - region.upper[0];
        }

        return -std::min(s - region.lower[0], region.upper[0] - s);
    }

    inline std::vector<std::pair<double, double>> merge_planar_intervals(
        std::vector<std::pair<double, double>> intervals
    )
    {
        if (intervals.empty()) {
            return intervals;
        }

        std::sort(intervals.begin(), intervals.end());

        std::vector<std::pair<double, double>> merged;
        merged.push_back(intervals.front());

        for (std::size_t i = 1; i < intervals.size(); ++i) {
            auto& back = merged.back();

            if (intervals[i].first <= back.second + 1e-14) {
                back.second = std::max(back.second, intervals[i].second);
            }
            else {
                merged.push_back(intervals[i]);
            }
        }

        return merged;
    }

    inline double signed_distance_to_merged_planar_intervals(
        double s,
        const std::vector<std::pair<double, double>>& intervals
    )
    {
        double best_positive = std::numeric_limits<double>::max();

        for (const auto& interval : intervals) {
            const double lower = interval.first;
            const double upper = interval.second;

            if (s >= lower && s <= upper) {
                return -std::min(s - lower, upper - s);
            }

            const double distance = (s < lower) ? lower - s : s - upper;
            best_positive = std::min(best_positive, distance);
        }

        return best_positive;
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

    if (cfg.initial_condition != "regions" &&
        cfg.initial_condition != "planar_regions" &&
        cfg.initial_condition != "explosion" &&
        cfg.initial_condition != "double_explosion" &&
        cfg.initial_condition != "shock_bubble" &&
        cfg.initial_condition != "coated_shock_bubble") {
        throw std::runtime_error(
            "initialise_phi_data_from_regions: unsupported initial condition for GFM level sets"
        );
    }

    if ((cfg.initial_condition == "regions" ||
         cfg.initial_condition == "planar_regions") &&
        cfg.regions.size() < 2) {
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

    if (cfg.initial_condition == "planar_regions") {
        const auto normal =
            initial_level_set_detail::normalised_planar_normal<DIM>(cfg);
        std::vector<int> tracked_materials;

        for (const auto& region : cfg.regions) {
            if (region.material_id == out.background_material_id) {
                continue;
            }

            if (std::find(tracked_materials.begin(), tracked_materials.end(), region.material_id) ==
                tracked_materials.end()) {
                tracked_materials.push_back(region.material_id);
            }
        }

        for (const int tracked_material : tracked_materials) {
            std::vector<std::pair<double, double>> intervals;
            for (const auto& region : cfg.regions) {
                if (region.material_id == tracked_material) {
                    intervals.emplace_back(region.lower[0], region.upper[0]);
                }
            }
            intervals = initial_level_set_detail::merge_planar_intervals(intervals);

            std::vector<double> phi(total_cells, 0.0);

            for (int id = 0; id < total_cells; ++id) {
                const std::array<int, DIM> idx = unflatten_index<DIM>(id, grid);
                const auto x =
                    initial_level_set_detail::cell_center<DIM>(
                        idx,
                        cfg.domain_min,
                        dx
                    );
                const double s =
                    initial_level_set_detail::planar_coordinate<DIM>(x, normal);

                phi[id] =
                    initial_level_set_detail::signed_distance_to_merged_planar_intervals(
                        s,
                        intervals
                    );
            }

            out.phi_list.push_back(phi);
            out.tracked_interfaces.push_back(
                TrackedInterface{tracked_material, static_cast<int>(out.tracked_interfaces.size())}
            );
        }

        return out;
    }

    if (cfg.initial_condition == "explosion" ||
        cfg.initial_condition == "double_explosion" ||
        cfg.initial_condition == "shock_bubble" ||
        cfg.initial_condition == "coated_shock_bubble") {
        const bool is_shock_bubble = (cfg.initial_condition == "shock_bubble");
        const bool is_coated_shock_bubble = (cfg.initial_condition == "coated_shock_bubble");

        if (is_coated_shock_bubble) {
            const auto add_radial_phi = [&](int tracked_material, double inner_radius, double outer_radius) {
                if (tracked_material == out.background_material_id) {
                    return;
                }

                std::vector<double> phi(total_cells, 0.0);

                for (int id = 0; id < total_cells; ++id) {
                    const std::array<int, DIM> idx = unflatten_index<DIM>(id, grid);
                    const auto x =
                        initial_level_set_detail::cell_center<DIM>(
                            idx,
                            cfg.domain_min,
                            dx
                        );

                    double r2 = 0.0;
                    for (int d = 0; d < DIM; ++d) {
                        const double delta = x[d] - cfg.bubble_center[d];
                        r2 += delta * delta;
                    }

                    const double r = std::sqrt(r2);
                    if (inner_radius <= 0.0) {
                        phi[id] = r - outer_radius;
                    }
                    else if (r < inner_radius) {
                        phi[id] = inner_radius - r;
                    }
                    else if (r <= outer_radius) {
                        phi[id] = -std::min(r - inner_radius, outer_radius - r);
                    }
                    else {
                        phi[id] = r - outer_radius;
                    }
                }

                out.phi_list.push_back(phi);
                out.tracked_interfaces.push_back(
                    TrackedInterface{
                        tracked_material,
                        static_cast<int>(out.tracked_interfaces.size())
                    }
                );
            };

            add_radial_phi(cfg.material_bubble, 0.0, cfg.bubble_radius);
            add_radial_phi(cfg.material_film, cfg.bubble_radius, cfg.film_radius);
            return out;
        }

        const double radius = is_shock_bubble
            ? cfg.bubble_radius
            : cfg.explosion_radius;
        const int inside_material = is_shock_bubble
            ? cfg.material_bubble
            : cfg.material_in;
        const int outside_material = is_shock_bubble
            ? cfg.material_right
            : cfg.material_out;

        const bool track_inside = (inside_material != out.background_material_id);
        const int tracked_material = track_inside ? inside_material : outside_material;

        if (tracked_material == out.background_material_id) {
            return out;
        }

        std::vector<double> phi(total_cells, 0.0);

        for (int id = 0; id < total_cells; ++id) {
            const std::array<int, DIM> idx = unflatten_index<DIM>(id, grid);
            const auto x =
                initial_level_set_detail::cell_center<DIM>(
                    idx,
                    cfg.domain_min,
                    dx
                );

            double r_min = std::numeric_limits<double>::max();

            if (is_shock_bubble || cfg.initial_condition == "explosion") {
                const auto center = is_shock_bubble
                    ? cfg.bubble_center
                    : cfg.explosion_center;
                double r2 = 0.0;

                for (int d = 0; d < DIM; ++d) {
                    const double delta = x[d] - center[d];
                    r2 += delta * delta;
                }

                r_min = std::sqrt(r2);
            }
            else {
                for (const auto& center : cfg.explosion_centers) {
                    double r2 = 0.0;

                    for (int d = 0; d < DIM; ++d) {
                        const double delta = x[d] - center[d];
                        r2 += delta * delta;
                    }

                    r_min = std::min(r_min, std::sqrt(r2));
                }
            }

            const double signed_distance = r_min - radius;
            phi[id] = track_inside ? signed_distance : -signed_distance;
        }

        out.phi_list.push_back(phi);
        out.tracked_interfaces.push_back(
            TrackedInterface{tracked_material, 0}
        );

        return out;
    }

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

        /* 
        Store the signed-distance field and the data needed later by
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

