#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

#include "src/dim/primitives.hpp"
#include "src/dim/solver/advance/geometry.hpp"
#include "src/io/config.hpp"

namespace dim {

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
            throw std::runtime_error("dim::normalised_planar_normal: zero planar normal");
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
    inline bool point_in_planar_region(
        const std::array<double, DIM>& x,
        const Region<DIM>& region,
        const std::array<double, DIM>& normal
    )
    {
        const double s = ::dim::planar_coordinate<DIM>(x, normal);
        return s >= region.lower[0] && s < region.upper[0];
    }

    // [1] Computes the signed distance from a point to a rectangular region
    template<int DIM>
    inline double signed_distance_to_box(
        const std::array<double, DIM>& x,
        const Region<DIM>& region,
        const std::array<double, DIM>& domain_min,
        const std::array<double, DIM>& domain_max
    )
    {
        double outside_sq = 0.0;
        double inside_dist = std::numeric_limits<double>::max();
        bool inside = true;
        bool has_interface_axis = false;

        for (int d = 0; d < DIM; ++d) {
            const bool lower_is_domain_boundary =
                std::abs(region.lower[d] - domain_min[d]) <= 1e-14;
            const bool upper_is_domain_boundary =
                std::abs(region.upper[d] - domain_max[d]) <= 1e-14;
            const bool spans_domain =
                lower_is_domain_boundary && upper_is_domain_boundary;

            if (spans_domain) {
                continue;
            }

            has_interface_axis = true;

            if (x[d] < region.lower[d]) {
                const double dist = region.lower[d] - x[d];
                outside_sq += dist * dist;
                inside = false;
            }
            else if (x[d] > region.upper[d]) {
                const double dist = x[d] - region.upper[d];
                outside_sq += dist * dist;
                inside = false;
            }
            else {
                const double dist_to_lower = x[d] - region.lower[d];
                const double dist_to_upper = region.upper[d] - x[d];
                inside_dist = std::min(inside_dist, std::min(dist_to_lower, dist_to_upper));
            }
        }

        if (!has_interface_axis) {
            return -std::numeric_limits<double>::max();
        }

        return inside ? -inside_dist : std::sqrt(outside_sq);
    }

    // [2] Computes the signed distance from a point to a circular/spherical interface
    template<int DIM>
    inline double signed_distance_to_sphere(
        const std::array<double, DIM>& x,
        const std::array<double, DIM>& center,
        double radius
    )
    {
        double radius_sq = 0.0;
        for (int d = 0; d < DIM; ++d) {
            const double delta = x[d] - center[d];
            radius_sq += delta * delta;
        }

        return std::sqrt(radius_sq) - radius;
    }

    template<int DIM>
    inline double signed_distance_to_planar_interval(
        const std::array<double, DIM>& x,
        const Region<DIM>& region,
        const std::array<double, DIM>& normal
    )
    {
        const double s = ::dim::planar_coordinate<DIM>(x, normal);

        if (s < region.lower[0]) {
            return region.lower[0] - s;
        }

        if (s > region.upper[0]) {
            return s - region.upper[0];
        }

        return -std::min(s - region.lower[0], region.upper[0] - s);
    }

    // [3] Maps a config material id to its index in the DIM material arrays
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

        throw std::runtime_error("dim::material_slot_from_id: material id not found");
    }

    // [4] Builds the primitive state at a point assuming a sharp interface
    template<int DIM>
    inline Primitive<DIM> sharp_region_primitive(
        const Config<DIM>& cfg,
        const std::array<double, DIM>& x,
        int nmat
    )
    {
        const bool use_planar_regions = (cfg.initial_condition == "planar_regions");
        const auto planar_normal = use_planar_regions
            ? ::dim::normalised_planar_normal<DIM>(cfg)
            : std::array<double, DIM>{};

        for (const auto& region : cfg.regions) {
            const bool inside_region = use_planar_regions
                ? ::dim::point_in_planar_region<DIM>(x, region, planar_normal)
                : ::dim::point_in_region<DIM>(x, region);

            if (!inside_region) {
                continue;
            }

            Primitive<DIM> P{};
            P.alpha.assign(nmat, 0.0);
            P.rho.assign(nmat, 1.0);
            P.vel = region.vel;
            P.p = region.p;

            const int slot = ::dim::material_slot_from_id<DIM>(cfg, region.material_id);
            P.alpha[slot] = 1.0;
            P.rho[slot] = region.rho;
            return P;
        }

        throw std::runtime_error("dim::sharp_region_primitive: point not covered by any region");
    }

    // [5] Converts signed distance into a smooth weight using a tanh profile
    inline double smooth_indicator(
        double signed_distance,
        double thickness,
        double sharpness_alpha
    )
    {
        if (thickness <= 0.0) {
            return (signed_distance <= 0.0) ? 1.0 : 0.0;
        }

        const double alpha = std::max(sharpness_alpha, 1.0e-12);
        return 0.5 * (1.0 - std::tanh(alpha * signed_distance / thickness));
    }

    // [6] Builds the primitive state at a point for the diffuse-interface case
    template<int DIM>
    inline Primitive<DIM> diffuse_region_primitive(
        const Config<DIM>& cfg,
        const std::array<double, DIM>& x,
        const EOSParams& params
    )
    {
        const int nmat = params.nmat();

        if (cfg.regions.empty()) {
            throw std::runtime_error("dim::diffuse_region_primitive: no regions defined");
        }

        if (cfg.interface_thickness <= 0.0) {
            return ::dim::sharp_region_primitive<DIM>(cfg, x, nmat);
        }

        std::vector<double> material_score(nmat, 0.0);
        std::vector<double> rho_sum(nmat, 0.0);
        double total_score = 0.0;
        double p_sum = 0.0;
        std::array<double, DIM> vel_sum{};

        const bool use_planar_regions = (cfg.initial_condition == "planar_regions");
        const auto planar_normal = use_planar_regions
            ? ::dim::normalised_planar_normal<DIM>(cfg)
            : std::array<double, DIM>{};

        for (const auto& region : cfg.regions) {
            const double signed_distance = use_planar_regions
                ? ::dim::signed_distance_to_planar_interval<DIM>(x, region, planar_normal)
                : ::dim::signed_distance_to_box<DIM>(
                    x,
                    region,
                    cfg.domain_min,
                    cfg.domain_max
                );
            const double score = smooth_indicator(
                signed_distance,
                cfg.interface_thickness,
                cfg.interface_sharpness_alpha
            );

            if (score <= 1e-14) {
                continue;
            }

            const int slot = ::dim::material_slot_from_id<DIM>(cfg, region.material_id);
            material_score[slot] += score;
            rho_sum[slot] += score * region.rho;
            p_sum += score * region.p;

            for (int d = 0; d < DIM; ++d) {
                vel_sum[d] += score * region.vel[d];
            }

            total_score += score;
        }

        if (total_score <= 1e-14) {
            return ::dim::sharp_region_primitive<DIM>(cfg, x, nmat);
        }

        Primitive<DIM> P{};
        P.alpha.assign(nmat, 0.0);
        P.rho.assign(nmat, 1.0);
        P.p = p_sum / total_score;

        for (int d = 0; d < DIM; ++d) {
            P.vel[d] = vel_sum[d] / total_score;
        }

        double alpha_sum = 0.0;
        for (int k = 0; k < nmat; ++k) {
            P.alpha[k] = material_score[k] / total_score;
            alpha_sum += P.alpha[k];

            if (material_score[k] > 1e-14) {
                P.rho[k] = rho_sum[k] / material_score[k];
            }
        }

        if (alpha_sum <= 0.0) {
            return ::dim::sharp_region_primitive<DIM>(cfg, x, nmat);
        }

        sanitise_alpha(P.alpha, 0.0);
        return P;
    }

    // [7] Builds the DIM primitive state for the shock-bubble validation case
    template<int DIM>
    inline Primitive<DIM> diffuse_shock_bubble_primitive(
        const Config<DIM>& cfg,
        const std::array<double, DIM>& x,
        const EOSParams& params
    )
    {
        const int nmat = params.nmat();
        const double signed_distance = ::dim::signed_distance_to_sphere<DIM>(
            x,
            cfg.bubble_center,
            cfg.bubble_radius
        );

        const bool post_shock = x[cfg.shock_axis] >= cfg.shock_position;
        const double rho_air = post_shock ? cfg.rho_right : cfg.rho_left;
        const auto vel_air = post_shock ? cfg.vel_right : cfg.vel_left;
        const double p_air = post_shock ? cfg.p_right : cfg.p_left;
        const int material_air = post_shock ? cfg.material_right : cfg.material_left;

        const int air_slot = ::dim::material_slot_from_id<DIM>(cfg, material_air);
        const int bubble_slot = ::dim::material_slot_from_id<DIM>(cfg, cfg.material_bubble);

        Primitive<DIM> P{};
        P.alpha.assign(nmat, 0.0);
        P.rho.assign(nmat, 1.0);

        if (cfg.interface_thickness <= 0.0) {
            const bool inside_bubble = signed_distance <= 0.0;
            const int slot = inside_bubble ? bubble_slot : air_slot;

            P.alpha[slot] = 1.0;
            P.rho[slot] = inside_bubble ? cfg.rho_bubble : rho_air;
            P.vel = inside_bubble ? cfg.vel_bubble : vel_air;
            P.p = inside_bubble ? cfg.p_bubble : p_air;
            return P;
        }

        const double bubble_weight = smooth_indicator(
            signed_distance,
            cfg.interface_thickness,
            cfg.interface_sharpness_alpha
        );
        const double air_weight = 1.0 - bubble_weight;

        std::vector<double> rho_sum(nmat, 0.0);
        std::vector<double> weight_sum(nmat, 0.0);

        auto add_material = [&](int slot, double weight, double rho) {
            if (weight <= 1e-14) {
                return;
            }

            P.alpha[slot] += weight;
            rho_sum[slot] += weight * rho;
            weight_sum[slot] += weight;
        };

        add_material(air_slot, air_weight, rho_air);
        add_material(bubble_slot, bubble_weight, cfg.rho_bubble);

        for (int k = 0; k < nmat; ++k) {
            if (weight_sum[k] > 1e-14) {
                P.rho[k] = rho_sum[k] / weight_sum[k];
            }
        }

        for (int d = 0; d < DIM; ++d) {
            P.vel[d] = air_weight * vel_air[d] + bubble_weight * cfg.vel_bubble[d];
        }

        P.p = air_weight * p_air + bubble_weight * cfg.p_bubble;
        sanitise_alpha(P.alpha, 0.0);
        return P;
    }

    // [7.1] Builds the DIM primitive state for a shock tube with a coated bubble
    template<int DIM>
    inline Primitive<DIM> diffuse_coated_shock_bubble_primitive(
        const Config<DIM>& cfg,
        const std::array<double, DIM>& x,
        const EOSParams& params
    )
    {
        const int nmat = params.nmat();
        const double inner_signed_distance = ::dim::signed_distance_to_sphere<DIM>(
            x,
            cfg.bubble_center,
            cfg.bubble_radius
        );
        const double outer_signed_distance = ::dim::signed_distance_to_sphere<DIM>(
            x,
            cfg.bubble_center,
            cfg.film_radius
        );

        const bool post_shock = x[cfg.shock_axis] >= cfg.shock_position;
        const double rho_carrier = post_shock ? cfg.rho_right : cfg.rho_left;
        const auto vel_carrier = post_shock ? cfg.vel_right : cfg.vel_left;
        const double p_carrier = post_shock ? cfg.p_right : cfg.p_left;
        const int material_carrier = post_shock ? cfg.material_right : cfg.material_left;

        const int carrier_slot = ::dim::material_slot_from_id<DIM>(cfg, material_carrier);
        const int film_slot = ::dim::material_slot_from_id<DIM>(cfg, cfg.material_film);
        const int bubble_slot = ::dim::material_slot_from_id<DIM>(cfg, cfg.material_bubble);

        Primitive<DIM> P{};
        P.alpha.assign(nmat, 0.0);
        P.rho.assign(nmat, 1.0);

        if (cfg.interface_thickness <= 0.0) {
            const bool inside_bubble = inner_signed_distance <= 0.0;
            const bool inside_film = outer_signed_distance <= 0.0;
            int slot = carrier_slot;

            if (inside_bubble) {
                slot = bubble_slot;
                P.rho[slot] = cfg.rho_bubble;
                P.vel = cfg.vel_bubble;
                P.p = cfg.p_bubble;
            }
            else if (inside_film) {
                slot = film_slot;
                P.rho[slot] = cfg.rho_film;
                P.vel = cfg.vel_film;
                P.p = cfg.p_film;
            }
            else {
                P.rho[slot] = rho_carrier;
                P.vel = vel_carrier;
                P.p = p_carrier;
            }

            P.alpha[slot] = 1.0;
            return P;
        }

        const double bubble_weight = smooth_indicator(
            inner_signed_distance,
            cfg.interface_thickness,
            cfg.interface_sharpness_alpha
        );
        const double coated_weight = smooth_indicator(
            outer_signed_distance,
            cfg.interface_thickness,
            cfg.interface_sharpness_alpha
        );
        const double film_weight = std::max(coated_weight - bubble_weight, 0.0);
        const double carrier_weight = std::max(1.0 - coated_weight, 0.0);

        std::vector<double> rho_sum(nmat, 0.0);
        std::vector<double> weight_sum(nmat, 0.0);

        auto add_material = [&](int slot, double weight, double rho) {
            if (weight <= 1e-14) {
                return;
            }

            P.alpha[slot] += weight;
            rho_sum[slot] += weight * rho;
            weight_sum[slot] += weight;
        };

        add_material(carrier_slot, carrier_weight, rho_carrier);
        add_material(film_slot, film_weight, cfg.rho_film);
        add_material(bubble_slot, bubble_weight, cfg.rho_bubble);

        const double total_weight = std::max(
            carrier_weight + film_weight + bubble_weight,
            1e-14
        );

        for (int k = 0; k < nmat; ++k) {
            if (weight_sum[k] > 1e-14) {
                P.rho[k] = rho_sum[k] / weight_sum[k];
            }
        }

        for (int d = 0; d < DIM; ++d) {
            P.vel[d] = (
                carrier_weight * vel_carrier[d] +
                film_weight * cfg.vel_film[d] +
                bubble_weight * cfg.vel_bubble[d]
            ) / total_weight;
        }

        P.p = (
            carrier_weight * p_carrier +
            film_weight * cfg.p_film +
            bubble_weight * cfg.p_bubble
        ) / total_weight;
        sanitise_alpha(P.alpha, 0.0);
        return P;
    }

    /*
    [8] Loops over every grid cell and does the following:
        - Computes the cell center
        - Builds the primitive state at that location
        - Converts it to conservative DIM variables with prim_to_cons())
    */
    template<int DIM>
    inline void initialise_dim_from_regions(
        std::vector<State<DIM>>& U,
        const Config<DIM>& cfg,
        const std::array<int, DIM>& N,
        const EOSParams& params
    )
    {
        const int total = total_cells<DIM>(N);
        U.resize(total, make_state<DIM>(params.nmat()));

        std::array<double, DIM> dx{};
        for (int d = 0; d < DIM; ++d) {
            dx[d] = (cfg.domain_max[d] - cfg.domain_min[d]) / static_cast<double>(N[d]);
        }

        for (int linear = 0; linear < total; ++linear) {
            const auto idx = unflatten_index<DIM>(linear, N);
            const auto x = compute_cell_center<DIM>(idx, cfg.domain_min, dx);
            const Primitive<DIM> P = ::dim::diffuse_region_primitive<DIM>(cfg, x, params);
            U[linear] = prim_to_cons<DIM>(P, params);
        }
    }

    // [9] Initializes the DIM shock-bubble validation case
    template<int DIM>
    inline void initialise_dim_from_shock_bubble(
        std::vector<State<DIM>>& U,
        const Config<DIM>& cfg,
        const std::array<int, DIM>& N,
        const EOSParams& params
    )
    {
        const int total = total_cells<DIM>(N);
        U.resize(total, make_state<DIM>(params.nmat()));

        std::array<double, DIM> dx{};
        for (int d = 0; d < DIM; ++d) {
            dx[d] = (cfg.domain_max[d] - cfg.domain_min[d]) / static_cast<double>(N[d]);
        }

        for (int linear = 0; linear < total; ++linear) {
            const auto idx = unflatten_index<DIM>(linear, N);
            const auto x = compute_cell_center<DIM>(idx, cfg.domain_min, dx);
            const Primitive<DIM> P = ::dim::diffuse_shock_bubble_primitive<DIM>(cfg, x, params);
            U[linear] = prim_to_cons<DIM>(P, params);
        }
    }

    // [9.1] Initializes the DIM coated shock-bubble validation case
    template<int DIM>
    inline void initialise_dim_from_coated_shock_bubble(
        std::vector<State<DIM>>& U,
        const Config<DIM>& cfg,
        const std::array<int, DIM>& N,
        const EOSParams& params
    )
    {
        const int total = total_cells<DIM>(N);
        U.resize(total, make_state<DIM>(params.nmat()));

        std::array<double, DIM> dx{};
        for (int d = 0; d < DIM; ++d) {
            dx[d] = (cfg.domain_max[d] - cfg.domain_min[d]) / static_cast<double>(N[d]);
        }

        for (int linear = 0; linear < total; ++linear) {
            const auto idx = unflatten_index<DIM>(linear, N);
            const auto x = compute_cell_center<DIM>(idx, cfg.domain_min, dx);
            const Primitive<DIM> P =
                ::dim::diffuse_coated_shock_bubble_primitive<DIM>(cfg, x, params);
            U[linear] = prim_to_cons<DIM>(P, params);
        }
    }

    // [10] Top-level entry point for DIM initialization from config
    template<int DIM>
    inline void initialise_dim_from_config(
        std::vector<State<DIM>>& U,
        const Config<DIM>& cfg,
        const std::array<int, DIM>& N,
        const EOSParams& params
    )
    {
        if (cfg.initial_condition == "regions" ||
            cfg.initial_condition == "planar_regions") {
            ::dim::initialise_dim_from_regions<DIM>(U, cfg, N, params);
        }
        else if (cfg.initial_condition == "shock_bubble") {
            ::dim::initialise_dim_from_shock_bubble<DIM>(U, cfg, N, params);
        }
        else if (cfg.initial_condition == "coated_shock_bubble") {
            ::dim::initialise_dim_from_coated_shock_bubble<DIM>(U, cfg, N, params);
        }
        else {
            throw std::runtime_error("dim::initialise_dim_from_config: unsupported initial_condition: " + cfg.initial_condition);
        }
    }

} 
