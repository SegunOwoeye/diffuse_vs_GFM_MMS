#pragma once

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
        for (const auto& region : cfg.regions) {
            if (!::dim::point_in_region<DIM>(x, region)) {
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
        double thickness
    )
    {
        if (thickness <= 0.0) {
            return (signed_distance <= 0.0) ? 1.0 : 0.0;
        }

        return 0.5 * (1.0 - std::tanh(2.0 * signed_distance / thickness));
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

        for (const auto& region : cfg.regions) {
            const double signed_distance = ::dim::signed_distance_to_box<DIM>(
                x,
                region,
                cfg.domain_min,
                cfg.domain_max
            );
            const double score = smooth_indicator(signed_distance, cfg.interface_thickness);

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

    /*
    [7] Loops over every grid cell and does the following:
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

    // [8] Top-level entry point for DIM initialization from config
    template<int DIM>
    inline void initialise_dim_from_config(
        std::vector<State<DIM>>& U,
        const Config<DIM>& cfg,
        const std::array<int, DIM>& N,
        const EOSParams& params
    )
    {
        if (cfg.initial_condition != "regions") {
            throw std::runtime_error("dim::initialise_dim_from_config: only region-based DIM initialisation is implemented");
        }

        ::dim::initialise_dim_from_regions<DIM>(U, cfg, N, params);
    }

} 


