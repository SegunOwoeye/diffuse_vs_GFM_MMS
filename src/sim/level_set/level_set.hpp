#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

#include "src/sim/level_set/level_set_core.hpp"
#include "src/sim/level_set/level_set_derivatives.hpp"
#include "src/sim/grid/grid_utils.hpp"


namespace level_set_detail {

// [0] Decode flat id to coordinates
template<int DIM>
inline void decode_index(
    int id,
    const LevelSetGrid<DIM>& grid,
    std::array<int, DIM>& idx
)
{
    int tmp = id;

    for (int d = DIM - 1; d >= 0; --d) {
        idx[d] = tmp / grid.stride[d];
        tmp %= grid.stride[d];
    }
}


// [1] Boundary check from coordinates
template<int DIM>
inline bool is_boundary_from_coords(
    const std::array<int, DIM>& idx,
    const LevelSetGrid<DIM>& grid
)
{
    for (int d = 0; d < DIM; ++d) {
        if (idx[d] == 0 || idx[d] == grid.N[d] - 1) {
            return true;
        }
    }

    return false;
}


// [2] Interior check from coordinates
template<int DIM>
inline bool is_interior_from_coords(
    const std::array<int, DIM>& idx,
    const LevelSetGrid<DIM>& grid
)
{
    for (int d = 0; d < DIM; ++d) {
        if (idx[d] <= 0 || idx[d] >= grid.N[d] - 1) {
            return false;
        }
    }

    return true;
}


// [3] Clamp coordinates to interior
template<int DIM>
inline void clamp_coords_to_interior(
    std::array<int, DIM>& idx,
    const LevelSetGrid<DIM>& grid
)
{
    for (int d = 0; d < DIM; ++d) {
        if (idx[d] == 0) {
            idx[d] = 1;
        }
        else if (idx[d] == grid.N[d] - 1) {
            idx[d] = grid.N[d] - 2;
        }
    }
}


// [4] Flatten coordinates without validation
template<int DIM>
inline int flatten_coords_unchecked(
    const std::array<int, DIM>& idx,
    const LevelSetGrid<DIM>& grid
)
{
    int id = 0;

    for (int d = 0; d < DIM; ++d) {
        id += idx[d] * grid.stride[d];
    }

    return id;
}

} // namespace level_set_detail


// [5] Sign Function
inline double sign_function(
    double x,
    double zero_tol = 1e-10
)
{
    if (x > zero_tol) return 1.0;
    if (x < -zero_tol) return -1.0;
    return 0.0;
}


// [6] Smoothed Sign Function (Sussman)
inline double sussman_sign(
    double phi0,
    double h
)
{
    return phi0 / std::sqrt(phi0 * phi0 + h * h);
}


template<int DIM>
inline double redistance_update_from_neighbours(
    const std::vector<double>& distance,
    const LevelSetGrid<DIM>& grid,
    const std::array<int, DIM>& idx
)
{
    std::array<std::pair<double, double>, DIM> candidates{};
    int n_candidates = 0;

    for (int d = 0; d < DIM; ++d) {
        double best = std::numeric_limits<double>::infinity();

        std::array<int, DIM> nb{};
        if (try_offset_index<DIM>(idx, d, -1, grid, nb)) {
            const int nb_id = flatten_index<DIM>(nb, grid);
            best = std::min(best, distance[nb_id]);
        }

        if (try_offset_index<DIM>(idx, d, +1, grid, nb)) {
            const int nb_id = flatten_index<DIM>(nb, grid);
            best = std::min(best, distance[nb_id]);
        }

        if (std::isfinite(best)) {
            candidates[n_candidates++] = {best, grid.dx[d]};
        }
    }

    if (n_candidates == 0) {
        return std::numeric_limits<double>::infinity();
    }

    std::sort(
        candidates.begin(),
        candidates.begin() + n_candidates,
        [](const auto& a, const auto& b) { return a.first < b.first; }
    );

    double accepted_a_sum = 0.0;
    double accepted_a2_sum = 0.0;
    double accepted_inv_h2_sum = 0.0;
    double value = candidates[0].first + candidates[0].second;

    for (int m = 0; m < n_candidates; ++m) {
        const double a = candidates[m].first;
        const double h = candidates[m].second;
        const double inv_h2 = 1.0 / (h * h);

        accepted_a_sum += a * inv_h2;
        accepted_a2_sum += a * a * inv_h2;
        accepted_inv_h2_sum += inv_h2;

        const double disc =
            accepted_a_sum * accepted_a_sum -
            accepted_inv_h2_sum * (accepted_a2_sum - 1.0);

        if (disc < 0.0) {
            continue;
        }

        value =
            (accepted_a_sum + std::sqrt(disc)) /
            accepted_inv_h2_sum;

        if (m + 1 == n_candidates ||
            value <= candidates[m + 1].first) {
            break;
        }
    }

    return value;
}


template<int DIM>
inline std::vector<double> redistance_preserving_zero_level_set(
    const std::vector<double>& phi0,
    const LevelSetGrid<DIM>& grid,
    int sweep_cycles
)
{
    const int Ntot = static_cast<int>(total_cells(grid));
    if (static_cast<int>(phi0.size()) != Ntot) {
        throw std::runtime_error("redistance_preserving_zero_level_set: phi size mismatch");
    }

    const double infinity = std::numeric_limits<double>::infinity();
    const double zero_tol = 1.0e-14;

    std::vector<double> distance(Ntot, infinity);
    std::vector<char> fixed(Ntot, 0);
    std::vector<double> sign(Ntot, 1.0);

    for (int id = 0; id < Ntot; ++id) {
        sign[id] = (phi0[id] < 0.0) ? -1.0 : 1.0;
    }

    int seed_count = 0;

    for (int id = 0; id < Ntot; ++id) {
        const auto idx = unflatten_index<DIM>(id, grid);

        if (std::abs(phi0[id]) <= zero_tol) {
            distance[id] = 0.0;
            fixed[id] = 1;
            ++seed_count;
            continue;
        }

        double local_distance = infinity;

        for (int d = 0; d < DIM; ++d) {
            for (int side : {-1, +1}) {
                std::array<int, DIM> nb{};
                if (!try_offset_index<DIM>(idx, d, side, grid, nb)) {
                    continue;
                }

                const int nb_id = flatten_index<DIM>(nb, grid);
                const double phi_a = phi0[id];
                const double phi_b = phi0[nb_id];

                if (std::abs(phi_b) <= zero_tol) {
                    local_distance = std::min(local_distance, grid.dx[d]);
                    continue;
                }

                if (phi_a * phi_b >= 0.0) {
                    continue;
                }

                const double denom = std::abs(phi_a) + std::abs(phi_b);
                if (denom <= zero_tol) {
                    local_distance = 0.0;
                    continue;
                }

                const double theta = std::abs(phi_a) / denom;
                local_distance = std::min(local_distance, theta * grid.dx[d]);
            }
        }

        if (std::isfinite(local_distance)) {
            distance[id] = local_distance;
            fixed[id] = 1;
            ++seed_count;
        }
    }

    if (seed_count == 0) {
        return phi0;
    }

    const int direction_count = 1 << DIM;
    const int cycles = std::max(1, sweep_cycles);

    for (int cycle = 0; cycle < cycles; ++cycle) {
        for (int direction_mask = 0; direction_mask < direction_count; ++direction_mask) {
            for (int rank = 0; rank < Ntot; ++rank) {
                int tmp = rank;
                std::array<int, DIM> idx{};

                for (int d = 0; d < DIM; ++d) {
                    const int coord = tmp % grid.N[d];
                    tmp /= grid.N[d];
                    idx[d] = (direction_mask & (1 << d))
                        ? coord
                        : (grid.N[d] - 1 - coord);
                }

                const int id = flatten_index<DIM>(idx, grid);
                if (fixed[id]) {
                    continue;
                }

                const double updated =
                    redistance_update_from_neighbours<DIM>(
                        distance,
                        grid,
                        idx
                    );

                if (updated < distance[id]) {
                    distance[id] = updated;
                }
            }
        }
    }

    std::vector<double> phi(Ntot, 0.0);

    for (int id = 0; id < Ntot; ++id) {
        if (!std::isfinite(distance[id])) {
            phi[id] = phi0[id];
        }
        else {
            phi[id] = sign[id] * distance[id];
        }
    }

    return phi;
}


// [7] Validate scalar field size
template<int DIM>
inline void validate_level_set_field_size(
    const std::vector<double>& phi,
    const LevelSetGrid<DIM>& grid,
    const char* where
)
{
    const int Ntot = static_cast<int>(total_cells(grid));

    if (static_cast<int>(phi.size()) != Ntot) {
        throw std::runtime_error(std::string(where) + ": phi size mismatch");
    }
}


// [8] Validate vector field size
template<int DIM>
inline void validate_level_set_velocity_size(
    const std::vector<std::array<double, DIM>>& vel,
    const LevelSetGrid<DIM>& grid,
    const char* where
)
{
    const int Ntot = static_cast<int>(total_cells(grid));

    if (static_cast<int>(vel.size()) != Ntot) {
        throw std::runtime_error(std::string(where) + ": velocity size mismatch");
    }
}


// [9] Validate scalar speed field size
template<int DIM>
inline void validate_level_set_speed_size(
    const std::vector<double>& speed,
    const LevelSetGrid<DIM>& grid,
    const char* where
)
{
    const int Ntot = static_cast<int>(total_cells(grid));

    if (static_cast<int>(speed.size()) != Ntot) {
        throw std::runtime_error(std::string(where) + ": speed size mismatch");
    }
}


// [10] Apply Neumann Boundary
template<int DIM>
inline void apply_neumann_bc(
    std::vector<double>& phi,
    const LevelSetGrid<DIM>& grid
)
{
    validate_level_set_field_size<DIM>(phi, grid, "apply_neumann_bc");

    const int Ntot = static_cast<int>(total_cells(grid));

    for (int id = 0; id < Ntot; ++id) {
        std::array<int, DIM> idx{};
        level_set_detail::decode_index<DIM>(id, grid, idx);

        if (!level_set_detail::is_boundary_from_coords<DIM>(idx, grid)) {
            continue;
        }

        level_set_detail::clamp_coords_to_interior<DIM>(idx, grid);
        const int nb_id = level_set_detail::flatten_coords_unchecked<DIM>(idx, grid);

        phi[id] = phi[nb_id];
    }
}


// [11] RHS for phi_t + u dot grad(phi) = 0 with TVD upwinding
template<int DIM>
inline std::vector<double> level_set_flow_rhs(
    const std::vector<double>& phi,
    const std::vector<std::array<double, DIM>>& vel,
    const LevelSetGrid<DIM>& grid,
    LevelSetDerivativeScheme derivative_scheme = LevelSetDerivativeScheme::Tvd
)
{
    validate_level_set_field_size<DIM>(phi, grid, "level_set_flow_rhs");
    validate_level_set_velocity_size<DIM>(vel, grid, "level_set_flow_rhs");

    const int Ntot = static_cast<int>(total_cells(grid));
    std::vector<double> rhs(Ntot, 0.0);

    for (int id = 0; id < Ntot; ++id) {
        std::array<int, DIM> idx{};
        level_set_detail::decode_index<DIM>(id, grid, idx);

        if (!level_set_detail::is_interior_from_coords<DIM>(idx, grid)) {
            continue;
        }

        double adv_term = 0.0;

        for (int d = 0; d < DIM; ++d) {
            const double grad = (vel[id][d] >= 0.0)
                ? level_set_detail::dminus_flat<DIM>(phi, grid, id, idx[d], d, derivative_scheme)
                : level_set_detail::dplus_flat<DIM>(phi, grid, id, idx[d], d, derivative_scheme);

            adv_term += vel[id][d] * grad;
        }

        rhs[id] = -adv_term;
    }

    return rhs;
}


// [12] RHS for phi_t + V_n |grad(phi)| = 0 with Godunov TVD derivatives
template<int DIM>
inline std::vector<double> level_set_normal_speed_rhs(
    const std::vector<double>& phi,
    const std::vector<double>& normal_speed,
    const LevelSetGrid<DIM>& grid,
    LevelSetDerivativeScheme derivative_scheme = LevelSetDerivativeScheme::Tvd
)
{
    validate_level_set_field_size<DIM>(phi, grid, "level_set_normal_speed_rhs");
    validate_level_set_speed_size<DIM>(normal_speed, grid, "level_set_normal_speed_rhs");

    const int Ntot = static_cast<int>(total_cells(grid));
    std::vector<double> rhs(Ntot, 0.0);

    for (int id = 0; id < Ntot; ++id) {
        std::array<int, DIM> idx{};
        level_set_detail::decode_index<DIM>(id, grid, idx);

        if (!level_set_detail::is_interior_from_coords<DIM>(idx, grid)) {
            continue;
        }

        const double Vn = normal_speed[id];
        double grad_sq = 0.0;

        for (int d = 0; d < DIM; ++d) {
            const double dm = level_set_detail::dminus_flat<DIM>(
                phi,
                grid,
                id,
                idx[d],
                d,
                derivative_scheme
            );
            const double dp = level_set_detail::dplus_flat<DIM>(
                phi,
                grid,
                id,
                idx[d],
                d,
                derivative_scheme
            );

            double term = 0.0;

            if (Vn >= 0.0) {
                term =
                    std::max(dm, 0.0) * std::max(dm, 0.0) +
                    std::min(dp, 0.0) * std::min(dp, 0.0);
            }
            else {
                term =
                    std::min(dm, 0.0) * std::min(dm, 0.0) +
                    std::max(dp, 0.0) * std::max(dp, 0.0);
            }

            grad_sq += term;
        }

        rhs[id] = -Vn * std::sqrt(grad_sq);
    }

    return rhs;
}


/*
    [13] Advect Level Set with vector velocity

    Solves:
        phi_t + u dot grad(phi) = 0

        using TVD-limited spatial derivatives and TVD-RK3 time stepping.
*/
template<int DIM>
inline std::vector<double> advect_phi(
    const std::vector<double>& phi,
    const std::vector<std::array<double, DIM>>& vel,
    const LevelSetGrid<DIM>& grid,
    double dt,
    LevelSetDerivativeScheme derivative_scheme = LevelSetDerivativeScheme::Tvd
)
{
    validate_level_set_field_size<DIM>(phi, grid, "advect_phi");
    validate_level_set_velocity_size<DIM>(vel, grid, "advect_phi");

    const int Ntot = static_cast<int>(total_cells(grid));
    const std::vector<double> rhs0 =
        level_set_flow_rhs<DIM>(phi, vel, grid, derivative_scheme);

    std::vector<double> phi1(Ntot, 0.0);

    for (int id = 0; id < Ntot; ++id) {
        phi1[id] = phi[id] + dt * rhs0[id];
    }

    apply_neumann_bc<DIM>(phi1, grid);

    const std::vector<double> rhs1 =
        level_set_flow_rhs<DIM>(phi1, vel, grid, derivative_scheme);

    std::vector<double> phi2(Ntot, 0.0);

    for (int id = 0; id < Ntot; ++id) {
        phi2[id] = 0.75 * phi[id] + 0.25 * (phi1[id] + dt * rhs1[id]);
    }

    apply_neumann_bc<DIM>(phi2, grid);

    const std::vector<double> rhs2 =
        level_set_flow_rhs<DIM>(phi2, vel, grid, derivative_scheme);

    std::vector<double> phi_new(Ntot, 0.0);

    for (int id = 0; id < Ntot; ++id) {
        phi_new[id] = (1.0 / 3.0) * phi[id] +
            (2.0 / 3.0) * (phi2[id] + dt * rhs2[id]);
    }

    apply_neumann_bc<DIM>(phi_new, grid);

    return phi_new;
}


/*
    [14] Advect Level Set with normal speed

    Solves:
        phi_t + V_n |grad(phi)| = 0

        using Godunov TVD-limited spatial derivatives and TVD-RK3 time stepping.
        This is the sharp-interface transport path when V_n is supplied by rGFM.
*/
template<int DIM>
inline std::vector<double> advect_phi_normal_speed(
    const std::vector<double>& phi,
    const std::vector<double>& normal_speed,
    const LevelSetGrid<DIM>& grid,
    double dt,
    LevelSetDerivativeScheme derivative_scheme = LevelSetDerivativeScheme::Tvd
)
{
    validate_level_set_field_size<DIM>(phi, grid, "advect_phi_normal_speed");
    validate_level_set_speed_size<DIM>(normal_speed, grid, "advect_phi_normal_speed");

    const int Ntot = static_cast<int>(total_cells(grid));
    const std::vector<double> rhs0 =
        level_set_normal_speed_rhs<DIM>(phi, normal_speed, grid, derivative_scheme);

    std::vector<double> phi1(Ntot, 0.0);

    for (int id = 0; id < Ntot; ++id) {
        phi1[id] = phi[id] + dt * rhs0[id];
    }

    apply_neumann_bc<DIM>(phi1, grid);

    const std::vector<double> rhs1 =
        level_set_normal_speed_rhs<DIM>(phi1, normal_speed, grid, derivative_scheme);

    std::vector<double> phi2(Ntot, 0.0);

    for (int id = 0; id < Ntot; ++id) {
        phi2[id] = 0.75 * phi[id] + 0.25 * (phi1[id] + dt * rhs1[id]);
    }

    apply_neumann_bc<DIM>(phi2, grid);

    const std::vector<double> rhs2 =
        level_set_normal_speed_rhs<DIM>(phi2, normal_speed, grid, derivative_scheme);

    std::vector<double> phi_new(Ntot, 0.0);

    for (int id = 0; id < Ntot; ++id) {
        phi_new[id] = (1.0 / 3.0) * phi[id] +
            (2.0 / 3.0) * (phi2[id] + dt * rhs2[id]);
    }

    apply_neumann_bc<DIM>(phi_new, grid);

    return phi_new;
}


/*
 [15] Reinitialisation (Godunov / Sussman)

    Solves:
        phi_tau + s(phi0) ( |grad(phi)| - 1 ) = 0

    Rebuilds a signed-distance field while preserving the zero-level-set location.
*/
template<int DIM>
inline std::vector<double> level_set_reinit_rhs(
    const std::vector<double>& phi,
    const std::vector<double>& phi0,
    const LevelSetGrid<DIM>& grid,
    LevelSetDerivativeScheme derivative_scheme = LevelSetDerivativeScheme::Tvd
)
{
    validate_level_set_field_size<DIM>(phi, grid, "level_set_reinit_rhs");
    validate_level_set_field_size<DIM>(phi0, grid, "level_set_reinit_rhs");

    const int Ntot = static_cast<int>(total_cells(grid));
    const double h = *std::min_element(grid.dx.begin(), grid.dx.end());
    std::vector<double> rhs(Ntot, 0.0);

    for (int id = 0; id < Ntot; ++id) {
        std::array<int, DIM> idx{};
        level_set_detail::decode_index<DIM>(id, grid, idx);

        if (!level_set_detail::is_interior_from_coords<DIM>(idx, grid)) {
            continue;
        }

        const double S = sussman_sign(phi0[id], h);
        double grad_sq = 0.0;

        for (int d = 0; d < DIM; ++d) {
            const double dm = level_set_detail::dminus_flat<DIM>(
                phi,
                grid,
                id,
                idx[d],
                d,
                derivative_scheme
            );
            const double dp = level_set_detail::dplus_flat<DIM>(
                phi,
                grid,
                id,
                idx[d],
                d,
                derivative_scheme
            );

            if (S >= 0.0) {
                grad_sq +=
                    std::max(dm, 0.0) * std::max(dm, 0.0) +
                    std::min(dp, 0.0) * std::min(dp, 0.0);
            }
            else {
                grad_sq +=
                    std::min(dm, 0.0) * std::min(dm, 0.0) +
                    std::max(dp, 0.0) * std::max(dp, 0.0);
            }
        }

        rhs[id] = -S * (std::sqrt(grad_sq) - 1.0);
    }

    return rhs;
}


template<int DIM>
inline std::vector<double> reinitialise_phi(
    const std::vector<double>& phi0,
    const LevelSetGrid<DIM>& grid,
    int iterations = 10,
    LevelSetDerivativeScheme derivative_scheme = LevelSetDerivativeScheme::Tvd
)
{
    validate_level_set_field_size<DIM>(phi0, grid, "reinitialise_phi");

    if (iterations < 0) {
        throw std::runtime_error("reinitialise_phi: iterations must be non-negative");
    }

    if (iterations == 0) {
        return phi0;
    }

    const int Ntot = static_cast<int>(total_cells(grid));
    const double min_dx = *std::min_element(grid.dx.begin(), grid.dx.end());
    const double dtau = 0.3 * min_dx;

    std::vector<double> phi = phi0;
    apply_neumann_bc<DIM>(phi, grid);

    for (int iter = 0; iter < iterations; ++iter) {
        const std::vector<double> rhs0 =
            level_set_reinit_rhs<DIM>(phi, phi0, grid, derivative_scheme);

        std::vector<double> phi1(Ntot, 0.0);

        for (int id = 0; id < Ntot; ++id) {
            phi1[id] = phi[id] + dtau * rhs0[id];
        }

        apply_neumann_bc<DIM>(phi1, grid);

        const std::vector<double> rhs1 =
            level_set_reinit_rhs<DIM>(phi1, phi0, grid, derivative_scheme);

        std::vector<double> phi2(Ntot, 0.0);

        for (int id = 0; id < Ntot; ++id) {
            phi2[id] = 0.75 * phi[id] + 0.25 * (phi1[id] + dtau * rhs1[id]);
        }

        apply_neumann_bc<DIM>(phi2, grid);

        const std::vector<double> rhs2 =
            level_set_reinit_rhs<DIM>(phi2, phi0, grid, derivative_scheme);

        for (int id = 0; id < Ntot; ++id) {
            phi[id] = (1.0 / 3.0) * phi[id] +
                (2.0 / 3.0) * (phi2[id] + dtau * rhs2[id]);
        }

        apply_neumann_bc<DIM>(phi, grid);
    }

    return phi;
}


/*  [14] Compute Normals
    Uses central differences for geometry diagnostics and interface-normal
    construction. Boundary values are copied from the nearest clamped interior cell.
*/
template<int DIM>
inline std::vector<std::array<double, DIM>> compute_normals(
    const std::vector<double>& phi,
    const LevelSetGrid<DIM>& grid
)
{
    validate_level_set_field_size<DIM>(phi, grid, "compute_normals");

    const int Ntot = static_cast<int>(total_cells(grid));

    std::vector<std::array<double, DIM>> normals(Ntot);
    std::vector<bool> computed(Ntot, false);

    // [14.1] Interior cells
    for (int id = 0; id < Ntot; ++id) {
        std::array<int, DIM> idx{};
        level_set_detail::decode_index<DIM>(id, grid, idx);

        if (!level_set_detail::is_interior_from_coords<DIM>(idx, grid)) {
            continue;
        }

        double mag = 0.0;

        for (int d = 0; d < DIM; ++d) {
            const double g = level_set_detail::dcenter_flat<DIM>(phi, grid, id, d);
            normals[id][d] = g;
            mag += g * g;
        }

        mag = std::sqrt(mag);

        if (mag > 1e-14) {
            for (int d = 0; d < DIM; ++d) {
                normals[id][d] /= mag;
            }
        }
        else {
            for (int d = 0; d < DIM; ++d) {
                normals[id][d] = 0.0;
            }
        }

        computed[id] = true;
    }

    // [14.2] Boundary cells
    for (int id = 0; id < Ntot; ++id) {
        if (computed[id]) {
            continue;
        }

        std::array<int, DIM> idx{};
        level_set_detail::decode_index<DIM>(id, grid, idx);
        level_set_detail::clamp_coords_to_interior<DIM>(idx, grid);

        const int nb_id = level_set_detail::flatten_coords_unchecked<DIM>(idx, grid);

        if (computed[nb_id]) {
            normals[id] = normals[nb_id];
        }
        else {
            for (int d = 0; d < DIM; ++d) {
                normals[id][d] = 0.0;
            }
        }
    }

    return normals;
}


