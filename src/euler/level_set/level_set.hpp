#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>
#include <vector>

#include "src/euler/level_set/level_set_core.hpp"
#include "src/euler/level_set/level_set_derivatives.hpp"
#include "src/euler/grid/grid_utils.hpp"


// [0] Sign Function
inline double sign_function(
    double x,
    double zero_tol = 1e-10
)
{
    if (x > zero_tol) return 1.0;
    if (x < -zero_tol) return -1.0;
    return 0.0;
}


// [1] Smoothed Sign Function (Sussman)
inline double sussman_sign(
    double phi0,
    double h
)
{
    return phi0 / std::sqrt(phi0 * phi0 + h * h);
}


// [2] Validate scalar field size
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


// [3] Validate vector field size
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


// [4] Validate scalar speed field size
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


// [5] Apply Neumann Boundary
template<int DIM>
inline void apply_neumann_bc(
    std::vector<double>& phi,
    const LevelSetGrid<DIM>& grid
)
{
    validate_level_set_field_size<DIM>(phi, grid, "apply_neumann_bc");

    const int Ntot = static_cast<int>(total_cells(grid));

    for (int id = 0; id < Ntot; ++id) {
        const std::array<int, DIM> idx = unflatten_index<DIM>(id, grid);

        if (!is_boundary_cell<DIM>(idx, grid)) {
            continue;
        }

        const std::array<int, DIM> nb = clamp_to_interior<DIM>(idx, grid);
        const int nb_id = flatten_index<DIM>(nb, grid);

        phi[id] = phi[nb_id];
    }
}

/*
    [6] Advect Level Set with vector velocity

    Solves:
        phi_t + u · grad(phi) = 0
        - using WENO one-sided derivatives from level_set_derivatives.hpp.
*/
template<int DIM>
inline std::vector<double> advect_phi(
    const std::vector<double>& phi,
    const std::vector<std::array<double, DIM>>& vel,
    const LevelSetGrid<DIM>& grid,
    double dt
)
{
    validate_level_set_field_size<DIM>(phi, grid, "advect_phi");
    validate_level_set_velocity_size<DIM>(vel, grid, "advect_phi");

    const int Ntot = static_cast<int>(total_cells(grid));
    std::vector<double> phi_new = phi;

    for (int id = 0; id < Ntot; ++id) {
        const std::array<int, DIM> idx = unflatten_index<DIM>(id, grid);

        if (!is_interior_cell<DIM>(idx, grid)) {
            continue;
        }

        double adv_term = 0.0;

        for (int d = 0; d < DIM; ++d) {
            const double grad = (vel[id][d] >= 0.0)
                ? dminus<DIM>(phi, grid, idx, d)
                : dplus<DIM>(phi, grid, idx, d);

            adv_term += vel[id][d] * grad;
        }

        phi_new[id] = phi[id] - dt * adv_term;
    }

    apply_neumann_bc<DIM>(phi_new, grid);

    return phi_new;
}


// [7] Advect Level Set with normal speed
//
// Solves:
//     phi_t + V_n |grad(phi)| = 0
//
// using Godunov selection of one-sided derivatives. This is the form you
// actually want for sharp-interface transport once V_n is supplied from the
// interface physics layer.
template<int DIM>
inline std::vector<double> advect_phi_normal_speed(
    const std::vector<double>& phi,
    const std::vector<double>& normal_speed,
    const LevelSetGrid<DIM>& grid,
    double dt
)
{
    validate_level_set_field_size<DIM>(phi, grid, "advect_phi_normal_speed");
    validate_level_set_speed_size<DIM>(normal_speed, grid, "advect_phi_normal_speed");

    const int Ntot = static_cast<int>(total_cells(grid));
    std::vector<double> phi_new = phi;

    for (int id = 0; id < Ntot; ++id) {
        const std::array<int, DIM> idx = unflatten_index<DIM>(id, grid);

        if (!is_interior_cell<DIM>(idx, grid)) {
            continue;
        }

        const double Vn = normal_speed[id];
        double grad_sq = 0.0;

        for (int d = 0; d < DIM; ++d) {
            const double dm = dminus<DIM>(phi, grid, idx, d);
            const double dp = dplus<DIM>(phi, grid, idx, d);

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

        phi_new[id] = phi[id] - dt * Vn * std::sqrt(grad_sq);
    }

    apply_neumann_bc<DIM>(phi_new, grid);

    return phi_new;
}


// [8] Reinitialisation (Godunov / Sussman)
//
// Solves:
//     phi_tau + s(phi0) ( |grad(phi)| - 1 ) = 0
//
// using Godunov selection of WENO-backed one-sided derivatives.
template<int DIM>
inline std::vector<double> reinitialise_phi(
    const std::vector<double>& phi0,
    const LevelSetGrid<DIM>& grid,
    int iterations = 10
)
{
    validate_level_set_field_size<DIM>(phi0, grid, "reinitialise_phi");

    if (iterations < 0) {
        throw std::runtime_error("reinitialise_phi: iterations must be non-negative");
    }

    const int Ntot = static_cast<int>(total_cells(grid));

    double h = grid.dx[0];
    for (int d = 1; d < DIM; ++d) {
        h = std::min(h, grid.dx[d]);
    }

    const double dtau = 0.3 * h;

    std::vector<double> phi = phi0;
    std::vector<double> phi_new = phi0;

    for (int it = 0; it < iterations; ++it) {

        for (int id = 0; id < Ntot; ++id) {

            const std::array<int, DIM> idx = unflatten_index<DIM>(id, grid);

            if (!is_interior_cell<DIM>(idx, grid)) {
                continue;
            }

            const double s = sussman_sign(phi0[id], h);

            double grad_sq = 0.0;

            for (int d = 0; d < DIM; ++d) {

                const double dm = dminus<DIM>(phi, grid, idx, d);
                const double dp = dplus<DIM>(phi, grid, idx, d);

                double term = 0.0;

                if (s >= 0.0) {
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

            phi_new[id] =
                phi[id] - dtau * s * (std::sqrt(grad_sq) - 1.0);
        }

        apply_neumann_bc<DIM>(phi_new, grid);
        phi.swap(phi_new);
    }

    return phi;
}


/*  [9] Compute Normals
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

    // [9.1] Interior cells
    for (int id = 0; id < Ntot; ++id) {
        const std::array<int, DIM> idx = unflatten_index<DIM>(id, grid);

        if (!is_interior_cell<DIM>(idx, grid)) {
            continue;
        }

        double mag = 0.0;

        for (int d = 0; d < DIM; ++d) {
            const double g = dcenter<DIM>(phi, grid, idx, d);
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

    // [9.2] Boundary cells
    for (int id = 0; id < Ntot; ++id) {
        if (computed[id]) {
            continue;
        }

        const std::array<int, DIM> idx = unflatten_index<DIM>(id, grid);
        const std::array<int, DIM> nb = clamp_to_interior<DIM>(idx, grid);
        const int nb_id = flatten_index<DIM>(nb, grid);

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



