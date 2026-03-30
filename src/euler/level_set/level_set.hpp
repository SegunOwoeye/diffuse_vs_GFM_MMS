#pragma once

#include <vector>
#include <cmath>
#include <stdexcept>
#include <algorithm>
#include <array>

#include "src/euler/level_set/level_set_core.hpp"
#include "src/euler/level_set/level_set_derivatives.hpp"


// [0] Sign Function
inline double sign_function(
    double x,
    double zero_tol = 1e-10
)
{
    if (x > zero_tol) {
        return 1.0;
    }

    if (x < -zero_tol) {
        return -1.0;
    }

    return 0.0;
}


// [1] Smoothed Sign Function
inline double sussman_sign(
    double phi0,
    double h
)
{
    return phi0 / std::sqrt(phi0 * phi0 + h * h);
}


// [2] Apply Neumann Boundary (generic DIM)
template<int DIM>
inline void apply_neumann_bc(
    std::vector<double>& phi,
    const LevelSetGrid<DIM>& grid
)
{
    std::array<int, DIM> idx;

    const int N = total_cells(grid);

    for (int id = 0; id < N; ++id) {

        int tmp = id;
        for (int d = 0; d < DIM; ++d) {
            idx[d] = tmp % grid.N[d];
            tmp /= grid.N[d];
        }

        for (int d = 0; d < DIM; ++d) {

            if (idx[d] == 0) {
                auto nb = idx;
                nb[d] = 1;
                phi[id] = phi[flatten_index(nb, grid)];
            }

            if (idx[d] == grid.N[d] - 1) {
                auto nb = idx;
                nb[d] = grid.N[d] - 2;
                phi[id] = phi[flatten_index(nb, grid)];
            }
        }
    }
}


// [3] Advect Level Set (Godunov Upwind)
template<int DIM>
inline std::vector<double> advect_phi(
    const std::vector<double>& phi,
    const std::vector<std::array<double, DIM>>& vel,
    const LevelSetGrid<DIM>& grid,
    double dt
)
{
    const int N = total_cells(grid);

    if (static_cast<int>(phi.size()) != N) {
        throw std::runtime_error("advect_phi: phi size mismatch");
    }

    if (static_cast<int>(vel.size()) != N) {
        throw std::runtime_error("advect_phi: velocity size mismatch");
    }

    std::vector<double> phi_new = phi;

    std::array<int, DIM> idx;

    for (int id = 0; id < N; ++id) {

        int tmp = id;
        for (int d = 0; d < DIM; ++d) {
            idx[d] = tmp % grid.N[d];
            tmp /= grid.N[d];
        }

        if (!is_interior_cell(idx, grid)) {
            continue;
        }

        double adv_term = 0.0;

        for (int d = 0; d < DIM; ++d) {

            double grad;

            if (vel[id][d] >= 0.0) {
                grad = dminus(phi, grid, idx, d);
            }
            else {
                grad = dplus(phi, grid, idx, d);
            }

            adv_term += vel[id][d] * grad;
        }

        phi_new[id] = phi[id] - dt * adv_term;
    }

    apply_neumann_bc(phi_new, grid);

    return phi_new;
}


// [4] Reinitialisation (Godunov / Sussman)
template<int DIM>
inline std::vector<double> reinitialise_phi(
    const std::vector<double>& phi0,
    const LevelSetGrid<DIM>& grid,
    int iterations = 10
)
{
    const int N = total_cells(grid);

    if (static_cast<int>(phi0.size()) != N) {
        throw std::runtime_error("reinitialise_phi: size mismatch");
    }

    double h = grid.dx[0];
    for (int d = 1; d < DIM; ++d) {
        h = std::min(h, grid.dx[d]);
    }

    double dtau = 0.3 * h;

    std::vector<double> phi = phi0;
    std::vector<double> phi_new = phi0;

    std::array<int, DIM> idx;

    for (int it = 0; it < iterations; ++it) {

        for (int id = 0; id < N; ++id) {

            int tmp = id;
            for (int d = 0; d < DIM; ++d) {
                idx[d] = tmp % grid.N[d];
                tmp /= grid.N[d];
            }

            if (!is_interior_cell(idx, grid)) {
                continue;
            }

            double s = sussman_sign(phi0[id], h);

            double grad_sq = 0.0;

            for (int d = 0; d < DIM; ++d) {

                double dm = dminus(phi, grid, idx, d);
                double dp = dplus(phi, grid, idx, d);

                double term;

                if (s > 0.0) {
                    term = std::max(dm, 0.0) * std::max(dm, 0.0)
                      + std::min(dp, 0.0) * std::min(dp, 0.0);
                }
                else {
                    term = std::min(dm, 0.0) * std::min(dm, 0.0)
                      + std::max(dp, 0.0) * std::max(dp, 0.0);
                }

                grad_sq += term;
            }

            double grad = std::sqrt(grad_sq);

            phi_new[id] = phi[id] - dtau * s * (grad - 1.0);
        }

        apply_neumann_bc(phi_new, grid);
        phi.swap(phi_new);
    }

    return phi;
}


// [5] Compute Normals
template<int DIM>
inline std::vector<std::array<double, DIM>> compute_normals(
    const std::vector<double>& phi,
    const LevelSetGrid<DIM>& grid
)
{
    const int N = total_cells(grid);

    std::vector<std::array<double, DIM>> normals(N);

    std::array<int, DIM> idx;

    for (int id = 0; id < N; ++id) {

        int tmp = id;
        for (int d = 0; d < DIM; ++d) {
            idx[d] = tmp % grid.N[d];
            tmp /= grid.N[d];
        }

        if (!is_interior_cell(idx, grid)) {
            continue;
        }

        double mag = 0.0;

        for (int d = 0; d < DIM; ++d) {
            double g = dcenter(phi, grid, idx, d);
            normals[id][d] = g;
            mag += g * g;
        }

        mag = std::sqrt(mag);

        if (mag > 1e-14) {
            for (int d = 0; d < DIM; ++d) {
                normals[id][d] /= mag;
            }
        }
    }

    return normals;
}






