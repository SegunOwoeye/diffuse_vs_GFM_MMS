#pragma once

// Barton tensor initial conditions and cylindrical references.

#include "src/solid/elastoplastic/barton/config.hpp"

// Tensor Cartesian and cylindrical initial conditions.

// Initial conditions for the Cartesian and cylindrical Barton tensor multidimensional validation runs.

namespace solid::barton {

inline std::vector<TensorState2D> initialise_tensor_cartesian_2d(
    const TensorSolverConfig& cfg,
    const TensorMaterial& mat)
{
    const int nx = cfg.cells[0];
    const int ny = cfg.cells[1];
    const double dx = (cfg.domain_max[0] - cfg.domain_min[0]) / nx;
    const double dy = (cfg.domain_max[1] - cfg.domain_min[1]) / ny;
    const double hot_rho = material_density_from_p_T(cfg.hot_pressure, cfg.hot_temperature, mat);
    std::vector<TensorState2D> U(nx * ny);
#pragma omp parallel for collapse(2) schedule(static) if(nx * ny > 4096)
    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            const double x = cfg.domain_min[0] + (i + 0.5) * dx;
            const double y = cfg.domain_min[1] + (j + 0.5) * dy;
            const double r = std::sqrt(x * x + y * y);
            const bool hot = r > cfg.hot_radius;
            const double rho = hot ? hot_rho : mat.rho0;
            const double J = mat.rho0 / rho;
            const double a = std::cbrt(J);
            const std::array<double, 9> F{a, 0.0, 0.0, 0.0, a, 0.0, 0.0, 0.0, a};
            U[hidx(i, j, nx)] = tensor_cons_from_F(rho, 0.0, 0.0,
                hot ? cfg.hot_temperature : cfg.cold_temperature, F, mat);
        }
    }
    return U;
}

inline std::vector<TensorState2D> initialise_tensor_cylindrical_reference(
    const TensorSolverConfig& cfg,
    const TensorMaterial& mat)
{
    const double dr = (cfg.domain_max[0] - cfg.domain_min[0]) / cfg.radial_cells;
    const double hot_rho = material_density_from_p_T(cfg.hot_pressure, cfg.hot_temperature, mat);
    std::vector<TensorState2D> U(cfg.radial_cells);
    for (int i = 0; i < cfg.radial_cells; ++i) {
        const double r = cfg.domain_min[0] + (i + 0.5) * dr;
        const bool hot = r > cfg.hot_radius;
        const double rho = hot ? hot_rho : mat.rho0;
        const double J = mat.rho0 / rho;
        const double a = std::cbrt(J);
        const std::array<double, 9> F{a, 0.0, 0.0, 0.0, a, 0.0, 0.0, 0.0, a};
        U[i] = tensor_cons_from_F(rho, 0.0, 0.0,
            hot ? cfg.hot_temperature : cfg.cold_temperature, F, mat);
    }
    return U;
}

inline std::vector<TensorState3D> initialise_tensor_cartesian_3d(
    const TensorSolverConfig& cfg,
    const TensorMaterial& mat)
{
    const int nx = cfg.cells[0];
    const int ny = cfg.cells[1];
    const int nz = cfg.cells[2];
    const double dx = (cfg.domain_max[0] - cfg.domain_min[0]) / nx;
    const double dy = (cfg.domain_max[1] - cfg.domain_min[1]) / ny;
    const double dz = (cfg.domain_max[2] - cfg.domain_min[2]) / nz;
    const double hot_rho = material_density_from_p_T(cfg.hot_pressure, cfg.hot_temperature, mat);
    std::vector<TensorState3D> U(nx * ny * nz);
#pragma omp parallel for collapse(3) schedule(static) if(nx * ny * nz > 4096)
    for (int k = 0; k < nz; ++k) {
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                const double x = cfg.domain_min[0] + (i + 0.5) * dx;
                const double y = cfg.domain_min[1] + (j + 0.5) * dy;
                const double z = cfg.domain_min[2] + (k + 0.5) * dz;
                const double r = std::sqrt(x * x + y * y + z * z);
                const bool hot = r > cfg.hot_radius;
                const double rho = hot ? hot_rho : mat.rho0;
                const double J = mat.rho0 / rho;
                const double a = std::cbrt(J);
                const std::array<double, 9> F{a, 0.0, 0.0, 0.0, a, 0.0, 0.0, 0.0, a};
                U[hidx3(i, j, k, nx, ny)] = tensor_cons_from_F(rho, 0.0, 0.0, 0.0,
                    hot ? cfg.hot_temperature : cfg.cold_temperature, F, mat);
            }
        }
    }
    return U;
}

inline std::vector<TensorState3D> initialise_tensor_spherical_reference(
    const TensorSolverConfig& cfg,
    const TensorMaterial& mat)
{
    const double dr = (cfg.domain_max[0] - cfg.domain_min[0]) / cfg.radial_cells;
    const double hot_rho = material_density_from_p_T(cfg.hot_pressure, cfg.hot_temperature, mat);
    std::vector<TensorState3D> U(cfg.radial_cells);
    for (int i = 0; i < cfg.radial_cells; ++i) {
        const double r = cfg.domain_min[0] + (i + 0.5) * dr;
        const bool hot = r > cfg.hot_radius;
        const double rho = hot ? hot_rho : mat.rho0;
        const double J = mat.rho0 / rho;
        const double a = std::cbrt(J);
        const std::array<double, 9> F{a, 0.0, 0.0, 0.0, a, 0.0, 0.0, 0.0, a};
        U[i] = tensor_cons_from_F(rho, 0.0, 0.0, 0.0,
            hot ? cfg.hot_temperature : cfg.cold_temperature, F, mat);
    }
    return U;
}

} // namespace solid::barton
