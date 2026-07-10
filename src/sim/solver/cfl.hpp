#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>
#include <vector>

#include "src/euler/state.hpp"
#include "src/euler/thermo_state.hpp"
#include "src/euler/thermo_compute.hpp"
#include "src/euler/eos.hpp"
#include "src/euler/eos_params.hpp"


// [0] Compute Stable CFL Timestep for a Single EOS Family
template<int DIM, typename EOS>
inline double compute_dt_cfl(
    const std::vector<Conserved<DIM>>& U,
    const std::array<double, DIM>& dx,
    const EOSParams& params,
    double cfl,
    double dt_max
)
{
    if (U.empty()) {
        throw std::runtime_error("compute_dt_cfl: empty state vector");
    }

    if (cfl <= 0.0) {
        throw std::runtime_error("compute_dt_cfl: cfl must be positive");
    }

    if (dt_max <= 0.0) {
        throw std::runtime_error("compute_dt_cfl: dt_max must be positive");
    }

    for (int d = 0; d < DIM; ++d) {
        if (dx[d] <= 0.0) {
            throw std::runtime_error("compute_dt_cfl: grid spacing must be positive");
        }
    }

    double dt = dt_max;

    #pragma omp parallel for reduction(min:dt)
    for (std::size_t i = 0; i < U.size(); ++i) {
        const auto& Ui = U[i];
        const ThermoState<DIM> T = compute_thermo<DIM, EOS>(Ui, params);

        for (int d = 0; d < DIM; ++d) {
            const double wave_speed = std::abs(T.vel[d]) + T.c;

            if (wave_speed > 1e-14) {
                dt = std::min(dt, cfl * dx[d] / wave_speed);
            }
        }
    }

    return dt;
}


// [1] Compute Stable CFL Timestep with Per-Cell Material Parameters
template<int DIM, typename EOS>
inline double compute_dt_cfl_materials(
    const std::vector<Conserved<DIM>>& U,
    const std::vector<int>& material_id,
    const std::vector<EOSParams>& material_params,
    const std::array<double, DIM>& dx,
    double cfl,
    double dt_max
)
{
    if (U.empty()) {
        throw std::runtime_error("compute_dt_cfl_materials: empty state vector");
    }

    if (static_cast<int>(material_id.size()) != static_cast<int>(U.size())) {
        throw std::runtime_error("compute_dt_cfl_materials: material_id size mismatch");
    }

    if (material_params.empty()) {
        throw std::runtime_error("compute_dt_cfl_materials: material_params is empty");
    }

    if (cfl <= 0.0) {
        throw std::runtime_error("compute_dt_cfl_materials: cfl must be positive");
    }

    if (dt_max <= 0.0) {
        throw std::runtime_error("compute_dt_cfl_materials: dt_max must be positive");
    }

    for (int d = 0; d < DIM; ++d) {
        if (dx[d] <= 0.0) {
            throw std::runtime_error("compute_dt_cfl_materials: grid spacing must be positive");
        }
    }

    double dt = dt_max;

    #pragma omp parallel for reduction(min:dt)
    for (std::size_t i = 0; i < U.size(); ++i) {
        const int mat = material_id[i];

        if (mat < 0 || mat >= static_cast<int>(material_params.size())) {
            continue;
        }

        const EOSParams& params = material_params[mat];
        const ThermoState<DIM> T = compute_thermo<DIM, EOS>(U[i], params);

        for (int d = 0; d < DIM; ++d) {
            const double wave_speed = std::abs(T.vel[d]) + T.c;

            if (wave_speed > 1e-14) {
                dt = std::min(dt, cfl * dx[d] / wave_speed);
            }
        }
    }

    return dt;
}


// [2] Unsplit CFL: one timestep sees all directional wave contributions
template<int DIM, typename EOS>
inline double compute_dt_cfl_materials_unsplit(
    const std::vector<Conserved<DIM>>& U,
    const std::vector<int>& material_id,
    const std::vector<EOSParams>& material_params,
    const std::array<double, DIM>& dx,
    double cfl,
    double dt_max
)
{
    if (U.empty()) {
        throw std::runtime_error("compute_dt_cfl_materials_unsplit: empty state vector");
    }

    if (static_cast<int>(material_id.size()) != static_cast<int>(U.size())) {
        throw std::runtime_error("compute_dt_cfl_materials_unsplit: material_id size mismatch");
    }

    if (material_params.empty()) {
        throw std::runtime_error("compute_dt_cfl_materials_unsplit: material_params is empty");
    }

    if (cfl <= 0.0) {
        throw std::runtime_error("compute_dt_cfl_materials_unsplit: cfl must be positive");
    }

    if (dt_max <= 0.0) {
        throw std::runtime_error("compute_dt_cfl_materials_unsplit: dt_max must be positive");
    }

    double dt = dt_max;

    #pragma omp parallel for reduction(min:dt)
    for (std::size_t i = 0; i < U.size(); ++i) {
        const int mat = material_id[i];

        if (mat < 0 || mat >= static_cast<int>(material_params.size())) {
            continue;
        }

        const ThermoState<DIM> T =
            compute_thermo<DIM, EOS>(U[i], material_params[mat]);

        double spectral_sum = 0.0;
        for (int d = 0; d < DIM; ++d) {
            if (dx[d] <= 0.0) {
                throw std::runtime_error("compute_dt_cfl_materials_unsplit: grid spacing must be positive");
            }
            spectral_sum += (std::abs(T.vel[d]) + T.c) / dx[d];
        }

        if (spectral_sum > 1e-14) {
            dt = std::min(dt, cfl / spectral_sum);
        }
    }

    return dt;
}

