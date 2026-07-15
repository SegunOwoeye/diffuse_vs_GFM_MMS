#pragma once

#include <algorithm>
#include <array>
#include <stdexcept>
#include <string>
#include <vector>

#include "src/core/openmp_policy.hpp"
#include "src/dim/primitives.hpp"

namespace dim {

    // [0] Compute Stable CFL Timestep
    template<int DIM>
    inline double compute_dt_cfl(
        const std::vector<State<DIM>>& U,
        const std::array<double, DIM>& dx,
        const EOSParams& params,
        const std::string& lambda_model,
        double cfl,
        double dt_max
    )
    {
        if (U.empty()) {
            throw std::runtime_error("dim::compute_dt_cfl: empty state vector");
        }

        if (cfl <= 0.0) {
            throw std::runtime_error("dim::compute_dt_cfl: cfl must be positive");
        }

        if (dt_max <= 0.0) {
            throw std::runtime_error("dim::compute_dt_cfl: dt_max must be positive");
        }

        for (int d = 0; d < DIM; ++d) {
            if (dx[d] <= 0.0) {
                throw std::runtime_error("dim::compute_dt_cfl: grid spacing must be positive");
            }
        }

        double dt = dt_max;

        #pragma omp parallel for reduction(min:dt) schedule(static) if(runtime::openmp_should_parallelize_cells(U.size()))
        for (int i = 0; i < static_cast<int>(U.size()); ++i) {
            const Primitive<DIM> P = cons_to_prim<DIM>(U[i], params);
            const double c = MixtureEOS::generic_mixture_sound_speed(
                total_density(U[i]),
                P.p,
                P.alpha,
                P.rho,
                params,
                lambda_model
            );

            for (int d = 0; d < DIM; ++d) {
                const double wave_speed = std::abs(P.vel[d]) + c;
                if (wave_speed > 1e-14) {
                    dt = std::min(dt, cfl * dx[d] / wave_speed);
                }
            }
        }

        return dt;
    }

    // [1] Compute Stable CFL Timestep for Unsplit Updates
    template<int DIM>
    inline double compute_dt_cfl_unsplit(
        const std::vector<State<DIM>>& U,
        const std::array<double, DIM>& dx,
        const EOSParams& params,
        const std::string& lambda_model,
        double cfl,
        double dt_max
    )
    {
        if (U.empty()) {
            throw std::runtime_error("dim::compute_dt_cfl_unsplit: empty state vector");
        }

        if (cfl <= 0.0) {
            throw std::runtime_error("dim::compute_dt_cfl_unsplit: cfl must be positive");
        }

        if (dt_max <= 0.0) {
            throw std::runtime_error("dim::compute_dt_cfl_unsplit: dt_max must be positive");
        }

        double dt = dt_max;

        #pragma omp parallel for reduction(min:dt) schedule(static) if(runtime::openmp_should_parallelize_cells(U.size()))
        for (int i = 0; i < static_cast<int>(U.size()); ++i) {
            const Primitive<DIM> P = cons_to_prim<DIM>(U[i], params);
            const double c = MixtureEOS::generic_mixture_sound_speed(
                total_density(U[i]),
                P.p,
                P.alpha,
                P.rho,
                params,
                lambda_model
            );

            double spectral_sum = 0.0;
            for (int d = 0; d < DIM; ++d) {
                if (dx[d] <= 0.0) {
                    throw std::runtime_error("dim::compute_dt_cfl_unsplit: grid spacing must be positive");
                }
                spectral_sum += (std::abs(P.vel[d]) + c) / dx[d];
            }

            if (spectral_sum > 1e-14) {
                dt = std::min(dt, cfl / spectral_sum);
            }
        }

        return dt;
    }

} 
