#pragma once

#include <algorithm>
#include <array>
#include <stdexcept>
#include <vector>

#include "src/dim/primitives.hpp"

namespace dim {

    // [0] Compute Stable CFL Timestep
    template<int DIM>
    inline double compute_dt_cfl(
        const std::vector<State<DIM>>& U,
        const std::array<double, DIM>& dx,
        const EOSParams& params,
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

        #pragma omp parallel for reduction(min:dt)
        for (int i = 0; i < static_cast<int>(U.size()); ++i) {
            const Primitive<DIM> P = cons_to_prim<DIM>(U[i], params);
            const double c = IdealGasEOS::mixture_sound_speed(
                total_density(U[i]),
                P.p,
                P.alpha,
                P.rho,
                params
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

} 
