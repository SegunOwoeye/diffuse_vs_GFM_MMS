#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "src/dim/state.hpp"
#include "src/dim/conservative.hpp"
#include "src/dim/eos.hpp"
#include "src/dim/eos_params.hpp"


// [0] Compute Stable CFL Timestep (Diffuse Interface)
template<int DIM, int NMAT, typename EOS>
inline double compute_dt_cfl(
    const std::vector<Conserved<DIM, NMAT>>& U,
    const std::array<double, DIM>& dx,
    const EOSParams<NMAT>& params,
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
    for (int i = 0; i < static_cast<int>(U.size()); ++i) {

        const Primitive<DIM, NMAT> P = cons_to_prim<DIM, NMAT, EOS>(
            U[i],
            params
        );

        const double c = EOS::template mixture_sound_speed<NMAT>(
            P.alpha,
            P.rho,
            P.p,
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

