#pragma once

#include <vector>
#include <array>
#include <stdexcept>

#include "src/dim/state.hpp"
#include "src/dim/eos.hpp"
#include "src/dim/eos_params.hpp"
#include "src/dim/solver/cfl.hpp"

// Advance modules
#include "src/dim/solver/advance/geometry.hpp"
#include "src/dim/solver/advance/line_ops.hpp"
#include "src/dim/solver/advance/sweep_core.hpp"
#include "src/dim/solver/advance/boundary.hpp"


// [0] Step Result
template<int DIM, int NMAT>
struct StepResult {
    std::vector<Conserved<DIM, NMAT>> U_new;
    double dt;
};


// [1] Advance One Step (Diffuse Interface)
template<int DIM, int NMAT, typename EOS>
inline StepResult<DIM, NMAT> advance_one_step(
    const std::vector<Conserved<DIM, NMAT>>& U,
    const std::array<int, DIM>& N,
    const std::array<double, DIM>& dx,
    const EOSParams<NMAT>& params,
    double cfl,
    double dt_max
)
{
    const int Ntot = static_cast<int>(U.size());

    // [1.1] Validation
    if (Ntot == 0) {
        throw std::runtime_error("advance_one_step: empty state");
    }

    int expected = 1;
    for (int d = 0; d < DIM; ++d) {
        expected *= N[d];
    }

    if (expected != Ntot) {
        throw std::runtime_error("advance_one_step: grid size mismatch");
    }


    // [1.2] CFL timestep
    const double dt = compute_dt_cfl<DIM, NMAT, EOS>(
        U,
        dx,
        params,
        cfl,
        dt_max
    );


    // [1.3] Directional splitting (Godunov)
    std::vector<Conserved<DIM, NMAT>> U_stage = U;

    for (int dir = 0; dir < DIM; ++dir) {

        std::vector<Conserved<DIM, NMAT>> U_next = U_stage;

        sweep_direction_dispatch<DIM, NMAT, EOS>(
            dir,
            U_stage,
            N,
            dx,
            params,
            dt,
            U_next
        );

        apply_boundary_conditions<DIM, NMAT>(
            U_next,
            N
        );

        U_stage.swap(U_next);
    }


    // [1.4] Return result
    return {U_stage, dt};
}