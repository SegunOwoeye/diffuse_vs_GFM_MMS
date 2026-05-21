#pragma once

#include <array>
#include <stdexcept>
#include <vector>

#include "src/dim/solver/cfl.hpp"
#include "src/dim/solver/advance/boundary.hpp"
#include "src/dim/solver/advance/geometry.hpp"
#include "src/dim/solver/advance/line_ops.hpp"
#include "src/dim/solver/advance/sweep_core.hpp"
#include "src/io/config.hpp"

namespace dim {
    // [0] Step Result
    template<int DIM>
    struct StepResult {
        std::vector<State<DIM>> U_new;
        double dt = 0.0;
    };

    // [1] Advance One Step (Diffuse Interface)
    template<int DIM>
    inline StepResult<DIM> advance_one_step(
        const std::vector<State<DIM>>& U,
        const std::array<int, DIM>& N,
        const std::array<double, DIM>& dx,
        const EOSParams& params,
        const Config<DIM>& cfg,
        double cfl,
        double dt_max
    )
    {
        if (U.empty()) {
            throw std::runtime_error("dim::advance_one_step: empty state");
        }

        const int expected = total_cells<DIM>(N);
        if (expected != static_cast<int>(U.size())) {
            throw std::runtime_error("dim::advance_one_step: grid size mismatch");
        }

        const double dt = compute_dt_cfl<DIM>(U, dx, params, cfl, dt_max);

        std::vector<State<DIM>> U_stage = U;
        for (int dir = 0; dir < DIM; ++dir) {
            std::vector<State<DIM>> U_next = U_stage;
            sweep_direction_dispatch<DIM>(dir, U_stage, N, dx, params, dt, U_next);
            apply_boundary_conditions<DIM>(U_next, N, cfg);
            U_stage.swap(U_next);
        }

        // Return result
        return {U_stage, dt};
    }

} 
