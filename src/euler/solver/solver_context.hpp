#pragma once

#include <array>
#include <stdexcept>
#include <vector>

#include "src/euler/eos_params.hpp"


// [0] Solver Context
// Holds all configuration and grid/material data required by the solver
template<int DIM>
struct SolverContext {

    // [0.1] Grid
    std::array<int, DIM> N;
    std::array<double, DIM> dx;

    // [0.2] Time stepping
    double cfl = 0.5;
    double dt_max = 1e-3;

    // [0.3] Level set
    int reinit_iterations = 10;

    // [0.4] Materials
    std::vector<int> material_id;
    std::vector<EOSParams> material_params;

    // [0.5] Basic validation
    inline void validate(int total_cells) const
    {
        if (total_cells <= 0) {
            throw std::runtime_error("SolverContext: invalid total cell count");
        }

        if (cfl <= 0.0) {
            throw std::runtime_error("SolverContext: cfl must be positive");
        }

        if (dt_max <= 0.0) {
            throw std::runtime_error("SolverContext: dt_max must be positive");
        }

        for (int d = 0; d < DIM; ++d) {
            if (N[d] <= 0) {
                throw std::runtime_error("SolverContext: invalid grid size");
            }

            if (dx[d] <= 0.0) {
                throw std::runtime_error("SolverContext: grid spacing must be positive");
            }
        }

        if (!material_id.empty() && static_cast<int>(material_id.size()) != total_cells) {
            throw std::runtime_error("SolverContext: material_id size mismatch");
        }

        if (!material_id.empty() && material_params.empty()) {
            throw std::runtime_error("SolverContext: material_params required");
        }
    }
};




