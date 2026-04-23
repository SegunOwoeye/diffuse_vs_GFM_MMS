#pragma once

#include <array>
#include <stdexcept>
#include <vector>

#include "src/euler/eos_params.hpp"
#include "src/euler/gfm/tracked_interface.hpp"
#include "src/euler/level_set/level_set_core.hpp"


// [0] Boundary Condition Type
enum class BoundaryConditionType {
    transmissive,
    reflective
};


// [1] Solver Context
template<int DIM>
struct SolverContext {

    // [1.1] Grid
    std::array<int, DIM> N{};
    std::array<double, DIM> dx{};

    // [1.2] Time stepping
    double cfl = 0.5;
    double dt_max = 1e-3;

    // [1.3] Level set / GFM
    LevelSetGrid<DIM> level_set_grid{};
    std::vector<std::vector<double>> phi_list;
    std::vector<TrackedInterface> tracked_interfaces;
    int background_material_id = 0;

    bool reinit_enabled = false;
    int reinit_frequency = 5;
    int reinit_iterations = 10;
    int completed_steps = 0;

    // [1.4] Interface-control flags
    bool advect_level_set = true;
    bool reassign_material_from_phi = true;
    bool use_axis_normals_in_1d = true;

    // [1.5] Physical boundary conditions
    std::array<BoundaryConditionType, DIM> bc_lo{};
    std::array<BoundaryConditionType, DIM> bc_hi{};

    // [1.6] Materials
    std::vector<int> material_id;
    std::vector<EOSParams> material_params;

    // [1.7] Build level set grid from solver grid
    inline void initialise_level_set_grid()
    {
        level_set_grid = make_level_set_grid<DIM>(N, dx);
    }

    // [1.8] Set default boundary conditions
    inline void initialise_boundary_conditions(
        BoundaryConditionType default_lo = BoundaryConditionType::transmissive,
        BoundaryConditionType default_hi = BoundaryConditionType::transmissive
    )
    {
        for (int d = 0; d < DIM; ++d) {
            bc_lo[d] = default_lo;
            bc_hi[d] = default_hi;
        }
    }

    // [1.9] Number of tracked interfaces
    inline int n_interfaces() const
    {
        return static_cast<int>(phi_list.size());
    }

    // [1.10] Validation
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

            if (level_set_grid.N[d] != N[d]) {
                throw std::runtime_error("SolverContext: level_set_grid size mismatch");
            }

            if (level_set_grid.dx[d] != dx[d]) {
                throw std::runtime_error("SolverContext: level_set_grid spacing mismatch");
            }
        }

        if (reinit_enabled && reinit_frequency <= 0) {
            throw std::runtime_error("SolverContext: reinit_frequency must be positive");
        }

        if (reinit_iterations < 0) {
            throw std::runtime_error("SolverContext: reinit_iterations must be non-negative");
        }

        if (!material_id.empty() && static_cast<int>(material_id.size()) != total_cells) {
            throw std::runtime_error("SolverContext: material_id size mismatch");
        }

        if (!material_id.empty() && material_params.empty()) {
            throw std::runtime_error("SolverContext: material_params required");
        }

        if (material_params.empty()) {
            throw std::runtime_error("SolverContext: material_params cannot be empty");
        }

        if (background_material_id < 0 ||
            background_material_id >= static_cast<int>(material_params.size())) {
            throw std::runtime_error("SolverContext: invalid background_material_id");
        }

        if (completed_steps < 0) {
            throw std::runtime_error("SolverContext: completed_steps must be non-negative");
        }

        if (static_cast<int>(tracked_interfaces.size()) != static_cast<int>(phi_list.size())) {
            throw std::runtime_error("SolverContext: tracked_interfaces size mismatch");
        }

        for (int k = 0; k < static_cast<int>(phi_list.size()); ++k) {
            if (static_cast<int>(phi_list[k].size()) != total_cells) {
                throw std::runtime_error("SolverContext: phi_list entry size mismatch");
            }

            if (tracked_interfaces[k].negative_material_id < 0 ||
                tracked_interfaces[k].negative_material_id >= static_cast<int>(material_params.size())) {
                throw std::runtime_error("SolverContext: invalid tracked negative material id");
            }

            if (tracked_interfaces[k].negative_material_id == background_material_id) {
                throw std::runtime_error("SolverContext: tracked interface cannot target background material");
            }

            if (tracked_interfaces[k].component_id < 0) {
                throw std::runtime_error("SolverContext: invalid tracked component id");
            }
        }
    }
};


