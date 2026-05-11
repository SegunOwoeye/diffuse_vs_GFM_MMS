#pragma once

#include <array>
#include <stdexcept>
#include <string>
#include <vector>

#include "src/app/levelset/initial_levelset.hpp"
#include "src/euler/solver/solver_context.hpp"
#include "src/euler/eos_params.hpp"
#include "src/io/config.hpp"


// [0] Translate config boundary-condition names to solver enum values
inline BoundaryConditionType boundary_condition_from_config(
    const std::string& name
)
{
    if (name == "transmissive" ||
        name == "zero_gradient" ||
        name == "outflow") {
        return BoundaryConditionType::transmissive;
    }

    if (name == "reflective" ||
        name == "reflection") {
        return BoundaryConditionType::reflective;
    }

    throw std::runtime_error("Unknown boundary condition: " + name);
}


// [0] Build solver context from config and initialised state data
template<int DIM>
inline SolverContext<DIM> build_solver_context(
    const Config<DIM>& cfg,
    const std::array<int, DIM>& N,
    const std::vector<int>& material_id,
    const std::vector<EOSParams>& material_params,
    const InitialLevelSetData<DIM>& ls_data
)
{
    SolverContext<DIM> ctx{};

    ctx.N = N;

    for (int d = 0; d < DIM; ++d) {
        ctx.dx[d] = (cfg.domain_max[d] - cfg.domain_min[d]) / static_cast<double>(N[d]);
    }

    ctx.cfl = cfg.cfl;
    ctx.dt_max = 1e-3;

    ctx.material_id = material_id;
    ctx.material_params = material_params;

    ctx.initialise_level_set_grid();

    for (int d = 0; d < DIM; ++d) {
        ctx.bc_lo[d] = boundary_condition_from_config(cfg.bc_lo[d]);
        ctx.bc_hi[d] = boundary_condition_from_config(cfg.bc_hi[d]);
    }

    ctx.phi_list = ls_data.phi_list;
    ctx.tracked_interfaces = ls_data.tracked_interfaces;
    ctx.background_material_id = ls_data.background_material_id;

    ctx.advect_level_set = (cfg.interface_method == "GFM" && cfg.use_level_set && !ctx.phi_list.empty());
    ctx.reassign_material_from_phi = ctx.advect_level_set;

    ctx.reinit_enabled = (
        cfg.interface_method == "GFM" &&
        cfg.use_level_set &&
        !ctx.phi_list.empty() &&
        cfg.reinit_interval > 0
    );
    ctx.reinit_frequency = ctx.reinit_enabled ? cfg.reinit_interval : 0;
    ctx.reinit_iterations = 5;
    ctx.completed_steps = 0;

    return ctx;
}
