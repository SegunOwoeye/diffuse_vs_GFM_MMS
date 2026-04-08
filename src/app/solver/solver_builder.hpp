#pragma once

#include <array>
#include <vector>

#include "src/app/levelset/initial_levelset.hpp"
#include "src/euler/solver/solver_context.hpp"
#include "src/euler/eos_params.hpp"
#include "src/io/config.hpp"


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

    ctx.phi_list = ls_data.phi_list;
    ctx.phi_material_ids = ls_data.phi_material_ids;
    ctx.background_material_id = ls_data.background_material_id;

    ctx.reinit_enabled = (cfg.interface_method == "GFM" && cfg.use_level_set && !ctx.phi_list.empty());
    ctx.reinit_frequency = (cfg.reinit_interval > 0) ? cfg.reinit_interval : 5;
    ctx.reinit_iterations = 5;

    return ctx;
}