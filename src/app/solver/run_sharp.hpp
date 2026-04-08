#pragma once

#include <array>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "src/setup/initial_conditions.hpp"
#include "src/euler/solver/advance_step.hpp"
#include "src/euler/solver/solver_context.hpp"
#include "src/euler/level_set/level_set.hpp"
#include "src/io/write_csv.hpp"
#include "src/io/compute_exact_solution.hpp"
#include "src/io/write_exact_csv.hpp"
#include "src/euler/eos_params.hpp"
#include "src/io/config.hpp"

#include "src/app/levelset/initial_levelset.hpp"
#include "src/app/solver/solver_builder.hpp"
#include "src/app/io/euler_output_utils.hpp"


// [0] Run SM or GFM sharp-interface case
template<int DIM, typename EOS>
inline void run_sharp_interface_case(
    const Config<DIM>& cfg,
    const std::array<int, DIM>& N,
    const std::vector<EOSParams>& material_params
)
{
    std::vector<Conserved<DIM>> U;
    std::vector<int> material_id;

    initialise_from_config<DIM, EOS>(
        U,
        material_id,
        cfg,
        N
    );

    InitialLevelSetData<DIM> ls_data{};

    if (cfg.interface_method == "GFM") {
        ls_data = initialise_phi_data_from_regions<DIM>(cfg, N);
    }

    SolverContext<DIM> ctx = build_solver_context<DIM>(
        cfg,
        N,
        material_id,
        material_params,
        ls_data
    );

    // [0.1] Make material assignment consistent with initial level sets
    if (ctx.reassign_material_from_phi && !ctx.phi_list.empty()) {
        assign_material_ids_from_phi<DIM>(
            ctx.phi_list,
            ctx.phi_material_ids,
            ctx.background_material_id,
            ctx.material_id,
            ctx.level_set_grid
        );
    }

    double time = 0.0;
    int step = 0;

    while (time < cfg.tfinal - 1e-14) {
        ctx.dt_max = cfg.tfinal - time;

        StepResult<DIM> result = advance_one_step<DIM, EOS>(U, ctx);

        if (result.dt <= 0.0) {
            throw std::runtime_error("run_sharp_interface_case: non-positive timestep");
        }

        U = result.U_new;
        ctx.phi_list = result.phi_list_new;

        if (ctx.reassign_material_from_phi && !ctx.phi_list.empty()) {
            assign_material_ids_from_phi<DIM>(
                ctx.phi_list,
                ctx.phi_material_ids,
                ctx.background_material_id,
                ctx.material_id,
                ctx.level_set_grid
            );
        }

        if (ctx.reinit_enabled &&
            !ctx.phi_list.empty() &&
            step > 0 &&
            step % ctx.reinit_frequency == 0) {

            for (int k = 0; k < ctx.n_interfaces(); ++k) {
                ctx.phi_list[k] = reinitialise_phi<DIM>(
                    ctx.phi_list[k],
                    ctx.level_set_grid,
                    ctx.reinit_iterations
                );
            }

            assign_material_ids_from_phi<DIM>(
                ctx.phi_list,
                ctx.phi_material_ids,
                ctx.background_material_id,
                ctx.material_id,
                ctx.level_set_grid
            );
        }

        time += result.dt;
        ++step;

        if (step % 25 == 0 || time >= cfg.tfinal - 1e-14) {
            std::cout << "step=" << step
                      << " time=" << time
                      << " dt=" << result.dt << "\n";
        }
    }

    write_numerical_output<DIM, EOS>(
        cfg,
        N,
        U,
        ctx.material_id,
        material_params
    );

    #if APP_DIM == 1
    if constexpr (DIM == 1) {
        write_exact_output_1d<EOS>(cfg);
    }
    #endif
}


