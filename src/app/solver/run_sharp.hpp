#pragma once

#include <array>
#include <chrono>
#include <optional>
#include <stdexcept>
#include <vector>

#include "src/app/io/conservation_report.hpp"
#include "src/app/io/runtime_report.hpp"
#include "src/setup/initial_conditions.hpp"
#include "src/sim/solver/advance_step.hpp"
#include "src/sim/solver/solver_context.hpp"
#include "src/sim/level_set/level_set.hpp"
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
        ls_data = initialise_phi_data_from_regions<DIM>(cfg, N, material_id);
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
            ctx.tracked_interfaces,
            ctx.background_material_id,
            ctx.material_id,
            ctx.level_set_grid
        );

        fill_small_enclosed_background_cavities<DIM>(
            ctx.material_id,
            ctx.tracked_interfaces,
            ctx.background_material_id,
            ctx.level_set_grid
        );
    }

    double time = 0.0;
    int step = 0;
    std::size_t next_output_index = 0;
    const auto wall_start = std::chrono::steady_clock::now();
    const bool track_conservation = app_io::conservation_tracking_enabled();
    const int conservation_interval = app_io::conservation_tracking_interval();
    std::optional<app_io::ConservationReport<DIM>> conservation_report;

    if (track_conservation) {
        const auto initial_totals =
            app_io::compute_sharp_conservation_totals<DIM>(
                U,
                ctx.material_id,
                ctx.dx,
                static_cast<int>(ctx.material_params.size())
            );

        conservation_report.emplace(cfg, N, initial_totals);
        conservation_report->write(step, time, initial_totals);
    }

    const auto* phi_output =
        (cfg.interface_method == "GFM" && !ctx.phi_list.empty())
            ? &ctx.phi_list
            : nullptr;

    auto write_snapshot = [&](double snapshot_time) {
        const std::string suffix = format_time_tag(snapshot_time);
        write_numerical_output<DIM, EOS>(
            cfg,
            N,
            U,
            ctx.material_id,
            material_params,
            phi_output,
            suffix
        );
    };

    while (time < cfg.tfinal - 1e-14) {
        const double next_output_time =
            (next_output_index < cfg.output_times.size())
                ? cfg.output_times[next_output_index]
                : cfg.tfinal;

        ctx.dt_max = next_output_time - time;

        StepResult<DIM> result = advance_one_step<DIM, EOS>(U, ctx);

        if (result.dt <= 0.0) {
            throw std::runtime_error("run_sharp_interface_case: non-positive timestep");
        }

        U = result.U_new;
        ctx.phi_list = result.phi_list_new;

        if (ctx.reassign_material_from_phi && !ctx.phi_list.empty()) {
            assign_material_ids_from_phi<DIM>(
                ctx.phi_list,
                ctx.tracked_interfaces,
                ctx.background_material_id,
                ctx.material_id,
                ctx.level_set_grid
            );

            fill_small_enclosed_background_cavities<DIM>(
                ctx.material_id,
                ctx.tracked_interfaces,
                ctx.background_material_id,
                ctx.level_set_grid
            );
        }

        time += result.dt;
        ++step;

        if (conservation_report.has_value()) {
            const auto boundary_flux =
                app_io::compute_sharp_boundary_flux<DIM, EOS>(
                    U,
                    ctx.material_id,
                    N,
                    ctx.dx,
                    ctx.material_params
                );
            const auto interface_flux_mismatch =
                app_io::compute_sharp_interface_flux_mismatch<DIM, EOS>(
                    U,
                    ctx.material_id,
                    N,
                    ctx.dx,
                    ctx.material_params
                );
            conservation_report->accumulate_fluxes(
                result.dt,
                boundary_flux,
                &interface_flux_mismatch
            );
        }

        phi_output = (cfg.interface_method == "GFM" && !ctx.phi_list.empty())
                ? &ctx.phi_list
                : nullptr;

        while (next_output_index < cfg.output_times.size() &&
               time >= cfg.output_times[next_output_index] - 1e-14)
        {
            write_snapshot(cfg.output_times[next_output_index]);
            ++next_output_index;
        }

        if (conservation_report.has_value() &&
            (step % conservation_interval == 0 ||
             time >= cfg.tfinal - 1e-14))
        {
            conservation_report->write(
                step,
                time,
                app_io::compute_sharp_conservation_totals<DIM>(
                    U,
                    ctx.material_id,
                    ctx.dx,
                    static_cast<int>(ctx.material_params.size())
                )
            );
        }
    }

    const auto wall_end = std::chrono::steady_clock::now();
    const double wall_seconds =
        std::chrono::duration<double>(wall_end - wall_start).count();

    if (cfg.output_times.empty()) {
        write_numerical_output<DIM, EOS>(
            cfg,
            N,
            U,
            ctx.material_id,
            material_params,
            phi_output
        );
    }

    app_io::write_runtime_report<DIM>(cfg, N, step, time, wall_seconds);

    #if APP_DIM == 1
    if constexpr (DIM == 1) {
        write_exact_output_1d<EOS>(cfg);
    }
    #endif
}

