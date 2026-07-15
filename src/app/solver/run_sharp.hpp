#pragma once

#include <array>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
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


// [0.1] Build rGFM diagnostic output path
template<int DIM>
inline std::filesystem::path build_rgfm_diagnostics_filename(
    const Config<DIM>& cfg,
    const std::array<int, DIM>& N
)
{
    std::string stem = cfg.output_prefix;

    for (int d = 0; d < DIM; ++d) {
        stem += "_N" + std::to_string(N[d]);
    }

    stem += "_rgfm_diagnostics.csv";

    const std::filesystem::path dir =
        std::filesystem::path(cfg.output_dir) / cfg.output_prefix;

    return dir / stem;
}


// [0.2] Append one sampled step of rGFM interface-pair diagnostics
template<int DIM>
inline void append_rgfm_diagnostics(
    const Config<DIM>& cfg,
    const std::array<int, DIM>& N,
    int step,
    double time,
    const std::vector<RGFMDiagnosticRow<DIM>>& rows
)
{
    if (rows.empty()) {
        return;
    }

    const std::filesystem::path filename =
        build_rgfm_diagnostics_filename<DIM>(cfg, N);

    if (!filename.parent_path().empty()) {
        std::filesystem::create_directories(filename.parent_path());
    }

    const bool write_header = !std::filesystem::exists(filename);
    std::ofstream file(filename, std::ios::app);

    if (!file) {
        throw std::runtime_error(
            "append_rgfm_diagnostics: failed to open " + filename.string()
        );
    }

    file << std::setprecision(17);

    if (write_header) {
        file << "step,time,interface_id,neg_id,pos_id";

        for (int d = 0; d < DIM; ++d) {
            file << ",neg_i" << d;
        }

        for (int d = 0; d < DIM; ++d) {
            file << ",pos_i" << d;
        }

        file << ",neg_mat,pos_mat,phi_neg,phi_pos";

        for (int d = 0; d < DIM; ++d) {
            file << ",normal" << d;
        }

        file << ",normal_magnitude,rho_neg,rho_pos,p_neg,p_pos";
        file << ",un_neg,un_pos,p_star,un_star,un_applied";
        file << ",pair_alignment,pair_distance,score_neg,score_pos\n";
    }

    for (const auto& row : rows) {
        file << step << "," << time << ","
             << row.interface_id << ","
             << row.neg_id << ","
             << row.pos_id;

        for (int d = 0; d < DIM; ++d) {
            file << "," << row.neg_idx[d];
        }

        for (int d = 0; d < DIM; ++d) {
            file << "," << row.pos_idx[d];
        }

        file << "," << row.neg_mat
             << "," << row.pos_mat
             << "," << row.phi_neg
             << "," << row.phi_pos;

        for (int d = 0; d < DIM; ++d) {
            file << "," << row.normal[d];
        }

        file << "," << row.normal_magnitude
             << "," << row.rho_neg
             << "," << row.rho_pos
             << "," << row.p_neg
             << "," << row.p_pos
             << "," << row.un_neg
             << "," << row.un_pos
             << "," << row.p_star
             << "," << row.un_star
             << "," << row.un_applied
             << "," << row.pair_alignment
             << "," << row.pair_distance
             << "," << row.score_neg
             << "," << row.score_pos
             << "\n";
    }
}


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
            ctx.level_set_grid,
            ctx.level_set_component_policy
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
    const auto openmp_runtime = runtime::observe_openmp_runtime();
    const auto wall_start = std::chrono::steady_clock::now();
    SolverPhaseTimings phase_timings{};
    const bool track_conservation = app_io::conservation_tracking_enabled();
    const int conservation_interval = app_io::conservation_tracking_interval();
    std::optional<app_io::ConservationReport<DIM>> conservation_report;

    if (track_conservation) {
        const auto conservation_start = std::chrono::steady_clock::now();
        const auto initial_totals =
            app_io::compute_sharp_conservation_totals<DIM>(
                U,
                ctx.material_id,
                ctx.dx,
                static_cast<int>(ctx.material_params.size())
            );

        conservation_report.emplace(cfg, N, initial_totals);
        conservation_report->write(step, time, initial_totals);
        const auto conservation_end = std::chrono::steady_clock::now();
        phase_timings.conservation_seconds +=
            std::chrono::duration<double>(conservation_end - conservation_start).count();
    }

    const auto* phi_output =
        (cfg.interface_method == "GFM" && !ctx.phi_list.empty())
            ? &ctx.phi_list
            : nullptr;

    auto write_snapshot = [&](double snapshot_time) {
        const std::string suffix = format_time_tag(snapshot_time);
        const auto output_start = std::chrono::steady_clock::now();
        write_numerical_output<DIM, EOS>(
            cfg,
            N,
            U,
            ctx.material_id,
            material_params,
            phi_output,
            suffix
        );
        const auto output_end = std::chrono::steady_clock::now();
        phase_timings.output_seconds +=
            std::chrono::duration<double>(output_end - output_start).count();
    };

    while (next_output_index < cfg.output_times.size() &&
           cfg.output_times[next_output_index] <= 1e-14)
    {
        write_snapshot(cfg.output_times[next_output_index]);
        ++next_output_index;
    }

    while (time < cfg.tfinal - 1e-14) {
        const double next_output_time =
            (next_output_index < cfg.output_times.size())
                ? cfg.output_times[next_output_index]
                : cfg.tfinal;

        ctx.dt_max = next_output_time - time;

        StepResult<DIM> result = advance_one_step<DIM, EOS>(U, ctx);
        phase_timings.add_step(result.phase_timings);

        if (result.dt <= 0.0) {
            throw std::runtime_error("run_sharp_interface_case: non-positive timestep");
        }

        if (cfg.rgfm_diagnostics &&
            (step % cfg.rgfm_diagnostics_interval == 0 ||
             time + result.dt >= cfg.tfinal - 1e-14))
        {
            append_rgfm_diagnostics<DIM>(
                cfg,
                N,
                step,
                time,
                result.rgfm_diagnostic_rows
            );
        }

        U = result.U_new;
        ctx.phi_list = result.phi_list_new;

        if (ctx.reassign_material_from_phi && !ctx.phi_list.empty()) {
            assign_material_ids_from_phi<DIM>(
                ctx.phi_list,
                ctx.tracked_interfaces,
                ctx.background_material_id,
                ctx.material_id,
                ctx.level_set_grid,
                ctx.level_set_component_policy
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
            const auto conservation_start = std::chrono::steady_clock::now();
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
            const auto conservation_end = std::chrono::steady_clock::now();
            phase_timings.conservation_seconds +=
                std::chrono::duration<double>(conservation_end - conservation_start).count();
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
            const auto conservation_start = std::chrono::steady_clock::now();
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
            const auto conservation_end = std::chrono::steady_clock::now();
            phase_timings.conservation_seconds +=
                std::chrono::duration<double>(conservation_end - conservation_start).count();
        }
    }

    const auto wall_end = std::chrono::steady_clock::now();
    const double wall_seconds =
        std::chrono::duration<double>(wall_end - wall_start).count();

    if (cfg.output_times.empty()) {
        const auto output_start = std::chrono::steady_clock::now();
        write_numerical_output<DIM, EOS>(
            cfg,
            N,
            U,
            ctx.material_id,
            material_params,
            phi_output
        );
        const auto output_end = std::chrono::steady_clock::now();
        phase_timings.output_seconds +=
            std::chrono::duration<double>(output_end - output_start).count();
    }

    app_io::write_runtime_report<DIM>(
        cfg,
        N,
        step,
        time,
        wall_seconds,
        &phase_timings,
        &openmp_runtime
    );

    #if APP_DIM == 1
    if constexpr (DIM == 1) {
        write_exact_output_1d<EOS>(cfg);
    }
    #endif
}
