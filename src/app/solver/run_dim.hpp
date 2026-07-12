#pragma once

#include <array>
#include <chrono>
#include <optional>
#include <stdexcept>
#include <vector>

#include "src/app/io/conservation_report.hpp"
#include "src/app/io/dim_output_utils.hpp"
#include "src/app/io/runtime_report.hpp"
#include "src/dim/eos_params.hpp"
#include "src/dim/solver/advance_step.hpp"
#include "src/io/config.hpp"
#include "src/setup/dim_initial_conditions.hpp"

// [0] Run Diffuse Interface Method (DIM)
template<int DIM>
inline void run_dim_case(
    const Config<DIM>& cfg,
    const std::array<int, DIM>& N,
    const dim::EOSParams& material_params
)
{
    if (cfg.barton_solid_material >= 0) {
        throw std::runtime_error(
            "run_dim_case: base DIM fluid solver does not run Barton solids; use the Barton DIM extension"
        );
    }

    const dim::EOSParams& active_material_params = material_params;
    std::vector<dim::State<DIM>> U;
    dim::initialise_dim_from_config<DIM>(U, cfg, N, active_material_params);

    std::array<double, DIM> dx{};
    for (int d = 0; d < DIM; ++d) {
        dx[d] = (cfg.domain_max[d] - cfg.domain_min[d]) / static_cast<double>(N[d]);
    }

    double time = 0.0;
    int step = 0;
    std::size_t next_output_index = 0;
    const auto wall_start = std::chrono::steady_clock::now();
    SolverPhaseTimings phase_timings{};
    const bool track_conservation = app_io::conservation_tracking_enabled();
    const int conservation_interval = app_io::conservation_tracking_interval();
    std::optional<app_io::ConservationReport<DIM>> conservation_report;

    if (track_conservation) {
        const auto conservation_start = std::chrono::steady_clock::now();
        const auto initial_totals =
            app_io::compute_dim_conservation_totals<DIM>(
                U,
                dx,
                active_material_params.nmat()
            );

        conservation_report.emplace(cfg, N, initial_totals);
        conservation_report->write(step, time, initial_totals);
        const auto conservation_end = std::chrono::steady_clock::now();
        phase_timings.conservation_seconds +=
            std::chrono::duration<double>(conservation_end - conservation_start).count();
    }

    auto write_snapshot = [&](double snapshot_time) {
        const auto output_start = std::chrono::steady_clock::now();
        dim_app::write_numerical_output<DIM>(
            cfg,
            N,
            U,
            active_material_params,
            dim_app::format_time_tag(snapshot_time)
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

        const dim::StepResult<DIM> result = dim::advance_one_step<DIM>(
            U,
            N,
            dx,
            active_material_params,
            cfg,
            cfg.cfl,
            next_output_time - time
        );
        phase_timings.add_step(result.phase_timings);

        if (result.dt <= 0.0) {
            throw std::runtime_error("run_dim_case: non-positive timestep");
        }

        U = result.U_new;
        time += result.dt;
        ++step;

        if (conservation_report.has_value()) {
            const auto conservation_start = std::chrono::steady_clock::now();
            conservation_report->accumulate_fluxes(
                result.dt,
                app_io::compute_dim_boundary_flux<DIM>(
                    U,
                    N,
                    dx,
                    active_material_params
                )
            );
            const auto conservation_end = std::chrono::steady_clock::now();
            phase_timings.conservation_seconds +=
                std::chrono::duration<double>(conservation_end - conservation_start).count();
        }

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
                app_io::compute_dim_conservation_totals<DIM>(
                    U,
                    dx,
                    active_material_params.nmat()
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
        dim_app::write_numerical_output<DIM>(cfg, N, U, active_material_params);
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
        &phase_timings
    );
}
