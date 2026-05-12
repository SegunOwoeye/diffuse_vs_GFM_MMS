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
    std::vector<dim::State<DIM>> U;
    dim::initialise_dim_from_config<DIM>(U, cfg, N, material_params);

    std::array<double, DIM> dx{};
    for (int d = 0; d < DIM; ++d) {
        dx[d] = (cfg.domain_max[d] - cfg.domain_min[d]) / static_cast<double>(N[d]);
    }

    double time = 0.0;
    int step = 0;
    const auto wall_start = std::chrono::steady_clock::now();
    const bool track_conservation = app_io::conservation_tracking_enabled();
    const int conservation_interval = app_io::conservation_tracking_interval();
    std::optional<app_io::ConservationReport<DIM>> conservation_report;

    if (track_conservation) {
        const auto initial_totals =
            app_io::compute_dim_conservation_totals<DIM>(
                U,
                dx,
                material_params.nmat()
            );

        conservation_report.emplace(cfg, N, initial_totals);
        conservation_report->write(step, time, initial_totals);
    }

    while (time < cfg.tfinal - 1e-14) {
        const dim::StepResult<DIM> result = dim::advance_one_step<DIM>(
            U,
            N,
            dx,
            material_params,
            cfg.cfl,
            cfg.tfinal - time
        );

        if (result.dt <= 0.0) {
            throw std::runtime_error("run_dim_case: non-positive timestep");
        }

        U = result.U_new;
        time += result.dt;
        ++step;

        if (conservation_report.has_value() &&
            (step % conservation_interval == 0 ||
             time >= cfg.tfinal - 1e-14))
        {
            conservation_report->write(
                step,
                time,
                app_io::compute_dim_conservation_totals<DIM>(
                    U,
                    dx,
                    material_params.nmat()
                )
            );
        }
    }

    const auto wall_end = std::chrono::steady_clock::now();
    const double wall_seconds =
        std::chrono::duration<double>(wall_end - wall_start).count();

    dim_app::write_numerical_output<DIM>(cfg, N, U, material_params);
    app_io::write_runtime_report<DIM>(cfg, N, step, time, wall_seconds);
}
