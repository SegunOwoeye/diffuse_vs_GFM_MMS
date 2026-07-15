#pragma once

#include <array>
#include <chrono>
#include <stdexcept>
#include <vector>

#include "src/core/phase_timings.hpp"
#include "src/core/openmp_policy.hpp"
#include "src/dim/solver/cfl.hpp"
#include "src/dim/solver/advance/boundary.hpp"
#include "src/dim/solver/advance/geometry.hpp"
#include "src/dim/solver/advance/line_ops.hpp"
#include "src/dim/solver/advance/sweep_core.hpp"
#include "src/fv/solver/directional_update.hpp"
#include "src/io/config.hpp"

namespace dim {
    // [0] Step Result
    template<int DIM>
    struct StepResult {
        std::vector<State<DIM>> U_new;
        double dt = 0.0;
        SolverPhaseTimings phase_timings;
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

        SolverPhaseTimings phase_timings{};
        const auto cfl_start = std::chrono::steady_clock::now();
        const double dt = (cfg.time_update == "unsplit")
            ? compute_dt_cfl_unsplit<DIM>(U, dx, params, cfg.dim_lambda_model, cfl, dt_max)
            : compute_dt_cfl<DIM>(U, dx, params, cfg.dim_lambda_model, cfl, dt_max);
        const auto cfl_end = std::chrono::steady_clock::now();
        phase_timings.cfl_seconds =
            std::chrono::duration<double>(cfl_end - cfl_start).count();

        std::vector<State<DIM>> U_stage = U;

        auto sweep = [&](int dir,
                         const std::vector<State<DIM>>& U_in,
                         std::vector<State<DIM>>& U_out)
        {
            sweep_direction_dispatch<DIM>(
                dir,
                U_in,
                N,
                dx,
                params,
                dt,
                cfg.dim_alpha_source_floor,
                cfg.dim_lambda_model,
                U_out
            );
        };

        if (cfg.time_update == "unsplit") {
            auto accumulate_delta = [&](int,
                                        std::vector<State<DIM>>& U_accum,
                                        const std::vector<State<DIM>>& U_dir,
                                        const std::vector<State<DIM>>& U_base)
            {
                #pragma omp parallel for if(runtime::openmp_should_parallelize_cells(U_base.size()))
                for (int id = 0; id < static_cast<int>(U_base.size()); ++id) {
                    for (int k = 0; k < static_cast<int>(U_base[id].partial_mass.size()); ++k) {
                        U_accum[id].partial_mass[k] +=
                            U_dir[id].partial_mass[k] - U_base[id].partial_mass[k];
                    }

                    for (int d = 0; d < DIM; ++d) {
                        U_accum[id].mom[d] += U_dir[id].mom[d] - U_base[id].mom[d];
                    }

                    U_accum[id].E += U_dir[id].E - U_base[id].E;

                    for (int k = 0; k < static_cast<int>(U_base[id].alpha.size()); ++k) {
                        U_accum[id].alpha[k] += U_dir[id].alpha[k] - U_base[id].alpha[k];
                    }

                    repair_state<DIM>(U_accum[id], params);
                }
            };

            auto after_accumulation = [&](std::vector<State<DIM>>& U_accum)
            {
                const auto boundary_start = std::chrono::steady_clock::now();
                apply_boundary_conditions<DIM>(U_accum, N, cfg, params);
                const auto boundary_end = std::chrono::steady_clock::now();
                phase_timings.boundary_seconds +=
                    std::chrono::duration<double>(boundary_end - boundary_start).count();
            };

            const auto sweep_start = std::chrono::steady_clock::now();
            fv::advance_unsplit_directions<DIM>(
                U,
                U_stage,
                sweep,
                accumulate_delta,
                after_accumulation
            );
            const auto sweep_end = std::chrono::steady_clock::now();
            phase_timings.sweep_seconds =
                std::chrono::duration<double>(sweep_end - sweep_start).count()
                - phase_timings.boundary_seconds;
        }
        else {
            auto after_direction = [&](int, std::vector<State<DIM>>& U_next)
            {
                const auto boundary_start = std::chrono::steady_clock::now();
                apply_boundary_conditions<DIM>(U_next, N, cfg, params);
                const auto boundary_end = std::chrono::steady_clock::now();
                phase_timings.boundary_seconds +=
                    std::chrono::duration<double>(boundary_end - boundary_start).count();
            };

            const auto sweep_start = std::chrono::steady_clock::now();
            fv::advance_split_directions<DIM>(
                U_stage,
                sweep,
                after_direction
            );
            const auto sweep_end = std::chrono::steady_clock::now();
            phase_timings.sweep_seconds =
                std::chrono::duration<double>(sweep_end - sweep_start).count()
                - phase_timings.boundary_seconds;
        }

        // Return result
        return {U_stage, dt, phase_timings};
    }

} 
