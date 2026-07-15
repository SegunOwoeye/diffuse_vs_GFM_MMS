#pragma once

#include <vector>
#include <array>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <stdexcept>

#include "src/euler/state.hpp"
#include "src/core/phase_timings.hpp"
#include "src/euler/eos.hpp"
#include "src/euler/eos_params.hpp"
#include "src/sim/solver/cfl.hpp"
#include "src/sim/solver/solver_context.hpp"
#include "src/euler/thermo_compute.hpp"
#include "src/sim/level_set/level_set.hpp"

// Split modules
#include "src/sim/solver/advance/geometry.hpp"
#include "src/sim/solver/advance/line_ops.hpp"
#include "src/sim/solver/advance/interface_speed.hpp"
#include "src/sim/solver/advance/sweep_core.hpp"
#include "src/sim/solver/advance/boundary.hpp"
#include "src/sim/solver/advance/material_transfer.hpp"
#include "src/sim/gfm/rgfm.hpp"
#include "src/fv/solver/directional_update.hpp"
#include "src/math/numerical_safety.hpp"



// [0] Step Result
template<int DIM>
struct StepResult {
    std::vector<Conserved<DIM>> U_new;
    std::vector<std::vector<double>> phi_list_new;
    double dt;
    std::vector<RGFMDiagnosticRow<DIM>> rgfm_diagnostic_rows;
    SolverPhaseTimings phase_timings;
};



// [1] Velocity field
template<int DIM, typename EOS>
inline std::vector<std::array<double, DIM>> build_velocity_field(
    const std::vector<Conserved<DIM>>& U,
    const std::vector<int>& material_id,
    const std::vector<EOSParams>& material_params
)
{
    const int Ntot = static_cast<int>(U.size());

    std::vector<std::array<double, DIM>> vel(Ntot);

    for (int i = 0; i < Ntot; ++i) {

        const int mat = material_id.empty() ? 0 : material_id[i];

        const EOSParams& params = material_params.empty()
            ? EOSParams{}
            : material_params[mat];

        const ThermoState<DIM> T = compute_thermo<DIM, EOS>(U[i], params);

        vel[i] = T.vel;
    }

    return vel;
}


// [1.1] Build level-set transport velocity field u_ls = Vn * n(phi)
template<int DIM>
inline std::vector<std::array<double, DIM>> build_level_set_transport_velocity(
    const std::vector<double>& normal_speed,
    const std::vector<std::array<double, DIM>>& normals
)
{
    const int Ntot = static_cast<int>(normal_speed.size());

    if (static_cast<int>(normals.size()) != Ntot) {
        throw std::runtime_error("build_level_set_transport_velocity: size mismatch");
    }

    std::vector<std::array<double, DIM>> vel_ls(Ntot);

    for (int id = 0; id < Ntot; ++id) {
        for (int d = 0; d < DIM; ++d) {
            vel_ls[id][d] = normal_speed[id] * normals[id][d];
        }
    }

    return vel_ls;
}


/* [1.2] Rebuild 1D tracked level sets from the current material map.
    This preserves component topology and removes spurious extra zero crossings.
*/
template<int DIM>
inline void rebuild_tracked_level_sets_from_material_map_1d(
    std::vector<std::vector<double>>& phi_list,
    const std::vector<int>& material_id,
    const SolverContext<DIM>& ctx
)
{
    if constexpr (DIM != 1) {
        return;
    }
    else {
        struct Component1D {
            int material_id = -1;
            int start = -1;
            int end = -1;
        };

        const int Nx = ctx.N[0];

        if (static_cast<int>(material_id.size()) != Nx) {
            throw std::runtime_error("rebuild_tracked_level_sets_from_material_map_1d: material size mismatch");
        }

        std::vector<Component1D> components;

        for (int i = 0; i < Nx; ) {
            if (material_id[i] == ctx.background_material_id) {
                ++i;
                continue;
            }

            const int mat = material_id[i];
            const int start = i;

            while (i + 1 < Nx && material_id[i + 1] == mat) {
                ++i;
            }

            components.push_back(Component1D{mat, start, i});
            ++i;
        }

        for (int k = 0; k < static_cast<int>(phi_list.size()); ++k) {
            const TrackedInterface& tracked = ctx.tracked_interfaces[k];

            if (tracked.component_id < 0 ||
                tracked.component_id >= static_cast<int>(components.size()) ||
                components[tracked.component_id].material_id != tracked.negative_material_id) {
                continue;
            }

            const Component1D& component = components[tracked.component_id];
            std::vector<double> boundaries;

            if (component.start > 0) {
                boundaries.push_back(
                    static_cast<double>(component.start) * ctx.dx[0]
                );
            }

            if (component.end < Nx - 1) {
                boundaries.push_back(
                    static_cast<double>(component.end + 1) * ctx.dx[0]
                );
            }

            if (boundaries.empty()) {
                throw std::runtime_error(
                    "rebuild_tracked_level_sets_from_material_map_1d: tracked component has no interior boundary"
                );
            }

            for (int i = 0; i < Nx; ++i) {
                const double x = (static_cast<double>(i) + 0.5) * ctx.dx[0];
                double best_dist = std::abs(x - boundaries[0]);

                for (int b = 1; b < static_cast<int>(boundaries.size()); ++b) {
                    best_dist = std::min(best_dist, std::abs(x - boundaries[b]));
                }

                phi_list[k][i] = (i >= component.start && i <= component.end)
                    ? -best_dist
                    : best_dist;
            }
        }
    }
}


/* 
    [1.3] Detect and preserve planar level sets in multidimensional
    validation runs. 
*/
template<int DIM>
inline int detect_planar_level_set_axis(
    const std::vector<double>& phi,
    const SolverContext<DIM>& ctx
)
{
    if constexpr (DIM == 1) {
        return 0;
    }
    else {
        const int Ntot = static_cast<int>(phi.size());

        if (Ntot == 0) {
            return -1;
        }

        double min_dx = ctx.dx[0];
        double scale = 0.0;

        for (int d = 1; d < DIM; ++d) {
            min_dx = std::min(min_dx, ctx.dx[d]);
        }

        for (const double value : phi) {
            scale = std::max(scale, std::abs(value));
        }

        const double tol = 1.0e-10 * std::max(scale, min_dx);

        for (int axis = 0; axis < DIM; ++axis) {
            std::vector<double> min_by_axis(
                ctx.N[axis],
                std::numeric_limits<double>::max()
            );
            std::vector<double> max_by_axis(
                ctx.N[axis],
                -std::numeric_limits<double>::max()
            );

            for (int id = 0; id < Ntot; ++id) {
                const auto idx = unflatten_index<DIM>(id, ctx.level_set_grid);
                const int a = idx[axis];

                min_by_axis[a] = std::min(min_by_axis[a], phi[id]);
                max_by_axis[a] = std::max(max_by_axis[a], phi[id]);
            }

            double max_transverse_range = 0.0;

            for (int a = 0; a < ctx.N[axis]; ++a) {
                max_transverse_range = std::max(
                    max_transverse_range,
                    max_by_axis[a] - min_by_axis[a]
                );
            }

            if (max_transverse_range <= tol) {
                return axis;
            }
        }

        return -1;
    }
}


template<int DIM>
inline std::vector<std::array<double, DIM>> compute_solver_normals(
    const std::vector<double>& phi,
    const SolverContext<DIM>& ctx
)
{
    if constexpr (DIM == 1) {
        if (ctx.use_axis_normals_in_1d) {
            const int N = static_cast<int>(phi.size());
            std::vector<std::array<double, DIM>> normals(N);

            for (int i = 0; i < N; ++i) {
                double grad = 0.0;

                if (N <= 1) {
                    grad = 1.0;
                }
                else if (i == 0) {
                    grad = phi[1] - phi[0];
                }
                else if (i == N - 1) {
                    grad = phi[N - 1] - phi[N - 2];
                }
                else {
                    grad = phi[i + 1] - phi[i - 1];

                    if (std::abs(grad) <= 1e-14) {
                        const double grad_left = phi[i] - phi[i - 1];
                        const double grad_right = phi[i + 1] - phi[i];
                        grad = (std::abs(grad_left) >= std::abs(grad_right))
                            ? grad_left
                            : grad_right;
                    }
                }

                normals[i][0] = (grad < 0.0) ? -1.0 : 1.0;
            }

            return normals;
        }
    }

    return compute_normals<DIM>(
        phi,
        ctx.level_set_grid
    );
}


template<int DIM>
inline std::vector<double> reinitialise_solver_phi(
    const std::vector<double>& phi,
    const SolverContext<DIM>& ctx
)
{
    if (ctx.level_set_reinit_method == "redistance") {
        return redistance_preserving_zero_level_set<DIM>(
            phi,
            ctx.level_set_grid,
            ctx.reinit_iterations
        );
    }

    return reinitialise_phi<DIM>(
        phi,
        ctx.level_set_grid,
        ctx.reinit_iterations,
        ctx.level_set_derivative_scheme
    );
}


template<int DIM>
inline void project_planar_level_set(
    std::vector<double>& phi,
    const SolverContext<DIM>& ctx,
    int axis
)
{
    if (axis < 0 || axis >= DIM) {
        return;
    }

    const int Ntot = static_cast<int>(phi.size());
    std::vector<double> sum(ctx.N[axis], 0.0);
    std::vector<int> count(ctx.N[axis], 0);

    for (int id = 0; id < Ntot; ++id) {
        const auto idx = unflatten_index<DIM>(id, ctx.level_set_grid);
        const int a = idx[axis];

        sum[a] += phi[id];
        count[a] += 1;
    }

    for (int a = 0; a < ctx.N[axis]; ++a) {
        if (count[a] > 0) {
            sum[a] /= static_cast<double>(count[a]);
        }
    }

    for (int id = 0; id < Ntot; ++id) {
        const auto idx = unflatten_index<DIM>(id, ctx.level_set_grid);
        phi[id] = sum[idx[axis]];
    }
}


template<int DIM>
inline int common_planar_level_set_axis(
    const std::vector<int>& planar_phi_axis
)
{
    int common_axis = -1;

    for (const int axis : planar_phi_axis) {
        if (axis < 0) {
            return -1;
        }

        if (common_axis < 0) {
            common_axis = axis;
        }
        else if (common_axis != axis) {
            return -1;
        }
    }

    return common_axis;
}


template<int DIM>
inline void zero_roundoff_planar_transverse_momentum(
    std::vector<Conserved<DIM>>& U,
    int axis
)
{
    if (axis < 0 || axis >= DIM) {
        return;
    }

    double max_axis_velocity = 0.0;
    std::array<double, DIM> max_transverse_velocity{};

    for (const auto& state : U) {
        if (state.rho <= 0.0 || !std::isfinite(state.rho)) {
            return;
        }

        max_axis_velocity = std::max(
            max_axis_velocity,
            std::abs(state.mom[axis] / state.rho)
        );

        for (int d = 0; d < DIM; ++d) {
            if (d == axis) {
                continue;
            }

            max_transverse_velocity[d] = std::max(
                max_transverse_velocity[d],
                std::abs(state.mom[d] / state.rho)
            );
        }
    }

    const double velocity_tol = std::max(
        1.0e-12,
        1.0e-8 * std::max(1.0, max_axis_velocity)
    );

    for (int d = 0; d < DIM; ++d) {
        if (d == axis || max_transverse_velocity[d] > velocity_tol) {
            continue;
        }

        for (int id = 0; id < static_cast<int>(U.size()); ++id) {
            U[id].mom[d] = 0.0;
        }
    }
}


template<int DIM>
inline void enforce_level_set_sign_from_material_map(
    std::vector<std::vector<double>>& phi_list,
    const std::vector<int>& material_id,
    const SolverContext<DIM>& ctx
)
{
    const int Ntot = static_cast<int>(material_id.size());

    if (static_cast<int>(phi_list.size()) !=
        static_cast<int>(ctx.tracked_interfaces.size())) {
        throw std::runtime_error(
            "enforce_level_set_sign_from_material_map: tracked interface mismatch"
        );
    }

    double min_dx = ctx.dx[0];
    for (int d = 1; d < DIM; ++d) {
        min_dx = std::min(min_dx, ctx.dx[d]);
    }

    const double sign_floor = geometry_tolerance(min_dx, 1e-12, 1e-14);

    for (int k = 0; k < static_cast<int>(phi_list.size()); ++k) {
        if (static_cast<int>(phi_list[k].size()) != Ntot) {
            throw std::runtime_error(
                "enforce_level_set_sign_from_material_map: level-set size mismatch"
            );
        }

        for (int id = 0; id < Ntot; ++id) {
            const double magnitude = std::max(std::abs(phi_list[k][id]), sign_floor);
            phi_list[k][id] = tracked_interface_contains_negative_material(
                    ctx.tracked_interfaces[k],
                    material_id[id])
                ? -magnitude
                : magnitude;
        }
    }
}



// [2] Advance One Step
template<int DIM, typename EOS>
inline StepResult<DIM> advance_one_step(
    const std::vector<Conserved<DIM>>& U,
    SolverContext<DIM>& ctx
)
{
    const int Ntot = static_cast<int>(U.size());

    if (Ntot == 0) {
        throw std::runtime_error("advance_one_step: empty state");
    }

    ctx.validate(Ntot);
    SolverPhaseTimings phase_timings{};


    // [2.1] Material map
    std::vector<int> material_id_current = ctx.material_id;

    if (ctx.reassign_material_from_phi) {
        assign_material_ids_from_phi<DIM>(
            ctx.phi_list,
            ctx.tracked_interfaces,
            ctx.background_material_id,
            material_id_current,
            ctx.level_set_grid,
            ctx.level_set_component_policy
        );

        fill_small_enclosed_background_cavities<DIM>(
            material_id_current,
            ctx.tracked_interfaces,
            ctx.background_material_id,
            ctx.level_set_grid
        );
    }


    // [2.2] CFL timestep
    const auto cfl_start = std::chrono::steady_clock::now();
    const double dt = (ctx.time_update == "unsplit")
        ? compute_dt_cfl_materials_unsplit<DIM, EOS>(
            U,
            material_id_current,
            ctx.material_params,
            ctx.dx,
            ctx.cfl,
            ctx.dt_max
        )
        : compute_dt_cfl_materials<DIM, EOS>(
            U,
            material_id_current,
            ctx.material_params,
            ctx.dx,
            ctx.cfl,
            ctx.dt_max
        );
    const auto cfl_end = std::chrono::steady_clock::now();
    phase_timings.cfl_seconds =
        std::chrono::duration<double>(cfl_end - cfl_start).count();


    // [2.3] Working level sets
    std::vector<std::vector<double>> phi_list_work = ctx.phi_list;
    std::vector<int> planar_phi_axis(phi_list_work.size(), -1);

    for (int k = 0; k < static_cast<int>(phi_list_work.size()); ++k) {
        planar_phi_axis[k] =
            detect_planar_level_set_axis<DIM>(phi_list_work[k], ctx);
    }

    const int planar_flow_axis =
        common_planar_level_set_axis<DIM>(planar_phi_axis);


    // [2.4] Normals
    std::vector<std::vector<std::array<double, DIM>>> normals_current(
        ctx.phi_list.size()
    );

    for (int k = 0; k < static_cast<int>(ctx.phi_list.size()); ++k) {
        normals_current[k] =
            compute_solver_normals<DIM>(
                ctx.phi_list[k],
                ctx
            );
    }

    const RGFMInterfaceData<DIM> rgfm_current =
        build_rgfm_interface_data<DIM, EOS>(
            U,
            material_id_current,
            ctx.phi_list,
            ctx.tracked_interfaces,
            normals_current,
            ctx
        );


    /*
        [2.5] Fluid update on the current interface.

        Split sweeps advance directions sequentially. Unsplit updates compute
        each directional delta from the same beginning-of-step state and then
        accumulate those deltas in one conservative update.
    */
    std::vector<Conserved<DIM>> U_stage = U;
    zero_roundoff_planar_transverse_momentum<DIM>(
        U_stage,
        planar_flow_axis
    );

    if (ctx.time_update == "unsplit") {
        const std::vector<Conserved<DIM>> U_base = U_stage;
        auto sweep = [&](int dir,
                         const std::vector<Conserved<DIM>>& U_in,
                         std::vector<Conserved<DIM>>& U_out)
        {
            sweep_direction_dispatch<DIM, EOS>(
                dir,
                U_in,
                material_id_current,
                ctx.phi_list,
                ctx.tracked_interfaces,
                normals_current,
                ctx,
                dt,
                U_out,
                nullptr
            );
        };

        auto accumulate_delta = [&](int,
                                    std::vector<Conserved<DIM>>& U_accum,
                                    const std::vector<Conserved<DIM>>& U_dir,
                                    const std::vector<Conserved<DIM>>& U_base)
        {
            #pragma omp parallel for if(Ntot > 512)
            for (int id = 0; id < Ntot; ++id) {
                U_accum[id] = U_accum[id] + (U_dir[id] - U_base[id]);
            }
        };

        auto after_accumulation = [&](std::vector<Conserved<DIM>>& U_accum)
        {
            const auto boundary_start = std::chrono::steady_clock::now();
            #pragma omp parallel for if(Ntot > 512)
            for (int id = 0; id < Ntot; ++id) {
                const int mat = material_id_current[id];
                enforce_positive_conserved<DIM, EOS>(
                    U_accum[id],
                    ctx.material_params[mat]
                );
            }

            apply_boundary_conditions<DIM, EOS>(U_accum, ctx);
            const auto boundary_end = std::chrono::steady_clock::now();
            phase_timings.boundary_seconds +=
                std::chrono::duration<double>(boundary_end - boundary_start).count();
            zero_roundoff_planar_transverse_momentum<DIM>(
                U_accum,
                planar_flow_axis
            );
        };

        const auto sweep_start = std::chrono::steady_clock::now();
        fv::advance_unsplit_directions<DIM>(
            U_base,
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
        auto sweep = [&](int dir,
                         const std::vector<Conserved<DIM>>& U_in,
                         std::vector<Conserved<DIM>>& U_out)
        {
            const RGFMInterfaceData<DIM> rgfm_stage =
                build_rgfm_interface_data<DIM, EOS>(
                    U_in,
                    material_id_current,
                    ctx.phi_list,
                    ctx.tracked_interfaces,
                    normals_current,
                    ctx
                );

            sweep_direction_dispatch<DIM, EOS>(
                dir,
                U_in,
                material_id_current,
                ctx.phi_list,
                ctx.tracked_interfaces,
                normals_current,
                ctx,
                dt,
                U_out,
                nullptr
            );
        };

        auto after_direction = [&](int, std::vector<Conserved<DIM>>& U_next)
        {
            const auto boundary_start = std::chrono::steady_clock::now();
            apply_boundary_conditions<DIM, EOS>(U_next, ctx);
            const auto boundary_end = std::chrono::steady_clock::now();
            phase_timings.boundary_seconds +=
                std::chrono::duration<double>(boundary_end - boundary_start).count();
            zero_roundoff_planar_transverse_momentum<DIM>(
                U_next,
                planar_flow_axis
            );
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

    // [2.6] Level-set advection using the current-interface extension speed
    const auto level_set_start = std::chrono::steady_clock::now();
    if (ctx.advect_level_set && !phi_list_work.empty()) {
        std::vector<std::array<double, DIM>> level_set_velocity_field;
        const bool use_vector_level_set_transport =
            ctx.level_set_advection == "flow" ||
            ctx.level_set_advection == "physical_flow" ||
            (ctx.level_set_advection == "normal_speed" && phi_list_work.size() > 1);

        if (ctx.level_set_advection == "flow") {
            level_set_velocity_field =
                build_velocity_field<DIM, EOS>(
                    rgfm_current.U_real,
                    material_id_current,
                    ctx.material_params
                );
        }
        else if (ctx.level_set_advection == "physical_flow" ||
                 (ctx.level_set_advection == "normal_speed" && phi_list_work.size() > 1)) {
            level_set_velocity_field =
                build_velocity_field<DIM, EOS>(
                    U,
                    material_id_current,
                    ctx.material_params
                );
        }

        for (int k = 0; k < static_cast<int>(phi_list_work.size()); ++k) {
            if (use_vector_level_set_transport) {
                phi_list_work[k] = advect_phi<DIM>(
                    ctx.phi_list[k],
                    level_set_velocity_field,
                    ctx.level_set_grid,
                    dt,
                    ctx.level_set_derivative_scheme
                );
            }
            else {
                phi_list_work[k] = advect_phi_normal_speed<DIM>(
                    ctx.phi_list[k],
                    rgfm_current.normal_speed_fields[k],
                    ctx.level_set_grid,
                    dt,
                    ctx.level_set_derivative_scheme
                );
            }

            project_planar_level_set<DIM>(
                phi_list_work[k],
                ctx,
                planar_phi_axis[k]
            );
        }
    }


    // [2.7] Reinitialisation
    if (ctx.reinit_enabled &&
        ((ctx.completed_steps + 1) % ctx.reinit_frequency == 0))
    {
        for (int k = 0; k < static_cast<int>(phi_list_work.size()); ++k) {

            phi_list_work[k] =
                reinitialise_solver_phi<DIM>(
                    phi_list_work[k],
                    ctx
                );

            project_planar_level_set<DIM>(
                phi_list_work[k],
                ctx,
                planar_phi_axis[k]
            );
        }
    }
    const auto level_set_end = std::chrono::steady_clock::now();
    phase_timings.level_set_seconds =
        std::chrono::duration<double>(level_set_end - level_set_start).count();


    // [2.8] Updated material map and state transfer after the interface move
    const auto transfer_start = std::chrono::steady_clock::now();
    std::vector<int> material_id_work = material_id_current;

    if (ctx.reassign_material_from_phi) {
        assign_material_ids_from_phi<DIM>(
            phi_list_work,
            ctx.tracked_interfaces,
            ctx.background_material_id,
            material_id_work,
            ctx.level_set_grid,
            ctx.level_set_component_policy
        );

        fill_small_enclosed_background_cavities<DIM>(
            material_id_work,
            ctx.tracked_interfaces,
            ctx.background_material_id,
            ctx.level_set_grid
        );

        enforce_level_set_sign_from_material_map<DIM>(
            phi_list_work,
            material_id_work,
            ctx
        );
    }

    std::vector<std::vector<Conserved<DIM>>> transfer_material_states;
    const std::vector<std::vector<Conserved<DIM>>>* transfer_material_states_ptr = nullptr;

    if (ctx.advect_level_set && !ctx.phi_list.empty()) {
        const RGFMInterfaceData<DIM> rgfm_transfer =
            build_rgfm_interface_data<DIM, EOS>(
                U_stage,
                material_id_current,
                ctx.phi_list,
                ctx.tracked_interfaces,
                normals_current,
                ctx
            );
        transfer_material_states = rgfm_transfer.material_states;
        transfer_material_states_ptr = &transfer_material_states;
    }

    std::vector<Conserved<DIM>> U_final =
        transfer_reassigned_material_states<DIM, EOS>(
            U_stage,
            material_id_current,
            material_id_work,
            ctx,
            transfer_material_states_ptr
        );

    zero_roundoff_planar_transverse_momentum<DIM>(
        U_final,
        planar_flow_axis
    );
    const auto transfer_end = std::chrono::steady_clock::now();
    phase_timings.material_transfer_seconds =
        std::chrono::duration<double>(transfer_end - transfer_start).count();

    ctx.completed_steps += 1;

    return {U_final, phi_list_work, dt, rgfm_current.diagnostic_rows, phase_timings};
}
