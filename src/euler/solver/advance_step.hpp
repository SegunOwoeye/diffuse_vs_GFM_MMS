#pragma once

#include <vector>
#include <array>
#include <stdexcept>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "src/euler/state.hpp"
#include "src/euler/eos.hpp"
#include "src/euler/eos_params.hpp"
#include "src/euler/solver/cfl.hpp"
#include "src/euler/solver/solver_context.hpp"
#include "src/euler/thermo_compute.hpp"
#include "src/euler/level_set/level_set.hpp"

// Split modules
#include "src/euler/solver/advance/geometry.hpp"
#include "src/euler/solver/advance/line_ops.hpp"
#include "src/euler/solver/advance/interface_speed.hpp"
#include "src/euler/solver/advance/sweep_core.hpp"
#include "src/euler/solver/advance/boundary.hpp"



// [0] Step Result
template<int DIM>
struct StepResult {
    std::vector<Conserved<DIM>> U_new;
    std::vector<std::vector<double>> phi_list_new;
    double dt;
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

    #pragma omp parallel for
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

    
    // [2.1] Material map
    std::vector<int> material_id_current = ctx.material_id;

    if (ctx.reassign_material_from_phi) {
        assign_material_ids_from_phi<DIM>(
            ctx.phi_list,
            ctx.tracked_interfaces,
            ctx.background_material_id,
            material_id_current,
            ctx.level_set_grid
        );
    }

    
    // [2.2] CFL timestep
    const double dt = compute_dt_cfl_materials<DIM, EOS>(
        U,
        material_id_current,
        ctx.material_params,
        ctx.dx,
        ctx.cfl,
        ctx.dt_max
    );

    
    // [2.3] Working level sets
    std::vector<std::vector<double>> phi_list_work = ctx.phi_list;

    
    // [2.4] Normals 
    std::vector<std::vector<std::array<double, DIM>>> normals_current(
        ctx.phi_list.size()
    );

    #pragma omp parallel for
    for (int k = 0; k < static_cast<int>(ctx.phi_list.size()); ++k) {
        normals_current[k] =
            compute_normals<DIM>(
                ctx.phi_list[k],
                ctx.level_set_grid
            );
    }

    
    // [2.5] Level-set advection
    if (ctx.advect_level_set && !phi_list_work.empty()) {

        const auto vel = build_velocity_field<DIM, EOS>(
            U,
            material_id_current,
            ctx.material_params
        );

        const auto Vn_fields = build_interface_normal_speed_fields<DIM, EOS>(
                U,
                material_id_current,
                vel,
                ctx.phi_list,
                ctx.tracked_interfaces,
                normals_current,
                ctx,
                dt
            );

        #pragma omp parallel for
        for (int k = 0; k < static_cast<int>(phi_list_work.size()); ++k) {
            const auto vel_ls = build_level_set_transport_velocity<DIM>(
                Vn_fields[k],
                normals_current[k]
            );

            phi_list_work[k] = advect_phi<DIM>(
                    ctx.phi_list[k],
                    vel_ls,
                    ctx.level_set_grid,
                    dt
                );
        }
    }

    
    // [2.6] Reinitialisation
    if (ctx.reinit_enabled &&
        ((ctx.completed_steps + 1) % ctx.reinit_frequency == 0))
    {
        #pragma omp parallel for
        for (int k = 0; k < static_cast<int>(phi_list_work.size()); ++k) {

            phi_list_work[k] =
                reinitialise_phi<DIM>(
                    phi_list_work[k],
                    ctx.level_set_grid,
                    ctx.reinit_iterations
                );
        }
    }

    
    // [2.7] Updated material map
    std::vector<int> material_id_work = material_id_current;

    if (ctx.reassign_material_from_phi) {
        assign_material_ids_from_phi<DIM>(
            phi_list_work,
            ctx.tracked_interfaces,
            ctx.background_material_id,
            material_id_work,
            ctx.level_set_grid
        );

        rebuild_tracked_level_sets_from_material_map_1d<DIM>(
            phi_list_work,
            material_id_work,
            ctx
        );
    }

    
    // [2.8] Normals 
    std::vector<std::vector<std::array<double, DIM>>> normals_list(
        phi_list_work.size()
    );

    #pragma omp parallel for
    for (int k = 0; k < static_cast<int>(phi_list_work.size()); ++k) {
        normals_list[k] =
            compute_normals<DIM>(
                phi_list_work[k],
                ctx.level_set_grid
            );
    }

    
    // [2.9] Directional splitting
    std::vector<Conserved<DIM>> U_stage = U;

    for (int dir = 0; dir < DIM; ++dir) {

        std::vector<Conserved<DIM>> U_next = U_stage;

        sweep_direction_dispatch<DIM, EOS>(
            dir,
            U_stage,
            material_id_work,
            phi_list_work,
            ctx.tracked_interfaces,
            normals_list,
            ctx,
            dt,
            U_next
        );

        apply_boundary_conditions<DIM>(U_next, ctx);

        U_stage.swap(U_next);
    }

    ctx.completed_steps += 1;

    return {U_stage, phi_list_work, dt};
}




